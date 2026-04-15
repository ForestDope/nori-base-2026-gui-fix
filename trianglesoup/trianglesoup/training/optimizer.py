"""Training loop orchestration for triangle soup optimization."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import drjit as dr
import mitsuba as mi
import numpy as np

from trianglesoup.data_io import CameraDataset, CameraView, load_dataset, select_progress_views
from trianglesoup.ray_loader import Rayloader
from trianglesoup.rendering import losses
from trianglesoup.rendering.integrator import register as register_integrator
from .config import RuntimeConfig, TriangleSoupConfig
from .variables import TriangleSoupVariables, create_random_triangles


@dataclass
class TrainingMetrics:
    iteration: int
    loss: float
    learning_rate: float
    num_triangles: int
    pruned_triangles: int = 0
    added_triangles: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class TriangleStatistics:
    avg_weight: np.ndarray
    visible_views: np.ndarray
    pixel_hits: np.ndarray


class TriangleSoupTrainer:
    """Main orchestrator for triangle soup optimization."""

    def __init__(
        self,
        config: TriangleSoupConfig,
        runtime: RuntimeConfig,
        train_dataset: Optional[CameraDataset] = None,
        test_dataset: Optional[CameraDataset] = None,
    ) -> None:
        register_integrator()
        self.config = config
        self.runtime = runtime
        self.iteration = 0
        self.metrics: List[TrainingMetrics] = []
        self.current_learning_rate = config.learning_rate
        self.topology_rng = np.random.default_rng(0)
        self.densification_events = 0

        if train_dataset is None or test_dataset is None:
            loaded_train_dataset, loaded_test_dataset = load_dataset(
                config.dataset_path, jitter_pixels=config.jitter_pixels
            )
            if train_dataset is None:
                train_dataset = loaded_train_dataset
            if test_dataset is None:
                test_dataset = loaded_test_dataset

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        if config.train_view_indices:
            indices = config.train_view_indices
            for idx in indices:
                if idx >= len(self.train_dataset.views):
                    raise ValueError(
                        f"--train-view {idx} is out of range "
                        f"(train dataset has {len(self.train_dataset.views)} views)"
                    )
            selected = [self.train_dataset.views[i] for i in indices]
            self.train_dataset = CameraDataset(views=selected)
            test_selected = [
                self.test_dataset.views[i]
                for i in indices
                if i < len(self.test_dataset.views)
            ]
            if not test_selected:
                test_selected = [self.train_dataset.views[0]]
            self.test_dataset = CameraDataset(views=test_selected)
            names = ", ".join(v.name for v in self.train_dataset.views)
            print(f"[train-views] Overfitting {len(indices)} view(s): {names}")
        if not self.train_dataset.views:
            raise ValueError("Training dataset is empty; need at least one view")
        self.progress_views = list(
            select_progress_views(self.train_dataset, config.progress_view_count)
        )

        if config.initial_state_path:
            state_path = Path(config.initial_state_path)
            state = np.load(state_path)
            vertices_np = state["vertex_positions"].astype(np.float32)
            faces_np = state["faces"].astype(np.uint32)
            # Accept either the new "colors" key or the legacy "sh_params"
            # key for backwards compatibility with older state files.
            if "colors" in state:
                initial_colors = state["colors"].astype(np.float32)
            elif "sh_params" in state:
                initial_colors = state["sh_params"].astype(np.float32)
            else:
                initial_colors = None
            initial_occ = state["occupancy"].astype(np.float32) if "occupancy" in state else None
        elif config.mesh_init is not None:
            from trianglesoup.mesh_init import mesh_init
            vertices_np, faces_np, mesh_colors = mesh_init(
                config.mesh_init,
                target_triangles=config.mesh_init_triangles,
            )
            initial_colors = mesh_colors
            initial_occ = None
        else:
            vertices_np, faces_np = create_random_triangles(
                config.n_triangles_per_unit_cube,
                scale=config.scene_scale,
                triangle_base_size_factor=config.triangle_base_size_factor,
            )
            initial_colors = None
            initial_occ = None

        self._initialize_optimization_state(
            vertices_np,
            faces_np,
            initial_colors,
            initial_occ,
        )

        target_ref = self.train_dataset.views[0].target
        total_pixels = (
            target_ref.shape[0] * target_ref.shape[1] * len(self.train_dataset.views)
        )
        pixels_per_batch = min(self.config.mini_batch_pixels, total_pixels)

        self.ray_loader = Rayloader(
            sensors=[view.sensor for view in self.train_dataset.views],
            target_images=[view.target for view in self.train_dataset.views],
            pixels_per_batch=pixels_per_batch,
            seed=0,
            regular_reshuffle=False,
            tile_size=4,
            jitter_pixels=config.jitter_pixels,
        )
        self._build_frozen_step()

    def _initialize_optimization_state(
        self,
        vertices_np: np.ndarray,
        faces_np: np.ndarray,
        initial_colors: np.ndarray | None,
        initial_occ: np.ndarray | None,
        initial_sigma: np.ndarray | None = None,
    ) -> None:
        self.scene, self.integrator = self._build_scene(vertices_np, faces_np)

        self.variables = TriangleSoupVariables(
            config=self.config,
            integrator=self.integrator,
            vertices=vertices_np,
            faces=faces_np,
            initial_colors=initial_colors,
            initial_occupancy=initial_occ,
        )
        if initial_sigma is not None:
            self.variables.triangle_model.sigma = mi.Float(initial_sigma.astype(np.float32))

        self.scene_params = mi.traverse(self.scene)
        self.scene_params.keep([
            "soup_mesh.vertex_positions",
            "soup_mesh.faces",
            "integrator.colors",
            "integrator.occupancy",
            "integrator.sigma",
        ])
        dr.enable_grad(self.scene_params["soup_mesh.vertex_positions"])
        dr.enable_grad(self.scene_params["integrator.colors"])
        dr.enable_grad(self.scene_params["integrator.occupancy"])
        dr.enable_grad(self.scene_params["integrator.sigma"])

        self.optimizer = mi.ad.Adam(lr=self.current_learning_rate)
        self.variables.register(self.optimizer)
        self.variables.apply_learning_rate_scales(
            self.optimizer, self.current_learning_rate
        )
        self.variables.update_scene(self.scene_params, self.optimizer)

        if hasattr(self, "ray_loader"):
            self._build_frozen_step()

    def _build_scene(self, vertices_np: np.ndarray, faces_np: np.ndarray):
        vertex_count = vertices_np.shape[0]
        face_count = faces_np.shape[0]

        mesh = mi.Mesh(
            "soup_mesh",
            int(vertex_count),
            int(face_count),
        )
        mesh_params = mi.traverse(mesh)
        mesh_params['vertex_positions'] = mi.Float(vertices_np.reshape(-1))
        mesh_params['faces'] = mi.UInt32(faces_np.reshape(-1))
        mesh_params.update()

        integrator_dict = {
            "type": "soup_emissive",
        }
        mesh.set_bsdf(mi.load_dict({"type": "null"}))
        scene_dict = {
            "type": "scene",
            "integrator": integrator_dict,
            "soup_mesh": mesh,
        }
        scene = mi.load_dict(scene_dict)
        integrator = scene.integrator()
        if self.config.background_radiance is not None:
            integrator.background_color = mi.Color3f(self.config.background_radiance)
        return scene, integrator

    def _schedule_multiplier(self, iteration: int) -> int:
        if not self.config.spp_multiplier_schedule:
            return 1
        multiplier = 1
        total = 0
        for factor, steps in self.config.spp_multiplier_schedule:
            total += steps
            multiplier = int(factor)
            if iteration < total:
                break
        return max(1, multiplier)

    def _update_learning_rate(self, iteration: int) -> None:
        if self.config.learning_rate_plateau_patience > 0:
            self._update_learning_rate_plateau(iteration)
        elif (
            self.config.learning_rate_decay_step > 0
            and iteration > 0
            and iteration % self.config.learning_rate_decay_step == 0
        ):
            self._apply_lr_decay()

    def _apply_lr_decay(self) -> None:
        self.current_learning_rate = max(
            self.current_learning_rate * self.config.learning_rate_decay_rate,
            self.config.learning_rate_min,
        )
        self.optimizer.set_learning_rate(self.current_learning_rate)
        self.variables.apply_learning_rate_scales(
            self.optimizer, self.current_learning_rate
        )

    def _update_learning_rate_plateau(self, iteration: int) -> None:
        """Reduce LR when loss stops improving over a window."""
        patience = self.config.learning_rate_plateau_patience
        window = self.config.learning_rate_plateau_window
        if iteration < window or iteration % window != 0:
            return

        # Compare average loss of current window vs previous window.
        recent = self.metrics[-window:]
        if len(recent) < window:
            return
        previous = self.metrics[-(2 * window):-window]
        if len(previous) < window:
            return

        avg_recent = sum(m.loss for m in recent) / len(recent)
        avg_previous = sum(m.loss for m in previous) / len(previous)

        if not hasattr(self, "_plateau_counter"):
            self._plateau_counter = 0

        # No improvement: loss didn't decrease by at least 1%
        if avg_recent >= avg_previous * 0.99:
            self._plateau_counter += 1
        else:
            self._plateau_counter = 0

        if self._plateau_counter >= patience:
            self._plateau_counter = 0
            self._apply_lr_decay()

    def _topology_updates_supported(self) -> bool:
        return hasattr(self.variables.triangle_model, "colors")

    def _collect_triangle_statistics(self) -> TriangleStatistics:
        n_triangles = self.variables.n_triangles
        avg_weight = np.zeros(n_triangles, dtype=np.float32)
        visible_views = np.zeros(n_triangles, dtype=np.int32)
        pixel_hits = np.zeros(n_triangles, dtype=np.int32)

        for view in self.train_dataset.views:
            self.integrator.begin_triangle_statistics(n_triangles)
            render = mi.render(
                self.scene,
                params=self.scene_params,
                sensor=view.sensor,
                spp=max(1, self.config.pruning_stats_spp),
                integrator=self.integrator,
            )
            dr.eval(render)
            view_avg_weight, view_pixel_hits = self.integrator.end_triangle_statistics()
            avg_weight = np.maximum(avg_weight, view_avg_weight.astype(np.float32, copy=False))
            pixel_hits += view_pixel_hits.astype(np.int32, copy=False)
            visible_views += (
                view_pixel_hits > self.config.pruning_min_pixels_per_view
            ).astype(np.int32, copy=False)

        return TriangleStatistics(
            avg_weight=avg_weight,
            visible_views=visible_views,
            pixel_hits=pixel_hits,
        )

    def _faces_for_sequential_vertices(self, n_triangles: int) -> np.ndarray:
        return np.arange(n_triangles * 3, dtype=np.uint32).reshape(n_triangles, 3)

    def _state_triangles(self, state: dict[str, np.ndarray | None]) -> np.ndarray:
        vertices = state["vertices"]
        faces = state["faces"]
        if vertices is None or faces is None:
            raise ValueError("Triangle state is missing geometry")
        return vertices[faces]

    def _filter_state(
        self,
        state: dict[str, np.ndarray | None],
        keep_mask: np.ndarray,
    ) -> dict[str, np.ndarray | None]:
        triangles = self._state_triangles(state)[keep_mask]
        n_triangles = triangles.shape[0]
        return {
            "vertices": triangles.reshape(-1, 3).astype(np.float32),
            "faces": self._faces_for_sequential_vertices(n_triangles),
            "colors": state["colors"][keep_mask].astype(np.float32),
            "occupancy": state["occupancy"][keep_mask].astype(np.float32),
            "sigma": state["sigma"][keep_mask].astype(np.float32),
        }

    def _triangle_areas(self, triangles: np.ndarray) -> np.ndarray:
        edge01 = triangles[:, 1] - triangles[:, 0]
        edge02 = triangles[:, 2] - triangles[:, 0]
        return 0.5 * np.linalg.norm(np.cross(edge01, edge02), axis=1)

    def _triangle_aspect_ratios(self, triangles: np.ndarray) -> np.ndarray:
        """Ratio of longest edge to shortest edge for each triangle.

        Thin slivers have high aspect ratios; an equilateral triangle is 1.
        """
        e01 = np.linalg.norm(triangles[:, 1] - triangles[:, 0], axis=1)
        e02 = np.linalg.norm(triangles[:, 2] - triangles[:, 0], axis=1)
        e12 = np.linalg.norm(triangles[:, 2] - triangles[:, 1], axis=1)
        edges = np.stack([e01, e02, e12], axis=-1)
        shortest = np.maximum(edges.min(axis=-1), 1e-12)
        return edges.max(axis=-1) / shortest

    def _clone_triangle_with_noise(self, triangle: np.ndarray) -> np.ndarray:
        edge01 = triangle[1] - triangle[0]
        edge02 = triangle[2] - triangle[0]
        normal = np.cross(edge01, edge02)
        normal_norm = np.linalg.norm(normal)
        edge01_norm = np.linalg.norm(edge01)
        if normal_norm < 1e-8 or edge01_norm < 1e-8:
            scale = self.config.densification_clone_noise_scale * max(
                self.config.triangle_base_size_factor,
                1e-3,
            )
            offset = self.topology_rng.normal(size=(1, 3)).astype(np.float32) * scale
            return triangle + offset

        tangent = edge01 / edge01_norm
        bitangent = np.cross(normal / normal_norm, tangent)
        bitangent_norm = np.linalg.norm(bitangent)
        if bitangent_norm < 1e-8:
            return triangle.copy()
        bitangent = bitangent / bitangent_norm

        mean_edge = (
            np.linalg.norm(triangle[1] - triangle[0])
            + np.linalg.norm(triangle[2] - triangle[1])
            + np.linalg.norm(triangle[0] - triangle[2])
        ) / 3.0
        scale = max(mean_edge, 1e-4) * self.config.densification_clone_noise_scale
        offset_2d = self.topology_rng.normal(size=2).astype(np.float32) * scale
        offset = tangent * offset_2d[0] + bitangent * offset_2d[1]
        return triangle + offset[None, :]

    def _split_triangle(self, triangle: np.ndarray) -> list[np.ndarray]:
        v0, v1, v2 = triangle
        m01 = 0.5 * (v0 + v1)
        m12 = 0.5 * (v1 + v2)
        m20 = 0.5 * (v2 + v0)
        return [
            np.stack([v0, m01, m20], axis=0),
            np.stack([m01, v1, m12], axis=0),
            np.stack([m20, m12, v2], axis=0),
            np.stack([m01, m12, m20], axis=0),
        ]

    def _split_triangle_colors(self, color_entry: np.ndarray) -> list[np.ndarray]:
        return [color_entry.copy() for _ in range(4)]

    def _extract_adam_state(self) -> dict[str, tuple] | None:
        """Extract Adam optimizer (m, v, t) for each parameter key."""
        if not hasattr(self, "optimizer"):
            return None
        adam_state: dict[str, tuple] = {}
        for key in [
            "soup_mesh.vertex_positions",
            "triangle.colors",
            "triangle.occupancy",
            "triangle.sigma",
        ]:
            if key not in self.optimizer.state:
                continue
            # State layout: (value, bool, None, (t, m, v, None))
            inner = self.optimizer.state[key][3]
            t, m, v = inner[0], inner[1], inner[2]
            adam_state[key] = (t, m, v)
        return adam_state

    def _apply_adam_state(
        self,
        old_adam: dict[str, tuple] | None,
        parent_index: np.ndarray,
    ) -> None:
        """Remap saved Adam (m, v) into the new optimizer using parent_index.

        ``parent_index[i]`` is the old-triangle index that new triangle
        ``i`` descends from.  For vertex positions the index is expanded
        3x (3 sequential vertices per triangle).
        """
        if old_adam is None:
            return

        idx_tri = mi.UInt32(parent_index.astype(np.uint32))
        # Vertex-level: each triangle owns 3 consecutive vertices.
        vtx_parent = np.repeat(parent_index * 3, 3) + np.tile([0, 1, 2], len(parent_index))
        # Positions are flat (x,y,z per vertex) so expand 3x further.
        pos_parent = np.repeat(vtx_parent * 3, 3) + np.tile([0, 1, 2], len(vtx_parent))
        idx_pos = mi.UInt32(pos_parent.astype(np.uint32))

        for key in [
            "soup_mesh.vertex_positions",
            "triangle.colors",
            "triangle.occupancy",
            "triangle.sigma",
        ]:
            if key not in old_adam or key not in self.optimizer.state:
                continue
            old_t, old_m, old_v = old_adam[key]
            idx = idx_pos if key == "soup_mesh.vertex_positions" else idx_tri
            m_type = type(old_m)
            v_type = type(old_v)
            new_m = dr.gather(m_type, old_m, idx)
            new_v = dr.gather(v_type, old_v, idx)
            inner = self.optimizer.state[key][3]
            # Mutate the inner tuple in-place: (t, m, v, None)
            self.optimizer.state[key] = (
                self.optimizer.state[key][0],
                self.optimizer.state[key][1],
                self.optimizer.state[key][2],
                (old_t, new_m, new_v, inner[3]),
            )

    def _rebuild_from_state(
        self,
        state: dict[str, np.ndarray | None],
        parent_index: np.ndarray | None = None,
    ) -> None:
        old_adam = self._extract_adam_state() if parent_index is not None else None
        self._initialize_optimization_state(
            state["vertices"],
            state["faces"],
            state["colors"],
            state["occupancy"],
            state.get("sigma"),
        )
        if parent_index is not None:
            self._apply_adam_state(old_adam, parent_index)

    def _apply_pruning(self, stats: TriangleStatistics) -> int:
        if self.variables.n_triangles <= 1:
            return 0

        # Cap the visibility requirement to the actual number of training
        # views — otherwise single- or two-view training prunes every
        # triangle because none can appear in ≥ min_view_count views.
        min_views = min(
            self.config.pruning_min_view_count, len(self.train_dataset.views)
        )
        keep_mask = (
            (stats.avg_weight >= self.config.pruning_occupancy_threshold)
            & (stats.visible_views >= min_views)
        )

        # Prune degenerate triangles: remove a triangle if any of the
        # configured geometry checks fail (too small or too elongated).
        if (
            self.config.pruning_min_area > 0.0
            or self.config.pruning_max_aspect_ratio > 0.0
        ):
            state = self.variables.export_state()
            triangles = self._state_triangles(state)
            degenerate = np.zeros_like(keep_mask)
            if self.config.pruning_min_area > 0.0:
                areas = self._triangle_areas(triangles)
                degenerate |= areas < self.config.pruning_min_area
            if self.config.pruning_max_aspect_ratio > 0.0:
                aspects = self._triangle_aspect_ratios(triangles)
                degenerate |= aspects > self.config.pruning_max_aspect_ratio
            keep_mask &= ~degenerate

        if np.all(keep_mask):
            return 0
        if not np.any(keep_mask):
            keep_mask[int(np.argmax(stats.avg_weight))] = True

        pruned = int((~keep_mask).sum())
        parent_index = np.where(keep_mask)[0]
        state = self.variables.export_state()
        self._rebuild_from_state(self._filter_state(state, keep_mask), parent_index)
        return pruned

    def _densification_weights(self, state: dict[str, np.ndarray | None]) -> np.ndarray:
        """Probabilistic selection weights, matching the paper's scheme.

        The Triangle Splatting paper (§3.2) alternates between two
        distributions across densification events:
          - **opacity (o)**: pick visible, contributing triangles.
          - **inverse sharpness (1/σ)**: pick solid triangles. Because
            the window function is bounded by the triangle's projected
            geometry, low σ marks regions where each triangle carries a
            lot of reconstruction weight and where more resolution is
            needed.
        """
        n = self.variables.n_triangles
        use_sigma = (self.densification_events % 2) == 1

        if use_sigma:
            sigma = state.get("sigma")
            if sigma is not None:
                # Clamp to avoid division by zero; favour low-σ (solid) triangles.
                weights = 1.0 / np.clip(sigma.astype(np.float32), 1e-2, None)
            else:
                weights = np.ones(n, dtype=np.float32)
        else:
            occupancy = state["occupancy"]
            if occupancy is not None:
                weights = np.clip(occupancy.astype(np.float32), 0.0, None)
            else:
                weights = np.ones(n, dtype=np.float32)

        weight_sum = float(weights.sum())
        if not np.isfinite(weight_sum) or weight_sum <= 0.0:
            return np.full(n, 1.0 / n, dtype=np.float32)
        return weights / weight_sum

    def _apply_densification(self) -> int:
        n_triangles = self.variables.n_triangles
        if n_triangles == 0:
            return 0
        if self.config.max_triangles > 0 and n_triangles >= self.config.max_triangles:
            return 0

        state = self.variables.export_state()
        triangles = self._state_triangles(state)
        colors = state["colors"]
        occupancy = state["occupancy"]
        sigma = state["sigma"]
        areas = self._triangle_areas(triangles)

        selection_budget = int(np.ceil(n_triangles * self.config.densification_budget_ratio))
        selection_budget = max(selection_budget, 1)
        selection_budget = min(selection_budget, n_triangles)
        probabilities = self._densification_weights(state)
        selected = self.topology_rng.choice(
            n_triangles,
            size=selection_budget,
            replace=False,
            p=probabilities,
        )
        selected_mask = np.zeros(n_triangles, dtype=bool)
        selected_mask[selected] = True

        triangle_out: list[np.ndarray] = []
        color_out: list[np.ndarray] = []
        occupancy_out: list[float] = []
        sigma_out: list[float] = []
        parent_index_out: list[int] = []
        added_triangles = 0
        hard_cap = self.config.densification_max_new_triangles

        for index in range(n_triangles):
            triangle = triangles[index]
            color_entry = colors[index]
            occ_value = float(occupancy[index])
            sig_value = float(sigma[index])

            if not selected_mask[index]:
                triangle_out.append(triangle)
                color_out.append(color_entry)
                occupancy_out.append(occ_value)
                sigma_out.append(sig_value)
                parent_index_out.append(index)
                continue

            if areas[index] < self.config.pruning_area_threshold:
                triangle_out.append(triangle)
                color_out.append(color_entry)
                triangle_out.append(self._clone_triangle_with_noise(triangle))
                color_out.append(color_entry.copy())
                occupancy_out.extend([occ_value, occ_value])
                sigma_out.extend([sig_value, sig_value])
                parent_index_out.extend([index, index])
                added_triangles += 1
            else:
                children = self._split_triangle(triangle)
                child_colors = self._split_triangle_colors(color_entry)
                for child_triangle, child_color in zip(children, child_colors):
                    triangle_out.append(child_triangle)
                    color_out.append(child_color)
                    occupancy_out.append(occ_value)
                    sigma_out.append(sig_value)
                    parent_index_out.append(index)
                added_triangles += 3

            if hard_cap > 0 and added_triangles >= hard_cap:
                selected_mask[index + 1 :] = False

        if added_triangles <= 0:
            return 0

        parent_index = np.asarray(parent_index_out, dtype=np.int64)
        new_triangles = np.stack(triangle_out, axis=0).astype(np.float32)
        new_state: dict[str, np.ndarray | None] = {
            "vertices": new_triangles.reshape(-1, 3),
            "faces": self._faces_for_sequential_vertices(new_triangles.shape[0]),
            "colors": np.stack(color_out, axis=0).astype(np.float32),
            "occupancy": np.asarray(occupancy_out, dtype=np.float32),
            "sigma": np.asarray(sigma_out, dtype=np.float32),
        }
        self._rebuild_from_state(new_state, parent_index)
        self.densification_events += 1
        return int(new_triangles.shape[0] - n_triangles)

    def _maybe_adapt_topology(self) -> tuple[int, int]:
        if not self._topology_updates_supported():
            return 0, 0

        next_iteration = self.iteration + 1
        pruned = 0
        added = 0

        if (
            self.config.pruning_step > 0
            and next_iteration % self.config.pruning_step == 0
        ):
            stats = self._collect_triangle_statistics()
            pruned = self._apply_pruning(stats)

        if (
            self.config.densification_step > 0
            and next_iteration % self.config.densification_step == 0
        ):
            added = self._apply_densification()

        return pruned, added

    def _build_frozen_step(self) -> None:
        """Build (or rebuild) the ``dr.freeze``-wrapped training kernels.

        Records the drjit computation graph for render → photometric
        loss → backward once, then replays it on every subsequent call
        with different DrJit array values. Topology changes rebuild
        the scene and must call this method again.

        The integrator reads its per-triangle state (colors, occupancy,
        sigma) from plain Python attributes on ``triangle_model``.
        ``dr.freeze`` can't discover those through the ``integrator``
        argument, so we thread them in as explicit tensor arguments and
        reattach them to the integrator inside the kernel before
        rendering. The triangle vertex positions are passed both through
        ``scene_params[PARAM_KEY_POS]`` (for the BVH). The window function
        reads vertex positions directly from the mesh via ``MeshPtr``.
        """
        pixel_batch_multiplier = self.ray_loader.pixel_batch_multiplier
        spp = self.config.primal_spp
        size_weight = self.config.loss_size_weight
        has_size_reg = size_weight > 0
        occ_weight = self.config.loss_occupancy_weight
        has_occ_reg = occ_weight > 0
        n_tris = self.variables.n_triangles

        def _bind_integrator_state(integrator, colors, occupancy, sigma):
            integrator.triangle_model.colors = colors
            integrator.triangle_model.occupancy = occupancy
            integrator.triangle_model.sigma = sigma

        def _regularizers(vertex_positions, occupancy):
            reg = mi.Float(0.0)
            if has_size_reg:
                reg = reg + size_weight * losses.size_regularization(
                    vertex_positions, n_tris
                )
            if has_occ_reg:
                reg = reg + occ_weight * losses.occupancy_regularization(occupancy)
            return reg

        def _step_kernel(
            scene, scene_params, sensor, integrator,
            colors, occupancy, sigma, vertex_positions, target_batch,
        ):
            _bind_integrator_state(integrator, colors, occupancy, sigma)
            rendered = mi.render(
                scene,
                params=scene_params,
                sensor=sensor,
                spp=spp,
                integrator=integrator,
            )
            photo = losses.photometric_loss(rendered, target_batch)
            loss = photo * pixel_batch_multiplier + _regularizers(vertex_positions, occupancy)
            dr.backward(loss)
            return photo

        def _full_view_kernel(
            scene, scene_params, sensor, integrator,
            colors, occupancy, sigma, vertex_positions, target,
        ):
            _bind_integrator_state(integrator, colors, occupancy, sigma)
            rendered = mi.render(
                scene,
                params=scene_params,
                sensor=sensor,
                spp=spp,
                integrator=integrator,
            )
            photo = losses.photometric_loss(rendered, target)
            loss = photo + _regularizers(vertex_positions, occupancy)
            dr.backward(loss)
            return photo

        self._frozen_step_fn = dr.freeze(_step_kernel)
        self._frozen_full_view_fn = dr.freeze(_full_view_kernel)

    def _current_integrator_state(self):
        """Fetch the current drjit variables the integrator reads.

        Returns ``(colors, occupancy, sigma, vertex_positions_flat)``.
        The vertex positions are the optimiser's source-of-truth buffer;
        the regularisers need it as a separate argument.
        """
        return (
            self.variables.triangle_model.colors,
            self.variables.triangle_model.occupancy,
            self.variables.triangle_model.sigma,
            self.variables.vertex_positions_flat,
        )

    def _step_unfrozen(self, target_batch, flat_sensor, spp):
        """Non-frozen training step: render -> loss -> backward."""
        rendered = mi.render(
            self.scene,
            params=self.scene_params,
            sensor=flat_sensor,
            seed=self.iteration,
            spp=spp,
            integrator=self.integrator,
        )
        photo = losses.photometric_loss(rendered, target_batch)
        loss = photo * self.ray_loader.pixel_batch_multiplier

        if self.config.loss_occupancy_weight > 0:
            loss = loss + self.config.loss_occupancy_weight * losses.occupancy_regularization(
                self.variables.triangle_model.occupancy
            )
        if self.config.loss_size_weight > 0:
            loss = loss + self.config.loss_size_weight * losses.size_regularization(
                self.variables.vertex_positions_flat,
                self.variables.n_triangles,
            )

        dr.backward(loss)
        return photo

    def _step_full_view(self) -> mi.Float:
        """Render one full training view, compute L1 loss, and backprop.

        Cycles through views round-robin. Uses the dr.freeze-wrapped
        kernel so the render/loss/backward graph is recorded once and
        replayed each iteration.
        """
        view_idx = self.iteration % len(self.train_dataset.views)
        view = self.train_dataset.views[view_idx]
        target = mi.TensorXf(view.target)
        colors, occupancy, sigma, vertex_positions = self._current_integrator_state()

        return self._frozen_full_view_fn(
            self.scene,
            self.scene_params,
            view.sensor,
            self.integrator,
            colors,
            occupancy,
            sigma,
            vertex_positions,
            target,
        )

    def step(self) -> TrainingMetrics:
        spp_multiplier = self._schedule_multiplier(self.iteration)
        spp = self.config.primal_spp * spp_multiplier

        if self.config.full_view_training:
            photo = self._step_full_view()
        else:
            target_batch, flat_sensor = self.ray_loader.next()
            if spp_multiplier == 1:
                colors, occupancy, sigma, vertex_positions = (
                    self._current_integrator_state()
                )
                photo = self._frozen_step_fn(
                    self.scene,
                    self.scene_params,
                    flat_sensor,
                    self.integrator,
                    colors,
                    occupancy,
                    sigma,
                    vertex_positions,
                    target_batch,
                )
            else:
                photo = self._step_unfrozen(target_batch, flat_sensor, spp)

        self.optimizer.step()
        self.variables.clamp(self.optimizer)
        self.variables.update_scene(self.scene_params, self.optimizer)
        pruned_triangles, added_triangles = self._maybe_adapt_topology()

        photo_detached = dr.detach(photo)
        loss_value = float(np.asarray(photo_detached).flat[0])

        metric = TrainingMetrics(
            iteration=self.iteration,
            loss=loss_value,
            learning_rate=self.current_learning_rate,
            num_triangles=self.variables.n_triangles,
            pruned_triangles=pruned_triangles,
            added_triangles=added_triangles,
        )
        self.metrics.append(metric)

        self.iteration += 1
        self._update_learning_rate(self.iteration)
        return metric

    def render_dataset(
        self,
        output_dir: Path,
        dataset_name: str,
        views: list[CameraView],
        spp: int,
        width: int | None = None,
        height: int | None = None,
        scale: int = 1,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for idx, view in enumerate(views):
            render_width = width
            render_height = height
            base_width = view.image_width
            base_height = view.image_height
            if scale != 1 and render_width is None and render_height is None:
                if base_width is not None and base_height is not None:
                    render_width = int(base_width * scale)
                    render_height = int(base_height * scale)
            elif base_width is not None and base_height is not None:
                if render_width is None and render_height is None:
                    render_width = base_width
                    render_height = base_height
                elif render_width is None:
                    render_width = int(base_width * render_height / base_height)
                elif render_height is None:
                    render_height = int(base_height * render_width / base_width)

            render_sensor = view.sensor
            if (render_width is not None) or (render_height is not None):
                if view.sensor_spec is not None:
                    spec = dict(view.sensor_spec)
                    film = dict(spec.get("film", {}))
                    if render_width is not None:
                        film["width"] = int(render_width)
                    if render_height is not None:
                        film["height"] = int(render_height)
                    spec["film"] = film
                    render_sensor = mi.load_dict(spec)
                else:
                    print(
                        f"[warn] No sensor spec for {view.name}; using existing resolution"
                    )

            render = mi.render(
                self.scene,
                params=self.scene_params,
                sensor=render_sensor,
                spp=spp,
                integrator=self.integrator,
            )
            dr.eval(render)
            safe_name = "".join(
                c if c.isalnum() or c in {"_", "-", ".", "+"} else "_"
                for c in view.name
            )
            filename = f"{dataset_name}_{idx:04d}_{safe_name}.exr"
            mi.Bitmap(render).write(str(output_dir / filename))

    def run(self) -> None:
        for _ in range(self.config.iterations):
            metric = self.step()
            if (
                self.runtime.log_interval > 0
                and metric.iteration % self.runtime.log_interval == 0
            ):
                extra = ""
                if metric.pruned_triangles or metric.added_triangles:
                    extra = (
                        f" prune=-{metric.pruned_triangles}"
                        f" densify=+{metric.added_triangles}"
                    )
                print(
                    f"[{metric.iteration:05d}] loss={metric.loss:.6f} "
                    f"lr={metric.learning_rate:.4e} tris={metric.num_triangles}{extra}"
                )
