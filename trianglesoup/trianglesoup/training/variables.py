"""Variable management for triangle soup optimization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import drjit as dr
import mitsuba as mi

from trianglesoup.rendering import triangle_model


@dataclass
class VariableGroup:
    key: str
    clamp_min: float | None
    clamp_max: float | None
    lr_scale: float


def create_random_triangles(
    n_triangles_per_unit_cube: int,
    scale: float = 1.0,
    triangle_base_size_factor: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """Create random equilateral triangles distributed in a cube."""
    import numpy as np

    num_triangles = int(n_triangles_per_unit_cube * (2 * scale) ** 3)
    if num_triangles <= 0:
        raise ValueError("Number of triangles must be positive")

    centers = (np.random.rand(num_triangles, 3) - 0.5) * 2 * scale
    triangle_side_length = triangle_base_size_factor * scale
    radius = triangle_side_length / np.sqrt(3.0)
    v0 = np.array([radius, 0.0, 0.0])
    v1 = np.array([
        radius * np.cos(2 * np.pi / 3.0),
        radius * np.sin(2 * np.pi / 3.0),
        0.0,
    ])
    v2 = np.array([
        radius * np.cos(4 * np.pi / 3.0),
        radius * np.sin(4 * np.pi / 3.0),
        0.0,
    ])
    canonical = np.stack([v0, v1, v2], axis=0)

    angles = np.random.uniform(0, 2 * np.pi, num_triangles)
    axes = np.random.randn(num_triangles, 3)
    norms = np.linalg.norm(axes, axis=1, keepdims=True)
    axes[norms[:, 0] < 1e-9] = np.array([1.0, 0.0, 0.0])
    norms[norms[:, 0] < 1e-9, 0] = 1.0
    axes = axes / norms

    k = np.zeros((num_triangles, 3, 3))
    k[:, 0, 1] = -axes[:, 2]
    k[:, 0, 2] = axes[:, 1]
    k[:, 1, 0] = axes[:, 2]
    k[:, 1, 2] = -axes[:, 0]
    k[:, 2, 0] = -axes[:, 1]
    k[:, 2, 1] = axes[:, 0]

    sin_a = np.sin(angles)[:, None, None]
    cos_a = np.cos(angles)[:, None, None]
    eye = np.repeat(np.eye(3)[None, ...], num_triangles, axis=0)
    rot = eye + sin_a * k + (1 - cos_a) * (k @ k)

    rotated = canonical[None, :, :] @ rot.transpose(0, 2, 1)
    translated = rotated + centers[:, None, :]
    vertices = translated.reshape(num_triangles * 3, 3).astype(np.float32)
    faces = np.arange(num_triangles * 3, dtype=np.uint32).reshape(num_triangles, 3)
    return vertices, faces


class TriangleSoupVariables:
    """Manage differentiable parameters for a triangle soup."""

    def __init__(
        self,
        config,
        integrator,
        vertices: np.ndarray,
        faces: np.ndarray,
        initial_colors: np.ndarray | None = None,
        initial_occupancy: np.ndarray | None = None,
    ) -> None:
        self.config = config
        self.integrator = integrator

        self.vertices_np = vertices.astype(np.float32)
        self.faces_np = faces.astype(np.uint32)
        self.n_vertices = self.vertices_np.shape[0]
        self.n_triangles = self.faces_np.shape[0]

        self.vertex_positions_flat = mi.Float(self.vertices_np.reshape(-1))
        self.faces = mi.UInt32(self.faces_np.reshape(-1))

        self.triangle_model = triangle_model.TriangleModel(self.n_triangles)

        if initial_colors is not None:
            init_array = np.ascontiguousarray(initial_colors, dtype=np.float32)
            assert init_array.ndim == 2 and init_array.shape[1] == 3, (
                f"initial_colors must be (N, 3) with last dim = RGB, "
                f"got shape {init_array.shape}"
            )
            assert init_array.shape[0] == self.n_triangles, (
                f"initial_colors has {init_array.shape[0]} rows; "
                f"expected {self.n_triangles}"
            )
            self.triangle_model.colors = mi.Color3f(init_array.T)
        if initial_occupancy is not None:
            self.triangle_model.occupancy = mi.Float(initial_occupancy.astype(np.float32))
        else:
            self.triangle_model.occupancy = dr.full(
                mi.Float,
                2 * config.occupancy_threshold,
                self.n_triangles,
            )

        self.integrator.triangle_model = self.triangle_model

        self.variable_groups: list[VariableGroup] = [
            VariableGroup(
                key="soup_mesh.vertex_positions",
                clamp_min=-config.scene_scale,
                clamp_max=config.scene_scale,
                lr_scale=config.positions_learning_rate_scale,
            ),
            VariableGroup(
                key="triangle.colors",
                clamp_min=0.0,
                clamp_max=1.0,
                lr_scale=config.color_learning_rate_scale,
            ),
            VariableGroup(
                key="triangle.occupancy",
                clamp_min=0.0,
                clamp_max=config.occupancy_upper_bound,
                lr_scale=config.occupancy_learning_rate_scale,
            ),
            VariableGroup(
                key="triangle.sigma",
                clamp_min=0.01,
                clamp_max=100.0,
                lr_scale=config.sigma_learning_rate_scale,
            ),
        ]

    def register(self, optimizer: mi.ad.Optimizer) -> None:
        dr.enable_grad(self.vertex_positions_flat)
        optimizer["soup_mesh.vertex_positions"] = self.vertex_positions_flat

        dr.enable_grad(self.triangle_model.colors)
        optimizer["triangle.colors"] = self.triangle_model.colors

        dr.enable_grad(self.triangle_model.occupancy)
        optimizer["triangle.occupancy"] = self.triangle_model.occupancy

        dr.enable_grad(self.triangle_model.sigma)
        optimizer["triangle.sigma"] = self.triangle_model.sigma

    def apply_learning_rate_scales(self, optimizer: mi.ad.Optimizer, base_lr: float) -> None:
        lr_updates: dict[str, float] = {}
        for group in self.variable_groups:
            if group.lr_scale != 1.0:
                lr_updates[group.key] = base_lr * group.lr_scale
        if lr_updates:
            optimizer.set_learning_rate(lr_updates)

    def clamp(self, optimizer: mi.ad.Optimizer) -> None:
        for group in self.variable_groups:
            if group.clamp_min is None and group.clamp_max is None:
                continue
            value = optimizer[group.key]
            min_val = mi.Float(group.clamp_min) if group.clamp_min is not None else None
            max_val = mi.Float(group.clamp_max) if group.clamp_max is not None else None
            if min_val is not None and max_val is not None:
                value = dr.clip(value, min_val, max_val)
            elif min_val is not None:
                value = dr.maximum(value, min_val)
            elif max_val is not None:
                value = dr.minimum(value, max_val)
            optimizer[group.key] = value

    def update_scene(self, params: mi.SceneParameters, optimizer: mi.ad.Optimizer) -> None:
        params["soup_mesh.vertex_positions"] = optimizer["soup_mesh.vertex_positions"]
        params["soup_mesh.faces"] = self.faces
        params.update()

        self.vertex_positions_flat = optimizer["soup_mesh.vertex_positions"]
        self.triangle_model.colors = optimizer["triangle.colors"]
        self.triangle_model.occupancy = optimizer["triangle.occupancy"]
        self.triangle_model.sigma = optimizer["triangle.sigma"]
        self.integrator.triangle_model = self.triangle_model

    def export_state(self) -> dict[str, np.ndarray | None]:
        vertices_np = self.vertex_positions_flat.numpy().reshape(
            self.n_vertices, 3
        ).astype(np.float32)
        return {
            "vertices": vertices_np,
            "faces": self.faces_np.astype(np.uint32, copy=True),
            "colors": self.triangle_model.colors.numpy().T.astype(
                np.float32, copy=True
            ),
            "occupancy": self.triangle_model.occupancy.numpy().astype(
                np.float32, copy=True
            ),
            "sigma": self.triangle_model.sigma.numpy().astype(
                np.float32, copy=True
            ),
        }
