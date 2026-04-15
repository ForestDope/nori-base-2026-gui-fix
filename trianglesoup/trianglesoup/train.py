#!/usr/bin/env python3
"""Unified entry point for triangle soup optimization."""

from __future__ import annotations

import json
from pathlib import Path

import drjit as dr
import mitsuba as mi
import numpy as np

from trianglesoup.training.config import parse_arguments


def save_metrics(output_dir: Path, trainer) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    data = [m.__dict__ for m in trainer.metrics]
    with metrics_path.open("w") as f:
        json.dump(data, f, indent=2)


def _render_view_to_numpy(trainer, view, spp: int) -> np.ndarray:
    """Render one view with the current triangle soup → sRGB uint8 numpy array."""
    render = mi.render(
        trainer.scene,
        params=trainer.scene_params,
        sensor=view.sensor,
        spp=spp,
        integrator=trainer.integrator,
    )
    dr.eval(render)
    bmp = mi.Bitmap(render).convert(
        mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, srgb_gamma=True
    )
    return np.array(bmp, copy=True)


def _resolve_progress_view(trainer, selector: str | None):
    """Resolve --progress-view into a concrete CameraView.

    Matches against (in order): view name, sensor hash (as printed in the
    EXR filenames), or plain integer index into the test set.
    """
    from trianglesoup import utils

    test_views = list(trainer.test_dataset.views)
    train_views = list(trainer.train_dataset.views)
    all_views = test_views + train_views

    if selector is None:
        return test_views[0] if test_views else (train_views[0] if train_views else None)

    sel = selector.strip()
    if sel.lstrip("-").isdigit():
        idx = int(sel)
        pool = test_views if test_views else train_views
        if 0 <= idx < len(pool):
            return pool[idx]
        print(f"[progress-grid] index {idx} out of range ({len(pool)} views); using first")
        return pool[0] if pool else None

    for v in all_views:
        if sel in v.name:
            return v
    for v in all_views:
        if sel in utils.sensor_hash_to_10_digits(v.sensor):
            return v
    print(f"[progress-grid] no view matches '{sel}'; using first test/train view")
    return test_views[0] if test_views else (train_views[0] if train_views else None)


def save_progress_grid(
    output_dir: Path,
    snapshots: list[tuple[int, np.ndarray]],
    gt_image: np.ndarray | None,
    total_iterations: int,
) -> None:
    """Save a 3x3 matplotlib grid showing optimisation progress on one view."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[progress-grid] matplotlib missing; skipping plot")
        return

    tiles: list[tuple[str, np.ndarray]] = [
        ("Initial" if it < 0 else f"Iteration {it + 1}", img)
        for it, img in snapshots
    ]
    if gt_image is not None:
        tiles = tiles + [("Ground Truth", gt_image)]
    tiles = tiles[:9]

    fig, axes = plt.subplots(
        3, 3, figsize=(9, 9),
        gridspec_kw={"wspace": 0.02, "hspace": 0.12},
    )
    for ax, (title, img) in zip(axes.flat, tiles):
        ax.imshow(img)
        ax.set_title(title, fontsize=10, pad=2)
        ax.axis("off")
    for ax in list(axes.flat)[len(tiles):]:
        ax.axis("off")

    fig.suptitle(
        f"Optimisation progress — {total_iterations} iterations",
        fontsize=12,
        y=0.995,
    )
    fig.subplots_adjust(left=0.01, right=0.99, top=0.955, bottom=0.01)
    out_path = output_dir / "progress_grid.png"
    fig.savefig(out_path, dpi=240, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved progress grid to {out_path}")


def run_headless(cfg, runtime, train_dataset=None, test_dataset=None) -> None:
    from trianglesoup.training.optimizer import TriangleSoupTrainer

    output_dir = cfg.make_output_directory()
    trainer = TriangleSoupTrainer(
        cfg,
        runtime,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
    )

    if runtime.render_initial:
        initial_root = output_dir / "initial_renders"
        trainer.render_dataset(
            initial_root / "train",
            "train",
            trainer.train_dataset.views,
            runtime.initial_render_spp,
            width=runtime.initial_render_width,
            height=runtime.initial_render_height,
            scale=runtime.initial_render_scale,
        )
        if trainer.test_dataset.views:
            trainer.render_dataset(
                initial_root / "test",
                "test",
                trainer.test_dataset.views,
                runtime.initial_render_spp,
                width=runtime.initial_render_width,
                height=runtime.initial_render_height,
                scale=runtime.initial_render_scale,
            )
        print(f"Saved initial configuration renders to {initial_root}")

    from tqdm import tqdm

    # Progress grid: 9 tiles = iter-0 snapshot, 7 evenly-spaced intermediate
    # snapshots (always including the final iteration), and a GT reference
    # tile when available. Falls back to 9 snapshots (no GT) otherwise.
    grid_view = None
    grid_snapshots: list[tuple[int, np.ndarray]] = []
    grid_milestones: set[int] = set()
    grid_spp = max(4, runtime.final_render_spp // 4)
    if runtime.progress_grid and cfg.iterations >= 2:
        grid_view = _resolve_progress_view(trainer, runtime.progress_view)
        if grid_view is not None:
            n_intermediate = 7 if grid_view.target is not None else 8
            # metric.iteration is 0-indexed (0..iters-1), so we target
            # iters-1 as the last captured step.
            milestones = np.linspace(
                max(0, cfg.iterations // (n_intermediate + 1)),
                cfg.iterations - 1,
                n_intermediate,
                dtype=int,
            )
            grid_milestones = set(int(m) for m in milestones)
            # Pre-loop snapshot represents the untrained init (labelled -1 so
            # the tile prints "init" instead of colliding with iter 0).
            grid_snapshots.append(
                (-1, _render_view_to_numpy(trainer, grid_view, grid_spp))
            )

    progress_render_interval = runtime.progress_render_interval
    pbar = tqdm(range(cfg.iterations), desc="Training", unit="it")
    for _ in pbar:
        metrics = trainer.step()
        pbar.set_postfix_str(
            f"loss={metrics.loss:.6f} tris={metrics.num_triangles} lr={metrics.learning_rate:.1e}"
        )

        if grid_view is not None and metrics.iteration in grid_milestones:
            grid_snapshots.append(
                (metrics.iteration, _render_view_to_numpy(trainer, grid_view, grid_spp))
            )

        if runtime.log_interval > 0 and metrics.iteration % runtime.log_interval == 0:
            extra = ""
            if metrics.pruned_triangles or metrics.added_triangles:
                extra = (
                    f" prune=-{metrics.pruned_triangles}"
                    f" densify=+{metrics.added_triangles}"
                )
            tqdm.write(
                f"Iteration {metrics.iteration:05d} | loss={metrics.loss:.6f} "
                f"lr={metrics.learning_rate:.4e} tris={metrics.num_triangles}{extra}"
            )

        if (
            progress_render_interval > 0
            and metrics.iteration > 0
            and metrics.iteration % progress_render_interval == 0
        ):
            progress_dir = output_dir / "progress" / f"iter_{metrics.iteration:06d}"
            trainer.render_dataset(
                progress_dir,
                "progress",
                trainer.progress_views,
                runtime.final_render_spp,
            )
            tqdm.write(f"  Saved progress renders to {progress_dir}")

    save_metrics(output_dir, trainer)

    if grid_view is not None and grid_snapshots:
        gt = None
        if grid_view.target is not None:
            try:
                gt_bmp = mi.Bitmap(grid_view.target).convert(
                    mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, srgb_gamma=True
                )
                gt = np.array(gt_bmp, copy=True)
            except Exception as e:
                print(f"[progress-grid] could not load GT: {e}")
        save_progress_grid(output_dir, grid_snapshots, gt, cfg.iterations)

    if runtime.render_final:
        final_root = output_dir / "final_renders"
        trainer.render_dataset(
            final_root / "train",
            "train",
            trainer.train_dataset.views,
            runtime.final_render_spp,
            width=runtime.final_render_width,
            height=runtime.final_render_height,
            scale=runtime.final_render_scale,
        )
        if trainer.test_dataset.views:
            trainer.render_dataset(
                final_root / "test",
                "test",
                trainer.test_dataset.views,
                runtime.final_render_spp,
                width=runtime.final_render_width,
                height=runtime.final_render_height,
                scale=runtime.final_render_scale,
            )
        print(f"Saved final renders to {final_root}")


def main(args: list[str] | None = None) -> None:
    mi.set_variant("cuda_ad_rgb", "llvm_ad_rgb")

    cfg, runtime = parse_arguments(args)
    run_headless(cfg, runtime)


if __name__ == "__main__":
    main()
