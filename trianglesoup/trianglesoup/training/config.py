"""Configuration dataclasses and CLI helpers for triangle soup optimization."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Optional


def _positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"Expected positive integer, got {ivalue}")
    return ivalue


def _non_negative_int(value: str) -> int:
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError(f"Expected non-negative integer, got {ivalue}")
    return ivalue


def _non_negative_float(value: str) -> float:
    fvalue = float(value)
    if fvalue < 0.0:
        raise argparse.ArgumentTypeError(f"Expected non-negative float, got {fvalue}")
    return fvalue


def _parse_rgb(value: str) -> tuple[float, float, float]:
    parts = [float(v.strip()) for v in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "Expected three comma-separated values for RGB radiance"
        )
    return (parts[0], parts[1], parts[2])


@dataclass
class TriangleSoupConfig:
    """Configuration parameters for triangle soup optimization."""

    dataset_path: Path
    output_dir: Path = Path("output/trianglesoup")
    experiment_name: Optional[str] = None

    # Scene and representation
    scene_scale: float = 2.0
    n_triangles_per_unit_cube: int = 128
    triangle_base_size_factor: float = 0.02
    sh_degree: int = 0
    initial_state_path: Path | None = None
    mesh_init: Path | None = None
    mesh_init_triangles: int = 5000
    background_radiance: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Training control
    iterations: int = 512
    warmup_iterations: int = 0
    render_state_spp: int = 32
    primal_spp: int = 1
    gradient_spp: int = 0
    mini_batch_pixels: int = 1000000
    full_view_training: bool = False
    train_view_indices: tuple[int, ...] = ()  # non-empty → train only on those view indices
    jitter_pixels: bool = False
    spp_multiplier_schedule: Optional[list[tuple[float, int]]] = None

    # Learning rate schedule
    learning_rate: float = 1e-3
    learning_rate_decay_step: int = 0  # 0 = constant LR (no decay)
    learning_rate_decay_rate: float = 0.5
    learning_rate_min: float = 1e-5
    learning_rate_plateau_patience: int = 0
    learning_rate_plateau_window: int = 500
    positions_learning_rate_scale: float = 1.0
    color_learning_rate_scale: float = 10.0
    occupancy_learning_rate_scale: float = 1.0
    sigma_learning_rate_scale: float = 1.0

    # Loss weights (Eq. 3 of the paper).
    loss_occupancy_weight: float = 0.0055      # β1: L_o opacity regularization (paper: 0.0055)
    loss_size_weight: float = 1e-8             # β4: L_s size regularization (paper: 1e-8)

    # Occupancy & clamping
    occupancy_threshold: float = 0.1
    occupancy_upper_bound: float = 1.0 - 1e-5
    clamp_attribute_sums_strict: bool = False

    # Pruning and densification
    pruning_step: int = 512
    pruning_occupancy_threshold: float = 0.01
    pruning_area_threshold: float = 1e-4
    pruning_min_area: float = 1e-7
    pruning_max_aspect_ratio: float = 20.0
    pruning_min_view_count: int = 2
    pruning_min_pixels_per_view: int = 2
    pruning_stats_spp: int = 1
    max_triangles: int = 0
    densification_step: int = 512
    densification_type: str = "upsample"
    densification_heuristic: str = "all"
    densification_budget_ratio: float = 0.05
    densification_max_new_triangles: int = 0
    densification_clone_noise_scale: float = 0.1
    occupancy_reset_step_offset: int = 0
    occupancy_reset_step: int = 0
    occupancy_reset_target: float = 0.1

    # Logging & outputs
    progress_interval: int = 50
    checkpoint_interval: Optional[int] = None
    progress_view_count: int = 1
    verbose: bool = False
    optim_kernel_log: bool = False

    def make_output_directory(self) -> Path:
        base = self.output_dir.expanduser().resolve()
        base.mkdir(parents=True, exist_ok=True)
        if self.experiment_name:
            run_dir = base / self.experiment_name
            run_dir.mkdir(parents=True, exist_ok=True)
            return run_dir
        return base


@dataclass
class RuntimeConfig:
    """Runtime options not tied directly to optimization."""

    headless: bool = True
    log_interval: int = 10
    render_final: bool = False
    final_render_spp: int = 64
    progress_render_interval: int = 0
    progress_grid: bool = True
    progress_view: Optional[str] = None
    final_render_scale: int = 1
    final_render_width: Optional[int] = None
    final_render_height: Optional[int] = None
    render_initial: bool = False
    initial_render_spp: int = 64
    initial_render_scale: int = 1
    initial_render_width: Optional[int] = None
    initial_render_height: Optional[int] = None


def _add_dataclass_arguments(
    parser: argparse.ArgumentParser,
    dataclass_type: type,
    prefix: str = "",
) -> None:
    for field_obj in fields(dataclass_type):
        if not field_obj.init:
            continue
        arg_name = f"--{prefix}{field_obj.name.replace('_', '-')}"
        default = field_obj.default
        help_text = field_obj.metadata.get("help") if field_obj.metadata else None
        arg_kwargs: dict[str, Any] = {}
        if field_obj.type in (int, Optional[int]):
            arg_kwargs["type"] = int
        elif field_obj.type in (float, Optional[float]):
            arg_kwargs["type"] = float
        elif field_obj.type in (bool, Optional[bool]):
            # For booleans, use store_true/store_false based on default
            if default is True:
                parser.add_argument(
                    f"--no-{prefix}{field_obj.name.replace('_', '-')}",
                    action="store_false",
                    dest=field_obj.name,
                    help=help_text or argparse.SUPPRESS,
                )
                continue
            parser.add_argument(
                arg_name,
                action="store_true",
                help=help_text or argparse.SUPPRESS,
            )
            continue
        elif field_obj.type in (Path, Optional[Path]):
            arg_kwargs["type"] = Path
        else:
            continue

        if help_text:
            arg_kwargs["help"] = help_text
        if field_obj.default is not field(default=None):
            arg_kwargs["default"] = default
        parser.add_argument(arg_name, **arg_kwargs)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Triangle soup inverse rendering optimization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to dataset directory containing train.json/test.json and images",
    )

    parser.add_argument(
        "--iterations",
        type=_positive_int,
        default=TriangleSoupConfig.iterations,
        help="Number of optimization iterations",
    )
    parser.add_argument(
        "--log-interval",
        type=_positive_int,
        default=10,
        help="Logging frequency in iterations",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=_positive_int,
        default=0,
        help="Checkpoint save interval (0 disables periodic checkpoints)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Optional experiment name (subdirectory under output dir)",
    )
    parser.add_argument(
        "--use-test-cube-config",
        action="store_true",
        help=(
            "Initialize optimizer/sampling hyperparameters to values used in "
            "unit_test/test_cube.py."
        ),
    )
    parser.add_argument(
        "--render-final",
        action="store_true",
        help="Render all train/test views after training",
    )
    parser.add_argument(
        "--final-render-spp",
        type=_positive_int,
        default=64,
        help="Samples per pixel for final train/test renders",
    )
    parser.add_argument(
        "--final-render-scale",
        type=_positive_int,
        default=1,
        help="Scale factor for final render resolution.",
    )
    parser.add_argument(
        "--final-render-width",
        type=_positive_int,
        default=None,
        help="Override final render width in pixels.",
    )
    parser.add_argument(
        "--final-render-height",
        type=_positive_int,
        default=None,
        help="Override final render height in pixels.",
    )
    parser.add_argument(
        "--render-initial",
        action="store_true",
        help="Render all train/test views before optimization starts",
    )
    parser.add_argument(
        "--initial-render-spp",
        type=_positive_int,
        default=64,
        help="Samples per pixel for initial train/test renders",
    )
    parser.add_argument(
        "--initial-render-scale",
        type=_positive_int,
        default=1,
        help="Scale factor for initial render resolution.",
    )
    parser.add_argument(
        "--initial-render-width",
        type=_positive_int,
        default=None,
        help="Override initial render width in pixels.",
    )
    parser.add_argument(
        "--initial-render-height",
        type=_positive_int,
        default=None,
        help="Override initial render height in pixels.",
    )
    parser.add_argument(
        "--background-radiance",
        type=_parse_rgb,
        default=None,
        help="Override background radiance as R,G,B (comma-separated).",
    )
    parser.add_argument(
        "--no-progress-grid",
        dest="progress_grid",
        action="store_false",
        help="Disable the 3x3 matplotlib optimisation-progress figure.",
    )
    parser.set_defaults(progress_grid=True)
    parser.add_argument(
        "--progress-view",
        type=str,
        default=None,
        help=(
            "Choose which view to render in the progress grid. Accepts a "
            "view name substring (e.g. '8455107cf5') or a plain integer "
            "index into the test set. Default: first test view."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=TriangleSoupConfig.output_dir,
        help="Base directory for experiment outputs",
    )
    parser.add_argument(
        "--sh-degree",
        type=_non_negative_int,
        default=TriangleSoupConfig.sh_degree,
        help="Spherical harmonics degree (0=DC only, 3=full as in paper)",
    )
    parser.add_argument(
        "--full-view-training", action="store_true",
        help="Render one full training view per iteration (round-robin) "
             "instead of sampling random pixel mini-batches. Gives cleaner "
             "gradients, closer to the paper's training loop.",
    )
    parser.add_argument(
        "--train-views", type=str, default="", metavar="INDICES",
        help="Overfit diagnostic: comma-separated list of view indices to train on "
             "(e.g. '0' for a single view, '0,3,7' for three). Empty string uses all views.",
    )
    parser.add_argument(
        "--jitter-pixels", action="store_true",
        help="Use the jittered independent sampler instead of the deterministic "
             "pixel-center stratified sampler. Enables temporal antialiasing at "
             "the cost of noisier per-iteration gradients.",
    )
    parser.add_argument(
        "--primal-spp", type=_positive_int,
        default=TriangleSoupConfig.primal_spp,
        help="Samples per pixel for the training render. Higher = less noise.",
    )
    parser.add_argument(
        "--learning-rate", type=float,
        default=TriangleSoupConfig.learning_rate,
        help="Base learning rate for Adam optimizer.",
    )
    parser.add_argument(
        "--learning-rate-decay-step", type=_non_negative_int,
        default=TriangleSoupConfig.learning_rate_decay_step,
        help="Apply LR decay every N iterations (0 = constant LR).",
    )
    parser.add_argument(
        "--learning-rate-decay-rate", type=float,
        default=TriangleSoupConfig.learning_rate_decay_rate,
        help="LR multiplier applied at each decay step (e.g. 0.5 halves the LR).",
    )
    parser.add_argument(
        "--learning-rate-min", type=float,
        default=TriangleSoupConfig.learning_rate_min,
        help="Lower bound for the decayed LR.",
    )
    parser.add_argument(
        "--sigma-learning-rate-scale", type=float,
        default=TriangleSoupConfig.sigma_learning_rate_scale,
        help="Per-group LR scale for the window-function sharpness σ.",
    )
    parser.add_argument(
        "--color-learning-rate-scale", type=float,
        default=TriangleSoupConfig.color_learning_rate_scale,
        help="Per-group LR scale for the color SH coefficients.",
    )
    parser.add_argument(
        "--positions-learning-rate-scale", type=float,
        default=TriangleSoupConfig.positions_learning_rate_scale,
        help="Per-group LR scale for the triangle vertex positions.",
    )
    parser.add_argument(
        "--occupancy-learning-rate-scale", type=float,
        default=TriangleSoupConfig.occupancy_learning_rate_scale,
        help="Per-group LR scale for per-triangle occupancy.",
    )
    parser.add_argument(
        "--mini-batch-pixels",
        type=_positive_int,
        default=TriangleSoupConfig.mini_batch_pixels,
        help="Pixels per training iteration. 512*512=262144 gives one full view per step.",
    )
    # Loss terms from Eq. 3 of the Triangle Splatting paper.
    parser.add_argument("--loss-occupancy-weight", type=float,
                        default=TriangleSoupConfig.loss_occupancy_weight,
                        help="Opacity regularization β1 (L_o). Paper uses 0.0055.")
    parser.add_argument("--loss-size-weight", type=float,
                        default=TriangleSoupConfig.loss_size_weight,
                        help="Size regularization β4 (L_s). Paper uses 1e-8.")
    parser.add_argument(
        "--mesh-init", type=Path, default=None,
        help="Path to a Mitsuba .xml scene file. Initialise the triangle soup "
             "from the GT mesh geometry instead of SfM or random.",
    )
    parser.add_argument(
        "--mesh-init-triangles", type=_positive_int,
        default=TriangleSoupConfig.mesh_init_triangles,
        help="Target triangle count when subsampling the GT mesh.",
    )
    _add_topology_arguments(parser)

    return parser


def _add_topology_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--pruning-step",
        type=_non_negative_int,
        default=TriangleSoupConfig.pruning_step,
        help="Run pruning every N iterations (0 disables pruning).",
    )
    parser.add_argument(
        "--pruning-occupancy-threshold",
        type=_non_negative_float,
        default=TriangleSoupConfig.pruning_occupancy_threshold,
        help="Prune triangles whose max observed T*o weight stays below this threshold.",
    )
    parser.add_argument(
        "--pruning-min-view-count",
        type=_non_negative_int,
        default=TriangleSoupConfig.pruning_min_view_count,
        help="Require a triangle to be rendered in at least this many views.",
    )
    parser.add_argument(
        "--pruning-min-pixels-per-view",
        type=_non_negative_int,
        default=TriangleSoupConfig.pruning_min_pixels_per_view,
        help="A view only counts towards pruning visibility if the triangle covers more than this many pixels.",
    )
    parser.add_argument(
        "--pruning-stats-spp",
        type=_positive_int,
        default=TriangleSoupConfig.pruning_stats_spp,
        help="Samples per pixel for the statistics pass used by pruning.",
    )
    parser.add_argument(
        "--pruning-min-area",
        type=_non_negative_float,
        default=TriangleSoupConfig.pruning_min_area,
        help="Prune triangles with area below this threshold (0 disables).",
    )
    parser.add_argument(
        "--pruning-max-aspect-ratio",
        type=_non_negative_float,
        default=TriangleSoupConfig.pruning_max_aspect_ratio,
        help="Prune triangles whose longest-to-shortest edge ratio exceeds this (0 disables).",
    )
    parser.add_argument(
        "--max-triangles",
        type=_non_negative_int,
        default=TriangleSoupConfig.max_triangles,
        help="Global cap on triangle count; densification is skipped when at this limit (0 disables the cap).",
    )
    parser.add_argument(
        "--densification-step",
        type=_non_negative_int,
        default=TriangleSoupConfig.densification_step,
        help="Run densification every N iterations (0 disables densification).",
    )
    parser.add_argument(
        "--densification-budget-ratio",
        type=_non_negative_float,
        default=TriangleSoupConfig.densification_budget_ratio,
        help="Fraction of current triangles to densify at each densification step.",
    )
    parser.add_argument(
        "--densification-max-new-triangles",
        type=_non_negative_int,
        default=TriangleSoupConfig.densification_max_new_triangles,
        help="Hard cap on the number of new triangles added at each densification step (0 disables the cap).",
    )
    parser.add_argument(
        "--densification-small-area-threshold",
        type=_non_negative_float,
        default=TriangleSoupConfig.pruning_area_threshold,
        help="Triangles smaller than this area are cloned with in-plane noise instead of split.",
    )
    parser.add_argument(
        "--densification-clone-noise-scale",
        type=_non_negative_float,
        default=TriangleSoupConfig.densification_clone_noise_scale,
        help="Relative in-plane offset used when cloning small triangles.",
    )


def parse_arguments(args: Optional[list[str]] = None) -> tuple[TriangleSoupConfig, RuntimeConfig]:
    parser = build_arg_parser()
    parsed = parser.parse_args(args=args)

    cfg = TriangleSoupConfig(
        dataset_path=parsed.dataset,
        output_dir=parsed.output_dir,
        experiment_name=parsed.experiment_name,
        iterations=parsed.iterations,
        sh_degree=parsed.sh_degree,
        full_view_training=parsed.full_view_training,
        train_view_indices=tuple(
            int(s) for s in parsed.train_views.split(",") if s.strip()
        ),
        jitter_pixels=parsed.jitter_pixels,
        primal_spp=parsed.primal_spp,
        learning_rate=parsed.learning_rate,
        learning_rate_decay_step=parsed.learning_rate_decay_step,
        learning_rate_decay_rate=parsed.learning_rate_decay_rate,
        learning_rate_min=parsed.learning_rate_min,
        sigma_learning_rate_scale=parsed.sigma_learning_rate_scale,
        color_learning_rate_scale=parsed.color_learning_rate_scale,
        positions_learning_rate_scale=parsed.positions_learning_rate_scale,
        occupancy_learning_rate_scale=parsed.occupancy_learning_rate_scale,
        mesh_init=parsed.mesh_init,
        mesh_init_triangles=parsed.mesh_init_triangles,
        mini_batch_pixels=parsed.mini_batch_pixels,
        background_radiance=(
            parsed.background_radiance
            if parsed.background_radiance is not None
            else TriangleSoupConfig.background_radiance
        ),
        checkpoint_interval=(
            parsed.checkpoint_interval if parsed.checkpoint_interval > 0 else None
        ),
        pruning_step=parsed.pruning_step,
        pruning_occupancy_threshold=parsed.pruning_occupancy_threshold,
        pruning_area_threshold=parsed.densification_small_area_threshold,
        pruning_min_area=parsed.pruning_min_area,
        pruning_max_aspect_ratio=parsed.pruning_max_aspect_ratio,
        pruning_min_view_count=parsed.pruning_min_view_count,
        pruning_min_pixels_per_view=parsed.pruning_min_pixels_per_view,
        pruning_stats_spp=parsed.pruning_stats_spp,
        max_triangles=parsed.max_triangles,
        densification_step=parsed.densification_step,
        densification_budget_ratio=parsed.densification_budget_ratio,
        densification_max_new_triangles=parsed.densification_max_new_triangles,
        densification_clone_noise_scale=parsed.densification_clone_noise_scale,
        loss_occupancy_weight=parsed.loss_occupancy_weight,
        loss_size_weight=parsed.loss_size_weight,
    )

    runtime = RuntimeConfig(
        headless=True,
        log_interval=parsed.log_interval,
        render_final=parsed.render_final,
        final_render_spp=parsed.final_render_spp,
        final_render_scale=parsed.final_render_scale,
        final_render_width=parsed.final_render_width,
        final_render_height=parsed.final_render_height,
        render_initial=parsed.render_initial,
        initial_render_spp=parsed.initial_render_spp,
        initial_render_scale=parsed.initial_render_scale,
        initial_render_width=parsed.initial_render_width,
        initial_render_height=parsed.initial_render_height,
        progress_grid=parsed.progress_grid,
        progress_view=parsed.progress_view,
    )

    if parsed.use_test_cube_config:
        cfg.scene_scale = 1.0
        cfg.n_triangles_per_unit_cube = 12
        cfg.triangle_base_size_factor = 0.25
        cfg.learning_rate = 5e-2
        cfg.color_learning_rate_scale = 1.0
        cfg.occupancy_learning_rate_scale = 1.0
        cfg.render_state_spp = 16
        cfg.primal_spp = 4
        cfg.gradient_spp = 0
        cfg.mini_batch_pixels = 4096
        cfg.learning_rate_decay_step = 256
        if parsed.background_radiance is None:
            cfg.background_radiance = (0.2, 0.2, 0.2)
        state_path = parsed.dataset / "reference_state.npz"
        if state_path.exists():
            cfg.initial_state_path = state_path

    return cfg, runtime


