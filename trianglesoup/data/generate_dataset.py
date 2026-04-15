#!/usr/bin/env python3
"""Render orbit views of a Mitsuba scene into a training dataset.

Takes a Mitsuba ``.xml`` scene file, places cameras on a Fibonacci
sphere orbit around the scene's bounding-box centre, renders each
view with the path tracer, and writes the result as a
trianglesoup-compatible dataset::

    <output>/
        train.json
        test.json
        train_images/  *.exr
        test_images/   *.exr

Usage::

    pixi run python data/generate_dataset.py \\
        --scene data/scenes/lego/scene.xml \\
        --output data/datasets/lego

The scene centre, orbit radius, and FOV are auto-detected from the
mesh bounding box unless overridden via ``--center`` / ``--radius``
/ ``--fov``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import drjit as dr
import mitsuba as mi
import numpy as np

from trianglesoup import utils


# ---------------------------------------------------------------------------
# Camera placement
# ---------------------------------------------------------------------------


def fibonacci_sphere(
    n_samples: int,
    radius: float,
    center: np.ndarray,
    min_elevation: float = -90.0,
    max_elevation: float = 90.0,
) -> List[np.ndarray]:
    """Return ``n_samples`` points on a sphere of given radius."""
    min_y = np.sin(np.deg2rad(min_elevation))
    max_y = np.sin(np.deg2rad(max_elevation))
    points: List[np.ndarray] = []
    golden_angle = np.pi * (3 - np.sqrt(5))
    for i in range(n_samples):
        y = max_y - (max_y - min_y) * (2 * i + 1) / (2 * n_samples)
        r = float(np.sqrt(max(0.0, 1 - y * y)))
        theta = golden_angle * i
        x = float(np.cos(theta) * r)
        z = float(np.sin(theta) * r)
        offset = radius * np.array([x, y, z], dtype=np.float64)
        points.append(center + offset)
    return points


def sensor_dict_for_json(
    origin: np.ndarray,
    target: np.ndarray,
    fov: float,
    resolution: int,
) -> dict:
    transform = mi.ScalarTransform4f().look_at(
        origin=list(origin), target=list(target), up=[0, 1, 0]
    )
    return {
        "type": "perspective",
        "fov": fov,
        "fov_axis": "x",
        "to_world": transform,
        "film": {
            "type": "hdrfilm",
            "width": int(resolution),
            "height": int(resolution),
            "sample_border": False,
            "filter": {"type": "box"},
        },
        "sampler": {"type": "independent", "sample_count": 1},
    }


def sensor_dict_to_json_friendly(spec: dict) -> dict:
    out = dict(spec)
    out["to_world"] = np.array(spec["to_world"].matrix).tolist()
    return out


# ---------------------------------------------------------------------------
# Auto-detection from scene bounding box
# ---------------------------------------------------------------------------


def _scene_bbox(scene) -> tuple[np.ndarray, np.ndarray]:
    """Compute the axis-aligned bounding box of all shapes in the scene."""
    all_min = np.full(3, np.inf)
    all_max = np.full(3, -np.inf)
    for shape in scene.shapes():
        params = mi.traverse(shape)
        verts = np.array(params["vertex_positions"]).reshape(-1, 3)
        all_min = np.minimum(all_min, verts.min(axis=0))
        all_max = np.maximum(all_max, verts.max(axis=0))
    return all_min, all_max


def _auto_orbit_params(scene) -> tuple[np.ndarray, float, float]:
    """Return (center, radius, fov) derived from the scene bounding box."""
    bbox_min, bbox_max = _scene_bbox(scene)
    center = 0.5 * (bbox_min + bbox_max)
    diagonal = float(np.linalg.norm(bbox_max - bbox_min))
    radius = diagonal * 1.2
    fov = 39.0  # reasonable default for most scenes
    return center, radius, fov


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(args: argparse.Namespace) -> None:
    mi.set_variant("cuda_ad_rgb", "llvm_ad_rgb")

    scene_path = args.scene.expanduser().resolve()
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene file not found: {scene_path}")

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "train_images").mkdir(exist_ok=True)
    (output / "test_images").mkdir(exist_ok=True)

    import sys
    (output / "generate_command.txt").write_text(" ".join(sys.argv) + "\n")

    print(f"Loading scene from {scene_path} ...")
    # Only pass parameters that the scene XML actually references as
    # $-variables; unknown parameters cause a Mitsuba parse error.
    # Try with common parameters first, fall back to bare load.
    try:
        scene = mi.load_file(
            str(scene_path),
            spp=args.spp,
            resx=args.resolution,
            resy=args.resolution,
            max_depth=args.max_depth,
        )
    except RuntimeError:
        scene = mi.load_file(str(scene_path))

    # Auto-detect orbit parameters from scene bounding box, or use overrides.
    auto_center, auto_radius, auto_fov = _auto_orbit_params(scene)

    if args.center is not None:
        center = np.array([float(c) for c in args.center.split(",")], dtype=np.float64)
    else:
        center = auto_center

    radius = args.radius if args.radius is not None else auto_radius
    fov = args.fov if args.fov is not None else auto_fov

    print(f"  -> orbit center = {center.tolist()}, radius = {radius:.2f}, fov = {fov:.1f} deg")

    # Generate orbit camera positions.
    train_origins = fibonacci_sphere(
        args.train_cameras, radius, center,
        min_elevation=args.min_elevation, max_elevation=args.max_elevation,
    )
    test_origins = fibonacci_sphere(
        args.test_cameras, radius * 1.05, center,
        min_elevation=args.min_elevation, max_elevation=args.max_elevation,
    )
    print(f"  -> {len(train_origins)} train + {len(test_origins)} test cameras")

    # Build (and write to JSON) the sensor specs for each split.
    def write_split_json(origins: List[np.ndarray], split: str) -> Path:
        data = {"w": int(args.resolution), "h": int(args.resolution)}
        for idx, origin in enumerate(origins):
            spec = sensor_dict_for_json(origin, center, fov, args.resolution)
            data[f"view{idx:03d}"] = sensor_dict_to_json_friendly(spec)
        json_path = output / f"{split}.json"
        with json_path.open("w") as f:
            json.dump(data, f, indent=2)
        return json_path

    train_json = write_split_json(train_origins, "train")
    test_json = write_split_json(test_origins, "test")
    print(f"  -> wrote {train_json.name} and {test_json.name}")

    train_sensors = utils.load_sensors_from_json(str(train_json))
    test_sensors = utils.load_sensors_from_json(str(test_json))

    def render_split(sensors: List[mi.Sensor], split: str) -> None:
        print(f"\nRendering {split} ({len(sensors)} views) at "
              f"{args.resolution}x{args.resolution}, spp={args.spp} ...")
        for i, sensor in enumerate(sensors):
            image = mi.render(scene, sensor=sensor, spp=args.spp)
            dr.eval(image)
            hash_id = utils.sensor_hash_to_10_digits(sensor)
            filename = f"{split}_view_{hash_id}.exr"
            out_path = output / f"{split}_images" / filename
            mi.Bitmap(image).write(str(out_path))
            print(f"  [{i + 1:>3}/{len(sensors)}] {filename}")

    render_split(train_sensors, "train")
    render_split(test_sensors, "test")

    print(f"\nDone. Dataset at {output}")
    print(f"\nTrain with:")
    print(f"  pixi run python python -m trianglesoup.train {output} \\")
    print(f"      --mesh-init {scene_path} \\")
    print(f"      --iterations 5000 --log-interval 100")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render orbit views of a Mitsuba scene and save as a "
                    "trianglesoup-compatible training dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scene", type=Path, default=Path("data/scenes/lego/scene.xml"),
        help="Path to the Mitsuba .xml scene to render.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output dataset directory. Defaults to data/datasets/<scene_name>.",
    )
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--spp", type=int, default=1024,
                        help="Samples per pixel for the path-tracer renders.")
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--train-cameras", type=int, default=32)
    parser.add_argument("--test-cameras", type=int, default=8)
    parser.add_argument(
        "--radius", type=float, default=None,
        help="Orbit camera distance from center. Auto-detected from the "
             "scene bounding box when not set.",
    )
    parser.add_argument(
        "--center", type=str, default=None,
        help="Orbit centre as 'x,y,z'. Auto-detected from the scene "
             "bounding box when not set.",
    )
    parser.add_argument(
        "--fov", type=float, default=None,
        help="Horizontal FOV in degrees. Default 39.",
    )
    parser.add_argument(
        "--min-elevation", type=float, default=10.0,
        help="Minimum camera elevation in degrees (0=equator, -90=below).",
    )
    parser.add_argument(
        "--max-elevation", type=float, default=80.0,
        help="Maximum camera elevation in degrees (90=directly above).",
    )
    args = parser.parse_args()

    # Auto-derive output from scene name if not specified.
    if args.output is None:
        scene_name = args.scene.parent.name
        args.output = Path(f"data/datasets/{scene_name}")

    main(args)
