#!/usr/bin/env python3
"""Render a triangle soup state from saved .npz file.

Usage::

    pixi run python -m trianglesoup.render state.npz --cameras cameras.json --output renders/
    pixi run python -m trianglesoup.render state.npz --orbit --output renders/

The state file should contain:
  - ``vertex_positions`` (N, 3)
  - ``faces`` (M, 3)
  - ``colors`` (M, 3)
  - ``occupancy`` (M,)
  - optionally ``sigma`` (M,)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np


def fibonacci_sphere(n: int, radius: float, center: np.ndarray) -> List[np.ndarray]:
    """Return ``n`` points on a sphere of given radius around center."""
    points: List[np.ndarray] = []
    golden_angle = np.pi * (3 - np.sqrt(5))
    for i in range(n):
        y = 1 - (2 * i + 1) / n
        r = float(np.sqrt(max(0.0, 1 - y * y)))
        theta = golden_angle * i
        x = float(np.cos(theta) * r)
        z = float(np.sin(theta) * r)
        offset = radius * np.array([x, y, z], dtype=np.float64)
        points.append(center + offset)
    return points


def main():
    parser = argparse.ArgumentParser(
        description="Render a triangle soup state file.",
    )
    parser.add_argument("state", type=Path, help="Path to .npz state file")
    parser.add_argument("--output", type=Path, default=Path("renders"),
                        help="Output directory for rendered images")
    parser.add_argument("--cameras", type=Path, default=None,
                        help="Camera JSON file (same format as train.json)")
    parser.add_argument("--orbit", action="store_true",
                        help="Generate orbit cameras around the mesh")
    parser.add_argument("--orbit-cameras", type=int, default=12,
                        help="Number of orbit cameras")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--spp", type=int, default=16)
    parser.add_argument("--fov", type=float, default=45.0)
    parser.add_argument("--background", type=str, default="0,0,0",
                        help="Background color as R,G,B")
    parser.add_argument("--format", choices=["exr", "png"], default="png")
    args = parser.parse_args()

    import mitsuba as mi
    mi.set_variant("cuda_ad_rgb", "llvm_ad_rgb")
    import drjit as dr
    from trianglesoup.rendering import triangle_model as tm
    from trianglesoup.rendering.integrator import register as register_integrator
    register_integrator()

    # Load state
    state = np.load(args.state)
    vertices = state["vertex_positions"].astype(np.float32)
    faces = state["faces"].astype(np.uint32)
    colors = state["colors"].astype(np.float32) if "colors" in state else \
             state["sh_params"].astype(np.float32)
    occupancy = state["occupancy"].astype(np.float32)
    sigma = state["sigma"].astype(np.float32) if "sigma" in state else None

    bg = tuple(float(v) for v in args.background.split(","))

    # Build scene
    n_triangles = faces.shape[0]
    mesh = mi.Mesh("soup", vertices.shape[0], n_triangles)
    mesh_params = mi.traverse(mesh)
    mesh_params["vertex_positions"] = mi.Float(vertices.reshape(-1))
    mesh_params["faces"] = mi.UInt32(faces.reshape(-1))
    mesh_params.update()
    mesh.set_bsdf(mi.load_dict({"type": "null"}))

    scene = mi.load_dict({
        "type": "scene",
        "integrator": {"type": "soup_emissive"},
        "shape": mesh,
    })
    integrator = scene.integrator()
    integrator.background_color = mi.Color3f(bg)
    integrator.triangle_model = tm.TriangleModel(n_triangles)
    integrator.triangle_model.colors = mi.Color3f(colors.T)
    integrator.triangle_model.occupancy = mi.Float(occupancy)
    if sigma is not None:
        integrator.triangle_model.sigma = mi.Float(sigma)

    params = mi.traverse(scene)
    params["shape.vertex_positions"] = mi.Float(vertices.reshape(-1))
    params["shape.faces"] = mi.UInt32(faces.reshape(-1))
    params["integrator.colors"] = mi.Color3f(colors.T)
    params["integrator.occupancy"] = mi.Float(occupancy)
    if sigma is not None:
        params["integrator.sigma"] = mi.Float(sigma)
    params.update()

    # Build sensors
    sensors = []
    names = []

    if args.cameras is not None:
        from trianglesoup.utils import load_sensors_from_json
        sensors = load_sensors_from_json(str(args.cameras))
        names = [f"view_{i:03d}" for i in range(len(sensors))]
    elif args.orbit:
        verts_3d = vertices.reshape(-1, 3)
        center = 0.5 * (verts_3d.min(axis=0) + verts_3d.max(axis=0))
        diagonal = float(np.linalg.norm(verts_3d.max(axis=0) - verts_3d.min(axis=0)))
        radius = diagonal * 1.5
        origins = fibonacci_sphere(args.orbit_cameras, radius, center)
        for i, origin in enumerate(origins):
            transform = mi.ScalarTransform4f().look_at(
                origin=list(origin), target=list(center), up=[0, 1, 0]
            )
            sensor = mi.load_dict({
                "type": "perspective", "fov": args.fov, "to_world": transform,
                "film": {"type": "hdrfilm", "width": args.resolution,
                         "height": args.resolution, "sample_border": False,
                         "filter": {"type": "box"}},
                "sampler": {"type": "independent", "sample_count": 1},
            })
            sensors.append(sensor)
            names.append(f"orbit_{i:03d}")
    else:
        parser.error("Provide --cameras or --orbit")

    # Render
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Rendering {len(sensors)} views at {args.resolution}x{args.resolution}, spp={args.spp}")

    for i, (sensor, name) in enumerate(zip(sensors, names)):
        image = mi.render(scene, sensor=sensor, spp=args.spp, integrator=integrator)
        dr.eval(image)

        if args.format == "png":
            bmp = mi.Bitmap(image).convert(
                mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, True
            )
            out_path = args.output / f"{name}.png"
        else:
            bmp = mi.Bitmap(image)
            out_path = args.output / f"{name}.exr"

        bmp.write(str(out_path))
        print(f"  [{i+1:>3}/{len(sensors)}] {out_path.name}")

    print(f"Done. {len(sensors)} images saved to {args.output}")


if __name__ == "__main__":
    main()
