#!/usr/bin/env python3
"""Generate a synthetic cube dataset for triangle soup tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np

import mitsuba as mi
import drjit as dr

from trianglesoup.rendering import triangle_model
from trianglesoup import utils


def create_cube_mesh() -> Tuple[np.ndarray, np.ndarray]:
    """Return vertices and triangle faces for a unit cube centered at origin.

    Vertices are unshared (3 per triangle) so the triangle model can
    index them as ``prim_index * 3 + {0, 1, 2}`` to compute edge vectors.
    """
    shared_v = np.array(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ],
        dtype=np.float32,
    )

    shared_faces = np.array(
        [
            [0, 1, 2], [0, 2, 3],  # bottom
            [4, 6, 5], [4, 7, 6],  # top
            [0, 4, 5], [0, 5, 1],  # front
            [1, 5, 6], [1, 6, 2],  # right
            [2, 6, 7], [2, 7, 3],  # back
            [3, 7, 4], [3, 4, 0],  # left
        ],
        dtype=np.uint32,
    )

    n_triangles = shared_faces.shape[0]
    vertices = shared_v[shared_faces].reshape(-1, 3).astype(np.float32)
    faces = np.arange(n_triangles * 3, dtype=np.uint32).reshape(n_triangles, 3)
    return vertices, faces


def fibonacci_sphere(samples: int, radius: float) -> List[np.ndarray]:
    points = []
    golden_angle = np.pi * (3 - np.sqrt(5))
    for i in range(samples):
        z = 1 - (2 * i + 1) / samples
        r = np.sqrt(max(0.0, 1 - z * z))
        theta = golden_angle * i
        x = np.cos(theta) * r
        y = np.sin(theta) * r
        points.append(radius * np.array([x, y, z], dtype=np.float32))
    return points


def sensor_dict(origin: np.ndarray, resolution: int) -> dict:
    transform = mi.ScalarTransform4f().look_at(
        origin=list(origin), target=[0, 0, 0], up=[0, 1, 0]
    )
    return {
        "type": "perspective",
        "fov": 45,
        "to_world": transform,
        "film": {
            "type": "hdrfilm",
            "width": resolution,
            "height": resolution,
            "sample_border": False,
            "filter": {"type": "box"},
        },
        "sampler": {"type": "independent", "sample_count": 1},
    }


def sensor_to_json(sensor_dict: dict) -> dict:
    result = dict(sensor_dict)
    result["to_world"] = np.array(sensor_dict["to_world"].matrix).tolist()
    return result


def configure_scene(
    vertices: np.ndarray,
    faces: np.ndarray,
    sh_colors: np.ndarray,
    occupancy: np.ndarray,
    sensor,
    background_radiance: tuple[float, float, float],
) -> mi.Scene:
    mesh = mi.Mesh("cube", vertices.shape[0], faces.shape[0])
    mesh_params = mi.traverse(mesh)
    mesh_params["vertex_positions"] = mi.Float(vertices.reshape(-1))
    mesh_params["faces"] = mi.UInt32(faces.reshape(-1))
    mesh_params.update()
    mesh.set_bsdf(mi.load_dict({"type": "null"}))

    if not isinstance(sensor, mi.Sensor):
        sensor_props = dict(sensor)
        if not isinstance(sensor_props.get("to_world"), mi.ScalarTransform4f):
            sensor_props["to_world"] = mi.ScalarTransform4f(sensor_props["to_world"])
        sensor = mi.load_dict(sensor_props)

    scene_dict = {
        "type": "scene",
        "integrator": {"type": "soup_emissive"},
        "shape": mesh,
        "sensor": sensor,
    }
    scene = mi.load_dict(scene_dict)

    integrator = scene.integrator()
    integrator.background_color = mi.Color3f(background_radiance)
    n_triangles = faces.shape[0]
    integrator.triangle_model = triangle_model.TriangleModel(n_triangles)
    integrator.triangle_model.occupancy = mi.Float(occupancy)

    params = mi.traverse(scene)
    params["shape.vertex_positions"] = mi.Float(vertices.reshape(-1))
    params["shape.faces"] = mi.UInt32(faces.reshape(-1))
    params["integrator.colors"] = mi.Color3f(sh_colors.T)
    params["integrator.occupancy"] = mi.Float(occupancy)
    params.update()
    return scene


def render_reference(scene: mi.Scene, spp: int) -> mi.TensorXf:
    image = mi.render(scene, spp=spp, integrator=scene.integrator())
    dr.eval(image)
    return image


def _parse_rgb(value: str) -> tuple[float, float, float]:
    parts = [float(v.strip()) for v in value.split(",")]
    if len(parts) != 3:
        raise ValueError("--background-radiance requires three comma-separated values")
    return (parts[0], parts[1], parts[2])


def main(args: argparse.Namespace) -> None:
    dataset_dir = Path(args.output).resolve()
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "train_images").mkdir(exist_ok=True)
    (dataset_dir / "test_images").mkdir(exist_ok=True)

    mi.set_variant("cuda_ad_rgb", "llvm_ad_rgb")
    from trianglesoup.rendering import integrator as integrator_module

    integrator_module.register()

    vertices, faces = create_cube_mesh()
    n_triangles = faces.shape[0]
    rng = np.random.default_rng(args.seed)
    colors = rng.uniform(0.1, 1.0, size=(n_triangles, 3)).astype(np.float32)
    occupancy = np.full(n_triangles, 0.5, dtype=np.float32)
    background = _parse_rgb(args.background_radiance)

    resolution = args.resolution
    spp = args.spp

    train_points = fibonacci_sphere(args.train_cameras, radius=args.radius)
    test_points = fibonacci_sphere(args.test_cameras, radius=args.radius * 1.05)

    def generate_views(points: List[np.ndarray], split: str) -> dict:
        data = {"w": resolution, "h": resolution}
        for idx, origin in enumerate(points):
            view_name = f"view{idx:03d}"
            data[view_name] = sensor_to_json(sensor_dict(origin, resolution))
        json_path = dataset_dir / f"{split}.json"
        with json_path.open("w") as f:
            json.dump(data, f, indent=2)
        return data

    train_json = generate_views(train_points, "train")
    test_json = generate_views(test_points, "test")

    def render_split(sensors: List[mi.Sensor], split: str) -> None:
        for sensor in sensors:
            scene = configure_scene(
                vertices, faces, colors, occupancy, sensor, background
            )
            image = render_reference(scene, spp)
            hash_id = utils.sensor_hash_to_10_digits(sensor)
            filename = f"{split}_view_{hash_id}.exr"
            out_path = dataset_dir / f"{split}_images" / filename
            mi.Bitmap(image).write(str(out_path))

    train_sensors = utils.load_sensors_from_json(str(dataset_dir / "train.json"))
    test_sensors = utils.load_sensors_from_json(str(dataset_dir / "test.json"))

    render_split(train_sensors, "train")
    render_split(test_sensors, "test")

    np.savez(
        dataset_dir / "reference_state.npz",
        vertex_positions=vertices,
        faces=faces,
        colors=colors,
        occupancy=occupancy,
    )

    ply_path = dataset_dir / "cube.ply"
    with ply_path.open("w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")
        for v in vertices:
            f.write(f"{v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

    xml_path = dataset_dir / "scene.xml"
    with xml_path.open("w") as f:
        f.write("""
<scene version="2.3.0">
  <integrator type="soup_emissive">
    <integer name="max_depth" value="16"/>
  </integrator>
  <emitter type="constant">
    <rgb name="radiance" value="0.2, 0.2, 0.2"/>
  </emitter>
  <shape type="ply">
    <string name="filename" value="cube.ply"/>
    <bsdf type="null"/>
  </shape>
</scene>
""".strip())

    print(f"Generated cube dataset at {dataset_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate cube dataset")
    parser.add_argument("--output", default="data/datasets/cube", help="output directory")
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--spp", type=int, default=64)
    parser.add_argument("--train-cameras", type=int, default=12)
    parser.add_argument("--test-cameras", type=int, default=6)
    parser.add_argument("--radius", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--background-radiance",
        default="0.2,0.2,0.2",
        help="Scene background radiance as R,G,B",
    )
    main(parser.parse_args())
