"""Initialise a triangle soup from a ground-truth Mitsuba scene mesh.

``mesh_init`` loads every shape in a Mitsuba ``.xml`` scene, extracts
its triangles and per-shape BSDF reflectance colour, subsamples to a
target count, and optionally scales each subsampled triangle outward
from its centroid so it reaches its nearest-neighbour triangle —
filling the gaps that sparse random subsampling leaves behind.

This is the recommended way to bootstrap training: starting from the
GT mesh gives the optimiser a head start compared to random init and
avoids the entire SfM detour.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def mesh_init(
    scene_xml: Path,
    target_triangles: int = 5000,
    scale_to_nn: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Initialise the triangle soup from the ground-truth Mitsuba scene mesh.

    Args:
        scene_xml: Path to the Mitsuba ``.xml`` scene file.
        target_triangles: Approximate number of triangles to keep.
        scale_to_nn: When ``True`` (default), scale each subsampled
            triangle so its radius roughly matches the distance to the
            nearest neighbouring triangle centroid.

    Returns:
        vertices: ``(M*3, 3)`` float32 vertex positions.
        faces: ``(M, 3)`` uint32 face indices (sequential).
        colors: ``(M, 3)`` float32 linear-RGB colour per triangle.
    """
    import mitsuba as mi  # type: ignore[import-not-found]

    if mi.variant() is None:
        mi.set_variant("scalar_rgb")

    scene = mi.load_file(str(scene_xml))

    all_triangles: list[np.ndarray] = []  # (3, 3) per triangle
    all_colors: list[np.ndarray] = []     # (3,) per triangle

    for shape in scene.shapes():
        params = mi.traverse(shape)
        verts = np.array(params["vertex_positions"]).reshape(-1, 3)
        face_idx = np.array(params["faces"]).reshape(-1, 3)
        tris = verts[face_idx]  # (N, 3, 3)

        # Per-shape BSDF reflectance as the initial colour.
        bsdf = shape.bsdf()
        bp = mi.traverse(bsdf)
        color = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        for k in bp.keys():
            if "reflectance" in k:
                c = np.array(bp[k]).flatten()
                if c.size >= 3:
                    color = c[:3].astype(np.float32)
                elif c.size == 1:
                    color = np.full(3, float(c[0]), dtype=np.float32)
                break

        all_triangles.append(tris.astype(np.float32))
        all_colors.append(
            np.broadcast_to(color, (len(tris), 3)).copy().astype(np.float32)
        )

    triangles = np.concatenate(all_triangles, axis=0)  # (T, 3, 3)
    colors = np.concatenate(all_colors, axis=0)         # (T, 3)
    print(f"GT mesh: {len(triangles)} triangles from {len(scene.shapes())} shapes")

    # Subsample to target count.
    if len(triangles) > target_triangles:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(triangles), target_triangles, replace=False)
        triangles = triangles[idx]
        colors = colors[idx]
        print(f"  -> subsampled to {len(triangles)} triangles")

    # Scale each triangle outward from its centroid so it fully covers the
    # local patch, overlapping its neighbours. Using the average distance
    # to the k nearest neighbours (instead of just the 1-NN) biases the
    # triangle to cover a ring of neighbours, which eliminates the
    # circumferential holes that random subsampling leaves behind.
    # The overlap factor gives a small safety margin so adjacent triangles
    # visibly intersect at init, closing any residual gaps.
    if scale_to_nn and len(triangles) > 1 and len(triangles) < 200000:
        centroids = triangles.mean(axis=1)  # (N, 3)

        offsets = triangles - centroids[:, None, :]  # (N, 3, 3)
        current_radius = np.linalg.norm(offsets, axis=2).max(axis=1)  # (N,)
        current_radius = np.maximum(current_radius, 1e-8)

        k = min(8, max(1, len(triangles) - 1))
        nn_dist = _knn_avg_dist(centroids, k=k)  # mean dist to k neighbours

        overlap = 1.6  # adjacent triangles overlap ~60%
        scale = np.clip(nn_dist * overlap / current_radius, 1.0, 100.0)

        triangles = centroids[:, None, :] + offsets * scale[:, None, None]
        triangles = triangles.astype(np.float32)
        print(
            f"  -> scaled triangles to k={k}-NN × {overlap} "
            f"(median scale {np.median(scale):.1f}x)"
        )

    n = len(triangles)
    vertices = triangles.reshape(-1, 3).astype(np.float32)
    faces = np.arange(n * 3, dtype=np.uint32).reshape(n, 3)
    colors = colors.astype(np.float32)

    print(f"  -> {n} triangles, mean color {colors.mean(axis=0)}")
    return vertices, faces, colors


def _knn_avg_dist(points: np.ndarray, k: int) -> np.ndarray:
    """Average distance to ``k`` nearest neighbours using numpy (no scipy)."""
    n = len(points)
    if n <= k:
        dists = np.linalg.norm(points[:, None] - points[None, :], axis=2)
        return np.mean(dists, axis=1)

    # We want the k+1 smallest distances per row (k neighbours + self at 0).
    kth = min(k, n - 1)
    batch_size = min(n, 4096)
    avg_dists = np.zeros(n, dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        diff = points[start:end, None, :] - points[None, :, :]
        dists = np.linalg.norm(diff, axis=2)
        partitioned = np.partition(dists, kth, axis=1)[:, : kth + 1]
        sorted_k = np.sort(partitioned, axis=1)[:, 1:]  # skip self (dist=0)
        avg_dists[start:end] = sorted_k.mean(axis=1)
    return avg_dists
