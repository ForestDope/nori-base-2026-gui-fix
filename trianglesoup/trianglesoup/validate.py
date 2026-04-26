#!/usr/bin/env python3
"""Validate student implementations part by part.

Compares your implementation against pre-generated reference data
in ``data/validation/``.

Usage::

    pixi run validate-part1
    pixi run validate-part2
    pixi run validate-part3
    pixi run validate-part4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


CUBE_STATE = Path("data/datasets/cube/reference_state.npz")
REF_DIR = Path("data/validation")
OUTPUT_DIR = Path("output/validation")
BACKGROUND_COLOR = (0.15, 0.18, 0.25)  # dim blue


def _check_refs(part_dir):
    if not part_dir.exists():
        print(f"ERROR: Reference data not found at {part_dir}/")
        print("  Run: pixi run generate-validation-refs")
        sys.exit(1)


def _load_cube_state():
    if not CUBE_STATE.exists():
        print(f"ERROR: {CUBE_STATE} not found. Run: pixi run make-cube-dataset")
        sys.exit(1)
    state = np.load(CUBE_STATE)
    return {
        "vertices": state["vertex_positions"].astype(np.float32),
        "faces": state["faces"].astype(np.uint32),
        "colors": state["colors"].astype(np.float32),
        "occupancy": state["occupancy"].astype(np.float32),
    }


def _build_scene(state, integrator_module, tm_module):
    import mitsuba as mi
    import drjit as dr

    vertices, faces = state["vertices"], state["faces"]
    colors, occupancy = state["colors"], state["occupancy"]
    n_triangles = faces.shape[0]

    integrator_module.register()

    mesh = mi.Mesh("soup", vertices.shape[0], n_triangles)
    mp = mi.traverse(mesh)
    mp["vertex_positions"] = mi.Float(vertices.reshape(-1))
    mp["faces"] = mi.UInt32(faces.reshape(-1))
    mp.update()
    mesh.set_bsdf(mi.load_dict({"type": "null"}))

    scene = mi.load_dict({
        "type": "scene",
        "integrator": {"type": "soup_emissive"},
        "shape": mesh,
    })
    integrator = scene.integrator()
    integrator.background_color = mi.Color3f(BACKGROUND_COLOR)

    integrator.triangle_model = tm_module.TriangleModel(n_triangles)
    integrator.triangle_model.colors = mi.Color3f(colors.T)
    integrator.triangle_model.occupancy = mi.Float(occupancy)

    params = mi.traverse(scene)
    params["integrator.colors"] = mi.Color3f(colors.T)
    params["integrator.occupancy"] = mi.Float(occupancy)
    params.update()

    return scene, integrator


def _make_sensor(origin=(0, 0, 3), resolution=128):
    import mitsuba as mi
    transform = mi.ScalarTransform4f().look_at(
        origin=list(origin), target=[0, 0, 0], up=[0, 1, 0]
    )
    return mi.load_dict({
        "type": "perspective", "fov": 45, "to_world": transform,
        "film": {"type": "hdrfilm", "width": resolution, "height": resolution,
                 "sample_border": False, "filter": {"type": "box"}},
        "sampler": {"type": "independent", "sample_count": 1},
    })


def _render(scene, integrator, sensor, spp=64):
    import mitsuba as mi
    import drjit as dr
    image = mi.render(scene, sensor=sensor, spp=spp, integrator=integrator)
    dr.eval(image)
    return image


def _save_image(image, path_without_ext):
    """Save image as both .exr and .png."""
    import mitsuba as mi
    p = Path(path_without_ext)
    p.parent.mkdir(parents=True, exist_ok=True)
    mi.util.write_bitmap(str(p.with_suffix(".exr")), image)
    bmp = mi.Bitmap(image).convert(mi.Bitmap.PixelFormat.RGB, mi.Struct.Type.UInt8, True)
    bmp.write(str(p.with_suffix(".png")))


def _load_exr(path):
    import mitsuba as mi
    return np.array(mi.Bitmap(str(path)))


def _compare_image(name, student_img, ref_path, atol=1e-5, rtol=1e-5):
    import mitsuba as mi
    ref = _load_exr(ref_path)
    student = np.array(mi.Bitmap(student_img))
    # Crop to matching channels (reference may be RGB or RGBA)
    c = min(student.shape[-1], ref.shape[-1])
    ref, student = ref[..., :c], student[..., :c]
    try:
        np.testing.assert_allclose(student, ref, atol=atol, rtol=rtol)
        print(f"  {name}: OK")
        return True
    except AssertionError:
        diff = np.abs(student - ref)
        n_bad = int(np.sum(diff > atol + rtol * np.abs(ref)))
        total = diff.size
        print(f"  {name}: FAIL ({n_bad}/{total} pixels exceed tolerance, "
              f"max_abs_diff={diff.max():.6f})")
        return False


# ---- Part-specific gradient helper (shared with generate script) ----

def _compute_gradients(integrator_module, tm_module, state, perturb_seed=7):
    """Render with perturbed params and return AD gradients for colors, occupancy, sigma, and vertex positions."""
    import mitsuba as mi
    import drjit as dr

    vertices, faces = state["vertices"], state["faces"]
    colors, occupancy = state["colors"], state["occupancy"]
    n_triangles = faces.shape[0]

    integrator_module.register()

    mesh = mi.Mesh("soup", vertices.shape[0], n_triangles)
    mp = mi.traverse(mesh)
    mp["vertex_positions"] = mi.Float(vertices.reshape(-1))
    mp["faces"] = mi.UInt32(faces.reshape(-1))
    mp.update()
    mesh.set_bsdf(mi.load_dict({"type": "null"}))

    scene = mi.load_dict({
        "type": "scene",
        "integrator": {"type": "soup_emissive"},
        "shape": mesh,
    })
    integrator = scene.integrator()
    integrator.background_color = mi.Color3f(BACKGROUND_COLOR)
    integrator.triangle_model = tm_module.TriangleModel(n_triangles)
    integrator.triangle_model.colors = mi.Color3f(colors.T)
    integrator.triangle_model.occupancy = mi.Float(occupancy)

    params = mi.traverse(scene)
    params["integrator.colors"] = mi.Color3f(colors.T)
    params["integrator.occupancy"] = mi.Float(occupancy)
    params.update()

    sensor = _make_sensor(origin=(0, 0, 3), resolution=128)
    spp, seed = 16, 42

    target = mi.render(scene, sensor=sensor, spp=spp, seed=seed, integrator=integrator)
    dr.eval(target)

    rng = np.random.default_rng(perturb_seed)

    # --- Colors ---
    perturbed = np.clip(colors + rng.normal(0, 0.08, colors.shape).astype(np.float32), 0.01, 0.99)
    perturbed_3xN = np.ascontiguousarray(perturbed.T, dtype=np.float32)

    leaf_c = mi.Color3f(perturbed_3xN)
    dr.enable_grad(leaf_c)
    params["integrator.colors"] = leaf_c
    integrator.triangle_model.colors = leaf_c
    params.update()

    img = mi.render(scene, sensor=sensor, spp=spp, seed=seed, integrator=integrator, params=params)
    loss = dr.mean(dr.abs(img - target))
    dr.backward(loss)
    grad_colors = np.array(dr.grad(leaf_c)).reshape(-1).copy()

    # Reset colors
    params["integrator.colors"] = mi.Color3f(colors.T)
    integrator.triangle_model.colors = mi.Color3f(colors.T)

    # --- Occupancy ---
    perturbed_occ = np.clip(occupancy + rng.normal(0, 0.08, occupancy.shape).astype(np.float32), 0.05, 0.95)
    leaf_o = mi.Float(perturbed_occ)
    dr.enable_grad(leaf_o)
    params["integrator.occupancy"] = leaf_o
    integrator.triangle_model.occupancy = leaf_o
    params.update()

    img = mi.render(scene, sensor=sensor, spp=spp, seed=seed, integrator=integrator, params=params)
    loss = dr.mean(dr.abs(img - target))
    dr.backward(loss)
    grad_occ = np.array(dr.grad(leaf_o)).reshape(-1).copy()

    # Reset occupancy
    params["integrator.occupancy"] = mi.Float(occupancy)
    integrator.triangle_model.occupancy = mi.Float(occupancy)

    # --- Sigma ---
    sigma_base = np.ones(n_triangles, dtype=np.float32)
    perturbed_sigma = np.clip(sigma_base + rng.normal(0, 0.2, n_triangles).astype(np.float32), 0.1, 5.0)
    leaf_s = mi.Float(perturbed_sigma)
    dr.enable_grad(leaf_s)
    params["integrator.sigma"] = leaf_s
    integrator.triangle_model.sigma = leaf_s
    params.update()

    img = mi.render(scene, sensor=sensor, spp=spp, seed=seed, integrator=integrator, params=params)
    loss = dr.mean(dr.abs(img - target))
    dr.backward(loss)
    grad_sigma = np.array(dr.grad(leaf_s)).reshape(-1).copy()

    # Reset sigma
    params["integrator.sigma"] = mi.Float(sigma_base)
    integrator.triangle_model.sigma = mi.Float(sigma_base)

    # --- Vertex positions ---
    perturbed_verts = vertices + rng.normal(0, 0.02, vertices.shape).astype(np.float32)
    leaf_p = mi.Float(perturbed_verts.reshape(-1))
    dr.enable_grad(leaf_p)
    params["shape.vertex_positions"] = leaf_p
    params.update()

    img = mi.render(scene, sensor=sensor, spp=spp, seed=seed, integrator=integrator, params=params)
    loss = dr.mean(dr.abs(img - target))
    dr.backward(loss)
    grad_pos = np.array(dr.grad(leaf_p)).reshape(-1).copy()

    return grad_colors, grad_occ, grad_sigma, grad_pos


# Derivative image parameter names
DERIV_PARAMS = [
    "dColor_red", "dColor_green", "dColor_blue",
    "dOcc", "dSigma",
    "dPos_v0x", "dPos_v0y", "dPos_v0z",
]


DERIV_TRI_INDICES = [8, 11]  # nearest and farthest from corner camera (2, 1.5, 2)


def _render_derivative_images(integrator_module, tm_module, state, out_dir,
                              tri_indices=None, resolution=128, suffix=""):
    """Render per-pixel derivative images using backward mode.

    For each pixel, renders with a fresh AD graph and calls
    ``dr.backward`` to read gradients on each parameter.
    Saves one EXR per (triangle, parameter) pair.
    """
    import mitsuba as mi
    import drjit as dr

    if tri_indices is None:
        tri_indices = DERIV_TRI_INDICES

    vertices, faces = state["vertices"], state["faces"]
    colors, occupancy = state["colors"], state["occupancy"]
    n_triangles = faces.shape[0]

    integrator_module.register()

    mesh = mi.Mesh("soup", vertices.shape[0], n_triangles)
    mp = mi.traverse(mesh)
    mp["vertex_positions"] = mi.Float(vertices.reshape(-1))
    mp["faces"] = mi.UInt32(faces.reshape(-1))
    mp.update()
    mesh.set_bsdf(mi.load_dict({"type": "null"}))
    scene = mi.load_dict({
        "type": "scene",
        "integrator": {"type": "soup_emissive"},
        "shape": mesh,
    })
    integ = scene.integrator()
    integ.background_color = mi.Color3f(BACKGROUND_COLOR)
    integ.triangle_model = tm_module.TriangleModel(n_triangles)
    integ.triangle_model.colors = mi.Color3f(np.ascontiguousarray(colors.T, dtype=np.float32))
    integ.triangle_model.occupancy = mi.Float(occupancy)
    integ.triangle_model.sigma = dr.full(mi.Float, 1.0, n_triangles)
    dr.eval(integ.triangle_model.colors,
            integ.triangle_model.occupancy,
            integ.triangle_model.sigma)
    params = mi.traverse(scene)

    out_dir.mkdir(parents=True, exist_ok=True)
    n_out = len(tri_indices)
    print(f"    Derivative images at {resolution}x{resolution} "
          f"for triangles {tri_indices}...")

    to_world = mi.ScalarTransform4f().look_at(
        origin=[2, 1.5, 2], target=[0, 0, 0], up=[0, 1, 0])

    pixel_sensor = mi.load_dict({
        "type": "perspective", "fov": 45, "to_world": to_world,
        "film": {
            "type": "hdrfilm",
            "width": resolution, "height": resolution,
            "crop_offset_x": 0, "crop_offset_y": 0,
            "crop_width": 1, "crop_height": 1,
            "sample_border": False, "filter": {"type": "box"},
        },
        "sampler": {"type": "independent", "sample_count": 1},
    })
    sensor_params = mi.traverse(pixel_sensor)

    deriv_color_r = np.zeros((n_out, resolution, resolution), dtype=np.float32)
    deriv_color_g = np.zeros((n_out, resolution, resolution), dtype=np.float32)
    deriv_color_b = np.zeros((n_out, resolution, resolution), dtype=np.float32)
    deriv_occ = np.zeros((n_out, resolution, resolution), dtype=np.float32)
    deriv_sigma = np.zeros((n_out, resolution, resolution), dtype=np.float32)
    deriv_pos_v0x = np.zeros((n_out, resolution, resolution), dtype=np.float32)
    deriv_pos_v0y = np.zeros((n_out, resolution, resolution), dtype=np.float32)
    deriv_pos_v0z = np.zeros((n_out, resolution, resolution), dtype=np.float32)

    leaf_c = mi.Color3f(np.ascontiguousarray(colors.T, dtype=np.float32))
    leaf_o = mi.Float(occupancy)
    leaf_s = dr.full(mi.Float, 1.0, n_triangles)
    leaf_p = mi.Float(vertices.reshape(-1))
    dr.enable_grad(leaf_c)
    dr.enable_grad(leaf_o)
    dr.enable_grad(leaf_s)
    dr.enable_grad(leaf_p)
    dr.eval(leaf_c, leaf_o, leaf_s, leaf_p)
    params["integrator.colors"] = leaf_c
    params["integrator.occupancy"] = leaf_o
    params["integrator.sigma"] = leaf_s
    params["shape.vertex_positions"] = leaf_p
    params.update()

    def _frozen_pixel_backward(leaf_c, leaf_o, leaf_s, leaf_p,
                               scene, scene_params, sensor, integrator):
        integrator.triangle_model.colors = leaf_c
        integrator.triangle_model.occupancy = leaf_o
        integrator.triangle_model.sigma = leaf_s
        img = mi.render(scene, sensor=sensor, spp=1, seed=42,
                        integrator=integrator, params=scene_params)
        dr.backward(dr.sum(img))
        return (dr.ravel(dr.grad(leaf_c)),
                dr.grad(leaf_o),
                dr.grad(leaf_s),
                dr.grad(leaf_p))

    frozen_fn = dr.freeze(_frozen_pixel_backward)

    from tqdm import tqdm
    for py in tqdm(range(resolution), desc="    Rows", unit="row"):
        for px in range(resolution):
            sensor_params["film.crop_offset"] = [px, py]
            sensor_params.update()
            g_c, g_o, g_s, g_p = frozen_fn(
                leaf_c, leaf_o, leaf_s, leaf_p,
                scene, params, pixel_sensor, integ,
            )

            for i, ti in enumerate(tri_indices):
                deriv_color_r[i, py, px] = g_c[ti * 3]
                deriv_color_g[i, py, px] = g_c[ti * 3 + 1]
                deriv_color_b[i, py, px] = g_c[ti * 3 + 2]
                deriv_occ[i, py, px] = g_o[ti]
                deriv_sigma[i, py, px] = g_s[ti]
                deriv_pos_v0x[i, py, px] = g_p[ti * 9]
                deriv_pos_v0y[i, py, px] = g_p[ti * 9 + 1]
                deriv_pos_v0z[i, py, px] = g_p[ti * 9 + 2]

            dr.set_grad(leaf_c, 0.0)
            dr.set_grad(leaf_o, 0.0)
            dr.set_grad(leaf_s, 0.0)
            dr.set_grad(leaf_p, 0.0)

    def _write_deriv(data, name):
        exr_path = out_dir / f"{name}{suffix}.exr"
        mi.util.write_bitmap(str(exr_path), data.astype(np.float32))
        png_path = out_dir / f"{name}{suffix}.png"
        mi.util.write_bitmap(str(png_path), data.astype(np.float32))

    for i, ti in enumerate(tri_indices):
        _write_deriv(deriv_color_r[i], f"dImage_dColor_tri{ti}_red")
        _write_deriv(deriv_color_g[i], f"dImage_dColor_tri{ti}_green")
        _write_deriv(deriv_color_b[i], f"dImage_dColor_tri{ti}_blue")
        _write_deriv(deriv_occ[i], f"dImage_dOcc_tri{ti}")
        _write_deriv(deriv_sigma[i], f"dImage_dSigma_tri{ti}")
        _write_deriv(deriv_pos_v0x[i], f"dImage_dPos_tri{ti}_v0x")
        _write_deriv(deriv_pos_v0y[i], f"dImage_dPos_tri{ti}_v0y")
        _write_deriv(deriv_pos_v0z[i], f"dImage_dPos_tri{ti}_v0z")
        print(f"    Triangle {ti}: saved {len(DERIV_PARAMS)} derivative images")

    return out_dir


# ---- Validation entry points ----

def _build_single_triangle_scene(integrator_module, tm_module,
                                  color, occupancy, sigma):
    """Build a scene with a single triangle facing the camera at the origin."""
    import mitsuba as mi
    import drjit as dr

    vertices = np.array([[-1, -1, 0], [1, -1, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.uint32)

    integrator_module.register()

    mesh = mi.Mesh("soup", 3, 1)
    mp = mi.traverse(mesh)
    mp["vertex_positions"] = mi.Float(vertices.reshape(-1))
    mp["faces"] = mi.UInt32(faces.reshape(-1))
    mp.update()
    mesh.set_bsdf(mi.load_dict({"type": "null"}))

    scene = mi.load_dict({
        "type": "scene",
        "integrator": {"type": "soup_emissive"},
        "shape": mesh,
    })
    integrator = scene.integrator()
    integrator.background_color = mi.Color3f(BACKGROUND_COLOR)
    integrator.triangle_model = tm_module.TriangleModel(1)
    integrator.triangle_model.colors = mi.Color3f(color)
    integrator.triangle_model.occupancy = mi.Float([occupancy])
    integrator.triangle_model.sigma = mi.Float([sigma])

    params = mi.traverse(scene)
    params["integrator.colors"] = mi.Color3f(color)
    params["integrator.occupancy"] = mi.Float([occupancy])
    params["integrator.sigma"] = mi.Float([sigma])
    params.update()

    return scene, integrator


def _generate_random_soup(seed=42):
    """Generate a deterministic random triangle-soup scene.

    Returns vertices (V,3), faces (N,3), colors (N,3), occupancy (N,),
    sigma (N,).  Some triangles are placed in front of the camera
    (visible), some behind it or far to the side (not visible).
    Camera assumed at (0, 0, 3) looking at origin.
    """
    rng = np.random.default_rng(seed)
    n_visible = 20
    n_behind = 8
    n_offscreen = 7
    n_total = n_visible + n_behind + n_offscreen

    verts_list = []
    for _ in range(n_visible):
        centre = rng.uniform(-0.8, 0.8, size=3).astype(np.float32)
        centre[2] = rng.uniform(-0.5, 0.5)
        size = rng.uniform(0.1, 0.5)
        tri = centre + rng.uniform(-size, size, size=(3, 3)).astype(np.float32)
        verts_list.append(tri)

    for _ in range(n_behind):
        centre = rng.uniform(-1, 1, size=3).astype(np.float32)
        centre[2] = rng.uniform(4.0, 6.0)
        size = rng.uniform(0.1, 0.4)
        tri = centre + rng.uniform(-size, size, size=(3, 3)).astype(np.float32)
        verts_list.append(tri)

    for _ in range(n_offscreen):
        centre = rng.uniform(-1, 1, size=3).astype(np.float32)
        axis = rng.choice([0, 1])
        centre[axis] = rng.choice([-1, 1]) * rng.uniform(4.0, 6.0)
        centre[2] = rng.uniform(-0.5, 0.5)
        size = rng.uniform(0.1, 0.4)
        tri = centre + rng.uniform(-size, size, size=(3, 3)).astype(np.float32)
        verts_list.append(tri)

    vertices = np.concatenate(verts_list, axis=0).astype(np.float32)
    faces = np.arange(n_total * 3, dtype=np.uint32).reshape(n_total, 3)

    colors = rng.uniform(0.1, 1.0, size=(n_total, 3)).astype(np.float32)
    occupancy = rng.uniform(0.1, 1.0, size=n_total).astype(np.float32)
    sigma = rng.uniform(0.5, 20.0, size=n_total).astype(np.float32)

    return {
        "vertices": vertices,
        "faces": faces,
        "colors": colors,
        "occupancy": occupancy,
        "sigma": sigma,
    }


def _build_random_soup_scene(integrator_module, tm_module, soup):
    """Build a Mitsuba scene from random soup state dict."""
    import mitsuba as mi
    import drjit as dr

    vertices, faces = soup["vertices"], soup["faces"]
    colors, occupancy, sigma = soup["colors"], soup["occupancy"], soup["sigma"]
    n_triangles = faces.shape[0]

    integrator_module.register()

    mesh = mi.Mesh("soup", vertices.shape[0], n_triangles)
    mp = mi.traverse(mesh)
    mp["vertex_positions"] = mi.Float(vertices.reshape(-1))
    mp["faces"] = mi.UInt32(faces.reshape(-1))
    mp.update()
    mesh.set_bsdf(mi.load_dict({"type": "null"}))

    scene = mi.load_dict({
        "type": "scene",
        "integrator": {"type": "soup_emissive"},
        "shape": mesh,
    })
    integrator = scene.integrator()
    integrator.background_color = mi.Color3f(BACKGROUND_COLOR)

    integrator.triangle_model = tm_module.TriangleModel(n_triangles)
    integrator.triangle_model.colors = mi.Color3f(colors.T)
    integrator.triangle_model.occupancy = mi.Float(occupancy)
    integrator.triangle_model.sigma = mi.Float(sigma)

    params = mi.traverse(scene)
    params["integrator.colors"] = mi.Color3f(colors.T)
    params["integrator.occupancy"] = mi.Float(occupancy)
    params["integrator.sigma"] = mi.Float(sigma)
    params.update()

    return scene, integrator


# Single-triangle test cases: (name, color, occupancy, sigma)
PART1_CASES = [
    ("sigma_1_occ_1",  [1.0, 0.0, 0.0], 1.0,  1.0),
    ("sigma_5_occ_1",  [1.0, 0.0, 0.0], 1.0,  5.0),
    ("sigma_10_occ_1", [1.0, 0.0, 0.0], 1.0, 10.0),
    ("sigma_5_occ_05", [0.0, 1.0, 0.0], 0.5,  5.0),
    ("sigma_5_occ_02", [0.0, 0.0, 1.0], 0.2,  5.0),
]


def validate_part1():
    """Part 1: Window function — single-triangle tests with different settings."""
    import mitsuba as mi
    mi.set_variant("cuda_ad_rgb", "llvm_ad_rgb")
    from trianglesoup.rendering import integrator as integrator_mod
    from trianglesoup.rendering import triangle_model as tm_mod

    ref_dir = REF_DIR / "part1"
    _check_refs(ref_dir)
    out = OUTPUT_DIR / "part1"

    print("Part 1: Window function (single-triangle scenes)")
    print("=" * 60)

    all_ok = True
    for name, color, occ, sigma in PART1_CASES:
        scene, integrator = _build_single_triangle_scene(
            integrator_mod, tm_mod, color, occ, sigma,
        )
        sensor = _make_sensor(origin=(0, 0, 3))
        image = _render(scene, integrator, sensor)
        _save_image(image, out / name)
        ok = _compare_image(f"{name} (sigma={sigma}, occ={occ})",
                            image, ref_dir / f"{name}_ref.exr")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("OK: All Part 1 renders match the reference.")
    else:
        print("Some renders differ from the reference.")
        print("  If all images look identical regardless of sigma, your")
        print("  window function is not implemented yet (still returns 1.0).")
        print("  sigma=1 should be very soft, sigma=50 nearly opaque.")
    print(f"\nYour renders saved to {out}/ (.exr and .png)")


def validate_part2():
    """Part 2: Alpha compositing — render and compare against reference."""
    import mitsuba as mi
    mi.set_variant("cuda_ad_rgb", "llvm_ad_rgb")
    import drjit as dr
    from trianglesoup.rendering import integrator as integrator_mod
    from trianglesoup.rendering import triangle_model as tm_mod

    ref_dir = REF_DIR / "part2"
    _check_refs(ref_dir)
    state = _load_cube_state()
    out = OUTPUT_DIR / "part2"

    print("Part 2: Alpha compositing")
    print("=" * 60)

    scene, integrator = _build_scene(state, integrator_mod, tm_mod)
    sensor = _make_sensor(origin=(2, 1.5, 2))
    image = _render(scene, integrator, sensor)
    _save_image(image, out / "corner")

    ok = _compare_image("corner", image, ref_dir / "corner_ref.exr")

    ref_stats = np.load(ref_dir / "stats_ref.npz")
    ref_arr = ref_stats["image"]
    student_arr = np.array(mi.Bitmap(image))

    n_nonzero = int(np.sum(student_arr.max(axis=-1) > 0.01))
    ref_nonzero = int(np.sum(ref_arr.max(axis=-1) > 0.01))
    total = student_arr.shape[0] * student_arr.shape[1]

    print(f"  Non-black pixels: {n_nonzero}/{total} (ref: {ref_nonzero}/{total})")

    print()
    if ok:
        print("OK: Your render matches the reference.")
    else:
        has_variety = student_arr[student_arr.max(axis=-1) > 0.01].std() > 0.02 if n_nonzero > 0 else False
        if not has_variety:
            print("WARNING: Image has very uniform colour. Back faces may not be")
            print("  visible — check that your loop continues past the first hit.")
        else:
            print("Render differs from reference. Check your alpha compositing logic.")
    print(f"\nYour render saved to {out}/ (.exr and .png)")


def _make_deriv_comparison(ref_dir, stu_dir, out_path,
                           tri_indices=None):
    """Create matplotlib comparison figures: Reference | Student | Difference."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if tri_indices is None:
        tri_indices = DERIV_TRI_INDICES

    # Map param name to EXR filename
    def _deriv_fname(param, ti, suffix=""):
        if param.startswith("dColor_"):
            return f"dImage_dColor_tri{ti}_{param.split('_')[1]}{suffix}.exr"
        if param.startswith("dPos_"):
            return f"dImage_dPos_tri{ti}_{param[len('dPos_'):]}{suffix}.exr"
        return f"dImage_{param}_tri{ti}{suffix}.exr"

    # Per-triangle derivative comparison (rows=Ref/Stu/Diff, cols=params)
    # Each param gets an image column + a thin colorbar column.
    # Ref/Student share one colorbar (spans rows 0-1), Difference has its own.
    from matplotlib.gridspec import GridSpec

    row_labels = ["Reference", "Student", "Difference"]
    cb_w = 0.08  # colorbar width relative to image column
    gap = 0.15   # extra gap between param groups
    fs = 9

    for ti in tri_indices:
        n_p = len(DERIV_PARAMS)
        # Width ratios: [img, cb, gap, img, cb, gap, ...] (no trailing gap)
        width_ratios = []
        for i in range(n_p):
            width_ratios.extend([1, cb_w])
            if i < n_p - 1:
                width_ratios.append(gap)
        total_cols = len(width_ratios)

        fig = plt.figure(figsize=(2.2 * n_p, 5.0))
        gs = GridSpec(3, total_cols, figure=fig,
                      width_ratios=width_ratios,
                      hspace=0.12, wspace=0.05)
        fig.suptitle(f"Derivative Comparison — Triangle {ti}", fontsize=fs + 2)

        for pi, param in enumerate(DERIV_PARAMS):
            base = pi * 3 if pi > 0 else 0
            if pi > 0:
                base = pi * 3  # skip gap columns
            img_c = base
            cb_c = base + 1

            ref_path = ref_dir / _deriv_fname(param, ti, "_ref")
            stu_path = stu_dir / _deriv_fname(param, ti)

            if not ref_path.exists() or not stu_path.exists():
                continue

            ref_img = _load_exr(ref_path)
            stu_img = _load_exr(stu_path)
            if ref_img.ndim == 3:
                ref_img = ref_img[..., 0]
            if stu_img.ndim == 3:
                stu_img = stu_img[..., 0]

            vmax = max(np.abs(ref_img).max(), np.abs(stu_img).max(), 1e-8)
            diff = stu_img - ref_img
            diff_vmax = max(np.abs(diff).max(), 1e-8)

            ax_ref = fig.add_subplot(gs[0, img_c])
            ax_stu = fig.add_subplot(gs[1, img_c])
            ax_dif = fig.add_subplot(gs[2, img_c])

            im_ref = ax_ref.imshow(ref_img, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax_stu.imshow(stu_img, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            im_diff = ax_dif.imshow(diff, cmap="PiYG", vmin=-diff_vmax, vmax=diff_vmax)

            ax_ref.set_title(param, fontsize=fs)
            for ax in (ax_ref, ax_stu, ax_dif):
                ax.set_xticks([])
                ax.set_yticks([])

            if pi == 0:
                ax_ref.set_ylabel(row_labels[0], fontsize=fs)
                ax_stu.set_ylabel(row_labels[1], fontsize=fs)
                ax_dif.set_ylabel(row_labels[2], fontsize=fs)

            # Colorbar for ref+student: spans rows 0-1, drawn into cax
            cax_rs = fig.add_subplot(gs[0:2, cb_c])
            cb_rs = fig.colorbar(im_ref, cax=cax_rs)
            cb_rs.ax.tick_params(labelsize=fs - 4)

            # Colorbar for difference: row 2
            cax_d = fig.add_subplot(gs[2, cb_c])
            cb_d = fig.colorbar(im_diff, cax=cax_d)
            cb_d.ax.tick_params(labelsize=fs - 4)
            cb_d.ax.yaxis.get_offset_text().set_visible(False)
            # Show max magnitude as title if values are tiny
            if diff_vmax < 1e-3:
                cb_d.ax.set_title(f"{diff_vmax:.0e}", fontsize=fs - 4, pad=2)

        fig_path = out_path / f"comparison_tri{ti}.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(fig_path), dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved comparison figure: {fig_path}")


def validate_part3():
    """Part 3: PRB backward — compare gradients and derivative images."""
    import mitsuba as mi
    mi.set_variant("cuda_ad_rgb", "llvm_ad_rgb")
    from trianglesoup.rendering import integrator as integrator_mod
    from trianglesoup.rendering import triangle_model as tm_mod

    ref_dir = REF_DIR / "part3"
    _check_refs(ref_dir)
    state = _load_cube_state()
    out = OUTPUT_DIR / "part3"

    # --- Gradient check ---
    ref_grad_c = np.load(ref_dir / "grad_colors_ref.npy")
    ref_grad_o = np.load(ref_dir / "grad_occupancy_ref.npy")
    ref_grad_s = np.load(ref_dir / "grad_sigma_ref.npy")
    ref_grad_p = np.load(ref_dir / "grad_positions_ref.npy")

    stu_grad_c, stu_grad_o, stu_grad_s, stu_grad_p = _compute_gradients(
        integrator_mod, tm_mod, state,
    )

    print("Part 3: Gradient check against reference")
    print("=" * 60)

    all_ok = True
    for name, ref_g, stu_g in [("colors", ref_grad_c, stu_grad_c),
                                ("occupancy", ref_grad_o, stu_grad_o),
                                ("sigma", ref_grad_s, stu_grad_s),
                                ("positions", ref_grad_p, stu_grad_p)]:
        nonzero = np.abs(ref_g) > 1e-5
        n_check = int(nonzero.sum())
        print(f"\n  {name} ({n_check} non-zero reference components):")

        if n_check == 0:
            print("    (no non-zero reference gradients to compare)")
            continue

        try:
            np.testing.assert_allclose(
                stu_g[nonzero], ref_g[nonzero], rtol=0.10, atol=1e-6,
            )
            print(f"    OK")
        except AssertionError:
            diff = np.abs(stu_g[nonzero] - ref_g[nonzero])
            rel = diff / np.maximum(np.abs(ref_g[nonzero]), 1e-6)
            n_bad = int(np.sum(rel > 0.10))
            print(f"    FAIL ({n_bad}/{n_check} components exceed 10% relative tolerance, "
                  f"max_rel_err={rel.max():.4f})")
            all_ok = False

        # Print a few sample components for visibility
        indices = np.where(nonzero)[0][:6]
        for idx in indices:
            r, s = float(ref_g[idx]), float(stu_g[idx])
            denom = max(abs(r), abs(s), 1e-6)
            rel = abs(r - s) / denom
            print(f"    [{idx:>2}] ref={r: .5e}  yours={s: .5e}  rel_err={rel:.3f}")

    print()
    if all_ok:
        print("OK: Your gradients match the reference.")
    else:
        stu_any = (np.any(np.abs(stu_grad_c) > 1e-6) or
                   np.any(np.abs(stu_grad_o) > 1e-6) or
                   np.any(np.abs(stu_grad_s) > 1e-6) or
                   np.any(np.abs(stu_grad_p) > 1e-6))
        if not stu_any:
            print("WARNING: All your gradients are zero — the backward pass is")
            print("  not implemented yet (Part 3 TODO in integrator).")
        else:
            print("WARNING: Some gradients differ from the reference.")
            print("  Check your PRB backward pass in the integrator.")

    # --- Derivative images ---
    ref_deriv_dir = ref_dir / "derivative_images"
    if ref_deriv_dir.exists():
        print(f"\nRendering student derivative images...")
        stu_deriv_dir = out / "student_derivatives"
        _render_derivative_images(integrator_mod, tm_mod, state, stu_deriv_dir)

        print(f"\nGenerating comparison figures...")
        _make_deriv_comparison(ref_deriv_dir, stu_deriv_dir, out)
    else:
        print(f"\nSkipping derivative image comparison (no reference at {ref_deriv_dir})")


def validate_part4():
    """Part 4: Triangle statistics — collect and compare against reference."""
    import mitsuba as mi
    mi.set_variant("cuda_ad_rgb", "llvm_ad_rgb")
    import drjit as dr
    from trianglesoup.rendering import integrator as integrator_mod
    from trianglesoup.rendering import triangle_model as tm_mod

    ref_dir = REF_DIR / "part4"
    _check_refs(ref_dir)

    soup = _generate_random_soup()
    scene, integrator = _build_random_soup_scene(
        integrator_mod, tm_mod, soup,
    )
    n_triangles = soup["faces"].shape[0]
    sensor = _make_sensor(origin=(0, 0, 3))

    print("Part 4: Triangle statistics")
    print("=" * 60)

    try:
        integrator.begin_triangle_statistics(n_triangles)
    except NotImplementedError:
        print("  begin_triangle_statistics not implemented yet.")
        return

    image = mi.render(scene, sensor=sensor, spp=16, integrator=integrator)
    dr.eval(image)

    try:
        avg_weight, pixel_count = integrator.end_triangle_statistics()
    except NotImplementedError:
        print("  end_triangle_statistics not implemented yet.")
        return

    ref_avg = np.load(ref_dir / "avg_weight_ref.npy")
    ref_count = np.load(ref_dir / "pixel_count_ref.npy")

    visible = pixel_count > 0
    ref_visible = ref_count > 0
    n_visible = int(visible.sum())
    ref_n_visible = int(ref_visible.sum())

    print(f"  Visible triangles: {n_visible} (ref: {ref_n_visible})")

    all_ok = True

    if n_visible != ref_n_visible:
        print(f"  FAIL: visible triangle count mismatch")
        all_ok = False

    if n_visible > 0 and ref_n_visible > 0:
        # Compare pixel counts on commonly visible triangles
        common = visible & ref_visible
        if common.any():
            try:
                np.testing.assert_array_equal(pixel_count[common], ref_count[common])
                print(f"  pixel_count: OK (exact match)")
            except AssertionError:
                diff = np.abs(pixel_count[common].astype(int) - ref_count[common].astype(int))
                n_bad = int(np.sum(diff > 0))
                print(f"  pixel_count: FAIL ({n_bad}/{int(common.sum())} triangles differ, "
                      f"max_diff={diff.max()})")
                all_ok = False

            try:
                np.testing.assert_allclose(
                    avg_weight[common], ref_avg[common], rtol=0.1, atol=1e-4,
                )
                print(f"  avg_weight: OK")
            except AssertionError:
                rel_err = np.abs(avg_weight[common] - ref_avg[common]) / np.maximum(np.abs(ref_avg[common]), 1e-6)
                n_bad = int(np.sum(rel_err > 0.1))
                print(f"  avg_weight: FAIL ({n_bad}/{int(common.sum())} triangles exceed tolerance, "
                      f"max_rel_err={rel_err.max():.4f})")
                all_ok = False

    print()
    if all_ok and n_visible > 0:
        print("OK: Your statistics match the reference.")
    elif n_visible == 0:
        print("WARNING: No statistics collected. Check your scatter_add calls")
        print("  inside the rendering loop.")
    else:
        print("Some statistics differ from the reference.")


def main():
    parser = argparse.ArgumentParser(description="Validate homework parts.")
    parser.add_argument("part", type=int, choices=[1, 2, 3, 4],
                        help="Which part to validate (1-4)")
    args = parser.parse_args()

    {1: validate_part1, 2: validate_part2, 3: validate_part3, 4: validate_part4}[args.part]()


if __name__ == "__main__":
    main()
