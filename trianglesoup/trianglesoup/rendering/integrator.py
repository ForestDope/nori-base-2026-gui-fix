"""Skeleton integrator for emissive triangle soup rendering.

Students will extend this in four steps:
  Part 1: Window function (in triangle_model.py)
  Part 2: Alpha compositing loop (forward/primal pass)
  Part 3: PRB backward pass (differentiable rendering)
  Part 4: Triangle statistics for pruning/densification
"""

from __future__ import annotations

import drjit as dr
import mitsuba as mi
from mitsuba.python.ad.integrators.common import RBIntegrator

from . import triangle_model


class TriangleSoupEmissive(RBIntegrator):
    """Integrator that renders emissive triangle soup.

    Currently only returns the flat colour of the first triangle hit.
    After completing Parts 2-4, rays will pass through multiple
    semi-transparent triangles, accumulating emission weighted by
    per-triangle occupancy (alpha compositing along the ray).
    """

    def __init__(self, props: mi.Properties = mi.Properties()) -> None:
        super().__init__(props)
        self.throughput_thres = props.get("throughput_thres", 1e-4)
        self.max_depth = props.get("max_depth", 64)

        self.triangle_model = triangle_model.TriangleModel(0)
        self.background_color = mi.Color3f(0.0)

        self.collect_triangle_stats = False
        self.triangle_stats_weight_sum = dr.zeros(mi.Float, 0)
        self.triangle_stats_pixel_count = dr.zeros(mi.UInt32, 0)

    @dr.syntax
    def sample(self, mode, scene, sampler, ray, δL, state_in, active, **kwargs):
        primal = mode == dr.ADMode.Primal
        active = mi.Bool(active)
        ray = mi.Ray3f(ray)

        throughput = mi.Float(1.0)
        radiance = mi.Color3f(0.0 if primal else state_in[0])
        δL = mi.Color3f(δL if δL is not None else 0.0)
        depth = mi.UInt32(0)

        while active & (depth < self.max_depth):
            active_next = mi.Bool(active)

            # Intersect the ray with the scene.
            si = scene.ray_intersect(ray, active_next)

            # Rays that escape the scene get the background colour.
            active_escaped = active_next & ~si.is_valid()
            radiance[active_escaped] += self.background_color * throughput
            active_next &= si.is_valid()

            # --- Skeleton: returns colour * occupancy of the first hit ----
            emission = self.triangle_model.eval_color(si, active_next)
            occupancy = self.triangle_model.get_occupancy(si, active_next)
            radiance[active_next] = occupancy * emission
            active_next = mi.Bool(False)
            # -------------------------------------------------------------

            # TODO (Part 2): Replace the skeleton block above with alpha
            #   compositing along the ray.

            # TODO (Part 3): Add the PRB backward pass for differentiable
            #   rendering.

            # Collect per-triangle statistics (primal only).
            if primal and self.collect_triangle_stats:
                # TODO (Part 4): Collect per-triangle statistics for
                #   pruning and densification.
                pass

            active = active_next
            depth += 1

        return (
            radiance if primal else δL,
            mi.Bool(True),
            [],
            [radiance],
        )

    def begin_triangle_statistics(self, n_triangles: int) -> None:
        """Allocate buffers for per-triangle statistics collection."""
        self.collect_triangle_stats = True
        self.triangle_stats_weight_sum = dr.zeros(mi.Float, n_triangles)
        self.triangle_stats_pixel_count = dr.zeros(mi.UInt32, n_triangles)

    def end_triangle_statistics(self) -> tuple:
        """Finalize statistics and return (avg_weight, pixel_count) arrays."""
        count_f = mi.Float(self.triangle_stats_pixel_count)
        avg_weight = dr.select(
            count_f > 0,
            self.triangle_stats_weight_sum / count_f,
            0.0,
        )
        dr.eval(avg_weight, self.triangle_stats_pixel_count)
        self.collect_triangle_stats = False
        return avg_weight.numpy(), self.triangle_stats_pixel_count.numpy()

    def aov_names(self):
        return []

    def traverse(self, callback):
        self.triangle_model.traverse(callback)


def register() -> None:
    mi.register_integrator("soup_emissive", lambda props: TriangleSoupEmissive(props))
