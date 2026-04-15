"""Skeleton per-triangle model for students.

Students will implement the window function (Part 1) in ``_window_function``.
All other methods are provided.
"""

from __future__ import annotations

import drjit as dr  # type: ignore
import mitsuba as mi  # type: ignore


class TriangleModel:
    """Per-triangle colour, occupancy, and window-function sharpness σ.

    The window function (Eq. 1 of the Triangle Splatting paper) should
    compute::

        I(p) = clip(min(b₀·w₀, b₁·w₁, b₂·w₂), 0, 1)^σ

    where ``wᵢ = perimeter / aᵢ`` and ``aᵢ`` is the edge length opposite
    vertex ``i``.  Vertex positions are read from the mesh via
    ``mi.MeshPtr(si.shape)``.
    """

    def __init__(self, n_primitives: int = 0, initial_sigma: float = 1.0) -> None:
        self.n_primitives = n_primitives
        self.colors = dr.full(mi.Color3f, 0.5, n_primitives)
        self.occupancy = dr.full(mi.Float, 0.5, n_primitives)
        self.sigma = dr.full(mi.Float, initial_sigma, n_primitives)

    def eval_color(self, si: mi.SurfaceInteraction3f, active: mi.Bool) -> mi.Color3f:
        """Per-triangle RGB emission — view-independent."""
        return dr.gather(mi.Color3f, self.colors, si.prim_index, active)

    def _window_function(self, si: mi.SurfaceInteraction3f, active: mi.Bool) -> mi.Float:
        """Compute the soft falloff from triangle centre to edges.

        Currently returns 1.0 (no falloff). After implementing Part 1,
        this should return clip(min(b0*w0, b1*w1, b2*w2), 0, 1)^sigma.
        """
        sigma = dr.gather(mi.Float, self.sigma, si.prim_index, active)

        # Get the three vertices of the hit triangle from the mesh.
        idx = si.prim_index
        mesh_ptr = mi.MeshPtr(si.shape)
        v0 = mesh_ptr.vertex_position(idx * 3 + 0, active)
        v1 = mesh_ptr.vertex_position(idx * 3 + 1, active)
        v2 = mesh_ptr.vertex_position(idx * 3 + 2, active)

        # Barycentric coordinates from the surface interaction.
        b0 = dr.maximum(0.0, 1.0 - si.uv.x - si.uv.y)
        b1 = dr.maximum(0.0, si.uv.x)
        b2 = dr.maximum(0.0, si.uv.y)

        # TODO (Part 1): Compute the window function I(p).
        # Replace the line below with your implementation.

        return mi.Float(1.0)

    def get_occupancy(self, si: mi.SurfaceInteraction3f, active: mi.Bool) -> mi.Float:
        base = dr.gather(mi.Float, self.occupancy, si.prim_index, active)
        return base * self._window_function(si, active)

    def get_transparency(self, si: mi.SurfaceInteraction3f, active: mi.Bool) -> mi.Color3f:
        """Differentiable ``1 − occupancy`` used by the PRB backward pass."""
        return mi.Color3f(1.0 - self.get_occupancy(si, active))

    def traverse(self, callback: mi.TraversalCallback) -> None:
        callback.put("colors", self.colors, mi.ParamFlags.Differentiable)
        callback.put("occupancy", self.occupancy, mi.ParamFlags.Differentiable)
        callback.put("sigma", self.sigma, mi.ParamFlags.Differentiable)
