"""Loss functions for triangle soup optimization (Eq. 3 of the paper)."""

from __future__ import annotations

import drjit as dr
import mitsuba as mi


def photometric_loss(rendered: mi.Color3f, target: mi.Color3f) -> mi.Float:
    """Per-pixel photometric loss (L1: mean absolute error)."""
    return dr.mean(dr.abs(rendered - target))


def occupancy_regularization(occupancy: mi.Float) -> mi.Float:
    """L_o: penalize high occupancy to encourage sparsity (from 3DGS-MCMC)."""
    return dr.mean(dr.abs(occupancy))


def size_regularization(vertex_positions: mi.Float, n_triangles: int) -> mi.Float:
    """L_s from eq. 3 of the Triangle Splatting paper.

        L_s = mean( 2 · ||(v1-v0) × (v2-v0)||₂⁻¹ )

    ||(v1-v0) × (v2-v0)|| is twice the triangle area, so the term equals
    mean(1/area). Minimizing L_s encourages larger triangles.
    """
    idx = dr.arange(mi.UInt32, n_triangles)

    v0x = dr.gather(mi.Float, vertex_positions, idx * 9 + 0)
    v0y = dr.gather(mi.Float, vertex_positions, idx * 9 + 1)
    v0z = dr.gather(mi.Float, vertex_positions, idx * 9 + 2)
    v1x = dr.gather(mi.Float, vertex_positions, idx * 9 + 3)
    v1y = dr.gather(mi.Float, vertex_positions, idx * 9 + 4)
    v1z = dr.gather(mi.Float, vertex_positions, idx * 9 + 5)
    v2x = dr.gather(mi.Float, vertex_positions, idx * 9 + 6)
    v2y = dr.gather(mi.Float, vertex_positions, idx * 9 + 7)
    v2z = dr.gather(mi.Float, vertex_positions, idx * 9 + 8)

    e1x, e1y, e1z = v1x - v0x, v1y - v0y, v1z - v0z
    e2x, e2y, e2z = v2x - v0x, v2y - v0y, v2z - v0z

    cx = e1y * e2z - e1z * e2y
    cy = e1z * e2x - e1x * e2z
    cz = e1x * e2y - e1y * e2x

    cross_norm = dr.sqrt(cx * cx + cy * cy + cz * cz + 1e-12)
    return dr.mean(2.0 / cross_norm)
