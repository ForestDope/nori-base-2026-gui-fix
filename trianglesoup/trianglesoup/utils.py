"""Shared helpers for triangle-soup camera datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, List

import mitsuba as mi
import numpy as np


def sensor_hash_to_10_digits(sensor: mi.Sensor) -> str:
    """Return a short deterministic hash for a Mitsuba sensor.

    The hash is intentionally short; it is used only for stable file basenames.
    """
    transform = sensor.world_transform()
    matrix = getattr(transform, "matrix", None)
    if matrix is None:
        matrix = np.array(sensor.parameters_dict()["to_world"])
    flat = np.array(matrix, dtype=np.float64).reshape(-1)
    digest = hashlib.sha1(flat.tobytes()).hexdigest()
    return digest[:10]


def _iter_sensor_entries(scene_dict: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    for name in sorted(k for k in scene_dict.keys() if k not in {"w", "h", "width", "height"}):
        value = scene_dict[name]
        if not isinstance(value, dict):
            continue
        if "to_world" not in value and "transform" not in value:
            continue
        yield name, value


def load_sensors_from_json(
    path: str | Path,
    *,
    return_specs: bool = False,
    jitter_pixels: bool = False,
) -> List[mi.Sensor] | tuple[List[mi.Sensor], List[dict]]:
    """Load all sensor definitions from a JSON dictionary.

    ``jitter_pixels=False`` (default) samples the exact pixel center every
    iteration via a non-jittered stratified sampler. ``jitter_pixels=True``
    uses the jittered ``independent`` sampler so each iteration sees a
    fresh sub-pixel offset — enables temporal antialiasing on triangle
    edges, at the cost of noisier per-iteration gradients.
    """
    with Path(path).open("r") as f:
        scene_dict = json.load(f)

    width = int(scene_dict.get("w", scene_dict.get("width", 512)))
    height = int(scene_dict.get("h", scene_dict.get("height", 512)))

    sampler_dict = (
        {"type": "independent", "sample_count": 1}
        if jitter_pixels
        else {"type": "stratified", "sample_count": 1, "jitter": False}
    )

    sensors: List[mi.Sensor] = []
    specs: List[dict] = []
    for _name, spec in _iter_sensor_entries(scene_dict):
        sensor_spec = dict(spec)
        to_world = sensor_spec.get("to_world")
        if to_world is not None:
            if isinstance(to_world, dict):
                raise ValueError("Expected to_world as a 4x4 matrix, got dict")
            sensor_spec["to_world"] = mi.ScalarTransform4f(to_world)

        sensor_spec.setdefault(
            "film",
            {
                "type": "hdrfilm",
                "width": width,
                "height": height,
                "filter": {"type": "box"},
                "sample_border": False,
            },
        )
        sensor_spec["sampler"] = dict(sampler_dict)
        sensors.append(mi.load_dict(sensor_spec))
        specs.append(sensor_spec)

    if return_specs:
        return sensors, specs
    return sensors
