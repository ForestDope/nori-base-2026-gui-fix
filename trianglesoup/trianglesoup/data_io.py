"""Data loading utilities for triangle soup optimization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import drjit as dr
import mitsuba as mi

from trianglesoup import utils


@dataclass
class CameraView:
    name: str
    sensor: mi.Sensor
    sensor_spec: dict | None = None
    target: mi.TensorXf | None = None
    image_width: int | None = None
    image_height: int | None = None


@dataclass
class CameraDataset:
    views: List[CameraView]

    @property
    def sensors(self) -> list[mi.Sensor]:
        return [view.sensor for view in self.views]

    @property
    def targets(self) -> list[mi.TensorXf]:
        return [view.target for view in self.views if view.target is not None]

    @property
    def names(self) -> list[str]:
        return [view.name for view in self.views]


def _load_sensor_names(json_path: Path) -> tuple[list[str], int, int]:
    with json_path.open("r") as f:
        data = json.load(f)
    width = int(data.get("w", data.get("width", 0)))
    height = int(data.get("h", data.get("height", 0)))
    names = [
        key for key, value in data.items() if isinstance(value, dict) and "to_world" in value
    ]
    names.sort()
    return names, width, height


def load_dataset(
    dataset_dir: Path,
    *,
    jitter_pixels: bool = False,
) -> tuple[CameraDataset, CameraDataset]:
    dataset_dir = dataset_dir.expanduser().resolve()
    train_json = dataset_dir / "train.json"
    test_json = dataset_dir / "test.json"
    train_img_dir = dataset_dir / "train_images"
    test_img_dir = dataset_dir / "test_images"

    if not train_json.exists():
        raise FileNotFoundError(f"Missing train.json at {train_json}")

    train_sensors, train_specs = utils.load_sensors_from_json(
        str(train_json), return_specs=True, jitter_pixels=jitter_pixels
    )
    train_names, train_width, train_height = _load_sensor_names(train_json)

    def _train_image_path(sensor) -> Path:
        hash_id = utils.sensor_hash_to_10_digits(sensor)
        if train_img_dir.exists():
            return train_img_dir / f"train_view_{hash_id}.exr"
        return dataset_dir / f"train_view_{hash_id}.exr"

    train_images = []
    for sensor in train_sensors:
        path = _train_image_path(sensor)
        if not path.exists():
            raise FileNotFoundError(f"Missing training image at {path}")
        train_images.append(mi.TensorXf(mi.Bitmap(str(path))))
    for tensor in train_images:
        dr.eval(tensor)

    train_views = [
        CameraView(
            name=train_names[i] if i < len(train_names) else f"train_{i}",
            sensor=sensor,
            sensor_spec=train_specs[i] if i < len(train_specs) else None,
            image_width=train_width,
            image_height=train_height,
            target=train_images[i],
        )
        for i, sensor in enumerate(train_sensors)
    ]

    test_views: list[CameraView] = []
    if test_json.exists():
        test_sensors, test_specs = utils.load_sensors_from_json(
            str(test_json), return_specs=True, jitter_pixels=jitter_pixels
        )
        test_names, test_width, test_height = _load_sensor_names(test_json)

        def _test_image_path(sensor) -> Path:
            hash_id = utils.sensor_hash_to_10_digits(sensor)
            if test_img_dir.exists():
                return test_img_dir / f"test_view_{hash_id}.exr"
            return dataset_dir / f"test_view_{hash_id}.exr"

        test_images = []
        for sensor in test_sensors:
            path = _test_image_path(sensor)
            if not path.exists():
                continue
            test_images.append(mi.TensorXf(mi.Bitmap(str(path))))
        for tensor in test_images:
            dr.eval(tensor)
        test_views = [
            CameraView(
                name=test_names[i] if i < len(test_names) else f"test_{i}",
                sensor=sensor,
                sensor_spec=test_specs[i] if i < len(test_specs) else None,
                image_width=test_width,
                image_height=test_height,
                target=(test_images[i] if i < len(test_images) else None),
            )
            for i, sensor in enumerate(test_sensors)
        ]

    return CameraDataset(train_views), CameraDataset(test_views)




def select_progress_views(dataset: CameraDataset, count: int) -> Iterable[CameraView]:
    if count <= 0 or len(dataset.views) == 0:
        return []
    count = min(count, len(dataset.views))
    step = max(1, len(dataset.views) // count)
    return [dataset.views[i] for i in range(0, len(dataset.views), step)][:count]
