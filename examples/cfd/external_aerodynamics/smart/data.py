# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PhysicsNeMo data pipeline for paired geometry, surface, and volume samples."""

from __future__ import annotations

from pathlib import Path

import torch
from omegaconf import DictConfig
from tensordict import TensorDict

from physicsnemo.datapipes import (
    Dataset,
    Normalize,
    NumpyReader,
    SubsamplePoints,
    Transform,
)
from physicsnemo.datapipes._rng import derive_seed


class ValidateSample(Transform):
    """Validate finite, unpadded point clouds before normalization and sampling."""

    def __init__(
        self,
        surface_channels: int,
        volume_channels: int,
        parameter_channels: int,
        targets: bool = True,
    ):
        super().__init__()
        self.channels = {"surface": surface_channels, "volume": volume_channels}
        self.parameter_channels = parameter_channels
        self.targets = targets

    def __call__(self, data: TensorDict) -> TensorDict:
        """Check the sample schema and standardize all numeric inputs to float32."""
        expected = {"geometry": 3}
        for field, channels in self.channels.items():
            if channels:
                expected[f"{field}_queries"] = 3
                if self.targets:
                    expected[f"{field}_targets"] = channels
        for key, width in expected.items():
            value = data[key]
            if value.ndim != 2 or value.shape[1] != width or value.shape[0] == 0:
                raise ValueError(
                    f"{key}: expected nonempty (N, {width}), got {tuple(value.shape)}"
                )
            if not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError(f"{key}: expected finite floating-point values")
            data[key] = value.float()
        for field, channels in self.channels.items():
            if (
                self.targets
                and channels
                and len(data[f"{field}_queries"]) != len(data[f"{field}_targets"])
            ):
                raise ValueError(f"{field}: query and target point counts differ")
        if self.parameter_channels:
            value = data["parameters"]
            if (
                value.shape != (self.parameter_channels,)
                or not value.is_floating_point()
                or not torch.isfinite(value).all()
            ):
                raise ValueError(
                    f"parameters: expected finite ({self.parameter_channels},)"
                )
            data["parameters"] = value.float()
        return data


def build_dataset(
    path: str | Path, cfg: DictConfig, *, training: bool, targets: bool = True
) -> Dataset:
    """Compose PhysicsNeMo's reader, validation, normalization, and subsampling.

    Sampling happens on the host so full-resolution targets never need to fit
    on the GPU. Validation preserves all query points for streamed evaluation.
    Field values remain in the user's chosen physical or nondimensional units.
    """
    if cfg.data.get("format", "numpy") in ("drivaerml_zarr", "drivaerml_mesh"):
        from drivaerml import build_drivaerml_dataset

        return build_drivaerml_dataset(path, cfg, training=training, targets=targets)
    if any(
        cfg.model.get(f"{field}_query_feature_channels", 0)
        for field in ("surface", "volume")
    ):
        raise ValueError("CFD query features currently require a DrivAerML reader")
    if cfg.data.get("format", "numpy") != "numpy":
        raise ValueError("Unsupported data format")
    fields = ["geometry"]
    coordinates = ["geometry"]
    for field in ("surface", "volume"):
        if cfg.model[f"{field}_channels"]:
            fields.append(f"{field}_queries")
            if targets:
                fields.append(f"{field}_targets")
            coordinates.append(f"{field}_queries")
    if cfg.model.parameter_channels:
        fields.append("parameters")
    center = torch.tensor(list(cfg.data.coordinate_center), dtype=torch.float32)
    scale = torch.tensor(list(cfg.data.coordinate_scale), dtype=torch.float32)
    if (
        center.shape != (3,)
        or scale.shape != (3,)
        or not torch.isfinite(center).all()
        or not torch.isfinite(scale).all()
        or (scale <= 0).any()
    ):
        raise ValueError(
            "Coordinate center/scale must be finite length-three vectors with positive scale"
        )
    transforms = [
        ValidateSample(
            cfg.model.surface_channels,
            cfg.model.volume_channels,
            cfg.model.parameter_channels,
            targets,
        ),
        Normalize(
            coordinates,
            means={k: center for k in coordinates},
            stds={k: scale for k in coordinates},
            method="mean_std",
        ),
    ]
    if cfg.model.parameter_channels:
        means = torch.tensor(list(cfg.data.parameter_mean))
        stds = torch.tensor(list(cfg.data.parameter_std))
        if (
            means.shape != (cfg.model.parameter_channels,)
            or stds.shape != means.shape
            or not torch.isfinite(means).all()
            or not torch.isfinite(stds).all()
            or (stds <= 0).any()
        ):
            raise ValueError(
                "Parameter mean/std must match parameter_channels and have finite, positive std"
            )
        transforms.append(
            Normalize(
                ["parameters"],
                means={"parameters": means},
                stds={"parameters": stds},
                method="mean_std",
            )
        )
    if training:
        for field in ("surface", "volume"):
            count = cfg.data[f"{field}_points"]
            if cfg.model[f"{field}_channels"]:
                if not isinstance(count, int) or count <= 0:
                    raise ValueError(f"{field}_points must be a positive integer")
                transforms.append(
                    SubsamplePoints(
                        [f"{field}_queries", f"{field}_targets"],
                        count,
                        algorithm="uniform",
                    )
                )
    # The existing reader returns TensorDicts with no point dimension in batch_size,
    # allowing different surface/volume sizes within a sample.
    return Dataset(
        NumpyReader(path, fields=fields),
        transforms=transforms,
        num_workers=cfg.data.workers,
    )


def model_inputs(batch: TensorDict) -> dict:
    """Map a batched sample to the model's public input signature."""
    result = {
        key: batch.get(key, None)
        for key in ("geometry", "surface_queries", "volume_queries", "parameters")
    }
    for field in ("surface", "volume"):
        key = f"{field}_query_features"
        if key in batch and batch[key] is not None:
            result[key] = batch[key]
    if "geometry_seed" in batch:
        seeds = batch["geometry_seed"].detach().cpu().reshape(-1).tolist()
        seed = seeds[0] if len(seeds) == 1 else derive_seed(seeds[0], *seeds[1:])
        result["geometry_generator"] = torch.Generator(
            device=batch["geometry"].device
        ).manual_seed(seed)
    return result
