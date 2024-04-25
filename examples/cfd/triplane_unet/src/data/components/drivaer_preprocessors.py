# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

from typing import List, Literal, Optional, Tuple, Union, Mapping, Any

import numpy as np
import torch
from src.data.mesh_utils import (
    Normalizer,
    bbox_to_centers,
    compute_drag_coefficient,
    point_cloud_to_sdf,
)


# DrivAer dataset
# Air density = 1.205 kg/m^3
# Stream velocity = 38.8889 m/s
DRIVAER_AIR_DENSITY = 1.205
DRIVAER_STREAM_VELOCITY = 38.8889

# DrivAer pressure mean and std
DRIVAER_PRESSURE_MEAN = -150.13066236223494
DRIVAER_PRESSURE_STD = 229.1046667362158
DRIVAER_AIR_COEFF = 2 / (DRIVAER_AIR_DENSITY * DRIVAER_STREAM_VELOCITY**2)


class DrivAerPreprocessingFunctor:
    """
    Preprocess DrivAer data that downsamples and normalizes the pressure field
    """

    def __init__(
        self,
        pressure_mean: float = DRIVAER_PRESSURE_MEAN,
        pressure_std: float = DRIVAER_PRESSURE_STD,
        every_n_data: int = 1,
    ):
        self.normalizer = Normalizer(pressure_mean, pressure_std)
        self.every_n_data = every_n_data

    def __call__(self, np_dict: Mapping[str, Any]) -> Mapping[str, Any]:
        if self.every_n_data > 1:
            # Downsample the data
            for k, v in np_dict.items():
                if (isinstance(v, np.ndarray) and v.size > 1) or (
                    isinstance(v, torch.Tensor) and v.numel() > 1
                ):
                    np_dict[k] = v[:: self.every_n_data]

        if "Snapshot" in np_dict:
            # array('EnSightXXX', dtype='<U10')
            # Convert it to an integer
            np_dict["Snapshot"] = int(
                np.char.replace(np_dict["Snapshot"], "EnSight", "")
            )

        np_dict["time_avg_pressure_whitened"] = self.normalizer.encode(
            np_dict["time_avg_pressure"]
        )
        return np_dict


class DrivAerDragPreprocessingFunctor:
    """
    Compute drag coefficient using area, normal, pressure and wall shear stress
    """

    def __init__(self, air_coeff: float = DRIVAER_AIR_COEFF):
        self.air_coeff = air_coeff

    def __call__(self, np_dict: Mapping[str, Any]) -> Mapping[str, Any]:
        # Compute drag coefficient using area, normal, pressure and wall shear stress
        drag_coef = compute_drag_coefficient(
            np_dict["cell_normals"],
            np_dict["cell_areas"],
            self.air_coeff / np_dict["proj_area_x"],
            np_dict["time_avg_pressure"],
            np_dict["time_avg_wall_shear_stress"],
        )
        # np_dict["c_d"] is computed on a finer mesh and the newly computed drag is on a coarser mesh so they are not equal
        np_dict["c_d_computed"] = drag_coef
        return np_dict


class DrivAerTDFPreprocessingFunctor:
    """
    Compute truncated distance field (TDF) from the object surface
    """

    def __init__(
        self,
        bbox_min: Tuple[float, float, float] = (-1.0, -1.0, -1.0),
        bbox_max: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        bbox_resolution: Tuple[int, int, int] = (128, 128, 128),
        dist_compute_device: str = "cuda",
        dist_chunk_size: int = 16384,
        point_sample_size: int = 16384,
        distance_threshold: float = 1.0,
    ):
        self.bbox_min = bbox_min
        self.bbox_max = bbox_max
        self.bbox_resolution = bbox_resolution
        self.dist_compute_device = dist_compute_device
        assert isinstance(dist_chunk_size, int), "dist_chunk_size must be an integer"
        self.dist_chunk_size = dist_chunk_size
        self.vox_centers = bbox_to_centers(
            torch.Tensor(self.bbox_min),
            torch.Tensor(self.bbox_max),
            self.bbox_resolution,
        ).to(self.dist_compute_device)
        self.point_sample_size = point_sample_size
        self.distance_threshold = distance_threshold

    def __call__(self, np_dict: Mapping[str, Any]) -> Mapping[str, Any]:
        assert (
            "time_avg_pressure_whitened" in np_dict
        ), "Please call DrivAerPreprocessingFunctor first"

        # compute bbox center
        vertices = np_dict["cell_centers"]
        obj_min = np.min(vertices, axis=0)
        obj_max = np.max(vertices, axis=0)
        obj_center = (obj_min + obj_max) / 2.0
        vertices = vertices - obj_center

        # Point sample index
        point_sample_idx = np.random.choice(
            len(vertices), self.point_sample_size, replace=True
        )

        vertices = vertices[point_sample_idx]
        vertices_device = torch.Tensor(vertices).to(self.dist_compute_device)

        # Distance computation
        dists = point_cloud_to_sdf(
            vertices_device,
            self.vox_centers.view(-1, 3),
            chunk_size=self.dist_chunk_size,
        )
        dists = torch.clamp(dists, -self.distance_threshold, self.distance_threshold)
        dists = dists.view(self.bbox_resolution).cpu().numpy()
        np_dict["sdf"] = dists
        np_dict["cell_centers"] = vertices
        np_dict["time_avg_pressure_whitened"] = np_dict["time_avg_pressure_whitened"][
            point_sample_idx
        ]
        return np_dict
