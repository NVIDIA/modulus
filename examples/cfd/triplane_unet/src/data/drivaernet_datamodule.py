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

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any
import yaml

import numpy as np
import pandas as pd
import pyvista as pv

import torch
from torch.utils.data import Dataset

from src.data.base_datamodule import BaseDataModule
from src.data.components.preprocessor_utils import (
    ComposePreprocessors,
    UnitGaussianNormalizer,
)
from src.data.mesh_utils import (
    compute_drag_coefficient,
)


# TODO: need to update/check these values.
# DrivAerNet dataset
# Air density = 1.205 kg/m^3
# Stream velocity = 30 m/s
DRIVAERNET_AIR_DENSITY = 1.205
DRIVAERNET_STREAM_VELOCITY = 38.8889
# DrivAerNet pressure mean and std
DRIVAERNET_PRESSURE_MEAN = -150.13066236223494
DRIVAERNET_PRESSURE_STD = 229.1046667362158
DRIVAERNET_AIR_COEFF = 2 / (DRIVAERNET_AIR_DENSITY * DRIVAERNET_STREAM_VELOCITY**2)


class DrivAerNetPreprocessor:
    def __init__(self, num_points: int = 16384) -> None:
        self.num_points = num_points

    def __call__(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        if self.num_points > 0:
            # Randomly sample points
            vertices = sample["cell_centers"]
            point_sample_idx = np.random.choice(
                len(vertices), self.num_points, replace=True
            )

            for k, v in sample.items():
                if (isinstance(v, np.ndarray) and v.size > 1) or (
                    isinstance(v, torch.Tensor) and v.numel() > 1
                ):
                    sample[k] = v[point_sample_idx]

        return sample


class DrivAerNetDragPreprocessor:
    def __init__(self) -> None:
        pass

    def __call__(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        # Compute drag coefficient using area, normal, pressure and wall shear stress
        drag_coef = compute_drag_coefficient(
            sample["cell_normals"],
            sample["cell_areas"],
            DRIVAERNET_AIR_COEFF / sample["proj_area_x"],
            sample["pressure"],
            sample["wall_shear_stress"],
        )
        # sample["c_d"] is computed on a finer mesh and the newly computed drag is on a coarser mesh so they are not equal
        sample["c_d_computed"] = drag_coef

        return sample


class DrivAerNetDataset(Dataset):
    """DrivAerNet dataset.

    For more information, see DrivAerNet code:
    https://github.com/Mohamedelrefaie/DrivAerNet
    """

    def __init__(
        self,
        data_path: str | Path,
        phase: str,
        coeff_filename: str = "coefficients.csv",
        preprocessors: Iterable[Callable] = None,
    ) -> None:
        """Initializes the dataset."""

        self.data_path = Path(data_path)
        if not self.data_path.is_dir():
            raise ValueError(
                f"Path {self.data_path} does not exist or is not a directory."
            )
        self.p_vtk_path = self.data_path / "SurfacePressureVTK"
        self.wss_vtk_path = self.data_path / "WallShearStressVTK"

        self.phase = phase.lower()
        phases = ["train", "val", "test"]
        if phase not in phases:
            raise ValueError(f"{phase = } is not supported, must be one of {phases}.")

        if preprocessors is None:
            preprocessors = []
        self.preprocessors = ComposePreprocessors(list(preprocessors))

        # Load phase ids used to select a corresponding data split.
        with open(self.data_path / f"{phase}_design_ids.txt", encoding="utf-8") as f:
            file_ids = set(l.rstrip() for l in f)

        # Read coefficients file which contains Cd, Cl etc.
        coeffs = pd.read_csv(self.data_path / coeff_filename, index_col="Design")
        coeffs = coeffs[coeffs.index.isin(file_ids)]

        # Read projected areas file which is in YAML-like format with entries that look like:
        # combined_DrivAer_F_D_WM_WW_1234.stl: 2.574603830871618
        with open(self.data_path / "projected_areas.txt", encoding="utf-8") as f:
            y = yaml.safe_load(f)
        proj_areas = pd.DataFrame.from_dict(
            {k.removeprefix("combined_").removesuffix(".stl"): v for k, v in y.items()},
            orient="index",
            columns=["proj_area_x"],
        )
        # Merge projected areas into the coeffs dataframe.
        coeffs = coeffs.join(proj_areas)
        # There are 10 entries missing in projected_areas.txt.
        # Check them by running `coeffs[coeffs.proj_area.isna()]`
        # train: DrivAer_F_D_WM_WW_0132, 0797, 1118, 1421, 1556, 1891, 2353, 2459.
        # val: DrivAer_F_D_WM_WW_0603, 3199.
        coeffs = coeffs.dropna()
        num_missing = {"train": 8, "val": 2, "test": 0}
        assert (
            len(file_ids) - len(coeffs) == num_missing[phase]
        ), f"{len(file_ids)=} {len(coeffs)=}"

        coeffs.sort_index(inplace=True)
        self.coeffs = coeffs

    def __len__(self) -> int:
        return len(self.coeffs)

    def __getitem__(self, index) -> dict[str, Any]:
        if not (0 <= index < len(self)):
            raise IndexError(f"Invalid {index = } expected in [0, {len(self)})")

        coeffs = self.coeffs.iloc[index]

        key = coeffs.name

        # Read pressure and WSS data.
        p_vtk_file = self.p_vtk_path / (key + ".vtk")
        mesh = pv.read(p_vtk_file)

        wss_vtk_file = self.wss_vtk_path / (key + ".vtk")
        wss = pv.read(wss_vtk_file).point_data["wallShearStress"]

        # Estimate normals.
        mesh.compute_normals(
            cell_normals=True, point_normals=True, flip_normals=True, inplace=True
        )

        # Extract cell centers and areas.
        cell_centers = np.array(mesh.cell_centers().points)
        cell_normals = np.array(mesh.cell_normals)
        cell_sizes = mesh.compute_cell_sizes(length=False, area=True, volume=False)
        cell_sizes = np.array(cell_sizes.cell_data["Area"])

        # Normalize cell normals.
        cell_normals = (
            cell_normals / np.linalg.norm(cell_normals, axis=1)[:, np.newaxis]
        )

        sample = {
            "mesh_nodes": np.array(mesh.points),
            "cell_centers": cell_centers,
            "cell_areas": cell_sizes,
            "cell_normals": cell_normals,
            **coeffs.to_dict(),
            "pressure": mesh.point_data["p"],
            "wall_shear_stress": wss,
        }

        return self.preprocessors(sample)


class DrivAerNetDataModule(BaseDataModule):
    """DrivAerNet data module."""

    def __init__(
        self,
        data_path: str | Path,
        preprocessors: Iterable[Callable] = None,
        **kwargs,
    ):
        self._train_dataset = DrivAerNetDataset(
            data_path,
            "train",
            preprocessors=preprocessors,
        )
        self._val_dataset = DrivAerNetDataset(
            data_path,
            "val",
            preprocessors=preprocessors,
        )
        self._test_dataset = DrivAerNetDataset(
            data_path,
            "test",
            preprocessors=preprocessors,
        )

    def encode(self, x):
        return self.normalizer.encode(x)

    def decode(self, x):
        return self.normalizer.decode(x)

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def val_dataset(self):
        return self._val_dataset

    @property
    def test_dataset(self):
        return self._test_dataset


def test_drivaernet_datamodule(data_path: str):
    preprocs = [DrivAerNetPreprocessor()]

    dm = DrivAerNetDataModule(data_path, preprocessors=preprocs)

    for x in dm.train_dataloader():
        break


def test_drivaernet_dataset(data_path: str, phase: str, size: int):
    dset = DrivAerNetDataset(data_path, phase)
    assert len(dset) == size

    x = dset[0]
    assert isinstance(x, dict)


if __name__ == "__main__":
    data_path = "/data/src/modulus/data/triplane_unet/DrivAerNet"
    test_mod = True
    if test_mod:
        test_drivaernet_datamodule(data_path)
    else:
        test_drivaernet_dataset(data_path, "train", 2768)
        test_drivaernet_dataset(data_path, "val", 593)
        test_drivaernet_dataset(data_path, "test", 595)
