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

from pathlib import Path
from typing import Any
import yaml

import numpy as np
import pandas as pd
import pyvista as pv

from torch.utils.data import Dataset

from src.data.base_datamodule import BaseDataModule


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
    ) -> None:
        """Initializes the dataset."""

        self.data_path = Path(data_path)
        if not self.data_path.is_dir():
            raise ValueError(
                f"Path {self.data_path} does not exist or is not a directory."
            )
        self.p_vtk_path = self.data_path / "SurfacePressureVTK"

        self.phase = phase.lower()
        phases = ["train", "val", "test"]
        if phase not in phases:
            raise ValueError(f"{phase = } is not supported, must be one of {phases}.")

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
            columns=["proj_area"],
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
        p_vtk_file = self.p_vtk_path / (key + ".vtk")
        mesh = pv.read(p_vtk_file)

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

        return {
            "mesh_nodes": np.array(mesh.points),
            "cell_centers": cell_centers,
            "cell_areas": cell_sizes,
            "cell_normals": cell_normals,
            **coeffs.to_dict(),
        }


def test_drivaernet_dataset(phase: str):
    dset = DrivAerNetDataset(
        "/data/src/modulus/data/triplane_unet/DrivAerNet",
        phase,
    )
    return dset


if __name__ == "__main__":
    train_dset = test_drivaernet_dataset("train")
    assert len(train_dset) == 2768
    x = train_dset[0]

    val_dset = test_drivaernet_dataset("val")
    assert len(val_dset) == 593

    test_dset = test_drivaernet_dataset("test")
    assert len(test_dset) == 595
