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

import warnings
from typing import Callable, Dict, Optional, Tuple, Union, Literal, List

try:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
except ImportError:
    warnings.warn("VTK not installed")
import glob
import os
import pickle
from pathlib import Path
import pandas as pd

import numpy as np
from torch.utils.data import Dataset, Subset
import webdataset as wds

from src.data.base_datamodule import BaseDataModule

from src.data.components.preprocessor_utils import (
    UnitGaussianNormalizer,
    UniformNormalizer,
    ComposePreprocessors,
)
from src.data.components.webdataset_utils import from_numpy, split_by_node_equal


# AhmedBody dataset stats
AHMED_BODY_PRESSURE_MEAN = -179.03588502774267
AHMED_BODY_PRESSURE_STD = 355.92119726212775
AHMED_BODY_VELOCITY_MAX = 60.0
AHMED_BODY_VELOCITY_MIN = 20.0


def _compute_cell_areas_and_centers(polydata):
    # Cell center and area
    num_cells = polydata.GetNumberOfCells()
    cell_areas = np.zeros(num_cells)
    cell_centers = np.zeros((num_cells, 3))

    # Loop through each cell and compute its area
    for i in range(num_cells):
        cell = polydata.GetCell(i)
        # Count the number of points in the cell
        num_points = cell.GetNumberOfPoints()
        # Get the points in the cell
        if num_points == 3:
            # If the cell is a triangle, we can just use the points
            p0 = np.array(cell.GetPoints().GetPoint(0))
            p1 = np.array(cell.GetPoints().GetPoint(1))
            p2 = np.array(cell.GetPoints().GetPoint(2))
            cell_areas[i] = np.linalg.norm(np.cross(p1 - p0, p2 - p0)) / 2
            cell_centers[i] = (p0 + p1 + p2) / 3
        else:
            # Loop through each triangle in the cell
            for j in range(num_points - 2):
                # Get the points of the triangle
                p0 = np.array(cell.GetPoints().GetPoint(0))
                p1 = np.array(cell.GetPoints().GetPoint(j + 1))
                p2 = np.array(cell.GetPoints().GetPoint(j + 2))
                # Compute the area of the triangle
                cell_areas[i] += np.linalg.norm(np.cross(p1 - p0, p2 - p0)) / 2
            # Compute cell center
            cell_centers[i] = np.mean(
                [cell.GetPoints().GetPoint(j) for j in range(num_points)], axis=0
            )
    return cell_areas, cell_centers


def read_vtp(path: str) -> Tuple[dict, dict]:
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(path)
    reader.Update()
    polydata = reader.GetOutput()

    # compute normals
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(polydata)
    normals.ComputePointNormalsOn()
    normals.ComputeCellNormalsOn()
    normals.Update()
    polydata = normals.GetOutput()

    point_data = polydata.GetPointData()
    field_count = point_data.GetNumberOfArrays()
    # Read points
    point_dict = {}
    for i in range(field_count):
        point_dict[point_data.GetArrayName(i)] = vtk_to_numpy(point_data.GetArray(i))
    point_dict["points"] = vtk_to_numpy(polydata.GetPoints().GetData())

    # read cells
    cell_data = polydata.GetCellData()
    cell_dict = {}
    field_count = cell_data.GetNumberOfArrays()
    for i in range(field_count):
        cell_dict[cell_data.GetArrayName(i)] = vtk_to_numpy(cell_data.GetArray(i))
    (
        cell_dict["cell_areas"],
        cell_dict["cell_centers"],
    ) = _compute_cell_areas_and_centers(polydata)
    return point_dict, cell_dict


class AhmedBodyDataset(Dataset):
    """Ahmed body dataset."""

    def __init__(
        self, data_path: Union[str, Path], transform: Optional[Callable] = None
    ):
        if isinstance(data_path, str):
            data_path = Path(data_path)
        self.data_path = data_path
        self.transform = transform

        assert self.data_path.exists(), f"Path {self.data_path} does not exist"
        assert self.data_path.is_dir(), f"Path {self.data_path} is not a directory"
        converged_cds = (
            self.data_path
            / "converged_cd_std_0.0e+00_2.7e-03_slope_1.0e-10_1.0e-05_510.csv"
        )
        assert converged_cds.exists(), f"Path {converged_cds} does not exist"

        # Count the number of case%d folders
        self.case_folders = glob.glob(str(self.data_path / "case*"))
        # Filter case folders that do not have simpleFoam_steady folder
        self.case_folders = [
            case_folder
            for case_folder in self.case_folders
            if os.path.exists(os.path.join(case_folder, "simpleFoam_steady"))
        ]
        self.case_folders.sort()
        self.converged_cds = pd.read_csv(converged_cds)

    def __len__(self):
        return len(self.converged_cds)

    def __getitem__(self, idx):
        converged_cd = self.converged_cds.iloc[idx]
        case_id = int(converged_cd["Unnamed: 0"])
        case_path = self.data_path / f"case{case_id}" / "simpleFoam_steady"
        assert case_path.exists(), f"Path {case_path} does not exist"

        # read case info and separate with ":" and convert it to a dictionary
        case_info_path = self.data_path / f"case{case_id}" / "case_info.txt"
        assert case_info_path.exists(), f"Path {case_info_path} does not exist"
        with open(str(case_info_path)) as f:
            case_info = f.readlines()

        case_info = [line.strip().split(":") for line in case_info]
        case_info = {key.strip().lower(): float(value) for key, value in case_info}

        point_data, cell_data = read_vtp(str(Path(case_path) / "ahmed_body.vtp"))

        # The dataset only save the positive Y values of the points due to symmetry.
        # We need to flip the points to get the full body

        # Copy all the non point data to have 2 copies
        for key, value in cell_data.items():
            cell_data[key] = np.concatenate([value, value], axis=0)
        for key, value in point_data.items():
            point_data[key] = np.concatenate([value, value], axis=0)

        # Flip the Y values of the points
        N = len(point_data["points"]) // 2
        point_data["points"][N:, 1] *= -1
        # Flip the Y values of the wallShearStress
        point_data["wallShearStress"][N:, 1] *= -1

        # Flip the Y values of the cell_centers
        N = len(cell_data["cell_centers"]) // 2
        cell_data["cell_centers"][N:, 1] *= -1
        # Flip the Y values of the wallShearStress
        cell_data["wallShearStress"][N:, 1] *= -1
        cds = {
            "cd_avg": converged_cd["cd_avg"],
            "cd_std": converged_cd["cd_std"],
        }
        data = {
            "point_data": point_data,
            "cell_data": cell_data,
            "case_info": case_info,
            "converged_cd": cds,
            "case_id": case_id,
        }
        if self.transform:
            data = self.transform(data)
        return data


AHMED_MAPPING = {
    "cell_data,wallShearStress": "wall_shear_stress",
    "cell_data,cell_areas": "cell_areas",
    "cell_data,cell_centers": "cell_centers",
    "cell_data,p": "pressure",
    "cell_data,Normals": "normals",
    "case_info,velocity": "velocity",
    "converged_cd,cd_avg": "c_d",
}


class AhmedBodyMappingFunctor:
    def __init__(self, mapping: dict):
        self.mapping = mapping

    def __call__(self, datum: dict) -> dict:
        # weird error fix: flatten all dictionary values
        for k, v in datum.items():
            if isinstance(v, np.ndarray):
                datum[k] = dict(enumerate(v.flatten(), 1))[1]

        new_datum = {}
        for key, new_key in self.mapping.items():
            if "," not in key:
                new_datum[new_key] = datum[k]
            else:
                keys = key.split(",")
                value = datum
                for k in keys:
                    value = value[k]
                new_datum[new_key] = value
        return new_datum


class AhmedBodyPreprocessingFunctor:
    """Ahmed body item transform."""

    def __init__(
        self,
        pressure_mean: float,
        pressure_std: float,
        velocity_max: float,
        velocity_min: float,
        num_points: Optional[int] = None,
    ):
        self.normalizer = UnitGaussianNormalizer(mean=pressure_mean, std=pressure_std)
        self.vel_normalizer = UniformNormalizer(
            min_val=velocity_min, max_val=velocity_max
        )
        self.num_points = num_points

    def __call__(self, datum: dict) -> dict:
        datum["normalized_pressure"] = self.normalizer.encode(datum["pressure"])
        datum["uniformized_velocity"] = self.vel_normalizer.encode(datum["velocity"])
        if self.num_points is not None:
            # Downsample the data
            indices = np.random.choice(
                np.arange(len(datum["pressure"])), self.num_points, replace=False
            )
            for k, v in datum.items():
                if isinstance(v, np.ndarray):
                    datum[k] = v[indices]
        return datum


class AhmedBodyDataModule(BaseDataModule):
    """Ahmed body data module."""

    def __init__(
        self,
        data_path: Union[Path, str],
        preprocessors: Optional[List[Callable]] = None,
        num_points: int = 16384,
    ) -> None:
        """
        Args:
            data_dir (Union[Path, str]): Path that contains train and test directories
        """
        super().__init__()

        if isinstance(data_path, str):
            data_path = Path(data_path)
        self.data_path = data_path

        # Assert the size
        assert self.data_path.exists(), f"Path {self.data_path} does not exist"
        assert self.data_path.is_dir(), f"Path {self.data_path} is not a directory"

        if preprocessors is None:
            preprocessors = []

        # Add normalization and downsampling first.
        preprocessors = [
            AhmedBodyMappingFunctor(AHMED_MAPPING),
            AhmedBodyPreprocessingFunctor(
                pressure_mean=AHMED_BODY_PRESSURE_MEAN,
                pressure_std=AHMED_BODY_PRESSURE_STD,
                velocity_max=AHMED_BODY_VELOCITY_MAX,
                velocity_min=AHMED_BODY_VELOCITY_MIN,
                num_points=num_points,
            ),
        ] + preprocessors
        self.normalizer = preprocessors[1].normalizer

        self.preprocessors = ComposePreprocessors(preprocessors)

        self._train_dataset = self._create_dataset("train")
        self._val_dataset = self._create_dataset("val")
        self._test_dataset = self._create_dataset("test")

    def _create_dataset(self, phase: str) -> wds.DataPipeline:
        # Create dataset with the processing pipeline.
        dataset = wds.DataPipeline(
            wds.SimpleShardList(str(self.data_path / f"{phase}.tar")),
            wds.tarfile_to_samples(),
            wds.map(lambda x: from_numpy(x, "npz")),
            wds.map(self.preprocessors),
        )

        return dataset

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


def test_datamodule(data_path: str, num_points: Optional[int] = None):
    datamodule = AhmedBodyDataModule(data_path, num_points=num_points)
    train_loader = datamodule.train_dataloader()
    for batch in train_loader:
        print(batch["wall_shear_stress"].shape)  # torch.Size([1, 149344, 3])
        print(batch["pressure"].shape)  # torch.Size([1, 149344])
        print(batch["normalized_pressure"].shape)  # torch.Size([1, 149344])
        print(batch["normalized_pressure"].mean())  # ~ 0
        print(batch["velocity"].shape)  # torch.Size([1])
        print(batch["uniformized_velocity"].shape)  # torch.Size([1])
        print(batch["uniformized_velocity"])  # [0, 1]
        print(batch["normals"].shape)  # torch.Size([1, 149344, 3])
        print(batch["cell_areas"].shape)  # torch.Size([1, 149344])
        print(batch["cell_centers"].shape)  # torch.Size([1, 149344, 3])
        break


if __name__ == "__main__":
    # Test the datamodule
    import fire

    fire.Fire(test_datamodule)
