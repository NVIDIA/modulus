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
from typing import Callable, Dict, Optional, Tuple, Union

try:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
except ImportError:
    warnings.warn("VTK not installed")
import glob
import os
import pickle
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset, Subset

from src.data.base_datamodule import BaseDataModule
from src.data.dict_dataset import MappingDatasetWrapper
from src.data.pickle_dataset import PickleDataset


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
    def __init__(self, data_path: Union[str, Path], transform=None):
        if isinstance(data_path, str):
            data_path = Path(data_path)
        self.data_path = data_path
        assert self.data_path.exists(), f"Path {self.data_path} does not exist"
        assert self.data_path.is_dir(), f"Path {self.data_path} is not a directory"
        self.transform = transform
        # Count the number of case%d folders
        self.case_folders = glob.glob(str(self.data_path / "case*"))
        # Filter case folders that do not have simpleFoam_steady folder
        self.case_folders = [
            case_folder
            for case_folder in self.case_folders
            if os.path.exists(os.path.join(case_folder, "simpleFoam_steady"))
        ]
        self.case_folders.sort()
        self.case_count = len(self.case_folders)

    def __len__(self):
        return self.case_count

    def __getitem__(self, idx):
        case_path = self.case_folders[idx]
        case_info_path = str(Path(case_path) / "case_info.txt")
        data_path = str(Path(case_path) / "simpleFoam_steady" / "ahmed_body.vtp")
        # read case info and separate with ":" and convert it to a dictionary
        with open(case_info_path) as f:
            case_info = f.readlines()
        case_info = [line.strip().split(":") for line in case_info]
        case_info = {key.strip().lower(): float(value) for key, value in case_info}
        point_data, cell_data = read_vtp(data_path)

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

        data = {"point_data": point_data, "cell_data": cell_data}
        data["case_info"] = case_info
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
}


class AhmedBodyDatumTransform:
    def __init__(self, info: Optional[dict] = None):
        if info is None:
            info = {}
        self.info = info

    def update_info(self, info: dict):
        self.info.update(info)

    def __call__(self, datum: dict) -> dict:
        # Normalize the pressure --> 0 mean 1 std normal distribution
        datum["normalized_pressure"] = (
            datum["pressure"] - self.info["mean_p"]
        ) / self.info["std_p"]
        # Uniform the velocity --> [0, 1] range
        datum["uniformized_velocity"] = (datum["velocity"] - self.info["min_vel"]) / (
            self.info["max_vel"] - self.info["min_vel"]
        )
        return datum


class AhmedBodyDataModule(BaseDataModule):
    def __init__(
        self,
        data_dir: Union[Path, str],
        every_n_data: Optional[int] = None,
        transform: Optional[Callable] = None,
        dataset_stats_path: Optional[Union[Path, str]] = "dataset_stats.pkl",
    ) -> None:
        """
        Args:
            data_dir (Union[Path, str]): Path that contains train and test directories
        """
        super().__init__()

        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
        if isinstance(dataset_stats_path, str):
            dataset_stats_path = data_dir / dataset_stats_path
        self.data_dir = data_dir

        # Assert the size
        assert self.data_dir.exists(), f"Path {self.data_dir} does not exist"
        assert self.data_dir.is_dir(), f"Path {self.data_dir} is not a directory"
        # Select files that have numeric names that start with 00
        dataset = PickleDataset(self.data_dir, file_format="00*", extension="pkl")
        assert len(dataset) == 805, f"Dataset size is {len(dataset)}, expected 805"

        if transform is None:
            transform = AhmedBodyDatumTransform()
        self.transform = transform

        # Split them into 600, 100, 105
        # Use the same seed to get the same split
        np.random.seed(42)
        indices = np.arange(len(dataset))
        np.random.shuffle(indices)
        train_indices = indices[:600]
        val_indices = indices[600:700]
        test_indices = indices[700:]

        def _get_subdataset(dataset, indices):
            return MappingDatasetWrapper(
                Subset(dataset, indices),
                mapping=AHMED_MAPPING,
                every_n_data=every_n_data,
                transform=transform,
            )

        # Create the train, val, test datasets
        self._train_dataset = _get_subdataset(dataset, train_indices)
        self._val_dataset = _get_subdataset(dataset, val_indices)
        self._test_dataset = _get_subdataset(dataset, test_indices)

        # Load dataset statistics. Create a new one if it does not exist
        if not dataset_stats_path.exists():
            dataset_stats = self.compute_dataset_statistics(self._train_dataset)
            with open(dataset_stats_path, "wb") as f:
                pickle.dump(dataset_stats, f)
        else:
            with open(dataset_stats_path, "rb") as f:
                dataset_stats = pickle.load(f)
        self.dataset_stats = dataset_stats
        self.transform.update_info(dataset_stats)

    # Compute the mean, min, max of p and vel
    def compute_dataset_statistics(self, dataset) -> Dict[str, float]:
        # compute the mean, min, max of data["p"], data["case_infp"]["Velocity"]
        # Use the running sum, running count method
        count_p = 0
        running_sum_p = 0
        running_square_sum_p = 0
        running_min_p = np.inf
        running_max_p = -np.inf
        count_vel = 0
        running_sum_vel = 0
        running_min_vel = np.inf
        running_max_vel = -np.inf

        print("Computing mean, min, max of p and vel")
        for i in range(len(dataset)):
            if i % 100 == 0:
                print(f"Processing {i}/{len(dataset)}")
            data = dataset[i]
            point_data = data["point_data"]
            p = point_data["p"]
            vel = data["case_info"]["velocity"]
            running_sum_p += p.sum()
            running_square_sum_p += (p**2).sum()
            running_min_p = min(running_min_p, p.min())
            running_max_p = max(running_max_p, p.max())
            count_p += len(p)

            running_sum_vel += vel
            running_min_vel = min(running_min_vel, vel)
            running_max_vel = max(running_max_vel, vel)
            count_vel += 1

        # Save the mean, min, max of p and vel into a file
        mean_p = running_sum_p / count_p
        std_p = np.sqrt((running_square_sum_p / count_p) - mean_p**2)
        mean_vel = running_sum_vel / count_vel
        # Create a return dictionary
        return {
            "mean_p": mean_p,
            "min_p": running_min_p,
            "max_p": running_max_p,
            "std_p": std_p,
            "mean_vel": mean_vel,
            "min_vel": running_min_vel,
            "max_vel": running_max_vel,
        }

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def val_dataset(self):
        return self._val_dataset

    @property
    def test_dataset(self):
        return self._test_dataset


if __name__ == "__main__":
    if False:
        data_path = Path("./datasets/ahmed_surface")
        dataset = AhmedBodyDataset(data_path)
        print(f"Dataset length: {len(dataset)}")
        point_data = dataset[0]["point_data"]
        print(point_data.keys())
        print(point_data["points"].shape)
        print(point_data["p"].shape)
        # Show width, height, depth of points
        print(np.ptp(point_data["points"], axis=0))

        # create a path
        path = Path("./datasets/ahmed_preprocessed")
        path.mkdir(parents=True, exist_ok=True)
        # Loop through each dataset and pickle it
        for i in range(len(dataset)):
            if i % 10 == 0:
                print(f"Processing {i}/{len(dataset)}")
            data = dataset[i]
            # Save the dictionary to a pickle file as 0 prepended to the index
            with open(path / f"{i:05d}.pkl", "wb") as f:
                pickle.dump(data, f)
    if True:
        # Test the datamodule
        data_dir = Path("./datasets/ahmed_preprocessed")
        datamodule = AhmedBodyDataModule(data_dir)
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
