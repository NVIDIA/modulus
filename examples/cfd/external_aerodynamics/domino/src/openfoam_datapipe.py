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

"""
This is the datapipe to read OpenFoam files (vtp/vtu/stl) and save them as point clouds 
in npy format. 

"""

import time, random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, List, Literal, Mapping, Optional, Union, Callable

import numpy as np
import pandas as pd
import pyvista as pv
import vtk
from modulus.utils.domino.utils import *
from torch.utils.data import Dataset


class DriveSimPaths:
    @staticmethod
    def geometry_path(car_dir: Path) -> Path:
        return car_dir / "body.stl"

    @staticmethod
    def volume_path(car_dir: Path) -> Path:
        return car_dir / "VTK/simpleFoam_steady_3000/internal.vtu"

    @staticmethod
    def surface_path(car_dir: Path) -> Path:
        return car_dir / "VTK/simpleFoam_steady_3000/boundary/aero_suv.vtp"


class DrivAerAwsPaths:
    @staticmethod
    def _get_index(car_dir: Path) -> str:
        return car_dir.name.removeprefix("run_")

    @staticmethod
    def geometry_path(car_dir: Path) -> Path:
        return car_dir / f"drivaer_{DrivAerAwsPaths._get_index(car_dir)}.stl"

    @staticmethod
    def volume_path(car_dir: Path) -> Path:
        return car_dir / f"volume_{DrivAerAwsPaths._get_index(car_dir)}.vtu"

    @staticmethod
    def surface_path(car_dir: Path) -> Path:
        return car_dir / f"boundary_{DrivAerAwsPaths._get_index(car_dir)}.vtp"


class OpenFoamDataset(Dataset):
    """
    Datapipe for converting openfoam dataset to npy

    """

    def __init__(
        self,
        data_path: Union[str, Path],
        kind: Literal["drivesim", "drivaer_aws"] = "drivesim",
        surface_variables: Optional[list] = [
            "pMean",
            "wallShearStress",
        ],
        volume_variables: Optional[list] = ["UMean", "pMean"],
        device: int = 0,
        model_type=None,
    ):
        if isinstance(data_path, str):
            data_path = Path(data_path)
        data_path = data_path.expanduser()

        self.data_path = data_path

        supported_kinds = ["drivesim", "drivaer_aws"]
        assert (
            kind in supported_kinds
        ), f"kind should be one of {supported_kinds}, got {kind}"
        self.path_getter = DriveSimPaths if kind == "drivesim" else DrivAerAwsPaths

        assert self.data_path.exists(), f"Path {self.data_path} does not exist"

        assert self.data_path.is_dir(), f"Path {self.data_path} is not a directory"

        self.filenames = get_filenames(self.data_path)
        random.shuffle(self.filenames)
        self.indices = np.array(len(self.filenames))

        self.surface_variables = surface_variables
        self.volume_variables = volume_variables
        self.device = device
        self.model_type = model_type

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        cfd_filename = self.filenames[idx]
        car_dir = self.data_path / cfd_filename

        stl_path = self.path_getter.geometry_path(car_dir)
        reader = pv.get_reader(stl_path)
        mesh_stl = reader.read()
        stl_vertices = mesh_stl.points
        stl_faces = np.array(mesh_stl.faces).reshape((-1, 4))[
            :, 1:
        ]  # Assuming triangular elements
        mesh_indices_flattened = stl_faces.flatten()
        stl_sizes = mesh_stl.compute_cell_sizes(length=False, area=True, volume=False)
        stl_sizes = np.array(stl_sizes.cell_data["Area"])
        stl_centers = np.array(mesh_stl.cell_centers().points)

        if self.model_type == "volume" or self.model_type == "combined":
            filepath = self.path_getter.volume_path(car_dir)
            reader = vtk.vtkXMLUnstructuredGridReader()
            reader.SetFileName(filepath)
            reader.Update()

            # Get the unstructured grid data
            polydata = reader.GetOutput()
            volume_coordinates, volume_fields = get_volume_data(
                polydata, self.volume_variables
            )
            volume_fields = np.concatenate(volume_fields, axis=-1)
        else:
            volume_fields = None
            volume_coordinates = None

        if self.model_type == "surface" or self.model_type == "combined":
            surface_filepath = self.path_getter.surface_path(car_dir)
            reader = vtk.vtkXMLPolyDataReader()
            reader.SetFileName(surface_filepath)
            reader.Update()
            polydata = reader.GetOutput()

            celldata_all = get_node_to_elem(polydata)
            celldata = celldata_all.GetCellData()
            surface_fields = get_fields(celldata, self.surface_variables)
            surface_fields = np.concatenate(surface_fields, axis=-1)

            mesh = pv.PolyData(polydata)
            surface_coordinates = np.array(mesh.cell_centers().points)

            surface_normals = np.array(mesh.cell_normals)
            surface_sizes = mesh.compute_cell_sizes(
                length=False, area=True, volume=False
            )
            surface_sizes = np.array(surface_sizes.cell_data["Area"])

            # Normalize cell normals
            surface_normals = (
                surface_normals / np.linalg.norm(surface_normals, axis=1)[:, np.newaxis]
            )
        else:
            surface_fields = None
            surface_coordinates = None
            surface_normals = None
            surface_sizes = None

        # Add the parameters to the dictionary
        return {
            "stl_coordinates": np.float32(stl_vertices),
            "stl_centers": np.float32(stl_centers),
            "stl_faces": np.float32(mesh_indices_flattened),
            "stl_areas": np.float32(stl_sizes),
            "surface_mesh_centers": np.float32(surface_coordinates),
            "surface_normals": np.float32(surface_normals),
            "surface_areas": np.float32(surface_sizes),
            "volume_fields": np.float32(volume_fields),
            "volume_mesh_centers": np.float32(volume_coordinates),
            "surface_fields": np.float32(surface_fields),
            "filename": cfd_filename,
        }


if __name__ == "__main__":
    fm_data = OpenFoamDataset(
        data_path="/code/aerofoundationdata/",
        phase="train",
        volume_variables=["UMean", "pMean", "nutMean"],
        surface_variables=["pMean", "wallShearStress", "nutMean"],
        sampling=False,
        sample_in_bbox=False,
    )
    d_dict = fm_data[1]
