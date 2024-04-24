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

from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pyvista as pv
import webdataset as wds
from torch.utils.data import DataLoader, Dataset

try:
    import ensightreader
except ImportError:
    print(
        "Could not import ensightreader. Please install it from `pip install ensight-reader`"
    )

from src.data.base_datamodule import BaseDataModule
from src.data.mesh_utils import Normalizer, convert_to_pyvista
from src.data.webdataset_base import Webdataset
from src.data.components.drivaer_webdataset_preprocessors import (
    DrivAerWebdatasetDragPreprocessingFunctor,
    DrivAerWebdatasetSDFPreprocessingFunctor,
)

# DrivAer dataset
# Air density = 1.205 kg/m^3
# Stream velocity = 38.8889 m/s
DRIVAER_AIR_DENSITY = 1.205
DRIVAER_STREAM_VELOCITY = 38.8889

# DrivAer pressure mean and std
DRIVAER_PRESSURE_MEAN = -150.13066236223494
DRIVAER_PRESSURE_STD = 229.1046667362158


class DrivAerDataset(Dataset):
    """DrivAer dataset.

    Data sets:
    Data set A: DrivAer without spoiler, 400 simulations (280 train, 60 validation, 60 test)
    Data set B: DrivAer with spoiler, 600 simulations (420 train, 90 validation, 90 test)

    1.       Train on A, test on  A
    2.       Train on B, test on B
    3.       Train on A + N samples of B (N is {0, 10, 50, 200, Max}), test on A and B
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        phase: Literal["train", "val", "test"] = "train",
        has_spoiler: bool = False,
        variables: Optional[list] = [
            "time_avg_pressure",
            # "time_avg_velocity", # data is invalid
            "time_avg_wall_shear_stress",
        ],
    ):
        if isinstance(data_path, str):
            data_path = Path(data_path)
        data_path = data_path.expanduser()
        self.data_path = data_path  # Path that contains data_set_A and data_set_B
        assert isinstance(
            has_spoiler, bool
        ), f"has_spoiler should be a boolean, got {has_spoiler}"
        assert phase in [
            "train",
            "val",
            "test",
        ], f"phase should be one of ['train', 'val', 'test'], got {phase}"
        assert self.data_path.exists(), f"Path {self.data_path} does not exist"
        assert self.data_path.is_dir(), f"Path {self.data_path} is not a directory"
        assert (
            self.data_path / "data_set_A"
        ).exists(), f"Path {self.data_path} does not contain data_set_A"
        self.has_spoiler = has_spoiler

        if has_spoiler:
            self.data_path = self.data_path / "data_set_B"
            self.TEST_INDS = np.array(range(510, 600))
            self.VAL_INDS = np.array(range(420, 510))
            self.TRAIN_INDS = np.array(range(420))
        else:
            self.data_path = self.data_path / "data_set_A"
            self.TEST_INDS = np.array(range(340, 400))
            self.VAL_INDS = np.array(range(280, 340))
            self.TRAIN_INDS = np.array(range(280))

        # Load parameters
        parameters = pd.read_csv(
            self.data_path / "ParameterFile.txt", delim_whitespace=True
        )
        self.phase = phase
        if phase == "train":
            self.indices = self.TRAIN_INDS
        elif phase == "val":
            self.indices = self.VAL_INDS
        elif phase == "test":
            self.indices = self.TEST_INDS
        self.parameters = parameters.iloc[self.indices]
        self.variables = variables

    @property
    def _attribute(self, variable, name):
        return self.data[variable].attrs[name]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]
        case = ensightreader.read_case(
            self.data_path / "snapshots" / f"EnSight{index}" / f"EnSight{index}.case"
        )
        geofile = case.get_geometry_model()
        ids = geofile.get_part_ids()  # list of part ids
        # remove id 49, which is the internalMesh for without spoiler
        if self.has_spoiler:
            ids.remove(50)
        else:
            ids.remove(49)

        pv_parts = []
        variable_data = defaultdict(list)
        for part_id in ids:
            # print(f"Reading part {part_id} / {len(ids)} for case {index}")
            part = geofile.get_part_by_id(part_id)
            element_blocks = part.element_blocks
            with geofile.open() as fp_geo:
                part_coordinates = part.read_nodes(fp_geo)

            # Read element data
            faces = []
            with geofile.open() as fp_geo:
                for block in part.element_blocks:
                    faces_block = convert_to_pyvista(block, fp_geo)
                    faces.append(faces_block)
            faces = np.concatenate(faces)
            part_mesh = pv.PolyData(part_coordinates, faces)
            pv_parts.append(part_mesh)

            # Get variables
            for variable_name in self.variables:
                variable = case.get_variable(variable_name)
                blocks = []
                for element_block in element_blocks:
                    with variable.mmap() as mm_var:
                        data = variable.read_element_data(
                            mm_var, part.part_id, element_block.element_type
                        )
                        if data is None:
                            print(
                                f"Variable {variable_name} is None in element block {element_block}"
                            )

                        blocks.append(data)
                data = np.concatenate(blocks)
                # scalar variables are transformed to N,1 arrays
                if len(data.shape) == 1:
                    data = data[:, np.newaxis]
                variable_data[variable_name].append(data)

            # Check if the data is consistent
            for k, v in variable_data.items():
                # The last item is the current part
                assert (
                    len(v[-1]) == part_mesh.n_faces_strict
                ), f"Length of {k} is not consistent"

        # Combine parts into one mesh
        mesh = (
            pv.MultiBlock(pv_parts).combine(merge_points=True).extract_surface().clean()
        )

        # Concatenate the variable_data
        for k, v in variable_data.items():
            variable_data[k] = np.concatenate(v).squeeze()
            assert (
                len(variable_data[k]) == mesh.n_faces_strict
            ), f"Length of {k} is not consistent"

        # Estimate normals
        mesh.compute_normals(
            cell_normals=True, point_normals=True, flip_normals=True, inplace=True
        )

        # Extract cell centers and areas
        cell_centers = np.array(mesh.cell_centers().points)
        cell_normals = np.array(mesh.cell_normals)
        cell_sizes = mesh.compute_cell_sizes(length=False, area=True, volume=False)
        cell_sizes = np.array(cell_sizes.cell_data["Area"])

        # Normalize cell normals
        cell_normals = (
            cell_normals / np.linalg.norm(cell_normals, axis=1)[:, np.newaxis]
        )

        # Get the idx'th row from pandas dataframe
        curr_params = self.parameters.iloc[idx]

        # Add the parameters to the dictionary
        return {
            "mesh_nodes": np.array(mesh.points),
            "cell_centers": cell_centers,
            "cell_areas": cell_sizes,
            "cell_normals": cell_normals,
            **curr_params.to_dict(),
            **variable_data,
        }


class DrivAerWebdataset(Webdataset):
    def __init__(
        self,
        paths: str | List[str],
        dataset_processor: Callable,
    ) -> None:
        if isinstance(paths, str):
            paths = [paths]
        super().__init__(paths, dataset_processor)


class DrivAerDataModule(BaseDataModule):
    def __init__(
        self,
        data_path: Union[Path, str],
        subsets_postfix: Optional[List[str]] = ["spoiler", "nospoiler"],
        webdataset_preprocessor: Callable = DrivAerWebdatasetDragPreprocessingFunctor(
            every_n_data=10
        ),
    ):
        """
        Args:
            data_path (Union[Path, str]): Path that contains train and test directories
            subsets_postfix (Optional[List[str]], optional): Postfixes for the subsets. Defaults to ["spoiler", "nospoiler"].

        """
        super().__init__()

        if isinstance(data_path, str):
            data_path = Path(data_path)
        assert (
            data_path.exists() and data_path.is_dir()
        ), f"{data_path} must exist and should be a directory"
        self.data_dir = data_path
        self.subsets_postfix = subsets_postfix
        self.webdataset_preprocessor = webdataset_preprocessor
        self.setup()

    def setup(self, stage: Optional[str] = None):
        subsets_postfix = self.subsets_postfix
        self._train_dataset = DrivAerWebdataset(
            [str(self.data_dir / f"train_{subset}.tar") for subset in subsets_postfix],
            self.webdataset_preprocessor,
        )
        self._val_dataset = DrivAerWebdataset(
            [str(self.data_dir / f"val_{subset}.tar") for subset in subsets_postfix],
            self.webdataset_preprocessor,
        )
        self._test_dataset = DrivAerWebdataset(
            [str(self.data_dir / f"test_{subset}.tar") for subset in subsets_postfix],
            self.webdataset_preprocessor,
        )
        self.normalizer = Normalizer(DRIVAER_PRESSURE_MEAN, DRIVAER_PRESSURE_STD)
        self.air_coeff = 2 / (DRIVAER_AIR_DENSITY * DRIVAER_STREAM_VELOCITY**2)

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

    def train_dataloader(self, **kwargs) -> wds.WebLoader:
        collate_fn = getattr(self, "collate_fn", None)
        # Remove shuffle from kwargs
        kwargs.pop("shuffle", None)
        buffer_size = kwargs.pop("buffer_size", 100)
        return DataLoader(
            self.train_dataset.shuffle(buffer_size), collate_fn=collate_fn, **kwargs
        )

    def val_dataloader(self, **kwargs) -> DataLoader:
        collate_fn = getattr(self, "collate_fn", None)
        return DataLoader(self.val_dataset, collate_fn=collate_fn, **kwargs)

    def test_dataloader(self, **kwargs) -> DataLoader:
        collate_fn = getattr(self, "collate_fn", None)
        return DataLoader(self.test_dataset, collate_fn=collate_fn, **kwargs)


def test_datamodule(
    data_dir: str,
    subset_postfix: Optional[List[str]] = ["spoiler", "nospoiler"],
    preprocessor: str = "DrivAerWebdatasetDragPreprocessingFunctor",
    every_n_data: Optional[int] = 100,
):
    # String to class
    preprocessor = globals()[preprocessor](every_n_data=every_n_data)
    datamodule = DrivAerDataModule(
        data_dir, subsets_postfix=subset_postfix, webdataset_preprocessor=preprocessor
    )
    for i, batch in enumerate(datamodule.val_dataloader()):
        print(i, batch["cell_centers"].shape, batch["time_avg_pressure_whitened"].shape)


if __name__ == "__main__":
    import sys

    test_datamodule(sys.argv[1], preprocessor=sys.argv[2])
