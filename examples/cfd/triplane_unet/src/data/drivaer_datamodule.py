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

import io
import itertools
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, List, Literal, Mapping, Optional, Union, Callable

import numpy as np
import pandas as pd
import pyvista as pv
import torch
import torch.utils
import torch.utils.data
import webdataset as wds
from torch.utils.data import DataLoader, Dataset

try:
    import ensightreader
except ImportError:
    print(
        "Could not import ensightreader. Please install it from `pip install ensight-reader`"
    )

from src.data.base_datamodule import BaseDataModule
from src.data.components import (
    ComposePreprocessors,
    DrivAerPreprocessingFunctor,
    DrivAerDragPreprocessingFunctor,
)
from src.data.mesh_utils import convert_to_pyvista

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


def split_by_node_equal(
    src: Iterable,
    drop_last: bool = False,
    group: "torch.distributed.ProcessGroup" = None,
):
    """Splits input iterable into equal-sized chunks according to multiprocessing configuration.

    Similar to `Webdataset.split_by_node`, but the resulting split is equal-sized.
    """

    rank, world_size, *_ = wds.utils.pytorch_worker_info(group=group)
    cur = iter(src)
    while len(next_items := list(itertools.islice(cur, world_size))) == world_size:
        yield next_items[rank]

    tail_size = len(next_items)
    assert tail_size < world_size
    # If drop_last is not set, handle the tail.
    if not drop_last and tail_size > 0:
        yield next_items[rank % tail_size]


def from_numpy(sample: Mapping[str, Any], key: str):
    """Loads numpy objects from .npy, .npz or pickled files."""

    np_obj = np.load(io.BytesIO(sample[key]), allow_pickle=True)
    return {k: np_obj[k] for k in np_obj.files}


class DrivAerDataModule(BaseDataModule):
    def __init__(
        self,
        data_path: Union[Path, str],
        subsets_postfix: Optional[List[str]] = ["spoiler", "nospoiler"],
        preprocessors: List[Callable] = None,
        every_n_data: int = 10,
    ):
        """
        Args:
            data_path (Union[Path, str]): Path that contains train and test directories
            subsets_postfix (Optional[List[str]], optional): Postfixes for the subsets.
            Defaults to ["spoiler", "nospoiler"].

        """
        super().__init__()

        if isinstance(data_path, str):
            data_path = Path(data_path)
        assert (
            data_path.exists() and data_path.is_dir()
        ), f"{data_path} must exist and should be a directory"

        self.data_dir = data_path
        self.subsets_postfix = subsets_postfix

        if preprocessors is None:
            preprocessors = []

        # Add normalization and downsampling first.
        default_preproc = DrivAerPreprocessingFunctor(
            pressure_mean=DRIVAER_PRESSURE_MEAN,
            pressure_std=DRIVAER_PRESSURE_STD,
            every_n_data=every_n_data,
        )
        preprocessors.insert(0, default_preproc)
        self.normalizer = default_preproc.normalizer

        self.preprocessors = ComposePreprocessors(preprocessors)

        self._train_dataset = self._create_dataset("train")
        self._val_dataset = self._create_dataset("val")
        self._test_dataset = self._create_dataset("test")

    def _create_dataset(self, prefix: str) -> wds.DataPipeline:
        paths = [
            str(self.data_dir / f"{prefix}_{subset}.tar")
            for subset in self.subsets_postfix
        ]

        # Create dataset with the processing pipeline.
        dataset = wds.DataPipeline(
            wds.SimpleShardList(sorted(paths)),
            wds.tarfile_to_samples(),
            split_by_node_equal,
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

    def _create_dataloader(self, dataset: wds.DataPipeline, **kwargs) -> wds.WebLoader:
        # Handle shuffling and batching.
        stages = []
        if (buf_size := kwargs.pop("shuffle_buffer_size", 0)) or kwargs.pop(
            "shuffle", False
        ):
            stages.append(wds.shuffle(buf_size if buf_size > 0 else 100))

        batch_size = kwargs.pop("batch_size", 1)
        stages.append(
            wds.batched(batch_size, collation_fn=torch.utils.data.default_collate)
        )

        # Create dataloader from the pipeline.
        # Use `compose` to avoid changing the original dataset.
        return wds.WebLoader(
            dataset.compose(*stages),
            batch_size=None,
            shuffle=False,
            **kwargs,
        )

    def train_dataloader(self, **kwargs) -> DataLoader:
        return self._create_dataloader(self.train_dataset, **kwargs)

    def val_dataloader(self, **kwargs) -> DataLoader:
        return self._create_dataloader(self.val_dataset, **kwargs)

    def test_dataloader(self, **kwargs) -> DataLoader:
        return self._create_dataloader(self.test_dataset, **kwargs)


def test_datamodule(
    data_dir: str,
    subset_postfix: Optional[List[str]] = ["spoiler", "nospoiler"],
):
    # String to class
    datamodule = DrivAerDataModule(
        data_dir, subsets_postfix=subset_postfix, preprocessors=[DrivAerDragPreprocessingFunctor()]
    )
    for i, batch in enumerate(datamodule.val_dataloader(num_workers=0)):
        print(i, batch["cell_centers"].shape, batch["time_avg_pressure_whitened"].shape)


if __name__ == "__main__":
    import sys

    test_datamodule(sys.argv[1])
