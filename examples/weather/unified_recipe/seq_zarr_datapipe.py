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

from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np
import torch
import zarr

try:
    import nvidia.dali as dali
    import nvidia.dali.plugin.pytorch as dali_pth
except ImportError:
    raise ImportError(
        "DALI dataset requires NVIDIA DALI package to be installed. "
        + "The package can be installed at:\n"
        + "https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html"
    )

from modulus.datapipes.datapipe import Datapipe
from modulus.datapipes.meta import DatapipeMetaData

Tensor = torch.Tensor


@dataclass
class MetaData(DatapipeMetaData):
    name: str = "SeqZarrDatapipe"
    # Optimization
    auto_device: bool = True
    cuda_graphs: bool = True
    # Parallel
    ddp_sharding: bool = True


class SeqZarrDatapipe(Datapipe):
    """
    DALI data pipeline for loading sequences from a Zarr array.

    This data pipeline is designed to be general given a Zarr dataset.

    Parameters
    ----------
    zarr_dataset : zarr.hierarchy.Group
        Zarr dataset to load from
    batch_size : int, optional
        Batch size, by default 1
    num_steps : int, optional
        Number of steps to predict, by default 2
    shuffle : bool, optional
        Shuffle data, by default True
    device : Union[str, torch.device], optional
        Device to use, by default "cuda"
    process_rank : int, optional
        Process rank, by default 0
    world_size : int, optional
        World size, by default 1
    """

    def __init__(
        self,
        zarr_dataset: zarr.hierarchy.Group,
        variables: list,
        batch_size: int = 1,
        num_steps: int = 2,
        shuffle: bool = True,
        device: Union[str, torch.device] = "cuda",
        process_rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__(meta=MetaData())

        # Store parameters
        self.zarr_dataset = zarr_dataset
        self.variables = variables
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.shuffle = shuffle

        # Set up device, needed for pipeline
        if isinstance(device, str):
            device = torch.device(device)
        if device.type == "cuda" and device.index == None:
            device = torch.device("cuda:0")
        self.device = device

        # Set up parallel
        self.process_rank = process_rank
        self.world_size = world_size

        # Outputs of pipeline
        self.pipe_outputs = self.variables

        # Create pipeline
        self.pipe = self._create_pipeline()

    def _create_pipeline(self) -> dali.Pipeline:
        """Create DALI pipeline

        Returns
        -------
        dali.Pipeline
            DALI pipeline
        """
        pipe = dali.Pipeline(
            batch_size=self.batch_size,
            num_threads=2,
            prefetch_queue_depth=2,
            py_num_workers=1,
            device_id=self.device.index,
            py_start_method="spawn",
        )

        with pipe:
            # Zarr source
            source = SeqZarrSource(
                self.zarr_dataset,
                self.variables,
                num_steps=self.num_steps,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                process_rank=self.process_rank,
                world_size=self.world_size,
            )

            # Update length of dataset
            self.total_length = len(source) // self.batch_size

            # Read current batch
            data = dali.fn.external_source(
                source,
                num_outputs=len(self.pipe_outputs),
                parallel=True,
                batch=False,
                device="cpu",
            )

            if self.device.type == "cuda":
                # Move tensors to GPU as external_source won't do that
                data = [d.gpu() for d in data]

            # Set outputs
            pipe.set_outputs(*data)

        return pipe

    def __iter__(self):
        # Reset the pipeline before creating an iterator to enable epochs.
        self.pipe.reset()
        # Create DALI PyTorch iterator.
        return dali_pth.DALIGenericIterator([self.pipe], self.pipe_outputs)

    def __len__(self):
        return self.total_length


class SeqZarrSource:
    """
    DALI Source for loading a zarr array.
    The arrays will be indexed along the first dimension (usually time).

    Parameters
    ----------
    zarr_dataset : zarr.hierarchy.Group
        Zarr dataset
    num_steps : int
        Number of steps to predict
    batch_size : int, optional
        Batch size, by default 1
    shuffle : bool, optional
        Shuffle data, by default True
    process_rank : int, optional
        Process rank, by default 0
    world_size : int, optional
        World size, by default 1
    """

    def __init__(
        self,
        zarr_dataset: zarr.hierarchy.Group,
        variables: list,
        num_steps: int,
        batch_size: int = 1,
        shuffle: bool = True,
        process_rank: int = 0,
        world_size: int = 1,
    ):
        # Set up parameters
        self.zarr_dataset = zarr_dataset
        self.variables = variables
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Check if all zarr arrays have the same first dimension
        self.first_dim = self.zarr_dataset[variables[0]].shape[0]
        for variable in self.variables:
            if zarr_dataset[variable].shape[0] != self.first_dim:
                raise ValueError("All zarr arrays must have the same first dimension.")

        # Get number of samples
        self.indices = np.arange(self.first_dim - self.num_steps)
        self.indices = np.array_split(self.indices, world_size)[process_rank]

        # Get number of full batches, ignore possible last incomplete batch for now.
        self.num_batches = len(self.indices) // self.batch_size

        # Set up last epoch
        self.last_epoch = None

    def __call__(
        self, sample_info: dali.types.SampleInfo
    ) -> Tuple[Tensor, Tensor, np.ndarray, np.ndarray, np.ndarray]:
        if sample_info.iteration >= self.num_batches:
            raise StopIteration()

        # Shuffle before the next epoch starts
        if self.shuffle and sample_info.epoch_idx != self.last_epoch:
            # All workers use the same rng seed so the resulting
            # indices are the same across workers
            np.random.default_rng(seed=sample_info.epoch_idx).shuffle(self.indices)
            self.last_epoch = sample_info.epoch_idx

        # Get local indices from global index
        idx = self.indices[sample_info.idx_in_epoch]

        # Make time indices
        time_idx = idx + np.arange(self.num_steps)

        # Get data
        data = []

        # Get slices
        for i, variable in enumerate(self.variables):
            data.append(self.zarr_dataset[variable][time_idx])

        return tuple(data)

    def __len__(self):
        return len(self.indices)
