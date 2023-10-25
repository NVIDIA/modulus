# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import h5py
import numpy as np
import torch
from datetime import datetime, timedelta
import netCDF4 as nc
import zarr
import kvikio.zarr

try:
    import nvidia.dali as dali
    import nvidia.dali.plugin.pytorch as dali_pth
except ImportError:
    raise ImportError(
        "DALI dataset requires NVIDIA DALI package to be installed. "
        + "The package can be installed at:\n"
        + "https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html"
    )

from dataclasses import dataclass
from typing import Iterable, List, Union, Tuple
from pathlib import Path
from torch.utils.data import Dataset

from modulus.datapipes.datapipe import Datapipe
from modulus.datapipes.meta import DatapipeMetaData

Tensor = torch.Tensor


@dataclass
class MetaData(DatapipeMetaData):
    name: str = "ClimateZarr"
    # Optimization
    auto_device: bool = True
    cuda_graphs: bool = True
    # Parallel
    ddp_sharding: bool = True


class ClimateZarrDatapipe(Datapipe):
    """
    A Climate DALI data pipeline for Zarr files. This pipeline loads data from
    Zarr files, which can include latitude, longitude, cosine of the solar zenith
    angle, geopotential, and land sea mask if specified. Additionally, it normalizes
    the data if a statistics file is provided. The pipeline returns a dictionary
    with the following structure:

    - `state_seq`: Tensor of shape (batch_size, num_steps, num_channels, height,
      width). This sequence is drawn from the HDF5 file and normalized if a
      statistics file is provided.
    - `timestamps`: Tensor of shape (batch_size, num_steps), containing
      timestamps for each timestep in the sequence.

    The Zarr files should contain the following variable
    with the corresponding name:
    - `fields`: Tensor of shape (num_timesteps, num_channels, height, width),
      containing climate data.
    - `time`: Tensor of shape (num_timesteps), containing timestamps for each
      field in the `fields` variable.
    - `lat`: Tensor of shape (height, width), containing latitude values.
    - `lon`: Tensor of shape (height, width), containing longitude values.

    Parameters
    ----------
    zarr_array : zarr.core.Array
        Zarr array to load data from
    channels : Union[List[int], None], optional
        Defines which climate variables to load, if None will use all in Zarr file, by default None
    batch_size : int, optional
        Batch size, by default 1
    stride : int, optional
        Number of time steps between input and output variables. For example, if the dataset
        contains data at every 6 hours, a stride 1 = 6 hour delta t and
        stride 2 = 12 hours delta t, by default 1
    num_steps : int, optional
        Number of timesteps to return, by default 2 (1 for input, 1 for output)
    shuffle : bool, optional
        Shuffle dataset, by default True
    device: Union[str, torch.device], optional
        Device for DALI pipeline to run on, by default cuda
    process_rank : int, optional
        Rank ID of local process, by default 0
    world_size : int, optional
        Number of training processes, by default 1
    gpu_decompression : bool, optional
        Whether the zarr file is GPU decompression compatible, by default True
    """

    def __init__(
        self,
        zarr_array: zarr.core.Array,
        channels: Union[List[int], None] = None,
        batch_size: int = 1,
        stride: int = 1,
        num_steps: int = 2,
        shuffle: bool = True,
        device: Union[str, torch.device] = "cuda",
        process_rank: int = 0,
        world_size: int = 1,
        gpu_decompression: bool = True,
    ):
        super().__init__(meta=MetaData())
        self.zarr_array = zarr_array
        self.channels = channels
        self.batch_size = batch_size
        self.stride = stride
        self.num_steps = num_steps
        self.shuffle = shuffle
        self.process_rank = process_rank
        self.gpu_decompression = gpu_decompression

        # Determine outputs of pipeline
        self.pipe_outputs = ["state_seq", "timestamps"]

        # Set up device, needed for pipeline
        if isinstance(device, str):
            device = torch.device(device)

        # Need a index id if cuda
        if device.type == "cuda" and device.index == None:
            device = torch.device("cuda:0")
        self.device = device

        # Parameters for gpu decompression
        if self.gpu_decompression:
            self.external_source_parallel = False
        else:
            self.external_source_parallel = True

        # Create pipeline
        self.pipe = self._create_pipeline()

    def _create_pipeline(self) -> dali.Pipeline:
        """Create DALI pipeline

        Returns
        -------
        dali.Pipeline
            HDF5 DALI pipeline
        """
        pipe = dali.Pipeline(
            batch_size=self.batch_size,
            num_threads=2,
            prefetch_queue_depth=2,
            py_num_workers=self.num_workers,
            device_id=self.device.index,
            py_start_method="spawn",
        )

        with pipe:
            # Zarr source
            source = ClimateDaliExternalSource(
                zarr_array=self.zarr_array,
                channels=self.channels,
                stride=self.stride,
                num_steps=self.num_steps,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                process_rank=self.process_rank,
                world_size=self.world_size,
            )

            # Update length of dataset
            self.total_length = len(source) // self.batch_size

            # Read current batch
            state_seq = dali.fn.external_source(
                source,
                num_outputs=1,
                parallel=self.external_source_parallel,
                batch=False,
            )

            # Make output list
            outputs = [state_seq]

            if self.device.type == "cuda":
                # Move tensors to GPU as external_source won't do that
                outputs = [o.gpu() for o in outputs]

            # Set outputs
            pipe.set_outputs(*outputs)

        return pipe

    def __iter__(self):
        # Reset the pipeline before creating an iterator to enable epochs.
        self.pipe.reset()
        # Create DALI PyTorch iterator.
        return dali_pth.DALIGenericIterator([self.pipe], self.pipe_outputs)

    def __len__(self):
        return self.total_length


class ClimateDaliExternalSource:
    """DALI Source for lazy-loading the Zarr climate files

    Parameters
    ----------
    zarr_array : zarr.core.Array
        Directory where climate data is stored
    channels : Iterable[int]
        List representing which climate variables to load
    stride : int
        Number of steps between input and output variables
    num_steps : int
        Number of timesteps to load
    batch_size : int, optional
        Batch size, by default 1
    shuffle : bool, optional
        Shuffle dataset, by default True
    process_rank : int, optional
        Rank ID of local process, by default 0
    world_size : int, optional
        Number of training processes, by default 1

    Note
    ----
    For more information about DALI external source operator:
    https://docs.nvidia.com/deeplearning/dali/archives/dali_1_13_0/user-guide/docs/examples/general/data_loading/parallel_external_source.html
    """

    def __init__(
        self,
        zarr_array: zarr.core.Array,
        num_samples: int,
        channels: Iterable[int],
        num_steps: int,
        stride: int,
        batch_size: int = 1,
        shuffle: bool = True,
        process_rank: int = 0,
        world_size: int = 1,
        gpu_decompression: bool = False,
    ):
        self.zarr_array = zarr_array
        self.channels = list(channels)
        self.num_steps = num_steps
        self.stride = stride
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.gpu_decompression = gpu_decompression
        self.last_epoch = None

        # Get number of samples
        self.indices = np.arange(self.zarr_array.shape[0])
        self.indices = np.array_split(self.indices, world_size)[process_rank]

        # Get number of full batches, ignore possible last incomplete batch for now.
        self.num_batches = len(self.indices) // self.batch_size

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

        # Load sequence of input variables
        time_idx = idx + np.arange(self.num_steps) * self.stride
        state_seq = self.zarr_array[time_idx, self.channels]

        return state_seq

    def __len__(self):
        return len(self.indices)
