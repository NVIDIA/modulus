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

import numpy as np
import torch
import zarr
import fsspec
from dataclasses import dataclass
from typing import Iterable, List, Union, Tuple, Dict
from pathlib import Path

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
    name: str = "ERA5Zarr"
    # Optimization
    auto_device: bool = True
    cuda_graphs: bool = True
    # Parallel
    ddp_sharding: bool = True


class ERA5Datapipe(Datapipe):
    """
    DALI data pipeline for loading ERA5 zarr arrays.
    """

    def __init__(
        self,
        static_variables: List[str],
        input_variables: List[str],
        base_path: str,
        fs: fsspec.AbstractFileSystem = None,
        batch_size: int = 1,
        stride: int = 1,
        num_steps: int = 2,
        shuffle: bool = True,
        device: Union[str, torch.device] = "cuda",
        process_rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__(meta=MetaData())

        # Set up variables
        self.static_variables = self._reformat_variables(static_variables)
        self.input_variables = self._reformat_variables(input_variables)

        # Load zarr arrays
        self.static_zarr_arrays = {}
        self.input_zarr_arrays = {}

        # Set up paths
        self.base_path = base_path
        if fs is None:
            fs = fsspec.filesystem("file")
        self.fs = fs

        # Set up parameters
        self.batch_size = batch_size
        self.stride = stride
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

        # Load zarr arrays
        self.static_zarr_arrays = self._load_zarr_array(self.static_variables)
        self.input_zarr_arrays = self._load_zarr_array(self.input_variables)

        # Outputs of pipeline
        self.pipe_outputs = ["static", "input"] # TODO: add timestamps

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
            # ERA5 source
            source = ERA5Source(
                self.static_zarr_arrays,
                self.input_zarr_arrays,
                num_steps=self.num_steps,
                stride=self.stride,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                process_rank=self.process_rank,
                world_size=self.world_size,
            )

            # Update length of dataset
            self.total_length = len(source) // self.batch_size

            # Read current batch
            static, input_ = dali.fn.external_source(
                source,
                num_outputs=2,
                parallel=True,
                batch=False,
                device="cpu",
            )

            # Make output list
            outputs = [static, input_]

            if self.device.type == "cuda":
                # Move tensors to GPU as external_source won't do that
                outputs = [o.gpu() for o in outputs]

            # Move this to remapping
            outputs = [o[:, :, :720, :1440] for o in outputs]

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

    def _variable_to_zarr_name(self, variable: str, pressure_level: int = None):
        # create zarr path for variable
        if pressure_level:
            zarr_path = f"{self.base_path}/pressure_levels/{variable}_pl{pressure_level}"
        else:
            zarr_path = f"{self.base_path}/single_levels/{variable}"
        zarr_path += ".zarr"
        return zarr_path

    def _reformat_variables(self, variables: List[str]) -> List[str]:
        reformatted_variables = []
        for variable in variables:
            if isinstance(variable, str):
                reformatted_variables.append((variable, None))
            else:
                reformatted_variables.append(tuple(variable))
        return reformatted_variables

    def _load_zarr_array(self, variables: List[str]):
        # Load zarr arrays
        zarr_arrays = {}
        for variable, pressure_level in variables:
            zarr_path = self._variable_to_zarr_name(variable, pressure_level)
            mapper = self.fs.get_mapper(zarr_path)
            zarr_dataset = zarr.open(mapper, mode="r")
            non_time_lat_lon_var = [var for var in zarr_dataset.array_keys() if var not in ["time", "latitude", "longitude"]]
            zarr_array = zarr_dataset[non_time_lat_lon_var[0]]
            zarr_arrays[(variable, pressure_level)] = zarr_array
        return zarr_arrays


class ERA5Source:
    """
    DALI Source for loading ERA5 data from zarr arrays.
    """

    def __init__(
        self,
        static_zarr_arrays: Dict[str, zarr.Array],
        input_zarr_arrays: Dict[str, zarr.Array],
        num_steps: int,
        stride: int,
        batch_size: int = 1,
        shuffle: bool = True,
        process_rank: int = 0,
        world_size: int = 1,
    ):

        # Set up parameters
        self.static_zarr_arrays = static_zarr_arrays
        self.input_zarr_arrays = input_zarr_arrays
        self.num_steps = num_steps
        self.stride = stride
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Check if all zarr arrays have the same length
        #static_lengths = [zarr_array.shape[0] for zarr_array in static_zarr_arrays.values()]
        #input_lengths = [zarr_array.shape[0] for zarr_array in input_zarr_arrays.values()]
        #assert len(set(static_lengths + input_lengths)) == 1, "All variables must have the same length."

        # Get number of static and input variables
        self.nr_static_variables = len(static_zarr_arrays)
        self.nr_input_variables = len(input_zarr_arrays)

        # Get shape of variables
        self.array_shape = list(static_zarr_arrays.values())[0].shape

        # Get number of samples
        self.indices = np.arange(self.array_shape[0] - self.num_steps * self.stride)
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
        time_idx = idx + np.arange(self.num_steps) * self.stride

        # Make blank array
        static = np.zeros((self.num_steps, self.nr_static_variables, *self.array_shape[1:]), dtype=np.float32)
        input_ = np.zeros((self.num_steps, self.nr_input_variables, *self.array_shape[1:]), dtype=np.float32)

        # Fill array
        for i, (variable, pressure_level) in enumerate(self.static_zarr_arrays.keys()):
            static[:, i] = self.static_zarr_arrays[(variable, pressure_level)][time_idx]
        for i, (variable, pressure_level) in enumerate(self.input_zarr_arrays.keys()):
            input_[:, i] = self.input_zarr_arrays[(variable, pressure_level)][time_idx]

        return static, input_

    def __len__(self):
        return len(self.indices)
