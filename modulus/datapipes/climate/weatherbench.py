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

import os

import numpy as np
import torch
import xarray as xr

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
from pathlib import Path
from typing import Iterable, List, Tuple, Union

from ..datapipe import Datapipe
from ..meta import DatapipeMetaData

Tensor = torch.Tensor


@dataclass
class MetaData(DatapipeMetaData):
    name: str = "WeatherBench"
    # Optimization
    auto_device: bool = True
    cuda_graphs: bool = True
    # Parallel
    ddp_sharding: bool = True


class WeatherBenchDatapipe(Datapipe):
    """WeatherBench DALI data pipeline

    Parameters
    ----------
    data_dir : str
        Directory where WeatherBench data is stored
    channels : Union[List[int], None], optional
        Defines which WeatherBench variables to load, if None will use all in WeatherBench file, by default None
    batch_size : int, optional
        Batch size, by default 1
    stride : int, optional
        Number of steps between input and output variables. For example, if the dataset
        contains data at every 6 hours, a stride 1 = 6 hour delta t and
        stride 2 = 12 hours delta t, by default 1
    num_steps : int, optional
        Number of timesteps are included in the output variables, by default 1
    patch_size : Union[Tuple[int, int], int, None], optional
        If specified, crops input and output variables so image dimensions are
        divisible by patch_size, by default None
    num_samples_per_year : int, optional
        Number of samples randomly taken from each year. If None, all will be use, by default None
    shuffle : bool, optional
        Shuffle dataset, by default True
    num_workers : int, optional
        Number of workers, by default 1
    device: Union[str, torch.device], optional
        Device for DALI pipeline to run on, by default cuda
    process_rank : int, optional
        Rank ID of local process, by default 0
    world_size : int, optional
        Number of training processes, by default 1
    """

    def __init__(
        self,
        data_dir: str,
        channels: Union[List[int], None] = None,
        constants_channels: Union[List[int], None] = None,
        batch_size: int = 1,
        num_steps: int = 1,
        stride: int = 1,
        patch_size: Union[Tuple[int, int], int, None] = None,
        t_num: int = 6,
        num_samples_per_year: Union[int, None] = None,
        shuffle: bool = True,
        num_workers: int = 1,
        device: Union[str, torch.device] = "cuda",
        process_rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__(meta=MetaData())
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.data_dir = Path(data_dir)
        self.channels = channels
        self.constants_channels = constants_channels
        self.stride = stride
        self.num_steps = num_steps
        self.num_samples_per_year = num_samples_per_year
        self.t_num = t_num
        self.process_rank = process_rank
        self.world_size = world_size

        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size

        # Set up device, needed for pipeline
        if isinstance(device, str):
            device = torch.device(device)
        # Need a index id if cuda
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda:0")
        self.device = device

        # check root directory exists
        if not self.data_dir.is_dir():
            raise IOError(f"Error, data directory {self.data_dir} does not exist")

        self.dir_name_list = [
            "geopotential",
            "temperature",
            "relative_humidity",
            "u_component_of_wind",
            "v_component_of_wind",
            "2m_temperature",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "total_precipitation",
            "geopotential_500",
            "potential_vorticity",
            "specific_humidity",
            "temperature_850",
            "toa_incident_solar_radiation",
            "total_cloud_cover",
            "vorticity",
        ]
        self.variable_name_list = [
            "z",
            "t",
            "r",
            "u",
            "v",
            "t2m",
            "u10",
            "v10",
            "tp",
            "z",
            "pv",
            "q",
            "t",
            "tisr",
            "tcc",
            "vo",
        ]
        self.constants_variable_name_list = [
            "lsm",
            "orography",
            "slt",
            "lat2d",
            "lon2d",
        ]

        self.parse_dataset_files()
        # self.load_statistics()

        self.pipe = self._create_pipeline()

    def parse_dataset_files(self) -> None:
        """Parses the data directory for valid nc files and determines training samples

        Raises
        ------
        ValueError
            In channels specified or number of samples per year is not valid
        """
        # get all input data files
        #         self.data_paths = sorted(self.data_dir.glob("*/*.nc"))
        #         for data_path in self.data_paths:
        #             self.logger.info(f"WeatherBench file found: {data_path}")
        self.n_years = 40

        self.logger.info(f"Number of years: {self.n_years}")

        # get total number of examples and image shape from the first file,
        # assuming other files have exactly the same format.
        self.logger.info(f"Getting file stats from {self.data_dir}")
        with xr.open_mfdataset(
            os.path.join(self.data_dir, self.dir_name_list[0], "*.nc")
        )[self.variable_name_list[0]] as f:
            # truncate the dataset to avoid out-of-range sampling
            data_samples_per_year = f.time.size // self.n_years
            self.img_shape = f.shape[-2:]

            # If channels not provided, use all of them
            if self.channels is None:
                self.channels = [i for i in range(len(self.dir_name_list))]

            # If num_samples_per_year use all
            if self.num_samples_per_year is None:
                self.num_samples_per_year = data_samples_per_year

            # Adjust image shape if patch_size defined
            if self.patch_size is not None:
                self.img_shape = [
                    s - s % self.patch_size[i] for i, s in enumerate(self.img_shape)
                ]
            self.logger.info(f"Input image shape: {self.img_shape}")

            # Get total length
            self.total_length = self.n_years * self.num_samples_per_year
            self.length = self.total_length

            # Sanity checks
            if self.num_samples_per_year > data_samples_per_year:
                raise ValueError(
                    f"num_samples_per_year ({self.num_samples_per_year}) > number of \
                    samples available ({data_samples_per_year})!"
                )

            self.logger.info(f"Number of samples/year: {self.num_samples_per_year}")

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
            py_num_workers=self.num_workers,
            device_id=self.device.index,
            py_start_method="spawn",
        )

        with pipe:
            source = WeatherBenchDaliExternalSource(
                data_dir=self.data_dir,
                dir_name_list=self.dir_name_list,
                variable_name_list=self.variable_name_list,
                constants_variable_name_list=self.constants_variable_name_list,
                num_samples=self.total_length,
                channels=self.channels,
                constants_channels=self.constants_channels,
                stride=self.stride,
                num_steps=self.num_steps,
                num_samples_per_year=self.num_samples_per_year,
                t_num=self.t_num,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                process_rank=self.process_rank,
                world_size=self.world_size,
            )
            # Update length of dataset
            self.length = len(source) // self.batch_size
            # Read current batch.
            invar, outvar = dali.fn.external_source(
                source,
                num_outputs=2,
                parallel=True,
                batch=False,
            )
            if self.device.type == "cuda":
                # Move tensors to GPU as external_source won't do that.
                invar = invar.gpu()
                outvar = outvar.gpu()

            # Crop.
            h, w = self.img_shape
            invar = invar[:, :, :h, :w]
            outvar = outvar[:, :, :h, :w]
            # Standardize.
            mu, sd = source.mean()
            invar = dali.fn.normalize(invar, mean=mu, stddev=sd)
            outvar = dali.fn.normalize(outvar, mean=mu, stddev=sd)

            # Set outputs.
            pipe.set_outputs(invar, outvar)

        return pipe

    def __iter__(self):
        # Reset the pipeline before creating an iterator to enable epochs.
        self.pipe.reset()
        # Create DALI PyTorch iterator.
        return dali_pth.DALIGenericIterator([self.pipe], ["invar", "outvar"])

    def __len__(self):
        return self.length


class WeatherBenchDaliExternalSource:
    """DALI Source for lazy-loading the WeatherBench files

    Parameters
    ----------
        Directory where WeatherBench data is stored
    num_samples : int
        Total number of training samples
    channels : Iterable[int]
        List representing which WeatherBench variables to load
    stride : int
        Number of steps between input and output variables
    num_steps : int
        Number of timesteps are included in the output variables
    num_samples_per_year : int
        Number of samples randomly taken from each year
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
        data_dir: str,
        dir_name_list: Iterable[str],
        variable_name_list: Iterable[str],
        constants_variable_name_list: Iterable[str],
        num_samples: int,
        channels: Iterable[int],
        constants_channels: Iterable[int],
        num_steps: int,
        stride: int,
        num_samples_per_year: int,
        t_num: int,
        batch_size: int = 1,
        shuffle: bool = True,
        process_rank: int = 0,
        world_size: int = 1,
    ):
        # Will be populated later once each worker starts running in its own process.
        self.data_dir = data_dir
        self.dir_name_list = dir_name_list
        self.variable_name_list = variable_name_list
        self.constants_variable_name_list = constants_variable_name_list
        self.num_samples = num_samples
        self.chans = list(channels)
        self.constants_chans = list(constants_channels)
        self.num_steps = num_steps
        self.stride = stride
        self.num_samples_per_year = num_samples_per_year
        self.t_num = t_num
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.last_epoch = None

        self.indices = np.arange(num_samples)
        # Shard from indices if running in parallel
        self.indices = np.array_split(self.indices, world_size)[process_rank]

        # Get number of full batches, ignore possible last incomplete batch for now.
        # Also, DALI external source does not support incomplete batches in parallel mode.
        self.num_batches = len(self.indices) // self.batch_size
        self.data_files = [
            xr.open_mfdataset(
                os.path.join(self.data_dir, self.dir_name_list[chan], "*.nc")
            )[self.variable_name_list[chan]]
            for chan in self.chans
        ]
        self.constants_data_files = [
            xr.open_mfdataset(os.path.join(self.data_dir, "constants/*.nc"))[
                self.constants_variable_name_list[constants_chan]
            ]
            for constants_chan in self.constants_chans
        ]
        if not os.path.exists(os.path.join(self.data_dir, "mean.npy")):
            self.mu = np.hstack(
                [
                    expand_dim_by1(
                        data_file.mean(("time", "lat", "lon")).compute().to_numpy(), 1
                    )
                    for data_file in self.data_files
                ]
                + [
                    expand_dim_by1(
                        data_file.mean(("lat", "lon")).compute().to_numpy(), 1
                    )
                    for data_file in self.constants_data_files
                ]
            )
            self.mu = np.expand_dims(self.mu, axis=[0, 2, 3])
            self.sd = np.hstack(
                [
                    expand_dim_by1(
                        data_file.std("time").mean(("lat", "lon")).compute().to_numpy(),
                        1,
                    )
                    for data_file in self.data_files
                ]
                + [
                    expand_dim_by1(
                        data_file.std(("lat", "lon")).compute().to_numpy(), 1
                    )
                    for data_file in self.constants_data_files
                ]
            )
            self.sd = np.expand_dims(self.sd, axis=[0, 2, 3])
            with open(os.path.join(self.data_dir, "mean.npy"), "wb") as f:
                np.save(f, self.mu)
                np.save(f, self.sd)
        else:
            with open(os.path.join(self.data_dir, "mean.npy"), "rb") as f:
                self.mu = np.load(f)
                self.sd = np.load(f)

    def __call__(self, sample_info: dali.types.SampleInfo) -> Tuple[Tensor, Tensor]:
        if sample_info.iteration >= self.num_batches:
            raise StopIteration()

        # Shuffle before the next epoch starts.
        if self.shuffle and sample_info.epoch_idx != self.last_epoch:
            # All workers use the same rng seed so the resulting
            # indices are the same across workers.
            np.random.default_rng(seed=sample_info.epoch_idx).shuffle(self.indices)
            self.last_epoch = sample_info.epoch_idx

        # Get local indices from global index.
        idx = self.indices[sample_info.idx_in_epoch]
        # year_idx = idx // self.num_samples_per_year
        in_idx = [idx - (self.t_num - i - 1) * self.stride for i in range(self.t_num)]

        data = [
            expand_dim_by1(data_file[in_idx].to_numpy(), 4, axis=1)
            for data_file in self.data_files
        ] + [
            np.repeat(
                expand_dim_by1(expand_dim_by1(data_file.to_numpy(), 3), 4),
                self.t_num,
                axis=0,
            )
            for data_file in self.constants_data_files
        ]
        # Has [C,H,W] shape.
        invar = np.concatenate(data, axis=1)

        # Has [T,C,H,W] shape.
        outvar = np.empty((self.num_steps,) + invar.shape[1:], dtype=invar.dtype)

        for i in range(self.num_steps):
            out_idx = in_idx[-1] + (i + 1) * self.stride
            data = [
                expand_dim_by1(data_file[out_idx].to_numpy(), 3)
                for data_file in self.data_files
            ] + [
                expand_dim_by1(data_file.to_numpy(), 3)
                for data_file in self.constants_data_files
            ]
            outvar[i] = np.vstack(data)

        return invar, outvar

    def __len__(self):
        return len(self.indices)

    def mean(self):
        return self.mu, self.sd


def expand_dim_by1(x, n, axis=0):
    if len(x.shape) == n - 1:
        return np.expand_dims(x, axis=axis)
    elif len(x.shape) == n:
        return x
    else:
        raise Exception("invalid dim")
