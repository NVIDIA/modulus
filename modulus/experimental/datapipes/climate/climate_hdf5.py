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

from datetime import datetime, timedelta

import h5py
import netCDF4 as nc
import numpy as np
import torch

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

from modulus.datapipes.datapipe import Datapipe
from modulus.datapipes.meta import DatapipeMetaData
from modulus.experimental.datapipes.climate.utils.zenith_angle import cos_zenith_angle

Tensor = torch.Tensor


@dataclass
class MetaData(DatapipeMetaData):
    name: str = "ClimateHDF5"
    # Optimization
    auto_device: bool = True
    cuda_graphs: bool = True
    # Parallel
    ddp_sharding: bool = True


class ClimateHDF5Datapipe(Datapipe):
    """
    A Climate DALI data pipeline for HDF5 files. This pipeline loads data from
    HDF5 files, which can include latitude, longitude, cosine of the solar zenith
    angle, geopotential, and land sea mask if specified. Additionally, it normalizes
    the data if a statistics file is provided. The pipeline returns a dictionary
    with the following structure:

    - `state_seq`: Tensor of shape (batch_size, num_steps, num_channels, height,
      width). This sequence is drawn from the HDF5 file and normalized if a
      statistics file is provided.
    - `timestamps`: Tensor of shape (batch_size, num_steps), containing
      timestamps for each timestep in the sequence.
    - `land_sea_mask`: Tensor of shape (batch_size, 1, height, width),
      containing the land sea mask if a path to a land sea mask file is
      provided.
    - `geopotential`: Tensor of shape (batch_size, 1, height, width), containing
      geopotential if a path to a geopotential file is provided.
    - `latlon`: Tensor of shape (batch_size, 2, height, width), containing
      latitude and longitude meshgrid if specified.
    - `cos_latlon`: Tensor of shape (batch_size, 3, height, width), containing
      `[cos(lat), sin(lon), cos(lon)]` if specified. This is required by many
      neural climate models.
    - `cos_zenith`: Tensor of shape (batch_size, num_steps, 1, height, width),
      containing the cosine of the solar zenith angle if specified.

    To use this data pipeline, your data directory must be structured as
    follows:
    ```
    ├── data_dir
    │   ├── 1980.h5
    │   ├── 1981.h5
    │   ├── 1982.h5
    │   ├── ...
    │   └── 2020.h5
    ├── stats_dir
    │
    ├── global_means.npy
    │
    ├── global_stds.npy
    ```
    The HDF5 files should contain the following variable
    with the corresponding name:
    - `fields`: Tensor of shape (num_timesteps, num_channels, height, width),
      containing climate data. The order of the channels should match the order
      of the channels in the statistics files. The statistics files should be
      `.npy` files with the shape (1, num_channels, 1, 1).

    This pipeline assumes the HDF5 files have no metadata, such as timestamps.
    Because of this, it's important to specify the `dt` parameter and the
    `start_year` parameter so that the pipeline can compute the correct
    timestamps for each timestep. These timestamps are then used to compute the
    cosine of the solar zenith angle, if specified.

    Parameters
    ----------
    data_dir : str
        Directory where climate data is stored
    stats_dir : Union[str, None], optional
        Directory to data statistic numpy files for normalization, if None, no normalization
        will be used, by default None
    channels : Union[List[int], None], optional
        Defines which climate variables to load, if None will use all in HDF5 file, by default None
    batch_size : int, optional
        Batch size, by default 1
    stride : int, optional
        Number of steps between input and output variables. For example, if the dataset
        contains data at every 6 hours, a stride 1 = 6 hour delta t and
        stride 2 = 12 hours delta t, by default 1
    dt : float, optional
        Time in hours between each timestep in the dataset, by default 6 hr
    start_year : int, optional
        Start year of dataset, by default 1980
    num_steps : int, optional
        Number of timesteps to return, by default 2 (1 for input, 1 for output)
    lsm_filename : str, optional
        Path to land sea mask file, by default None
    geopotential_filename : str, optional
        Path to geopotential file, by default None
    use_latlon : bool, optional
        Include latitude and longitude meshgrid, by default False
    use_cos_zenith : bool, optional
        Include cosine of the solar zenith angle, by default False. If True then latitude and longitude
        will also be computed.
    latlon_lower_bound : Tuple[float, float], optional
        Lower bound of latitude and longitude, by default (-90, -180)
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
        stats_dir: Union[str, None] = None,
        channels: Union[List[int], None] = None,
        batch_size: int = 1,
        stride: int = 1,
        dt: float = 6.0,
        start_year: int = 1980,
        num_steps: int = 2,
        lsm_filename: str = None,
        geopotential_filename: str = None,
        use_latlon: bool = False,
        use_cos_zenith: bool = False,
        latlon_lower_bound: Tuple[float, float] = (-90, -180),
        patch_size: Union[Tuple[int, int], int, None] = None,
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
        self.stats_dir = Path(stats_dir) if stats_dir is not None else None
        self.channels = channels
        self.stride = stride
        self.dt = dt
        self.start_year = start_year
        self.num_steps = num_steps
        self.lsm_filename = lsm_filename
        self.geopotential_filename = geopotential_filename
        if use_cos_zenith:
            use_latlon = True
        self.use_latlon = use_latlon
        self.use_cos_zenith = use_cos_zenith
        self.latlon_lower_bound = latlon_lower_bound
        self.process_rank = process_rank
        self.world_size = world_size
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.num_samples_per_year = num_samples_per_year

        # Determine outputs of pipeline
        self.pipe_outputs = ["state_seq", "timestamps"]
        if self.lsm_filename is not None:
            self.pipe_outputs.append("land_sea_mask")
        if self.geopotential_filename is not None:
            self.pipe_outputs.append("geopotential")
        if self.use_latlon:
            self.pipe_outputs.append("latlon")
            self.pipe_outputs.append("cos_latlon")
        if self.use_cos_zenith:
            self.pipe_outputs.append("cos_zenith")

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
        if self.stats_dir is not None and not self.stats_dir.is_dir():
            raise IOError(f"Error, stats directory {self.stats_dir} does not exist")
        if self.stats_dir is None:
            self.logger.warning(
                "Warning, no stats directory specified, this will result in no normalisation"
            )

        # Load all data files and statistics
        self._parse_dataset_files()
        self._load_statistics()
        if self.lsm_filename is not None:
            self._load_land_sea_mask()
        if self.geopotential_filename is not None:
            self._load_geopotential()
        if self.use_latlon:
            self._load_latlon()

        # Create pipeline
        self.pipe = self._create_pipeline()

    def _parse_dataset_files(self) -> None:
        """Parses the data directory for valid HDF5 files and determines training samples

        Raises
        ------
        ValueError
            In channels specified or number of samples per year is not valid
        """
        # get all input data files
        self.data_paths = sorted(self.data_dir.glob("*.h5"))
        for data_path in self.data_paths:
            self.logger.info(f"Climate file found: {data_path}")
        self.n_years = len(self.data_paths)
        self.logger.info(f"Number of years: {self.n_years}")

        # get total number of examples and image shape from the first file,
        # assuming other files have exactly the same format.
        self.logger.info(f"Getting file stats from {self.data_paths[0]}")
        with h5py.File(self.data_paths[0], "r") as f:
            # truncate the dataset to avoid out-of-range sampling
            data_samples_per_year = (
                f["fields"].shape[0] - (self.num_steps - 1) * self.stride
            )
            self.data_shape = f["fields"].shape[2:]

            # If channels not provided, use all of them
            if self.channels is None:
                self.channels = [i for i in range(f["fields"].shape[1])]

            # If num_samples_per_year use all
            if self.num_samples_per_year is None:
                self.num_samples_per_year = data_samples_per_year

            # Adjust image shape if patch_size defined
            if self.patch_size is not None:
                self.cropped_data_shape = [
                    s - s % self.patch_size[i] for i, s in enumerate(self.data_shape)
                ]
            else:
                self.cropped_data_shape = self.data_shape
            self.logger.info(f"Input data shape: {self.cropped_data_shape}")

            # Get total length
            self.total_length = self.n_years * self.num_samples_per_year

            # Sanity checks
            if max(self.channels) >= f["fields"].shape[1]:
                raise ValueError(
                    f"Provided channel has indexes greater than the number \
                of fields {f['fields'].shape[1]}"
                )

            if self.num_samples_per_year > data_samples_per_year:
                raise ValueError(
                    f"num_samples_per_year ({self.num_samples_per_year}) > number of \
                    samples available ({data_samples_per_year})!"
                )

            self.logger.info(f"Number of samples/year: {self.num_samples_per_year}")
            self.logger.info(f"Number of channels available: {f['fields'].shape[1]}")

    def _load_statistics(self) -> None:
        """Loads climate statistics from pre-computed numpy files

        The statistic files should be of name global_means.npy and global_std.npy with
        a shape of [1, C, 1, 1] located in the stat_dir.

        Raises
        ------
        IOError
            If mean or std numpy files are not found
        AssertionError
            If loaded numpy arrays are not of correct size
        """
        # If no stats dir we just skip loading the stats
        if self.stats_dir is None:
            self.mu = None
            self.std = None
            return
        # load normalisation values
        mean_stat_file = self.stats_dir / Path("global_means.npy")
        std_stat_file = self.stats_dir / Path("global_stds.npy")

        if not mean_stat_file.exists():
            raise IOError(f"Mean statistics file {mean_stat_file} not found")
        if not std_stat_file.exists():
            raise IOError(f"Std statistics file {std_stat_file} not found")

        # has shape [1, C, 1, 1]
        self.mu = np.load(str(mean_stat_file))[:, self.channels]
        # has shape [1, C, 1, 1]
        self.sd = np.load(str(std_stat_file))[:, self.channels]

        if not self.mu.shape == self.sd.shape == (1, len(self.channels), 1, 1):
            raise AssertionError("Error, normalisation arrays have wrong shape")

    def _load_land_sea_mask(self) -> None:
        """Load land-sea mask from netCDF file."""
        ds = nc.Dataset(self.lsm_filename)
        lsm = np.array(ds["lsm"]).astype(np.float32)
        lsm = np.flip(
            lsm, axis=1
        )  # flip latitude axis, TODO hacky fix and we should get this from the file
        if lsm.shape[1:] != self.data_shape:
            raise AssertionError(
                "Land-sea mask shape {lsm.shape} does not match data shape {self.data_shape}"
            )
        lsm = lsm[:, : self.cropped_data_shape[0], : self.cropped_data_shape[1]]
        self.lsm = dali.types.Constant(lsm)

    def _load_geopotential(self, normalize: bool = True) -> None:
        """Get geopotential from netCDF file."""
        ds = nc.Dataset(self.geopotential_filename)
        geop = np.array(ds["z"]).astype(np.float32)
        geop = np.flip(
            geop, axis=1
        )  # flip latitude axis, TODO hacky fix and we should get this from the file
        if geop.shape[1:] != self.data_shape:
            raise AssertionError(
                f"Geopotential shape {geop.shape} does not match data shape {self.data_shape}"
            )
        geop = geop[:, : self.cropped_data_shape[0], : self.cropped_data_shape[1]]
        if normalize:
            geop = (geop - geop.mean()) / geop.std()
        self.geopotential = dali.types.Constant(geop)

    def _load_latlon(self) -> None:
        """Load latitude and longitude coordinates from data shape and compute cos/sin versions."""

        # get latitudes and longitudes from data shape
        lat = np.linspace(
            self.latlon_lower_bound[0],
            self.latlon_lower_bound[0] + 180,
            self.cropped_data_shape[0],
        ).astype(np.float32)
        lon = np.linspace(
            self.latlon_lower_bound[1],
            self.latlon_lower_bound[1] + 360,
            self.cropped_data_shape[1] + 1,
        ).astype(np.float32)[1:]
        lat, lon = np.meshgrid(lat, lon, indexing="ij")
        self.latlon = dali.types.Constant(np.stack((lat, lon), axis=0))

        # cos/sin latitudes and longitudes
        cos_lat = np.cos(np.deg2rad(lat))
        sin_lon = np.sin(np.deg2rad(lon))
        cos_lon = np.cos(np.deg2rad(lon))
        self.cos_latlon = dali.types.Constant(
            np.stack((cos_lat, sin_lon, cos_lon), axis=0)
        )

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
            # HDF5 source
            source = ClimateDaliExternalSource(
                data_paths=self.data_paths,
                num_samples=self.total_length,
                channels=self.channels,
                stride=self.stride,
                dt=self.dt,
                start_year=self.start_year,
                num_steps=self.num_steps,
                num_samples_per_year=self.num_samples_per_year,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                process_rank=self.process_rank,
                world_size=self.world_size,
            )

            # Update length of dataset
            self.total_length = len(source) // self.batch_size

            # Read current batch
            state_seq, timestamps = dali.fn.external_source(
                source,
                num_outputs=2,
                parallel=True,
                batch=False,
            )

            # Crop
            h, w = self.cropped_data_shape
            state_seq = state_seq[:, :, :h, :w]

            # Normalize
            if self.stats_dir is not None:
                state_seq = dali.fn.normalize(state_seq, mean=self.mu, stddev=self.sd)

            # Make output list
            outputs = [state_seq, timestamps]

            # Get static inputs
            if self.lsm_filename is not None:
                outputs.append(self.lsm)
            if self.geopotential_filename is not None:
                outputs.append(self.geopotential)
            if self.use_latlon:
                outputs.append(self.latlon)
                outputs.append(self.cos_latlon)

            # Get cosine zenith angle
            if self.use_cos_zenith:
                cos_zenith = cos_zenith_angle(timestamps, latlon=self.latlon)
                outputs.append(cos_zenith)

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
    """DALI Source for lazy-loading the HDF5 climate files

    Parameters
    ----------
    data_paths : Iterable[str]
        Directory where climate data is stored
    num_samples : int
        Total number of training samples
    channels : Iterable[int]
        List representing which climate variables to load
    stride : int
        Number of steps between input and output variables
    num_steps : int
        Number of timesteps to load
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
        data_paths: Iterable[str],
        num_samples: int,
        channels: Iterable[int],
        num_steps: int,
        stride: int,
        dt: float,
        start_year: int,
        num_samples_per_year: int,
        batch_size: int = 1,
        shuffle: bool = True,
        process_rank: int = 0,
        world_size: int = 1,
    ):
        self.data_paths = list(data_paths)
        # Will be populated later once each worker starts running in its own process.
        self.data_files = None
        self.num_samples = num_samples
        self.chans = list(channels)
        self.num_steps = num_steps
        self.stride = stride
        self.dt = dt
        self.start_year = start_year
        self.num_samples_per_year = num_samples_per_year
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.last_epoch = None

        self.indices = np.arange(num_samples)
        # Shard from indices if running in parallel
        self.indices = np.array_split(self.indices, world_size)[process_rank]

        # Get number of full batches, ignore possible last incomplete batch for now.
        # Also, DALI external source does not support incomplete batches in parallel mode.
        self.num_batches = len(self.indices) // self.batch_size

    def __call__(
        self, sample_info: dali.types.SampleInfo
    ) -> Tuple[Tensor, Tensor, np.ndarray, np.ndarray, np.ndarray]:
        if sample_info.iteration >= self.num_batches:
            raise StopIteration()

        if self.data_files is None:
            # This will be called once per worker. Workers are persistent,
            # so there is no need to explicitly close the files - this will be done
            # when corresponding pipeline/dataset is destroyed
            self.data_files = [h5py.File(path, "r") for path in self.data_paths]

        # Shuffle before the next epoch starts
        if self.shuffle and sample_info.epoch_idx != self.last_epoch:
            # All workers use the same rng seed so the resulting
            # indices are the same across workers
            np.random.default_rng(seed=sample_info.epoch_idx).shuffle(self.indices)
            self.last_epoch = sample_info.epoch_idx

        # Get local indices from global index
        # TODO: This is very hacky, but it works for now
        idx = self.indices[sample_info.idx_in_epoch]
        year_idx = idx // self.num_samples_per_year
        in_idx = idx % self.num_samples_per_year

        # Get data for the current year
        data = self.data_files[year_idx]["fields"]

        # Load sequence of input variables
        state_seq = np.empty(
            (self.num_steps, len(self.chans)) + data.shape[2:], dtype=data.dtype
        )
        for i in range(self.num_steps):
            ind = in_idx + i * self.stride
            state_seq[i] = data[ind, self.chans]

        # Load sequence of timestamps
        year = self.start_year + year_idx
        timestamps = np.array(
            [
                (
                    datetime(year, 1, 1)
                    + timedelta(hours=int(in_idx) * self.dt)
                    + timedelta(hours=i * self.stride * self.dt)
                ).timestamp()
                for i in range(self.num_steps)
            ]
        ).astype(np.float32)

        return state_seq, timestamps

    def __len__(self):
        return len(self.indices)
