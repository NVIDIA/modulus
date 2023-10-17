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

import sys
import os
import glob
import numpy as np
import cupy as cp
import cupyx as cpx
import h5py
import zarr
import logging
from itertools import groupby, accumulate
import operator
from bisect import bisect_right
from multiprocessing.pool import ThreadPool
from threading import Lock, Condition, Thread
from modulus.datapipes.datapipe import Datapipe
import modulus.datapipes.climate.sfno.dataloaders.h5_helpers as h5_helpers
import modulus.datapipes.climate.sfno.dataloaders.zarr_helpers as zarr_helpers
from modulus.datapipes.meta import DatapipeMetaData

from torch.utils.data import IterableDataset
import gc

# for nvtx annotation
import torch

# we need this for the zenith angle feature
import datetime

# @dataclass
class MetaData(DatapipeMetaData):
    name: str = "CachingDataset"
    # Optimization
    auto_device: bool = False
    cuda_graphs: bool = False
    # Parallel
    ddp_sharding: bool = True


def _get_slices(lst):
    for a, b in groupby(enumerate(lst), lambda pair: pair[1] - pair[0]):
        b = list(b)
        yield slice(b[0][1], b[-1][1] + 1)


class CachingDataset(Datapipe, IterableDataset):
    # very important: the seed has to be constant across the workers, or otherwise mayhem:
    def __init__(self, params, location: str, train: bool):

        IterableDataset.__init__(self)

        self.logger = logging.getLogger("core.datapipe.sfno.caching_datapipe")
        Datapipe.__init__(
            self,
            meta=MetaData(),
        )

        # set these to defaults for now, iter handles the rest
        self.worker_id = 0
        self.num_workers = 1
        self.batch_size: int = int(params.batch_size)  # number of samples per batch
        self.location: str = location  # location of data files
        self.train: bool = train  # if training samples should be randomly shuffled
        # maximum number of samples to use from full set
        if self.train:
            self.max_samples: int = (
                params.n_train_samples if hasattr(params, "n_train_samples") else None
            )
        else:
            self.max_samples: int = (
                params.n_eval_samples if hasattr(params, "n_eval_samples") else None
            )
        self.shuffle = train

        # The stride of timesteps, eg dt = 2 would use every 2nd timestep
        self.dt = params.dt if hasattr(params, "dt") else 1
        # number of timesteps in the past to fetch
        self.n_history = params.n_history
        # number of timesteps in the future to  fetch
        self.n_future = params.n_future if train else params.valid_autoreg_steps
        self.in_channels = params.in_channels  # which numerical input channels to load
        self.out_channels = (
            params.out_channels
        )  # which numerical output channels to load
        self.n_in_channels = len(self.in_channels)  # number of input channels
        self.n_out_channels = len(self.out_channels)  # number of output channels

        # get cropping:
        self.crop_size = [
            params.crop_size_x if hasattr(params, "crop_size_x") else None,
            params.crop_size_y if hasattr(params, "crop_size_y") else None,
        ]
        self.crop_anchor = [
            params.crop_anchor_x if hasattr(params, "crop_anchor_x") else 0,
            params.crop_anchor_y if hasattr(params, "crop_anchor_y") else 0,
        ]

        # seed used for shuffling, needs to be consistent across spatial ranks otherwise mayhem
        self.base_seed: int = params.seed if hasattr(params, "seed") else 333
        self.num_shards: int = params.data_num_shards  # number of data shards

        # cuda device setup
        self.device_id = torch.cuda.current_device()  # cuda device to use
        self.device = cp.cuda.Device(self.device_id)
        self.device.use()

        self.shard_id: int = params.data_shard_id  # shard id
        self.normalize: bool = False
        self.zenith_angle: bool = (
            params.add_zenith if hasattr(params, "add_zenith") else False
        )
        # The path of the dataset within the file, left as h5 for compatibility
        self.dataset_path: str = params.h5_path if hasattr(params, "h5_path") else False
        self.enable_logging: bool = params.log_to_screen
        # the number of hours between timesteps
        self.timestep_hours: int = (
            params.timestep_hours if hasattr(params, "timestep_hours") else 6
        )

        # cache config
        self.cache_num = params.cache_num  # number of samples in the cache
        self.eviction_rate = params.eviction_rate  # cache eviction rate

        # set the read slices
        # we do not support channel parallelism yet
        assert params.io_grid[0] == 1
        assert len(params.io_grid) == 3
        self.io_grid: list[int] = params.io_grid[1:]  # size of the spatial grid
        self.io_rank: lisy[int] = params.io_rank[1:]  # rank in the spatial grid

        # parse the files
        self._get_files_stats()

        # convert in_channels to list of slices:
        self.in_channels_slices = list(_get_slices(self.in_channels))
        self.out_channels_slices = list(_get_slices(self.out_channels))

        # startup the rng
        self.rng = np.random.default_rng(seed=self.base_seed)

        # get the stats for norms
        if params.get("normalization", None) == "minmax":
            self.normalize = True
            mins = np.load(params.min_path)
            maxes = np.load(params.max_path)
            # inputs
            self.in_bias = mins[:, self.in_channels]
            self.in_scale = maxes[:, self.in_channels] - mins[:, self.in_channels]
            # outputs
            self.out_bias = mins[:, self.out_channels]
            self.out_scale = maxes[:, self.out_channels] - mins[:, self.out_channels]
        elif params.get("normalization", None) == "zscore":
            self.normalize = True
            means = np.load(params.global_means_path)
            stds = np.load(params.global_stds_path)
            # inputs
            self.in_bias = means[:, self.in_channels]
            self.in_scale = stds[:, self.in_channels]
            # outputs
            self.out_bias = means[:, self.out_channels]
            self.out_scale = stds[:, self.out_channels]
        else:
            self.normalize = False
            # input
            self.in_bias = np.zeros((1, self.n_in_channels, 1, 1))
            self.in_scale = np.ones((1, self.n_in_channels, 1, 1))
            # outputs
            self.out_bias = np.zeros((1, self.n_out_channels, 1, 1))
            self.out_scale = np.ones((1, self.n_out_channels, 1, 1))

        # setup local lat/lon for calculating zenith angle
        if self.zenith_angle:
            if (
                hasattr(params, "lat")
                and hasattr(params, "lon")
                and params.lat
                and params.lon
            ):
                lattitude = params["lat"]
                longitude = params["lon"]
            else:
                longitude = np.linspace(0, 360, self.img_shape[1], endpoint=False)
                latitude = np.linspace(90, -90, self.img_shape[0])

            self.lon_grid, self.lat_grid = np.meshgrid(longitude, latitude)
            self.lat_grid_local = self.lat_grid[
                self.read_anchor[0] : self.read_anchor[0] + self.read_shape[0],
                self.read_anchor[1] : self.read_anchor[1] + self.read_shape[1],
            ]
            self.lon_grid_local = self.lon_grid[
                self.read_anchor[0] : self.read_anchor[0] + self.read_shape[0],
                self.read_anchor[1] : self.read_anchor[1] + self.read_shape[1],
            ]

        # prepare buffers
        self._init_buffers()

        if self.file_format == "h5":
            self.get_year_handle = h5_helpers.get_year_h5
            self.get_data_handle = h5_helpers.get_data_h5
        else:
            self.get_year_handle = zarr_helpers.get_year_zarr
            self.get_data_handle = zarr_helpers.get_data_zarr
        self.cache_update_lock = Lock()
        self.cache_wait_lock = Condition()
        self.prefill_lock = Lock()

    def _get_files_stats(self):
        # check for hdf5 files
        self.files_paths = []
        self.location = (
            [self.location] if not isinstance(self.location, list) else self.location
        )
        for location in self.location:
            self.files_paths = self.files_paths + glob.glob(
                os.path.join(location, "????.h5")
            )
        self.file_format = "h5"

        if not self.files_paths:
            raise IOError(
                f"Error, the specified file path {self.location} does neither container h5 nor zarr files."
            )

        self.files_paths.sort()

        # extract the years from filenames
        self.years = [
            int(os.path.splitext(os.path.basename(x))[0]) for x in self.files_paths
        ]

        # get stats
        self.n_years = len(self.files_paths)

        # get stats from first file
        if self.file_format == "h5":
            h5_helpers.get_stats_h5(self, self.enable_logging)
        else:
            zarr_helpers.get_stats_zarr(self, self.enable_logging)

        # determine local read size:
        # sanitize the crops first
        if self.crop_size[0] is None:
            self.crop_size[0] = self.img_shape[0]
        if self.crop_size[1] is None:
            self.crop_size[1] = self.img_shape[1]
        assert self.crop_anchor[0] + self.crop_size[0] <= self.img_shape[0]
        assert self.crop_anchor[1] + self.crop_size[1] <= self.img_shape[1]
        # for x
        read_shape_x = (self.crop_size[0] + self.io_grid[0] - 1) // self.io_grid[0]
        read_anchor_x = self.crop_anchor[0] + read_shape_x * self.io_rank[0]
        read_shape_x = min(read_shape_x, self.img_shape[0] - read_anchor_x)
        # for y
        read_shape_y = (self.crop_size[1] + self.io_grid[1] - 1) // self.io_grid[1]
        read_anchor_y = self.crop_anchor[1] + read_shape_y * self.io_rank[1]
        read_shape_y = min(read_shape_y, self.img_shape[1] - read_anchor_y)
        self.read_anchor = [read_anchor_x, read_anchor_y]
        self.read_shape = [read_shape_x, read_shape_y]

        # compute padding
        read_pad_x = (self.crop_size[0] + self.io_grid[0] - 1) // self.io_grid[
            0
        ] - read_shape_x
        read_pad_y = (self.crop_size[1] + self.io_grid[1] - 1) // self.io_grid[
            1
        ] - read_shape_y
        self.read_pad = [read_pad_x, read_pad_y]

        # do some sample indexing gymnastics
        self.year_offsets = list(accumulate(self.n_samples_year, operator.add))[:-1]
        self.year_offsets.insert(0, 0)
        self.n_samples_available = sum(self.n_samples_year)
        if self.max_samples is not None:
            self.n_samples_total = min(self.n_samples_available, self.max_samples)
        else:
            self.n_samples_total = self.n_samples_available

        # do the sharding
        self.n_samples_shard = self.n_samples_total // self.num_shards
        self.n_samples_offset = self.n_samples_available - self.n_samples_total

        # number of steps per epoch
        self.num_steps_per_cycle = self.n_samples_shard // self.batch_size
        self.num_steps_per_epoch = self.n_samples_total // (
            self.batch_size * self.num_shards
        )

        # we need those here
        self.num_samples_per_cycle_shard = self.num_steps_per_cycle * self.batch_size
        self.num_samples_per_epoch_shard = self.num_steps_per_epoch * self.batch_size
        # prepare file lists
        self.files = [None for _ in range(self.n_years)]
        self.dsets = [None for _ in range(self.n_years)]
        if self.enable_logging:
            self.logger.info(
                "Average number of samples per year: {:.1f}".format(
                    float(self.n_samples_total) / float(self.n_years)
                )
            )
            if self.zenith_angle:
                self.logger.info("Adding cosine zenith angle")
            self.logger.info(
                "Found data at path {}. Number of examples: {}. Full image Shape: {} x {} x {}. Read Shape: {} x {} x {}".format(
                    self.location,
                    self.n_samples_available,
                    self.img_shape[0],
                    self.img_shape[1],
                    self.total_channels,
                    self.read_shape[0],
                    self.read_shape[1],
                    self.n_in_channels,
                )
            )
            self.logger.info(
                "Using {} from the total number of available samples with {} samples per epoch (corresponds to {} steps for {} shards with local batch size {})".format(
                    self.n_samples_total,
                    self.n_samples_total,
                    self.num_steps_per_epoch,
                    self.num_shards,
                    self.batch_size,
                )
            )
            self.logger.info("Delta t: {} hours".format(self.timestep_hours * self.dt))
            self.logger.info(
                "Including {} hours of past history in training at a frequency of {} hours".format(
                    self.timestep_hours * self.dt * self.n_history,
                    self.timestep_hours * self.dt,
                )
            )
            self.logger.info(
                "Including {} hours of future targets in training at a frequency of {} hours".format(
                    self.timestep_hours * self.dt * self.n_future,
                    self.timestep_hours * self.dt,
                )
            )

        # some state variables
        self.last_cycle_epoch = None
        self.index_permutation = None

    def _init_buffers(self):
        # cache starts empty, this might change in the future
        self.inp_cache = [None for _ in range(self.cache_num)]
        self.tar_cache = [None for _ in range(self.cache_num)]

        if self.zenith_angle:
            # these are computed (relatively cheaply) on the fly so no need to store them
            self.zen_inp = cpx.zeros_pinned(
                (self.n_history + 1, 1, self.read_shape[0], self.read_shape[1]),
                dtype=np.float32,
            )
            self.zen_tar = cpx.zeros_pinned(
                (self.n_future + 1, 1, self.read_shape[0], self.read_shape[1]),
                dtype=np.float32,
            )

    def _compute_zenith_angle(self, zen_inp, zen_tar, local_idx, year_idx):

        # nvtx range
        torch.cuda.nvtx.range_push("GeneralES:_compute_zenith_angle")

        # import
        from utils.zenith_angle import cos_zenith_angle

        # compute hours into the year
        year = self.years[year_idx]
        jan_01_epoch = datetime.datetime(year, 1, 1, 0, 0, 0)

        # zenith angle for input
        inp_times = np.asarray(
            [
                jan_01_epoch + datetime.timedelta(hours=idx * self.timestep_hours)
                for idx in range(
                    local_idx - self.dt * self.n_history, local_idx + 1, self.dt
                )
            ]
        )
        cos_zenith_inp = np.expand_dims(
            cos_zenith_angle(
                inp_times, self.lon_grid_local, self.lat_grid_local
            ).astype(np.float32),
            axis=1,
        )
        zen_inp[...] = cos_zenith_inp[...]

        tar_times = np.asarray(
            [
                jan_01_epoch + datetime.timedelta(hours=idx * self.timestep_hours)
                for idx in range(
                    local_idx + self.dt,
                    local_idx + self.dt * (self.n_future + 1) + 1,
                    self.dt,
                )
            ]
        )
        cos_zenith_tar = np.expand_dims(
            cos_zenith_angle(
                tar_times, self.lon_grid_local, self.lat_grid_local
            ).astype(np.float32),
            axis=1,
        )
        zen_tar[...] = cos_zenith_tar[...]

        torch.cuda.nvtx.range_pop()

        return

    def __len__(self):
        return self.cache_num * self.num_workers

    def update_permutation(self):
        # updates for a new epoch
        self.replacement_idx = self.worker_id

        # update full epoch so clear the buffers
        self._init_buffers()

        # shuffle if requested
        if self.shuffle:
            # generate a unique seed and permutation:
            self.index_permutation = self.n_samples_offset + self.rng.permutation(
                self.n_samples_total
            )
        else:
            self.index_permutation = self.n_samples_offset + np.arange(
                self.n_samples_total
            )

        # shard the data
        start = self.n_samples_shard * self.shard_id
        end = start + self.n_samples_shard

        self.index_permutation = self.index_permutation[start:end]

        # everything's updated, start refilling the cache
        self.start_refill(0)

    def update_cache(self):
        """Evicts entries from the cache and starts refilling in the background"""
        self.sample_idx = 0
        num_prefill = 0

        # empty items from the cache
        with self.cache_update_lock:
            del self.inp_cache[: self.eviction_length]
            del self.tar_cache[: self.eviction_length]

            self.inp_cache.extend(self.replacements)
            self.tar_cache.extend(self.replacements)

        # run a gc first clear any space
        gc.collect()

        self.start_refill(self.cache_num - self.eviction_length - num_prefill)

    def refill_cache(self, start_idx):
        """Helper function that handles the actual refill"""
        torch.cuda.nvtx.range_push("GeneralES:_do_cache_refill")
        for idx in range(start_idx, self.cache_num):
            self.fill_cache_entry(idx)
        self.replace_done = True

        # clear any potential waiting consumers
        with self.cache_wait_lock:
            self.cache_wait_lock.notify_all()

        torch.cuda.nvtx.range_pop()

    def fill_cache_entry(self, cache_idx):
        """Helper function that handles the actual refill"""
        if self.inp_cache[cache_idx] is None:
            if self.zenith_angle:
                inp, tar, zen_inp, zen_tar = self._get_sample()
            else:
                inp, tar = self._get_sample()

            if self.zenith_angle:
                inp_entry = [inp, zen_inp]
                tar_entry = [tar, zen_tar]
            else:
                inp_entry = [inp]
                tar_entry = [tar]

            # lock the cache and update necessary entries
            with self.cache_update_lock:
                self.replacement_idx += self.num_workers
                self.inp_cache[cache_idx] = inp_entry
                self.tar_cache[cache_idx] = tar_entry

            # in case consumer is waiting on an update
            with self.cache_wait_lock:
                self.cache_wait_lock.notify_all()

    def start_refill(self, start_idx):
        """helper that spawns a thread to refill empty cache entries"""
        t = Thread(target=self.refill_cache, args=(start_idx,))
        t.start()

    def update_cache_pass(self):
        """Updates cache entries"""
        self.sample_idx = 0

        # if there are enough samples left evict and update with new
        # samples otherwise update perturbations
        if self.replacement_idx < self.n_samples_shard - (
            self.eviction_length * self.num_workers
        ):
            self.update_cache()
        else:
            self.update_permutation()

    def _get_sample(self):
        """reads the data from a global idx"""
        torch.cuda.nvtx.range_push("GeneralES:__get_sample")
        local_idx, year_idx = self._get_indices(self.replacement_idx)
        if self.enable_logging:
            print(f"local {local_idx} year {year_idx}")

        inp = np.zeros(
            (
                self.n_history + 1,
                self.n_in_channels,
                self.read_shape[0],
                self.read_shape[1],
            ),
            dtype=np.float32,
        )
        tar = np.zeros(
            (
                self.n_future + 1,
                self.n_out_channels,
                self.read_shape[0],
                self.read_shape[1],
            ),
            dtype=np.float32,
        )

        # do the read
        dset = self.dsets[year_idx]

        # load slice of data:
        start_x = self.read_anchor[0]
        end_x = start_x + self.read_shape[0]

        start_y = self.read_anchor[1]
        end_y = start_y + self.read_shape[1]

        # read data
        inp, tar = self.get_data_handle(
            self, inp, tar, dset, local_idx, start_x, end_x, start_y, end_y
        )

        # augments
        if (self.read_pad[0] > 0) or (self.read_pad[1] > 0):
            inp = np.pad(
                inp, [(0, 0), (0, 0), (0, self.read_pad[0]), (0, self.read_pad[1])]
            )
            tar = np.pad(
                tar, [(0, 0), (0, 0), (0, self.read_pad[0]), (0, self.read_pad[1])]
            )

        before = inp[0, 0, 0, 0]
        if self.normalize:
            inp = (inp - self.in_bias) / self.in_scale
            tar = (tar - self.out_bias) / self.out_scale
        after = inp[0, 0, 0, 0]

        if self.enable_logging:
            print(f"before {before} after {after}")

        result = inp, tar

        if self.zenith_angle:
            zen_inp = np.zeros(
                (self.n_history + 1, 1, self.read_shape[0], self.read_shape[1]),
                dtype=np.float32,
            )
            zen_tar = np.zeros(
                (self.n_future + 1, 1, self.read_shape[0], self.read_shape[1]),
                dtype=np.float32,
            )
            self._compute_zenith_angle(zen_inp, zen_tar, local_idx, year_idx)
            result = inp, tar, zen_inp, zen_tar

        result = tuple(torch.as_tensor(arr) for arr in result)

        torch.cuda.nvtx.range_pop()
        return result

    def _get_indices(self, global_sample_idx):
        """helper to get year and local indices"""

        # compute global iteration index:
        cycle_sample_idx = global_sample_idx % self.num_samples_per_cycle_shard

        # determine local and sample idx
        sample_idx = self.index_permutation[cycle_sample_idx]
        year_idx = (
            bisect_right(self.year_offsets, sample_idx) - 1
        )  # subtract 1 because we do 0-based indexing
        local_idx = sample_idx - self.year_offsets[year_idx]

        # if we are not at least self.dt*n_history timesteps into the prediction
        if local_idx < self.dt * self.n_history:
            local_idx += self.dt * self.n_history

        if local_idx >= (self.n_samples_year[year_idx] - self.dt * (self.n_future + 1)):
            local_idx = (
                self.n_samples_year[year_idx] - self.dt * (self.n_future + 1) - 1
            )

        if self.files[year_idx] is None:
            self.get_year_handle(self, year_idx)

        return local_idx, year_idx

    def __iter__(self):
        return self

    def worker_init(self, worker_id):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.worker_id = 0
            self.num_workers = 1
        else:
            self.worker_id = worker_info.id
            self.num_workers = worker_info.num_workers

        # caching info
        # we do this here because this is where worker info is available
        self.cache_num = min(self.cache_num, self.n_samples_shard) // self.num_workers
        self.eviction_length = int(self.cache_num * self.eviction_rate)
        self.replacements = [None for _ in range(self.eviction_length)]
        self.replacement_idx = self.worker_id
        self.sample_idx = 0

        if (
            self.num_steps_per_epoch < self.cache_num * self.num_workers
            and self.enable_logging
        ):
            self.logger.warning(
                f"Cache size of {self.cache_num * self.num_workers} is greater than number of available samples {self.num_steps_per_epoch}, clamping max size"
            )
            self.cache_num = self.num_steps_per_epoch
        if self.enable_logging:
            self.logger.info(
                f"Cache size {self.cache_num * self.num_workers} entries ({self.cache_num} per worker), evicting {self.eviction_length} per epoch"
            )

        self.update_permutation()

    def __next__(self):
        """Gets a sample from the cache"""
        # check if the samples in the cache have been used
        if self.sample_idx >= self.cache_num:
            self.update_cache_pass()
            raise StopIteration()

        torch.cuda.nvtx.range_push("GeneralES:__next__")

        # wait for this entry to be refilled if necessary
        while self.inp_cache[self.sample_idx] is None:
            # if so wait for the reader to complete
            with self.cache_wait_lock:
                self.cache_wait_lock.wait()

        # get data
        inp = self.inp_cache[self.sample_idx][0]
        tar = self.tar_cache[self.sample_idx][0]
        result = inp, tar
        if self.zenith_angle:
            zen_inp = self.inp_cache[self.sample_idx][1]
            zen_tar = self.tar_cache[self.sample_idx][1]
            result = inp, tar, zen_inp, zen_tar

        self.sample_idx += 1
        torch.cuda.nvtx.range_pop()
        return result

    def get_input_normalization(self):  # pragma: no cover
        """Returns the input normalization parameters"""
        return self.in_bias, self.in_scale

    def get_output_normalization(self):  # pragma: no cover
        """Returns the output normalization parameters"""
        return self.out_bias, self.out_scale
