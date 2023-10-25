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

# we need this for the zenith angle feature
import datetime
import glob
import logging
import operator
import os
from bisect import bisect_right
from itertools import accumulate, groupby

import cupy as cp
import cupyx as cpx
import h5py
import numpy as np

# for nvtx annotation
import torch
import zarr

from modulus.utils.zenith_angle import cos_zenith_angle


class GeneralES(object):
    """Dali helper class"""

    def _get_slices(self, lst):  # pragma: no cover
        for a, b in groupby(enumerate(lst), lambda pair: pair[1] - pair[0]):
            b = list(b)
            yield slice(b[0][1], b[-1][1] + 1)

    # very important: the seed has to be constant across the workers, or otherwise mayhem:
    def __init__(
        self,
        location,
        max_samples,
        samples_per_epoch,
        train,
        batch_size,
        dt,
        n_history,
        n_future,
        in_channels,
        out_channels,
        crop_size,
        crop_anchor,
        num_shards,
        shard_id,
        io_grid,
        io_rank,
        device_id=0,
        truncate_old=True,
        enable_logging=True,
        zenith_angle=True,
        seed=333,
        is_parallel=True,
        host_prefetch_buffers=False,
        timestep_hours=6,
    ):  # pragma: no cover
        self.batch_size = batch_size
        self.location = location
        self.max_samples = max_samples
        self.n_samples_per_epoch = samples_per_epoch
        self.truncate_old = truncate_old
        self.train = train
        self.dt = dt
        self.n_history = n_history
        self.n_future = n_future
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_in_channels = len(in_channels)
        self.n_out_channels = len(out_channels)
        self.crop_size = crop_size
        self.crop_anchor = crop_anchor
        self.base_seed = seed
        self.num_shards = num_shards
        self.device_id = device_id
        self.shard_id = shard_id
        self.is_parallel = is_parallel
        self.host_prefetch_buffers = host_prefetch_buffers
        self.zenith_angle = zenith_angle
        self.timestep_hours = timestep_hours

        # set the read slices
        # we do not support channel parallelism yet
        if io_grid[0] != 1:
            raise AssertionError("Channel parallelism is not yet supported")
        self.io_grid = io_grid[1:]
        self.io_rank = io_rank[1:]

        # parse the files
        self._get_files_stats(enable_logging)
        self.shuffle = True if train else False

        # convert in_channels to list of slices:
        self.in_channels_slices = list(self._get_slices(self.in_channels))
        self.out_channels_slices = list(self._get_slices(self.out_channels))

        # we need some additional static fields in this case
        if self.zenith_angle:
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

        # these things we want to read from a descriptor file ultimately:
        self.dt_samples = 6

    # HDF5 routines
    def _get_stats_h5(self, enable_logging):  # pragma: no cover
        with h5py.File(self.files_paths[0], "r") as _f:
            if enable_logging:
                logging.info("Getting file stats from {}".format(self.files_paths[0]))
            # original image shape (before padding)
            self.img_shape = _f["fields"].shape[
                2:4
            ]  # - 1 #just get rid of one of the pixels
            self.total_channels = _f["fields"].shape[1]

        # get all sample counts
        self.n_samples_year = []
        for filename in self.files_paths:
            with h5py.File(filename, "r") as _f:
                self.n_samples_year.append(_f["fields"].shape[0])
        return

    def _get_year_h5(self, year_idx):  # pragma: no cover
        self.files[year_idx] = h5py.File(self.files_paths[year_idx], "r")
        self.dsets[year_idx] = self.files[year_idx]["fields"]
        return

    def _get_data_h5(
        self, inp, tar, dset, local_idx, start_x, end_x, start_y, end_y
    ):  # pragma: no cover
        off = 0
        for slice_in in self.in_channels_slices:
            start = off
            end = start + (slice_in.stop - slice_in.start)
            # inp[:, start:end, ...] = dset[(local_idx-self.dt*self.n_history):(local_idx+1):self.dt, slice_in, start_x:end_x, start_y:end_y]
            dset.read_direct(
                inp,
                np.s_[
                    (local_idx - self.dt * self.n_history) : (local_idx + 1) : self.dt,
                    slice_in,
                    start_x:end_x,
                    start_y:end_y,
                ],
                np.s_[:, start:end, ...],
            )
            off = end

        off = 0
        for slice_out in self.out_channels_slices:
            start = off
            end = start + (slice_out.stop - slice_out.start)
            # tar[:, start:end, ...] = dset[(local_idx + self.dt):(local_idx + self.dt * (self.n_future + 1) + 1):self.dt, slice_out, start_x:end_x, start_y:end_y]
            dset.read_direct(
                tar,
                np.s_[
                    (local_idx + self.dt) : (
                        local_idx + self.dt * (self.n_future + 1) + 1
                    ) : self.dt,
                    slice_out,
                    start_x:end_x,
                    start_y:end_y,
                ],
                np.s_[:, start:end, ...],
            )
            off = end

        return inp, tar

    # zarr functions
    def _get_stats_zarr(self, enable_logging):  # pragma: no cover
        with zarr.convenience.open(self.files_paths[0], "r") as _f:
            if enable_logging:
                logging.info("Getting file stats from {}".format(self.files_paths[0]))
            # original image shape (before padding)
            self.img_shape = _f["/fields"].shape[
                2:4
            ]  # - 1 #just get rid of one of the pixels
            self.total_channels = _f["/fields"].shape[1]

        self.n_samples_year = []
        for filename in self.files_paths:
            with zarr.convenience.open(filename, "r") as _f:
                self.n_samples_year.append(_f["/fields"].shape[0])

        return

    def _get_year_zarr(self, year_idx):  # pragma: no cover
        self.files[year_idx] = zarr.convenience.open(self.files_paths[year_idx], "r")
        self.dsets[year_idx] = self.files[year_idx]["/fields"]
        return

    def _get_data_zarr(
        self, inp, tar, dset, local_idx, start_x, end_x, start_y, end_y
    ):  # pragma: no cover
        off = 0
        for slice_in in self.in_channels_slices:
            start = off
            end = start + (slice_in.stop - slice_in.start)
            inp[:, start:end, ...] = dset[
                (local_idx - self.dt * self.n_history) : (local_idx + 1) : self.dt,
                slice_in,
                start_x:end_x,
                start_y:end_y,
            ]
            off = end

        off = 0
        for slice_out in self.out_channels_slices:
            start = off
            end = start + (slice_out.stop - slice_out.start)
            tar[:, start:end, ...] = dset[
                (local_idx + self.dt) : (
                    local_idx + self.dt * (self.n_future + 1) + 1
                ) : self.dt,
                slice_out,
                start_x:end_x,
                start_y:end_y,
            ]
            off = end

        return inp, tar

    def _get_files_stats(self, enable_logging):  # pragma: no cover
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

        # # TODO: probably requires fix to re-enable zarr
        # if not self.files_paths:
        #     self.files_paths = glob.glob(os.path.join(self.location, "*.zarr"))
        #     self.file_format = "zarr"

        if not self.files_paths:
            raise IOError(
                f"Error, the specified file path {self.location} does neither container h5 nor zarr files."
            )

        self.files_paths.sort()

        # extract the years from filenames
        self.years = [
            int(os.path.splitext(os.path.basename(x))[0][-4:]) for x in self.files_paths
        ]

        # get stats
        self.n_years = len(self.files_paths)

        # get stats from first file
        if self.file_format == "h5":
            self._get_stats_h5(enable_logging)
        else:
            self._get_stats_zarr(enable_logging)

        # determine local read size:
        # sanitize the crops first
        if self.crop_size[0] is None:
            self.crop_size[0] = self.img_shape[0]
        if self.crop_size[1] is None:
            self.crop_size[1] = self.img_shape[1]
        if self.crop_anchor[0] + self.crop_size[0] > self.img_shape[0]:
            raise AssertionError
        if self.crop_anchor[1] + self.crop_size[1] > self.img_shape[1]:
            raise AssertionError
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
        if self.truncate_old:
            self.n_samples_offset = self.n_samples_available - self.n_samples_total
        else:
            self.n_samples_offset = 0

        # number of steps per epoch
        self.num_steps_per_cycle = self.n_samples_shard // self.batch_size
        if self.n_samples_per_epoch is None:
            self.n_samples_per_epoch = self.n_samples_total
        self.num_steps_per_epoch = self.n_samples_per_epoch // (
            self.batch_size * self.num_shards
        )

        # we need those here
        self.num_samples_per_cycle_shard = self.num_steps_per_cycle * self.batch_size
        self.num_samples_per_epoch_shard = self.num_steps_per_epoch * self.batch_size
        # prepare file lists
        self.files = [None for _ in range(self.n_years)]
        self.dsets = [None for _ in range(self.n_years)]
        if enable_logging:
            logging.info(
                "Average number of samples per year: {:.1f}".format(
                    float(self.n_samples_total) / float(self.n_years)
                )
            )
            logging.info(
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
            logging.info(
                "Using {} from the total number of available samples with {} samples per epoch (corresponds to {} steps for {} shards with local batch size {})".format(
                    self.n_samples_total,
                    self.n_samples_per_epoch,
                    self.num_steps_per_epoch,
                    self.num_shards,
                    self.batch_size,
                )
            )
            logging.info("Delta t: {} hours".format(self.timestep_hours * self.dt))
            logging.info(
                "Including {} hours of past history in training at a frequency of {} hours".format(
                    self.timestep_hours * self.dt * self.n_history,
                    self.timestep_hours * self.dt,
                )
            )
            logging.info(
                "Including {} hours of future targets in training at a frequency of {} hours".format(
                    self.timestep_hours * self.dt * self.n_future,
                    self.timestep_hours * self.dt,
                )
            )

        # some state variables
        self.last_cycle_epoch = None
        self.index_permutation = None

        # prepare buffers for double buffering
        if not self.is_parallel:
            self._init_buffers()

    def _init_double_buff_host(self, n_tsteps, n_channels):  # pragma: no cover
        buffs = [
            np.zeros(
                (
                    n_tsteps,
                    n_channels,
                    self.read_shape[0],
                    self.read_shape[1],
                ),
                dtype=np.float32,
            ),
            np.zeros(
                (
                    n_tsteps,
                    n_channels,
                    self.read_shape[0],
                    self.read_shape[1],
                ),
                dtype=np.float32,
            ),
        ]
        return buffs

    def _init_double_buff_gpu(self, n_tsteps, n_channels):  # pragma: no cover
        buffs = [
            cpx.zeros_pinned(
                (
                    n_tsteps,
                    n_channels,
                    self.read_shape[0],
                    self.read_shape[1],
                ),
                dtype=np.float32,
            ),
            cpx.zeros_pinned(
                (
                    n_tsteps,
                    n_channels,
                    self.read_shape[0],
                    self.read_shape[1],
                ),
                dtype=np.float32,
            ),
        ]
        return buffs

    def _init_buffers(self):  # pragma: no cover
        # set device
        self.device = cp.cuda.Device(self.device_id)
        self.device.use()
        self.current_buffer = 0
        if self.host_prefetch_buffers:
            self.inp_buffs = self._init_double_buff_host(
                self.n_history + 1, self.n_in_channels
            )
            self.tar_buffs = self._init_double_buff_host(
                self.n_future + 1, self.n_out_channels
            )
        else:
            self.inp_buffs = self._init_double_buff_gpu(
                self.n_history + 1, self.n_in_channels
            )
            self.tar_buffs = self._init_double_buff_gpu(
                self.n_future + 1, self.n_out_channels
            )
        if self.zenith_angle:
            if self.host_prefetch_buffers:
                self.zen_inp_buffs = self._init_double_buff_host(self.n_history + 1, 1)
                self.zen_tar_buffs = self._init_double_buff_host(self.n_future + 1, 1)
            else:
                self.zen_inp_buffs = self._init_double_buff_gpu(self.n_history + 1, 1)
                self.zen_tar_buffs = self._init_double_buff_gpu(self.n_future + 1, 1)
        return

    def _compute_zenith_angle(
        self, zen_inp, zen_tar, local_idx, year_idx
    ):  # pragma: no cover
        """Computes solar zenith angle"""

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
        cos_zenith_inp = np.asarray(
            [
                np.expand_dims(
                    cos_zenith_angle(
                        inp_time, self.lon_grid_local, self.lat_grid_local
                    ).astype(np.float32),
                    axis=0,
                )
                for inp_time in inp_times
            ]
        )
        zen_inp[...] = cos_zenith_inp[...]

        # zenith angle for target
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
        cos_zenith_tar = np.asarray(
            [
                np.expand_dims(
                    cos_zenith_angle(
                        tar_time, self.lon_grid_local, self.lat_grid_local
                    ).astype(np.float32),
                    axis=0,
                )
                for tar_time in tar_times
            ]
        )
        zen_tar[...] = cos_zenith_tar[...]

        return

    def __getstate__(self):  # pragma: no cover
        return self.__dict__.copy()

    def __setstate__(self, state):  # pragma: no cover
        self.__dict__.update(state)

        if self.file_format == "h5":
            self.get_year_handle = self._get_year_h5
            self.get_data_handle = self._get_data_h5
        else:
            self.get_year_handle = self._get_year_zarr
            self.get_data_handle = self._get_data_zarr

        if self.is_parallel:
            self._init_buffers()

    def __len__(self):  # pragma: no cover
        return self.n_samples_shard

    def __del__(self):  # pragma: no cover
        for f in self.files:
            if f is not None:
                f.close()

    def __call__(self, sample_info):  # pragma: no cover
        # compute global iteration index:
        global_sample_idx = (
            sample_info.idx_in_epoch
            + sample_info.epoch_idx * self.num_samples_per_epoch_shard
        )
        cycle_sample_idx = global_sample_idx % self.num_samples_per_cycle_shard
        cycle_epoch_idx = global_sample_idx // self.num_samples_per_cycle_shard

        # print(f'{"TRAIN" if self.train else "VALIDATION"} ITER INFO:', sample_info.idx_in_epoch, self.num_samples_per_epoch_shard)

        # check if epoch is done
        if sample_info.iteration >= self.num_steps_per_epoch:
            # print(f'{"TRAIN" if self.train else "VALIDATION"} END OF EPOCH TRIGGERED FOR', sample_info.idx_in_epoch, self.num_samples_per_epoch_shard, sample_info.iteration, self.num_steps_per_epoch)
            raise StopIteration

        torch.cuda.nvtx.range_push("GeneralES:__call__")

        # shuffle the data and shard
        if cycle_epoch_idx != self.last_cycle_epoch:
            self.last_cycle_epoch = cycle_epoch_idx
            # generate a unique seed and permutation:
            rng = np.random.default_rng(seed=self.base_seed + cycle_epoch_idx)
            if self.shuffle:
                self.index_permutation = self.n_samples_offset + rng.permutation(
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
            self.get_year_handle(year_idx)

        # handles to buffers
        inp = self.inp_buffs[self.current_buffer]
        tar = self.tar_buffs[self.current_buffer]
        if self.zenith_angle:
            zen_inp = self.zen_inp_buffs[self.current_buffer]
            zen_tar = self.zen_tar_buffs[self.current_buffer]
        self.current_buffer = (self.current_buffer + 1) % 2

        # do the read
        dset = self.dsets[year_idx]

        # load slice of data:
        start_x = self.read_anchor[0]
        end_x = start_x + self.read_shape[0]

        start_y = self.read_anchor[1]
        end_y = start_y + self.read_shape[1]

        # read data
        inp, tar = self.get_data_handle(
            inp, tar, dset, local_idx, start_x, end_x, start_y, end_y
        )

        # get time grid
        if self.zenith_angle:
            self._compute_zenith_angle(zen_inp, zen_tar, local_idx, year_idx)
            result = inp, tar, zen_inp, zen_tar
        else:
            result = inp, tar

        torch.cuda.nvtx.range_pop()

        return result
