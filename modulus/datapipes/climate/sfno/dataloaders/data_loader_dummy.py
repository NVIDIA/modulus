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

import glob
import logging

import h5py
import numpy as np
import torch

# distributed stuff
from modulus.utils.sfno.distributed import comm


class DummyLoader(object):
    """
    Dummy data loader for testing purposes
    """

    def __init__(self, params, location, train, device):  # pragma: no cover
        self.params = params
        self.location = location
        self.train = train
        self.dt = params.dt
        self.batch_size = int(params.batch_size)
        self.n_history = params.n_history
        self.n_future = params.n_future if train else params.valid_autoreg_steps
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels
        self.n_in_channels = len(self.in_channels)
        self.n_out_channels = len(self.out_channels)
        self.roll = params.roll
        self.device = device
        self.io_grid = params.io_grid[1:]
        self.io_rank = params.io_rank[1:]

        if train:
            self.n_samples_per_epoch = (
                params.n_train_samples_per_epoch
                if hasattr(params, "n_train_samples_per_epoch")
                else None
            )
        else:
            self.n_samples_per_epoch = (
                params.n_eval_samples_per_epoch
                if hasattr(params, "n_eval_samples_per_epoch")
                else None
            )

        # get cropping:
        self.img_crop_shape_x = (
            params.crop_size_x if hasattr(params, "crop_size_x") else None
        )
        self.img_crop_shape_y = (
            params.crop_size_y if hasattr(params, "crop_size_y") else None
        )
        self.img_crop_offset_x = (
            params.crop_anchor_x if hasattr(params, "crop_anchor_x") else 0
        )
        self.img_crop_offset_y = (
            params.crop_anchor_y if hasattr(params, "crop_anchor_y") else 0
        )

        self._get_files_stats()

        # zenith angle yes or no?
        self.add_zenith = self.params.add_zenith
        if self.add_zenith:
            self.zen_dummy = torch.zeros(
                (
                    self.batch_size,
                    self.n_history + 1,
                    1,
                    self.img_local_shape_x,
                    self.img_local_shape_y,
                ),
                dtype=torch.float32,
                device=self.device,
            )

    def _get_files_stats(self):  # pragma: no cover
        self.files_paths = glob.glob(self.location + "/*.h5")

        if not self.files_paths:
            logging.info(
                "No input files found, specifying dataset properties from parameter inputs"
            )
            self.n_years = self.params.n_years
            self.n_samples_per_year = self.params.n_samples_per_year
            self.img_shape_x = self.params.img_shape_x
            self.img_shape_y = self.params.img_shape_y

        else:
            self.files_paths.sort()
            self.n_years = len(self.files_paths)
            with h5py.File(self.files_paths[0], "r") as _f:
                logging.info("Getting file stats from {}".format(self.files_paths[0]))
                self.n_samples_per_year = _f["fields"].shape[0]
                # original image shape (before padding)
                self.img_shape_x = _f["fields"].shape[2]
                self.img_shape_y = _f["fields"].shape[3]

        # determine local read size:
        # sanitize the crops first
        if self.img_crop_shape_x is None:
            self.img_crop_shape_x = self.img_shape_x
        if self.img_crop_shape_y is None:
            self.img_crop_shape_y = self.img_shape_y
        if self.img_crop_offset_x + self.img_crop_shape_x > self.img_shape_x:
            raise AssertionError
        if self.img_crop_offset_y + self.img_crop_shape_y > self.img_shape_y:
            raise AssertionError

        # for x
        read_shape_x = (self.img_crop_shape_x + self.io_grid[0] - 1) // self.io_grid[0]
        read_anchor_x = self.img_crop_offset_x + read_shape_x * self.io_rank[0]
        read_shape_x = min(read_shape_x, self.img_shape_x - read_anchor_x)
        # for y
        read_shape_y = (self.img_crop_shape_y + self.io_grid[1] - 1) // self.io_grid[1]
        read_anchor_y = self.img_crop_offset_y + read_shape_y * self.io_rank[1]
        read_shape_y = min(read_shape_y, self.img_shape_y - read_anchor_y)

        # compute padding
        self.img_local_pad_x = (
            self.img_crop_shape_x + self.io_grid[0] - 1
        ) // self.io_grid[0] - read_shape_x
        self.img_local_pad_y = (
            self.img_crop_shape_y + self.io_grid[1] - 1
        ) // self.io_grid[1] - read_shape_y

        self.img_local_offset_x = read_anchor_x
        self.img_local_offset_y = read_anchor_y
        self.img_local_shape_x = read_shape_x + self.img_local_pad_x
        self.img_local_shape_y = read_shape_y + self.img_local_pad_y

        # sharding
        self.n_samples_total = (
            self.n_samples_per_epoch
            if self.n_samples_per_epoch is not None
            else self.n_years * self.n_samples_per_year
        )
        self.n_samples_shard = self.n_samples_total // comm.get_size("data")

        # channels
        self.n_in_channels_local = self.n_in_channels
        self.n_out_channels_local = self.n_out_channels

        self.files = [None for _ in range(self.n_years)]
        logging.info("Number of samples per year: {}".format(self.n_samples_per_year))
        logging.info(
            "Found data at path {}. Number of examples: {}. Image Shape: {} x {} x {}".format(
                self.location,
                self.n_samples_total,
                self.img_shape_x,
                self.img_shape_y,
                self.n_in_channels_local,
            )
        )
        logging.info(
            "Including {} hours of past history in training at a frequency of {} hours".format(
                6 * self.dt * self.n_history, 6 * self.dt
            )
        )
        logging.info("WARNING: using dummy data")

        # create tensors for dummy data
        self.device = torch.device(f"cuda:{comm.get_local_rank()}")
        self.inp = torch.zeros(
            (
                self.batch_size,
                self.n_history + 1,
                self.n_in_channels,
                self.img_local_shape_x,
                self.img_local_shape_y,
            ),
            dtype=torch.float32,
            device=self.device,
        )
        self.tar = torch.zeros(
            (
                self.batch_size,
                self.n_future + 1,
                self.n_out_channels_local,
                self.img_local_shape_x,
                self.img_local_shape_y,
            ),
            dtype=torch.float32,
            device=self.device,
        )

        # initialize output
        self.inp.uniform_()
        self.tar.uniform_()

        self.in_bias = np.zeros((1, self.n_in_channels, 1, 1)).astype(np.float32)
        self.in_scale = np.ones((1, self.n_in_channels, 1, 1)).astype(np.float32)
        self.out_bias = np.zeros((1, self.n_out_channels_local, 1, 1)).astype(
            np.float32
        )
        self.out_scale = np.ones((1, self.n_out_channels_local, 1, 1)).astype(
            np.float32
        )

    def get_input_normalization(self):  # pragma: no cover
        """Returns the input normalization parameters"""
        return self.in_bias, self.in_scale

    def get_output_normalization(self):  # pragma: no cover
        """Returns the output normalization parameters"""
        return self.out_bias, self.out_scale

    def __len__(self):  # pragma: no cover
        return self.n_samples_shard

    def __iter__(self):  # pragma: no cover
        self.sample_idx = 0
        return self

    def __next__(self):  # pragma: no cover
        if self.sample_idx < self.n_samples_shard:
            self.sample_idx += 1

            if self.add_zenith:
                return self.inp, self.tar, self.zen_dummy, self.zen_dummy
            else:
                return self.inp, self.tar
        else:
            raise StopIteration()
