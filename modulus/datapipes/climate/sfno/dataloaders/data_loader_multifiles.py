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
import random

import h5py
import numpy as np
from torch.utils.data import Dataset

# import cv2
from modulus.utils.sfno.img_utils import reshape_fields


class MultifilesDataset(Dataset):
    """
    Dataset class for loading data from multiple files
    """

    def __init__(self, params, location, train):  # pragma: no cover
        self.params = params
        self.location = location
        self.train = train
        self.dt = params.dt
        self.n_history = params.n_history
        self.in_channels = np.array(params.in_channels)
        self.out_channels = np.array(params.out_channels)
        self.n_in_channels = len(self.in_channels)
        self.n_out_channels = len(self.out_channels)
        self.crop_size_x = params.crop_size_x
        self.crop_size_y = params.crop_size_y
        self.roll = params.roll
        self._get_files_stats()

    def _get_files_stats(self):  # pragma: no cover
        self.files_paths = glob.glob(self.location + "/*.h5")
        self.files_paths.sort()
        self.n_years = len(self.files_paths)
        with h5py.File(self.files_paths[0], "r") as _f:
            logging.info("Getting file stats from {}".format(self.files_paths[0]))
            self.n_samples_per_year = _f["fields"].shape[0]
            # original image shape (before padding)
            self.img_shape_x = _f["fields"].shape[2]
            self.img_shape_y = _f["fields"].shape[3]
            self.img_crop_shape_x = self.img_shape_x
            self.img_crop_shape_y = self.img_shape_y

        # set these for compatibility with the distributed dataloader. Doesn't support distributed mode as of now
        self.img_local_offset_x = 0
        self.img_local_offset_y = 0
        self.img_local_shape_x = self.img_shape_x
        self.img_local_shape_y = self.img_shape_y

        self.n_samples_total = self.n_years * self.n_samples_per_year
        self.files = [None for _ in range(self.n_years)]
        logging.info("Number of samples per year: {}".format(self.n_samples_per_year))
        logging.info(
            "Found data at path {}. Number of examples: {}. Image Shape: {} x {} x {}".format(
                self.location,
                self.n_samples_total,
                self.img_shape_x,
                self.img_shape_y,
                self.n_in_channels,
            )
        )
        logging.info("Delta t: {} hours".format(6 * self.dt))
        logging.info(
            "Including {} hours of past history in training at a frequency of {} hours".format(
                6 * self.dt * self.n_history, 6 * self.dt
            )
        )

    def _open_file(self, year_idx):  # pragma: no cover
        _file = h5py.File(self.files_paths[year_idx], "r")
        self.files[year_idx] = _file["fields"]

    def __len__(self):  # pragma: no cover
        return self.n_samples_total

    def __getitem__(self, global_idx):  # pragma: no cover
        year_idx = int(global_idx / self.n_samples_per_year)  # which year we are on
        local_idx = int(
            global_idx % self.n_samples_per_year
        )  # which sample in that year we are on - determines indices for centering

        # open image file
        if self.files[year_idx] is None:
            self._open_file(year_idx)

        # if we are not at least self.dt*n_history timesteps into the prediction
        if local_idx < self.dt * self.n_history:
            local_idx += self.dt * self.n_history

        # if we are on the last image in a year predict identity, else predict next timestep
        step = 0 if local_idx >= self.n_samples_per_year - self.dt else self.dt

        if self.train and self.roll:
            y_roll = random.randint(0, self.img_shape_y)
        else:
            y_roll = 0

        if self.train and (self.crop_size_x or self.crop_size_y):
            rnd_x = random.randint(0, self.img_shape_x - self.crop_size_x)
            rnd_y = random.randint(0, self.img_shape_y - self.crop_size_y)
        else:
            rnd_x = 0
            rnd_y = 0

        return reshape_fields(
            self.files[year_idx][
                (local_idx - self.dt * self.n_history) : (local_idx + 1) : self.dt,
                self.in_channels,
            ],
            "inp",
            self.crop_size_x,
            self.crop_size_y,
            rnd_x,
            rnd_y,
            self.params,
            y_roll,
            self.train,
        ), reshape_fields(
            self.files[year_idx][local_idx + step, self.out_channels],
            "tar",
            self.crop_size_x,
            self.crop_size_y,
            rnd_x,
            rnd_y,
            self.params,
            y_roll,
            self.train,
        )
