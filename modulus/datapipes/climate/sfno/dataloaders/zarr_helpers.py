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

import logging
import h5py


def get_stats_zarr(params, enable_logging):  # pragma: no cover
    """loads image shape and number of samples from a specified directory"""
    with zarr.convenience.open(params.files_paths[0], "r") as _f:
        if enable_logging:
            logging.info("Getting file stats from {}".format(params.files_paths[0]))
        # original image shape (before padding)
        params.img_shape = _f[f"/{params.dataset_path}"].shape[
            2:4
        ]  # - 1 #just get rid of one of the pixels
        params.total_channels = _f[f"/{params.dataset_path}"].shape[1]

    params.n_samples_year = []
    for filename in params.files_paths:
        with zarr.convenience.open(filename, "r") as _f:
            params.n_samples_year.append(_f[f"/{params.dataset_path}"].shape[0])

    return


def get_year_zarr(params, year_idx):  # pragma: no cover
    """Open a dataset that points to a specic year"""
    _file = zarr.convenience.open(params.files_paths[year_idx], "r")
    params.dsets[year_idx] = _file[f"/{params.dataset_path}"]
    return


def get_data_zarr(
    params, inp, tar, dset, local_idx, start_x, end_x, start_y, end_y
):  # pragma: no cover
    """Loads a set of images from a zarr file"""
    off = 0
    for slice_in in params.in_channels_slices:
        start = off
        end = start + (slice_in.stop - slice_in.start)
        inp[:, start:end, ...] = dset[
            (local_idx - params.dt * params.n_history) : (local_idx + 1) : params.dt,
            slice_in,
            start_x:end_x,
            start_y:end_y,
        ]
        off = end

    off = 0
    for slice_out in params.out_channels_slices:
        start = off
        end = start + (slice_out.stop - slice_out.start)
        tar[:, start:end, ...] = dset[
            (local_idx + params.dt) : (
                local_idx + params.dt * (params.n_future + 1) + 1
            ) : params.dt,
            slice_out,
            start_x:end_x,
            start_y:end_y,
        ]
        off = end

    return inp, tar
