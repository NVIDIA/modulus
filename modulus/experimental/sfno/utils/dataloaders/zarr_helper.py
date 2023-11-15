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

from typing import Protocol
import xarray
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Any
from modulus.experimental.sfno.utils import comm
import numpy as np


class Params(Protocol):
    """A protocol with the required input parameters

    Useful for typechecking or editor autocompletion.
    """

    in_channels: Any
    out_channels: Any
    batch_size: int
    global_means_path: str
    global_stds_path: str


@dataclass
class Metadata:
    """Image metadata required to initialize the model"""

    img_shape_x: int
    img_shape_y: int
    in_channels: Any
    out_channels: Any

    img_crop_shape_x: int
    img_crop_shape_y: int
    img_crop_offset_x: int
    img_crop_offset_y: int
    img_local_shape_x: int
    img_local_shape_y: int
    img_local_offset_x: int
    img_local_offset_y: int


def get_data_loader(params: Params, files_pattern: str, train: bool):
    """Matches interface used in trainer.py:Trainer"""
    ds = xarray.open_zarr(files_pattern)
    dataset = _xarray_to_dataset(params, ds, train=train)

    # shape is (1, channel, 1, 1)
    mean = np.load(params.global_means_path)
    std = np.load(params.global_stds_path)

    assert mean.shape == (1, len(ds.channel), 1, 1), mean.shape
    assert not np.any(np.isnan(mean)), np.ravel(std)

    assert std.shape == (1, len(ds.channel), 1, 1), std.shape
    assert not np.any(np.isnan(std)), np.ravel(std)

    def reset_pipeline():
        pass

    def get_output_normalization():
        return mean[:, params.out_channels], std[:, params.out_channels]

    def get_input_normalization():
        return mean[:, params.in_channels], std[:, params.in_channels]

    def center(args):
        x, y = args

        xmean = mean[0, params.in_channels]
        xstd = std[0, params.in_channels]

        ymean = mean[0, params.out_channels]
        ystd = std[0, params.out_channels]

        return (x - xmean) / xstd, (y - ymean) / ystd

    dataset = Map(dataset, center)

    sampler = (
        DistributedSampler(
            dataset,
            shuffle=train,
            num_replicas=params.data_num_shards,
            rank=params.data_shard_id
        )
        if (params.data_num_shards > 1)
        else None
    )

    dataloader = DataLoader(
        dataset,
        batch_size=int(params.batch_size),
        num_workers=params.num_data_workers,
        shuffle=False,
        sampler=sampler if train else None,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )

    dataloader.get_output_normalization = get_output_normalization
    dataloader.get_input_normalization = get_input_normalization
    dataloader.reset_pipeline = reset_pipeline

    shape = ds.fields.shape
    nlon = shape[-1]
    nlat = shape[-2]

    metadata = Metadata(
        img_shape_y=nlon,
        img_shape_x=nlat,
        in_channels=params.in_channels,
        out_channels=params.out_channels,
        img_crop_shape_x=nlat,
        img_crop_shape_y=nlon,
        img_crop_offset_x=0,
        img_crop_offset_y=0,
        img_local_shape_x=nlat,
        img_local_shape_y=nlon,
        img_local_offset_x=0,
        img_local_offset_y=0,
    )

    if train:
        return dataloader, metadata, sampler
    else:
        return dataloader, metadata


def _xarray_to_dataset(params: Params, ds: xarray.Dataset, train: bool):

    year = ds.time.dt.year
    if train:
        mask = (year <= 2015) & (year >= 1980)
        ds = ds.sel(time=mask)
    else:
        mask = (2015 < year) & (year <= 2017)
        ds = ds.sel(time=mask)

    return XarrayDataset(ds.fields, params.in_channels, params.out_channels)


class Map(Dataset):
    def __init__(self, data, func):
        self.data = data
        self.func = func

    def __getitem__(self, i):
        return self.func(self.data[i])

    def __len__(self):
        return len(self.data)


@dataclass
class XarrayDataset(Dataset):
    data: xarray.DataArray
    in_channels: Any = slice(None)
    out_channels: Any = slice(None)

    def _to_array(self, x):
        return x.values

    def __getitem__(self, i):
        input_ = self.data.isel(time=i, channel=self.in_channels)
        target = self.data.isel(time=i + 1, channel=self.out_channels)
        x = self._to_array(input_)
        y = self._to_array(target)
        return x, y

    def __len__(self):
        times = self.data.time
        if len(times) > 1:
            return len(times) - 1
        else:
            return 0
