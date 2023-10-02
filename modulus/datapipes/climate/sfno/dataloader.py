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

import math
import types

import torch

# distributed stuff
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from modulus.utils.sfno.distributed import comm


def init_distributed_io(params):  # pragma: no cover
    """
    Initialize distributed IO
    """
    # set up sharding
    if dist.is_initialized():
        # this should always be safe now that data comm is orthogonal to
        # model comms
        params.data_num_shards = comm.get_size("data")
        params.data_shard_id = comm.get_rank("data")
        params.io_grid = [1, comm.get_size("h"), comm.get_size("w")]
        params.io_rank = [0, comm.get_rank("h"), comm.get_rank("w")]
    else:
        params.data_num_shards = 1
        params.data_shard_id = 0
        params.io_grid = [1, 1, 1]
        params.io_rank = [0, 0, 0]
        return

    # define IO grid:
    params.io_grid = [1, 1, 1] if not hasattr(params, "io_grid") else params.io_grid

    # to simplify, the number of total IO ranks has to be 1 or equal to the model parallel size
    num_io_ranks = math.prod(params.io_grid)
    if not ((num_io_ranks == 1) or (num_io_ranks == comm.get_size("spatial"))):
        raise AssertionError
    if not (params.io_grid[1] == comm.get_size("h")) or (params.io_grid[1] == 1):
        raise AssertionError
    if not (params.io_grid[2] == comm.get_size("w")) or (params.io_grid[2] == 1):
        raise AssertionError

    params.io_rank = [0, 0, 0]
    if params.io_grid[1] > 1:
        params.io_rank[1] = comm.get_rank("h")
    if params.io_grid[2] > 1:
        params.io_rank[2] = comm.get_rank("w")

    return


def get_dataloader(
    params, files_pattern, device, train=True, final_eval=False
):  # pragma: no cover
    """
    Get the dataloader
    """
    init_distributed_io(params)

    if params.get("data_type", "not zarr") == "zarr":
        from utils.dataloaders import zarr_helper as zarr

        return zarr.get_data_loader(params, files_pattern, train)

    elif params.get("multifiles", False):
        from utils.dataloaders.data_loader_multifiles import (
            MultifilesDataset as MultifilesDataset2D,
        )

        # multifiles dataset
        dataset = MultifilesDataset2D(params, files_pattern, train)

        sampler = (
            DistributedSampler(
                dataset,
                shuffle=train,
                num_replicas=params.data_num_shards,
                rank=params.data_shard_id,
            )
            if (params.data_num_shards > 1)
            else None
        )

        dataloader = DataLoader(
            dataset,
            batch_size=int(params.batch_size),
            num_workers=params.num_data_workers,
            shuffle=False,  # (sampler is None),
            sampler=sampler if train else None,
            drop_last=True,
            pin_memory=torch.cuda.is_available(),
        )

    elif params.enable_synthetic_data:
        from modulus.datapipes.climate.sfno.dataloaders.data_loader_dummy import (
            DummyLoader,
        )

        dataloader = DummyLoader(params, files_pattern, train, device)

        dataset = types.SimpleNamespace(
            in_channels=dataloader.in_channels,
            out_channels=dataloader.out_channels,
            img_shape_x=dataloader.img_shape_x,
            img_shape_y=dataloader.img_shape_y,
            img_crop_shape_x=dataloader.img_crop_shape_x,
            img_crop_shape_y=dataloader.img_crop_shape_y,
            img_crop_offset_x=dataloader.img_crop_offset_x,
            img_crop_offset_y=dataloader.img_crop_offset_y,
            img_local_shape_x=dataloader.img_local_shape_x,
            img_local_shape_y=dataloader.img_local_shape_y,
            img_local_offset_x=dataloader.img_local_offset_x,
            img_local_offset_y=dataloader.img_local_offset_y,
        )

        # not needed for the no multifiles case
        sampler = None

    else:
        from modulus.datapipes.climate.sfno.dataloaders.data_loader_dali_2d import (
            ERA5DaliESDataloader as ERA5DaliESDataloader2D,
        )

        dataloader = ERA5DaliESDataloader2D(
            params, files_pattern, train, final_eval=final_eval
        )

        dataset = types.SimpleNamespace(
            in_channels=dataloader.in_channels,
            out_channels=dataloader.out_channels,
            img_shape_x=dataloader.img_shape_x,
            img_shape_y=dataloader.img_shape_y,
            img_crop_shape_x=dataloader.img_crop_shape_x,
            img_crop_shape_y=dataloader.img_crop_shape_y,
            img_crop_offset_x=dataloader.img_crop_offset_x,
            img_crop_offset_y=dataloader.img_crop_offset_y,
            img_local_shape_x=dataloader.img_local_shape_x,
            img_local_shape_y=dataloader.img_local_shape_y,
            img_local_offset_x=dataloader.img_local_offset_x,
            img_local_offset_y=dataloader.img_local_offset_y,
        )

        if params.enable_benchy and train:
            from benchy.torch import BenchmarkGenericIteratorWrapper

            dataloader = BenchmarkGenericIteratorWrapper(dataloader, params.batch_size)

        # not needed for the no multifiles case
        sampler = None

    if train:
        return dataloader, dataset, sampler
    else:
        return dataloader, dataset
