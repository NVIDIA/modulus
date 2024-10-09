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

from typing import Tuple

import pytest
import torch
from pytest_utils import import_or_fail

from . import common

Tensor = torch.Tensor


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_darcy_2d_fix_constructor(device, pytestconfig):

    from modulus.datapipes.benchmarks.darcy_fix import Darcy2D_fix

    # construct data pipe
    datapipe = Darcy2D_fix(
        resolution=85,
        batch_size=4,
        normaliser=None,
        train_path="/data/fno/piececonst_r421_N1024_smooth1.mat",
        is_test=False,
    )

    # iterate datapipe is iterable
    assert common.check_datapipe_iterable(datapipe)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_darcy_2d_fix_device(device, pytestconfig):

    from modulus.datapipes.benchmarks.darcy_fix import Darcy2D_fix

    # construct data pipe
    datapipe = Darcy2D_fix(
        resolution=85,
        batch_size=4,
        normaliser=None,
        train_path="/data/fno/piececonst_r421_N1024_smooth1.mat",
        is_test=False,
        device=device,
    )

    # iterate datapipe is iterable
    for data in datapipe:
        for p in data:
            assert common.check_datapipe_device(p, device)
        break


@pytest.mark.parametrize(
    "resolution",
    [
        85,
    ],
)
@pytest.mark.parametrize(
    "batch_size",
    [
        1,
    ],
)
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_darcy_2d_fix_shape(resolution, batch_size, device, pytestconfig):

    from modulus.datapipes.benchmarks.darcy_fix import Darcy2D_fix

    # construct data pipe
    datapipe = Darcy2D_fix(
        resolution=resolution,
        batch_size=batch_size,
        normaliser=None,
        train_path="/data/fno/piececonst_r421_N1024_smooth1.mat",
        is_test=False,
    )

    # test single sample
    for data in datapipe:
        pos, x, y = data
        pos = pos.reshape(-1, resolution, resolution, 2)
        x = x.unsqueeze(-1)
        y = y.unsqueeze(-1)
        # check batch size
        assert common.check_batch_size([pos, x, y], batch_size)

        # check channels
        assert common.check_channels([x, y], 1, axis=-1)

        # check grid number
        assert common.check_grid([pos], (resolution, resolution), axis=(1, 2))
        break


@pytest.mark.parametrize("device", ["cuda:0"])
def test_darcy_2d_fix_cudagraphs(device, pytestconfig):

    from modulus.datapipes.benchmarks.darcy_fix import Darcy2D_fix

    # Preprocess function to convert dataloader output into Tuple of tensors
    def input_fn(data) -> Tuple[Tensor, ...]:
        return (data[0], data[1], data[2])

    # construct data pipe
    datapipe = Darcy2D_fix(
        resolution=85,
        batch_size=4,
        normaliser=None,
        train_path="/data/fno/piececonst_r421_N1024_smooth1.mat",
        is_test=False,
    )

    assert common.check_cuda_graphs(datapipe, input_fn)
