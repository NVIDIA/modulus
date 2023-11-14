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

import os, sys

import torch
import random
from modulus.experimental.sfno.utils.YParams import YParams
from modulus.experimental.sfno.networks.sfnonet import SphericalFourierNeuralOperatorNet as SFNO

import pytest


@pytest.mark.xfail
def test_superres():
    params = YParams('config/sfnonet.yaml', 'sfno_dhealy')

    params.batch_size = 1
    params.levels = 0 
    params.padding = 0
    params.in_chans = 10
    params.out_chans = 11
    params.img_shape = (721, 1440)
    params.embed_dim = 16
    params.scale_factor = 4
    params.out_shape = (900, 2000)
    params.num_layers = 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = SFNO(params, img_shape=params.img_shape, #modes=params.modes,
                 in_chans=params.in_chans, out_chans=params.out_chans, 
                 num_layers=params.num_layers, embed_dim=params.embed_dim).to(device)

    in_tensor = torch.randn((1, params.in_chans, *params.img_shape)).to(device)
    out = model(in_tensor) 
    target_shape = [1, params.out_chans, *params.out_img_shape]
    assert list(out.shape) == target_shape
