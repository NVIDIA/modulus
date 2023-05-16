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

"""An implementation of the fcn-mip plugin interface

This interface is documented here: https://gitlab-master.nvidia.com/earth-2/fcn-mip/-/blob/main/docs/plugin.md

Support loading checkpoints with non-distributed models, and models with
cos-zenith angle models.

See this example notebook: https://drive.google.com/file/d/18eJIrZScTJSuRx7EjBYLurejseUUoQ9J/view?usp=share_link
"""
import json
import numpy as np
import torch

from utils.YParams import ParamsBase
from modulus.models.sfno import sfnonet
from modulus.utils.sfno.zenith_angle import cos_zenith_angle
from modulus.utils.sfno.YParams import ParamsBase

import logging

logger = logging.getLogger(__name__)


def get_model(params: ParamsBase):
    if params.nettype == "sfno":
        return sfnonet.SphericalFourierNeuralOperatorNet(params)

    raise NotImplementedError(params.nettype)


class _DummyModule(torch.nn.Module):
    """Hack to handle that checkpoint parameter names begin with "module." """

    def __init__(self, model):
        super().__init__()
        self.module = model


class CosZenWrapper(torch.nn.Module):
    def __init__(self, model, lon, lat):
        super().__init__()
        self.model = model
        self.lon = lon
        self.lat = lat

    def forward(self, x, time):
        lon_grid, lat_grid = np.meshgrid(self.lon, self.lat)
        cosz = cos_zenith_angle(time, lon_grid, lat_grid)
        cosz = cosz.astype(np.float32)
        z = torch.from_numpy(cosz).to(device=x.device)
        # assume no history
        x = torch.cat([x, z[None, None]], dim=1)
        return self.model(x)


def sfno(package, pretrained=True):
    """Load SFNO model from checkpoints trained with era5_wind"""
    path = package.get("config.json")
    params = ParamsBase.from_json(path)
    logger.info(str(params.to_dict()))
    model = get_model(params)

    if pretrained:
        weights = package.get("weights.tar")
        checkpoint = torch.load(weights)
        load_me = _DummyModule(model)
        state = checkpoint["model_state"]
        state = {"module.device_buffer": model.device_buffer, **state}
        load_me.load_state_dict(state)

    if params.add_zenith:
        nlat = params.img_shape_x
        nlon = params.img_shape_y
        lat = 90 - np.arange(nlat) * 0.25
        lon = np.arange(nlon) * 0.25
        model = CosZenWrapper(model, lon, lat)

    return model
