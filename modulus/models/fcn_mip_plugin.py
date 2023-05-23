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
import json
import numpy as np
import torch

from modulus.models.sfno import sfnonet
from modulus.utils.sfno.zenith_angle import cos_zenith_angle
from modulus.utils.sfno.YParams import ParamsBase

from modulus.models.graphcast.graph_cast_net import GraphCastNet

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


class _CosZenWrapper(torch.nn.Module):
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
        model = _CosZenWrapper(model, lon, lat)

    return model


class GraphCastWrapper(torch.nn.Module):
    def __init__(self, model, dtype):
        super().__init__()
        self.model = model
        self.dtype = dtype

    def forward(self, x):
        x = x.to(self.dtype)
        y = self.model(x)
        return y


def graphcast_34ch(package, pretrained=True):
    num_channels = 34

    icospheres_path = package.get("icospheres.json")
    static_data_path = package.get("static", recursive=True)

    # instantiate the model, set dtype and move to device
    base_model = (
        GraphCastNet(
            meshgraph_path=icospheres_path,
            static_dataset_path=static_data_path,
            input_dim_grid_nodes=num_channels,
            input_dim_mesh_nodes=3,
            input_dim_edges=4,
            output_dim_grid_nodes=num_channels,
            processor_layers=16,
            hidden_dim=512,
            do_concat_trick=True,
        )
        .to(dtype=torch.bfloat16)
        .to(dist.device)
    )

    # set model to inference mode
    base_model.eval()

    model = GraphCastWrapper(base_model, torch.bfloat16)

    if pretrained:
        path = package.get("weights.tar")
        checkpoint = torch.load(path)
        weights = checkpoint["model_state_dict"]
        weights = _fix_state_dict_keys(weights, add_module=False)
        model.model.load_state_dict(weights, strict=True)

    return model


def _fix_state_dict_keys(state_dict, add_module=False):
    """Add or remove 'module.' from state_dict keys

    Parameters
    ----------
    state_dict : Dict
        Model state_dict
    add_module : bool, optional
        If True, will add 'module.' to keys, by default False

    Returns
    -------
    Dict
        Model state_dict with fixed keys
    """
    fixed_state_dict = {}
    for key, value in state_dict.items():
        if add_module:
            new_key = "module." + key
        else:
            new_key = key.replace("module.", "")
        fixed_state_dict[new_key] = value
    return fixed_state_dict
