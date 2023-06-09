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
import xarray
import datetime

import modulus
from modulus.models.sfno import sfnonet
from modulus.utils import filesystem
from modulus.utils.sfno.zenith_angle import cos_zenith_angle
from modulus.utils.sfno.YParams import ParamsBase

from modulus.models.graphcast.graph_cast_net import GraphCastNet
from modulus.models.dlwp import DLWP

import logging

logger = logging.getLogger(__name__)


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
        x, z = torch.broadcast_tensors(x, z)
        x = torch.cat([x, z], dim=1)
        return self.model(x)


def sfno(package: filesystem.Package, pretrained: bool = True) -> torch.nn.Module:
    """Load SFNO model from checkpoints trained with era5_wind"""
    path = package.get("config.json")
    params = ParamsBase.from_json(path)
    model = sfnonet.SphericalFourierNeuralOperatorNet(params)
    logger.info(str(params.to_dict()))

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


class _GraphCastWrapper(torch.nn.Module):
    def __init__(self, model, dtype):
        super().__init__()
        self.model = model
        self.dtype = dtype

    def forward(self, x):
        x = x.to(self.dtype)
        y = self.model(x)
        return y


def graphcast_34ch(
    package: filesystem.Package, pretrained: bool = True
) -> torch.nn.Module:
    """Load Graphcast 34 channel model from a checkpoint"""
    num_channels = 34

    icospheres_path = package.get("icospheres.json")
    static_data_path = package.get("static", recursive=True)

    # instantiate the model, set dtype
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
        .to("cuda")  # TODO hardcoded
    )

    # set model to inference mode
    base_model.eval()

    model = _GraphCastWrapper(base_model, torch.bfloat16)

    if pretrained:
        path = package.get("weights.tar")
        checkpoint = torch.load(path)
        weights = checkpoint["model_state_dict"]
        weights = _fix_state_dict_keys(weights, add_module=False)
        model.model.load_state_dict(weights, strict=True)

    return model


class _DLWPWrapper(torch.nn.Module):
    def __init__(
        self,
        model,
        lsm,
        longrid,
        latgrid,
        topographic_height,
        ll_to_cs_mapfile_path,
        cs_to_ll_mapfile_path,
    ):
        super(_DLWPWrapper, self).__init__()
        self.model = model
        self.lsm = lsm
        self.longrid = longrid
        self.latgrid = latgrid
        self.topographic_height = topographic_height

        # load map weights
        self.input_map_wts = xarray.open_dataset(ll_to_cs_mapfile_path)
        self.output_map_wts = xarray.open_dataset(cs_to_ll_mapfile_path)

    def prepare_input(self, input, time):
        device = input.device
        dtype = input.dtype
        num_chans = input.size(2)
        input_list = list(torch.split(input, 1, dim=1))
        i = self.input_map_wts.row.values - 1
        j = self.input_map_wts.col.values - 1
        data = self.input_map_wts.S.values
        M = torch.sparse_coo_tensor(np.array((i, j)), data).to(device).type(dtype)

        for i in range(len(input_list)):
            input_list[i] = input_list[i].reshape(num_chans, -1) @ M.T
            input_list[i] = input_list[i].reshape(1, num_chans, 6, 64, 64)

        for i in range(len(input_list)):
            tisr = np.maximum(
                cos_zenith_angle(
                    time + datetime.timedelta(hours=6 * i), self.longrid, self.latgrid
                ),
                0,
            ) - (
                1 / np.pi
            )  # subtract mean value
            tisr = (
                torch.tensor(tisr, dtype=dtype)
                .to(device)
                .unsqueeze(dim=0)
                .unsqueeze(dim=0)
            )  # add channel and batch size dimension
            input_list[i] = torch.cat(
                (input_list[i], tisr), dim=1
            )  # concat along channel dim

        input_model = torch.cat(
            input_list, dim=1
        )  # concat the time dimension into channels
        repeat_vals = (1, -1, -1, -1, -1)  # repeat along batch dimension
        lsm_tensor = torch.tensor(self.lsm, dtype=dtype).to(device).unsqueeze(dim=0)
        lsm_tensor = lsm_tensor.expand(*repeat_vals)
        topographic_height_tensor = (
            torch.tensor((self.topographic_height - 3.724e03) / 8.349e03, dtype=dtype)
            .to(device)
            .unsqueeze(dim=0)
        )
        topographic_height_tensor = topographic_height_tensor.expand(*repeat_vals)

        input_model = torch.cat(
            (input_model, lsm_tensor, topographic_height_tensor), dim=1
        )

        return input_model

    def prepare_output(self, output):
        device = output.device
        dtype = output.dtype
        output = torch.split(output, output.shape[1] // 2, dim=1)
        output = torch.stack(output, dim=1)  # add time dimension back in
        output_list = list(torch.split(output, 1, dim=1))
        num_chans = output_list[0].shape[2]
        i = self.output_map_wts.row.values - 1
        j = self.output_map_wts.col.values - 1
        data = self.output_map_wts.S.values
        M = torch.sparse_coo_tensor(np.array((i, j)), data).to(device).type(dtype)

        for i in range(len(output_list)):
            output_list[i] = output_list[i].reshape(num_chans, -1) @ M.T
            output_list[i] = output_list[i].reshape(1, num_chans, 721, 1440)
        output = torch.stack(output_list, dim=1)

        return output

    def forward(self, x, time):
        x = self.prepare_input(x, time)
        y = self.model(x)
        return self.prepare_output(y)


def dlwp(package, pretrained=True):
    # load static datasets
    lsm = xarray.open_dataset(package.get("land_sea_mask_rs_cs.nc"))["lsm"].values
    topographic_height = xarray.open_dataset(package.get("geopotential_rs_cs.nc"))[
        "z"
    ].values
    latlon_grids = xarray.open_dataset(package.get("latlon_grid_field_rs_cs.nc"))
    latgrid, longrid = latlon_grids["latgrid"].values, latlon_grids["longrid"].values

    # load maps
    ll_to_cs_mapfile_path = package.get("map_LL721x1440_CS64.nc")
    cs_to_ll_mapfile_path = package.get("map_CS64_LL721x1440.nc")

    with open(package.get("config.json")) as json_file:
        config = json.load(json_file)
        core_model = DLWP(
            nr_input_channels=config["nr_input_channels"],
            nr_output_channels=config["nr_output_channels"],
        )

        if pretrained:
            weights_path = package.get("weights.pt")
            weights = torch.load(weights_path)
            fixed_weights = _fix_state_dict_keys(weights, add_module=False)
            core_model.load_state_dict(fixed_weights)

        model = _DLWPWrapper(
            core_model,
            lsm,
            longrid,
            latgrid,
            topographic_height,
            ll_to_cs_mapfile_path,
            cs_to_ll_mapfile_path,
        )

        model.eval()

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
