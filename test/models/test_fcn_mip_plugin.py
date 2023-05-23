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

import fsspec
from modulus.utils.sfno.YParams import ParamsBase
from modulus.models.fcn_mip_plugin import sfno, graphcast_34ch, _CosZenWrapper
from modulus.utils.filesystem import Package, download_cached
from modulus.models.sfno.sfnonet import SphericalFourierNeuralOperatorNet
import numpy as np
import datetime
import torch
import json

import pytest


def save_ddp_checkpoint(model, check_point_path):
    """Save checkpoint with similar structure to the training checkpoints

    The keys are prefixed with "module."
    """
    model_state = {f"module.{k}": v for k, v in model.state_dict().items()}
    # This buffer is not present in some trained model checkpoints
    del model_state["module.device_buffer"]
    checkpoint = {"model_state": model_state}
    torch.save(checkpoint, check_point_path)


def save_untrained_sfno(path):

    config = {
        "N_in_channels": 2,
        "N_out_channels": 1,
        "img_shape_x": 4,
        "img_shape_y": 5,
        "scale_factor": 1,
        "num_layers": 2,
        "num_blocks": 2,
        "embed_dim": 2,
        "nettype": "sfno",
        "add_zenith": True,
    }
    params = ParamsBase()
    params.update_params(config)
    model = SphericalFourierNeuralOperatorNet(params)

    config_path = path / "config.json"
    with config_path.open("w") as f:
        json.dump(params.to_dict(), f)

    check_point_path = path / "weights.tar"
    save_ddp_checkpoint(model, check_point_path)

    url = f"file://{path.as_posix()}"
    package = Package(url, seperator="/")
    return package


def test_sfno(tmp_path):
    package = save_untrained_sfno(tmp_path)

    model = sfno(package, pretrained=True)
    x = torch.ones(1, 1, model.model.h, model.model.w)
    time = datetime.datetime(2018, 1, 1)
    with torch.no_grad():
        out = model(x, time=time)

    assert out.shape == x.shape


@pytest.mark.skip("graphcast test is extremely slow, when instatiating model.")
def test_graphcast(tmp_path):
    fs = fsspec.filesystem("https")

    version = "ede0fcbfaf7a8131668620a9aba19970774a4785"

    url = f"https://raw.githubusercontent.com/NVIDIA/modulus-launch/{version}/recipes/gnn/graphcast/icospheres.json"
    dest = tmp_path / "icospheres.json"
    fs.get(url, dest.as_posix())

    # download static data
    static = tmp_path / "static"
    static.mkdir()
    for file in ["geopotential.nc", "land_sea_mask.nc"]:
        root = f"https://media.githubusercontent.com/media/NVIDIA/modulus-launch/{version}/recipes/gnn/graphcast/datasets/static"
        fs.get(f"{root}/{file}", str(static / file))

    package = Package(tmp_path.as_posix(), "/")
    model = graphcast_34ch(package, pretrained=False)


@pytest.mark.parametrize("batch_size", [1, 2])
def test__CozZenWrapper(batch_size):
    class I(torch.nn.Module):
        def forward(self, x):
            return x

    model = I()
    nx, ny = (3, 4)
    lat = np.arange(nx)
    lon = np.arange(ny)

    x = torch.ones((batch_size, 1, nx, ny))
    time = datetime.datetime(2018, 1, 1)
    wrapper = _CosZenWrapper(model, lon, lat)
    wrapper(x, time=time)
