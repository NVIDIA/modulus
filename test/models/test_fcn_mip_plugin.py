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

from modulus.utils.sfno.YParams import ParamsBase
from modulus.models.fcn_mip_plugin import sfno, graphcast_34ch
from modulus.utils.filesystem import Package
from modulus.models.sfno.sfnonet import SphericalFourierNeuralOperatorNet
import datetime
import torch
import json


def save_checkpoint(model, check_point_path):
    model_state = {f"module.{k}": v for k, v in model.state_dict().items()}
    # This buffer is not present in some trained model checkpoints
    del model_state["module.device_buffer"]
    checkpoint = {"model_state": model_state}
    torch.save(checkpoint, check_point_path)


def save_mock_package(path):

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
    save_checkpoint(model, check_point_path)

    url = f"file://{path.as_posix()}"
    package = Package(url, seperator="/")
    return package


def test_sfno(tmp_path):
    # Can be tested against this URL too (but it is slow):
    # url = "s3://sw_climate_fno/nbrenowitz/model_packages/sfno_coszen"
    # package = Package(url, "/")
    package = save_mock_package(tmp_path)

    model = sfno(package, pretrained=True)
    x = torch.ones(1, 1, model.model.h, model.model.w)
    time = datetime.datetime(2018, 1, 1)
    with torch.no_grad():
        out = model(x, time=time)

    assert out.shape == x.shape


def test_graphcast(tmp_path):
    # TODO fix this test...the icosaspheres data used to be a pickle, but now it
    # is a json.
    url = "s3://sw_climate_fno/nbrenowitz/model_packages/graphcast_34ch"
    package = Package(url, "/")
    model = graphcast_34ch(package, pretrained=True)
