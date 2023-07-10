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

import sys, os

script_path = os.path.abspath(__file__)
sys.path.append(os.path.join(os.path.dirname(script_path), ".."))

import pytest

from utils import fix_random_seeds, create_random_input

import common
from utils import get_icosphere_path
from modulus.models.graphcast.graph_cast_net import GraphCastNet

icosphere_path = get_icosphere_path()


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_graphcast_forward(device, num_channels=2, res_h=10, res_w=20):
    """Test graphcast forward pass"""

    model_kwds = {
        "meshgraph_path": icosphere_path,
        "static_dataset_path": None,
        "input_res": (res_h, res_w),
        "input_dim_grid_nodes": num_channels,
        "input_dim_mesh_nodes": 3,
        "input_dim_edges": 4,
        "output_dim_grid_nodes": num_channels,
        "processor_layers": 3,
        "hidden_dim": 4,
        "do_concat_trick": True,
    }

    fix_random_seeds()
    x = create_random_input(model_kwds["input_res"], model_kwds["input_dim_grid_nodes"])
    x = x.to(device)

    # Construct graphcast model
    model = GraphCastNet(**model_kwds).to(device)

    assert common.validate_forward_accuracy(model, (x,), rtol=1e-2)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_graphcast_constructor(
    device, num_channels_1=2, num_channels_2=3, res_h=10, res_w=20
):
    """Test graphcast constructor options"""
    # Define dictionary of constructor args
    arg_list = [
        {
            "meshgraph_path": icosphere_path,
            "static_dataset_path": None,
            "input_res": (res_h, res_w),
            "input_dim_grid_nodes": num_channels_1,
            "input_dim_mesh_nodes": 3,
            "input_dim_edges": 4,
            "output_dim_grid_nodes": num_channels_1,
            "processor_layers": 3,
            "hidden_dim": 4,
            "do_concat_trick": True,
        },
        {
            "meshgraph_path": icosphere_path,
            "static_dataset_path": None,
            "input_res": (res_h, res_w),
            "input_dim_grid_nodes": num_channels_2,
            "input_dim_mesh_nodes": 3,
            "input_dim_edges": 4,
            "output_dim_grid_nodes": num_channels_2,
            "processor_layers": 4,
            "hidden_dim": 8,
            "do_concat_trick": False,
        },
    ]
    for kw_args in arg_list:
        # Construct GraphCast model
        model = GraphCastNet(**kw_args).to(device)

        x = create_random_input(
            kw_args["input_res"], kw_args["input_dim_grid_nodes"]
        ).to(device)
        outvar = model(x)
        assert outvar.shape == (
            1,
            kw_args["output_dim_grid_nodes"],
            *kw_args["input_res"],
        )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_GraphCast_optims(device, num_channels=2, res_h=10, res_w=20):
    """Test GraphCast optimizations"""

    def setup_model():
        """Set up fresh model and inputs for each optim test"""
        model_kwds = {
            "meshgraph_path": icosphere_path,
            "static_dataset_path": None,
            "input_res": (res_h, res_w),
            "input_dim_grid_nodes": num_channels,
            "input_dim_mesh_nodes": 3,
            "input_dim_edges": 4,
            "output_dim_grid_nodes": num_channels,
            "processor_layers": 3,
            "hidden_dim": 2,
            "do_concat_trick": True,
        }
        fix_random_seeds()
        x = create_random_input(
            model_kwds["input_res"], model_kwds["input_dim_grid_nodes"]
        )
        x = x.to(device)

        # Construct GraphCast model
        model = GraphCastNet(**model_kwds).to(device)
        return model, (x,)

    # Ideally always check graphs first
    model, invar = setup_model()
    assert common.validate_cuda_graphs(model, (*invar,))
    # Check JIT
    model, invar = setup_model()
    assert common.validate_jit(model, (*invar,))
    # Check AMP
    model, invar = setup_model()
    assert common.validate_amp(model, (*invar,))
    # Check Combo
    model, invar = setup_model()
    assert common.validate_combo_optims(model, (*invar,))


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_graphcast_checkpoint(device, num_channels=2, res_h=10, res_w=20):
    """Test GraphCast checkpoint save/load"""

    model_kwds = {
        "meshgraph_path": icosphere_path,
        "static_dataset_path": None,
        "input_res": (res_h, res_w),
        "input_dim_grid_nodes": num_channels,
        "input_dim_mesh_nodes": 3,
        "input_dim_edges": 4,
        "output_dim_grid_nodes": num_channels,
        "processor_layers": 3,
        "hidden_dim": 2,
        "do_concat_trick": True,
    }

    # Construct GraphCast model
    model_1 = GraphCastNet(**model_kwds).to(device)
    model_2 = GraphCastNet(**model_kwds).to(device)

    x = create_random_input(model_kwds["input_res"], model_kwds["input_dim_grid_nodes"])
    x = x.to(device)

    assert common.validate_checkpoint(
        model_1,
        model_2,
        (x,),
    )


@common.check_ort_version()
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_GraphCast_deploy(device, num_channels=2, res_h=10, res_w=20):
    """Test GraphCast deployment support"""

    model_kwds = {
        "meshgraph_path": icosphere_path,
        "static_dataset_path": None,
        "input_res": (res_h, res_w),
        "input_dim_grid_nodes": num_channels,
        "input_dim_mesh_nodes": 3,
        "input_dim_edges": 4,
        "output_dim_grid_nodes": num_channels,
        "processor_layers": 3,
        "hidden_dim": 2,
        "do_concat_trick": True,
    }

    # Construct GraphCast model
    model = GraphCastNet(**model_kwds).to(device)

    x = create_random_input(model_kwds["input_res"], model_kwds["input_dim_grid_nodes"])
    x = x.to(device)

    assert common.validate_onnx_export(model, x)
    assert common.validate_onnx_runtime(model, x)
