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

import torch
import pytest

from utils import fix_random_seeds, create_random_input, get_icosphere_path
from modulus.models.graphcast.graph_cast_net import GraphCastNet


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_grad_checkpointing(device, num_channels=2, res_h=15, res_w=15):
    """Test gradient checkpointing"""
    icosphere_path = get_icosphere_path()

    # constants
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
    num_steps = 2

    # Fix random seeds
    fix_random_seeds()

    # Random input
    x = create_random_input(
        model_kwds["input_res"], model_kwds["input_dim_grid_nodes"]
    ).to(device)

    # Instantiate the model
    model = GraphCastNet(**model_kwds).to(device)

    # Set gradient checkpointing
    model.set_checkpoint_model(False)
    model.set_checkpoint_encoder(True)
    model.set_checkpoint_processor(2)
    model.set_checkpoint_decoder(True)

    # Forward pass with checkpointing
    y_pred_checkpointed = x
    for i in range(num_steps):
        y_pred_checkpointed = model(y_pred_checkpointed)

    # dummy loss
    loss = y_pred_checkpointed.sum()

    # compute gradients
    loss.backward()
    computed_grads_checkpointed = {}
    for name, param in model.named_parameters():
        computed_grads_checkpointed[name] = param.grad.clone()

    # Fix random seeds
    fix_random_seeds()

    # Random input
    x = create_random_input(
        model_kwds["input_res"], model_kwds["input_dim_grid_nodes"]
    ).to(device)

    # Instantiate the model
    model = GraphCastNet(**model_kwds).to(device)

    # Set gradient checkpointing
    model.set_checkpoint_model(False)
    model.set_checkpoint_encoder(False)
    model.set_checkpoint_processor(1)
    model.set_checkpoint_decoder(False)

    # Forward pass without checkpointing
    y_pred = x
    for i in range(num_steps):
        y_pred = model(y_pred)

    # dummy loss
    loss = y_pred.sum()

    # compute gradients
    loss.backward()
    computed_grads = {}
    for name, param in model.named_parameters():
        computed_grads[name] = param.grad.clone()

    # Compare the gradients
    for name in computed_grads:
        torch.allclose(
            computed_grads_checkpointed[name], computed_grads[name]
        ), "Gradient do not match. Checkpointing failed!"

    # Check that the results are the same
    assert torch.allclose(
        y_pred_checkpointed, y_pred
    ), "Outputs do not match. Checkpointing failed!"
