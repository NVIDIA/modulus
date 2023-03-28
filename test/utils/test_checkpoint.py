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
import shutil
import torch.nn as nn

import modulus
from typing import Callable
from modulus.models.mlp import FullyConnected
from modulus.launch.utils import save_checkpoint, load_checkpoint


@pytest.fixture()
def checkpoint_folder() -> str:
    return "./checkpoints"


@pytest.fixture(params=["modulus", "pytorch"])
def model_generator(request) -> Callable:
    # Create fully-connected NN generator function
    if request.param == "modulus":
        generator = lambda x: FullyConnected(
            in_features=x,
            out_features=x,
            num_layers=2,
            layer_size=8,
        )
    else:
        generator = lambda x: nn.Sequential(
            nn.Linear(x, 8),
            nn.ReLU(),
            nn.Linear(8, x),
        )
    return generator


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_model_checkpointing(
    device, model_generator, checkpoint_folder, rtol: float = 1e-3, atol: float = 1e-3
):
    """Test checkpointing util for model"""
    mlp_model_1 = model_generator(8).to(device)
    mlp_model_2 = model_generator(4).to(device)

    input_1 = torch.randn(4, 8).to(device)
    input_2 = torch.randn(4, 4).to(device)

    output_1 = mlp_model_1(input_1)
    output_2 = mlp_model_2(input_2)
    # Save model weights to checkpoint
    save_checkpoint(checkpoint_folder, models=[mlp_model_1, mlp_model_2])

    # Load twin set of models for importing weights
    mlp_model_1 = model_generator(8).to(device)
    mlp_model_2 = model_generator(4).to(device)

    new_output_1 = mlp_model_1(input_1)
    new_output_2 = mlp_model_2(input_2)
    # Assert models are now different
    assert not torch.allclose(output_1, new_output_1, rtol, atol)
    assert not torch.allclose(output_2, new_output_2, rtol, atol)

    # Load model weights from checkpoint
    load_checkpoint(checkpoint_folder, models=[mlp_model_1, mlp_model_2], device=device)

    new_output_1 = mlp_model_1(input_1)
    new_output_2 = mlp_model_2(input_2)

    assert torch.allclose(output_1, new_output_1, rtol, atol)
    assert torch.allclose(output_2, new_output_2, rtol, atol)

    # Clean up
    shutil.rmtree(checkpoint_folder)
