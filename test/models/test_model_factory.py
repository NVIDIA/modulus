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

import pytest
import torch

from modulus.models import Module


class MockModel(Module):
    def __init__(self, layer_size=16):
        super().__init__()
        self.layer_size = layer_size
        self.layer = torch.nn.Linear(layer_size, layer_size)

    def forward(self, x):
        return self.layer(x)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_register_and_factory(device):
    # Register the MockModel
    Module.register(MockModel, "mock_model")

    # Use factory to get the MockModel
    RetrievedModel = Module.factory("mock_model")

    # Check if the retrieved model is the same as the one registered
    assert RetrievedModel == MockModel

    # Check forward pass of RetrievedModel
    layer_size = 16
    invar = torch.randn(1, layer_size).to(device)
    model = RetrievedModel(layer_size=layer_size).to(device)
    outvar = model(invar)
    assert outvar.shape == invar.shape
    assert outvar.device == invar.device
    Module._clear_model_registry()
