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
import torch.nn.functional as F
from torch import nn

from modulus.models.sfno.activations import ComplexReLU, ComplexActivation


def test_ComplexReLU_cartesian():
    relu = ComplexReLU(mode="cartesian")
    z = torch.randn(2, 2, dtype=torch.cfloat)
    output = relu(z)
    assert torch.allclose(output.real, F.relu(z.real)) and torch.allclose(
        output.imag, F.relu(z.imag)
    )


def test_ComplexReLU_real():
    relu = ComplexReLU(mode="real")
    z = torch.randn(2, 2, dtype=torch.cfloat)
    output = relu(z)
    assert torch.allclose(output.real, F.relu(z.real)) and torch.allclose(
        output.imag, z.imag
    )


def test_ComplexActivation_cartesian():
    activation = ComplexActivation(nn.ReLU(), mode="cartesian")
    z = torch.randn(2, 2, dtype=torch.cfloat)
    output = activation(z)
    assert torch.allclose(output.real, F.relu(z.real)) and torch.allclose(
        output.imag, F.relu(z.imag)
    )


def test_ComplexActivation_modulus():
    activation = ComplexActivation(nn.ReLU(), mode="modulus")
    z = torch.randn(2, 2, dtype=torch.cfloat)
    output = activation(z)
    assert torch.allclose(output.abs(), F.relu(z.abs()))
