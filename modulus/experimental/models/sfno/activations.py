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

from warnings import warn

import torch
from torch import nn


class ComplexReLU(nn.Module):
    """
    Complex-valued variants of the ReLU activation function
    """

    def __init__(
        self, negative_slope=0.0, mode="real", bias_shape=None, scale=1.0
    ):  # pragma: no cover
        super(ComplexReLU, self).__init__()

        # store parameters
        self.mode = mode
        if self.mode in ["modulus", "halfplane"]:
            if bias_shape is not None:
                self.bias = nn.Parameter(
                    scale * torch.ones(bias_shape, dtype=torch.float32)
                )
            else:
                self.bias = nn.Parameter(scale * torch.ones((1), dtype=torch.float32))
        else:
            self.bias = 0

        self.negative_slope = negative_slope
        self.act = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        if self.mode == "cartesian":
            zr = torch.view_as_real(z)
            za = self.act(zr)
            out = torch.view_as_complex(za)

        elif self.mode == "modulus":
            zabs = torch.sqrt(torch.square(z.real) + torch.square(z.imag))
            out = torch.where(zabs + self.bias > 0, (zabs + self.bias) * z / zabs, 0.0)
            # out = self.act(zabs - self.bias) * torch.exp(1.j * z.angle())

        elif self.mode == "halfplane":
            # bias is an angle parameter in this case
            modified_angle = torch.angle(z) - self.bias
            condition = torch.logical_and(
                (0.0 <= modified_angle), (modified_angle < torch.pi / 2.0)
            )
            out = torch.where(condition, z, self.negative_slope * z)

        elif self.mode == "real":
            zr = torch.view_as_real(z)
            outr = zr.clone()
            outr[..., 0] = self.act(zr[..., 0])
            out = torch.view_as_complex(outr)

        else:
            raise NotImplementedError

        return out


class ComplexActivation(nn.Module):
    """
    A module implementing complex-valued activation functions.
    The module supports different modes of operation, depending on how
    the complex numbers are treated for the activation function:
    - "cartesian": the activation function is applied separately to the
       real and imaginary parts of the complex input.
    - "modulus": the activation function is applied to the modulus of
       the complex input, after adding a learnable bias.
    - any other mode: the complex input is returned as-is (identity operation).
    """

    def __init__(
        self, activation, mode="cartesian", bias_shape=None
    ):  # pragma: no cover
        super(ComplexActivation, self).__init__()

        # store parameters
        self.mode = mode
        if self.mode == "modulus":
            if bias_shape is not None:
                self.bias = nn.Parameter(torch.zeros(bias_shape, dtype=torch.float32))
            else:
                self.bias = nn.Parameter(torch.zeros((1), dtype=torch.float32))
        else:
            bias = torch.zeros((1), dtype=torch.float32)
            self.register_buffer("bias", bias)

        # real valued activation
        self.act = activation

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        if self.mode == "cartesian":
            zr = torch.view_as_real(z)
            za = self.act(zr)
            out = torch.view_as_complex(za)
        elif self.mode == "modulus":
            zabs = torch.sqrt(torch.square(z.real) + torch.square(z.imag))
            out = self.act(zabs + self.bias) * torch.exp(1.0j * z.angle())
        else:
            # identity
            warn(
                "Unknown complex activation mode. Setting the activation to identity operation"
            )
            out = z

        return out
