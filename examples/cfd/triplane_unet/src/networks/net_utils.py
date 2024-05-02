# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import numpy as np
import torch
import torch.nn as nn


# TODO(cchooy) move to .component.mlp
class MLP(torch.nn.Module):
    """A simple feedforward neural network."""

    def __init__(self, channels: list, nonlinearity: nn.Module = nn.ReLU):
        super().__init__()

        n_layers = len(channels) - 1
        assert n_layers >= 1, "MLP must have at least 2 layers"

        # Create the MLP
        layers = []
        for layer_idx in range(n_layers - 1):
            layers.append(nn.Linear(channels[layer_idx], channels[layer_idx + 1]))
            layers.append(nonlinearity())
        layers.append(nn.Linear(channels[-2], channels[-1]))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# TODO(cchooy) move to .component.mlp
class MLPBlock(nn.Module):
    """MLPBlock."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = None,
        out_channels: int = None,
        activation=nn.GELU,
    ):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = in_channels
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.shortcut = nn.Linear(in_channels, out_channels)
        self.activation = activation()

    def forward(self, x):
        out = self.activation(self.norm1(self.fc1(x)))
        out = self.norm2(self.fc2(out))
        # add skip connection
        out = self.activation(out + self.shortcut(x))
        return out


# TODO(cchooy) move to .component.encoding
class SinusoidalEncoding(nn.Module):
    """PositionalEncoding."""

    def __init__(self, num_channels: int, data_range: float = 2.0):
        super().__init__()
        assert (
            num_channels % 2 == 0
        ), f"num_channels must be even for sin/cos, got {num_channels}"
        self.num_channels = num_channels
        self.data_range = data_range

    def forward(self, x):
        freqs = 2 ** torch.arange(
            start=0, end=self.num_channels // 2, device=x.device
        ).to(x.dtype)
        freqs = (2 * np.pi / self.data_range) * freqs
        x = x.unsqueeze(-1)
        # Make freq to have the same dimensions as x. X can be of any shape
        freqs = freqs.reshape((1,) * (len(x.shape) - 1) + freqs.shape)
        x = x * freqs
        x = torch.cat([x.cos(), x.sin()], dim=-1).flatten(start_dim=-2)
        return x
