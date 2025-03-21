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

# ruff: noqa: S101
import numpy as np
import torch
import torch.nn as nn


class SinusoidalEncoding(nn.Module):
    """SinusoidalEncoding."""

    def __init__(self, num_channels: int, data_range: float = 2.0):
        super().__init__()
        assert (
            num_channels % 2 == 0
        ), f"num_channels must be even for sin/cos, got {num_channels}"
        self.num_channels = num_channels
        self.data_range = data_range

        freqs = 2 ** torch.arange(start=0, end=self.num_channels // 2)
        freqs = (2 * np.pi / self.data_range) * freqs

        self.register_buffer("freqs", freqs)

    def forward(self, x):

        x = x.unsqueeze(-1)
        # Reshape frequencies to match input dimensions for broadcasting
        # Create a shape with 1s for all dimensions except the last one
        # For example, if x is (batch, points, 3), create (1, 1, freqs.shape)
        broadcast_shape = (1,) * (len(x.shape) - 1) + self.freqs.shape
        freqs = self.freqs.reshape(broadcast_shape)
        x = x * freqs
        x = torch.cat([x.cos(), x.sin()], dim=-1).flatten(start_dim=-2)
        return x
