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

import math

import torch
from torch.utils.checkpoint import checkpoint_sequential

from ..layers import DownSample3D, FuserLayer, UpSample3D
from ..module import Module


class PanguProcessor(Module):
    """
    Processor sub-component for the Pangu DLNWP model. This model contains the
    layers corresponding to both the encoder and decoder portions of the 3D
    Earth-Specific Transformer from the Pangu paper (see link below).

    Parameters
    ----------
    embed_dim: int
        Embedded dimension of the transformer layers.
    patched_inp_shape: tuple[int]
        Tuple containing the shape of the patched embedding inputs.
    num_heads: tuple[int]
        The number of attention heads for the contained transformers.
        Expected to have 4 entries, corresponding to the 4 Fuser Layers.
    window_size: tuple[int]
        Window size in the Earth-Specific transformer.
    drop_path: list
        Stochastic depth rate
    number_upsampled_blocks: int
        The number of upsampling (and downsampling) blocks to use.
    checkpoint_flag: bool
        Whether to use gradient checkpointing during training.
    """

    def __init__(
        self,
        embed_dim: int,
        patched_inp_shape: tuple[int],
        num_heads: tuple[int],
        window_size: tuple[int],
        drop_path: list,
        number_upsampled_blocks: int,
        checkpoint_flag: bool,
    ):
        super().__init__()

        self.layer1 = FuserLayer(
            dim=embed_dim,
            input_resolution=patched_inp_shape,
            depth=number_upsampled_blocks,
            num_heads=num_heads[0],
            window_size=window_size,
            drop_path=drop_path[:number_upsampled_blocks],
        )

        patched_inp_shape_downsample = (
            8,
            math.ceil(patched_inp_shape[1] / 2),
            math.ceil(patched_inp_shape[2] / 2),
        )

        self.layers = torch.nn.Sequential(
            DownSample3D(
                in_dim=embed_dim,
                input_resolution=patched_inp_shape,
                output_resolution=patched_inp_shape_downsample,
            ),
            FuserLayer(
                dim=embed_dim * 2,
                input_resolution=patched_inp_shape_downsample,
                depth=6,
                num_heads=num_heads[1],
                window_size=window_size,
                drop_path=drop_path[number_upsampled_blocks:],
            ),
            FuserLayer(
                dim=embed_dim * 2,
                input_resolution=patched_inp_shape_downsample,
                depth=6,
                num_heads=num_heads[2],
                window_size=window_size,
                drop_path=drop_path[number_upsampled_blocks:],
            ),
            UpSample3D(
                embed_dim * 2,
                embed_dim,
                patched_inp_shape_downsample,
                patched_inp_shape,
            ),
            FuserLayer(
                dim=embed_dim,
                input_resolution=patched_inp_shape,
                depth=2,
                num_heads=num_heads[3],
                window_size=window_size,
                drop_path=drop_path[:number_upsampled_blocks],
            ),
        )

        self.checkpoint_flag = checkpoint_flag

    def checkpointed_model(self, x: torch.Tensor):
        """Utility function to support gradient checkpointing."""
        modules = [module for k, module in self.layers._modules.items()]
        return checkpoint_sequential(modules, 5, x, use_reentrant=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        "Forward model pass."
        x = self.layer1(x)

        skip = x

        if self.checkpoint_flag:
            x = self.checkpointed_model(x)
        else:
            x = self.layers(x)

        return torch.concat([x, skip], dim=-1)
