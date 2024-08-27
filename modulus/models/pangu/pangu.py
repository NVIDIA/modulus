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
from dataclasses import dataclass

import numpy as np
import torch

from modulus.models.meta import ModelMetaData
from modulus.models.module import Module
from modulus.models.pangu.pangu_processor import PanguProcessor
from modulus.models.utils import (
    PatchEmbed2D,
    PatchEmbed3D,
    PatchRecovery2D,
    PatchRecovery3D,
)


@dataclass
class MetaData(ModelMetaData):
    name: str = "Pangu"
    # Optimization
    jit: bool = False  # ONNX Ops Conflict
    cuda_graphs: bool = True
    amp: bool = True
    # Inference
    onnx_cpu: bool = False  # No FFT op on CPU
    onnx_gpu: bool = True
    onnx_runtime: bool = True
    # Physics informed
    var_dim: int = 1
    func_torch: bool = False
    auto_grad: bool = False


class Pangu(Module):
    """
    Pangu A PyTorch impl of: `Pangu-Weather: A 3D High-Resolution Model for Fast and Accurate Global Weather Forecast`
    - https://arxiv.org/abs/2211.02556

    Parameters
    img_size: tuple[int]
        Image size [lat, lon]
    patch_size: tuple[int]
        Patch-embedding shape
    embed_dim: int
        Embedding dimension size, be default 192.
    num_heads: tuple[int]
        Number of attention heads to use for each Fuser Layer.
    window_size: tuple[int]
        Window size in 3D attention window mechanism.
    number_constant_variables: int
        The number of constant variables (do not change in time).
    number_surface_variables: int
        The number of surface variables (not including constant variables).
        By default 4
    number_atmosphere_variables: int
        The number of atmosphere variables per atmosphere level.
        By default 5
    number_atmosphere_levels: int
        The number of pressure levels in the atmosphere.
        By default 13.
    number_up_sampled_blocks: int
        The number of upsampled blocks in the Earth-specific Transformer blocks.
    number_down_sampled_blocks: int
        The number of downsampled blocks in the Earth-specific Transformer blocks.
    checkpoint_flag: int
        Whether to use gradient checkpointing in training.
    """

    def __init__(
        self,
        img_size=(721, 1440),
        patch_size=(2, 4, 4),
        embed_dim=192,
        num_heads=(6, 12, 12, 6),
        window_size=(2, 6, 12),
        number_constant_variables=3,
        number_surface_variables=4,
        number_atmosphere_variables=5,
        number_atmosphere_levels=13,
        number_up_sampled_blocks=2,
        number_down_sampled_blocks=6,
        checkpoint_flag: bool = False,
    ):
        super().__init__(meta=MetaData())
        drop_path = np.linspace(
            0, 0.2, number_up_sampled_blocks + number_down_sampled_blocks
        ).tolist()
        # In addition, three constant masks(the topography mask, land-sea mask and soil type mask)
        self.number_constant_variables = number_constant_variables
        self.number_surface_variables = number_surface_variables
        self.number_air_variables = number_atmosphere_variables
        self.number_air_levels = number_atmosphere_levels
        self.patchembed2d = PatchEmbed2D(
            img_size=img_size,
            patch_size=patch_size[1:],
            in_chans=self.number_surface_variables
            + self.number_constant_variables,  # add
            embed_dim=embed_dim,
        )
        self.patchembed3d = PatchEmbed3D(
            img_size=(number_atmosphere_levels, img_size[0], img_size[1]),
            patch_size=patch_size,
            in_chans=number_atmosphere_variables,
            embed_dim=embed_dim,
        )
        patched_inp_shape = (
            8,
            math.ceil(img_size[0] / patch_size[1]),
            math.ceil(img_size[1] / patch_size[2]),
        )

        self.processor = PanguProcessor(
            embed_dim,
            patched_inp_shape,
            num_heads,
            window_size,
            drop_path,
            number_up_sampled_blocks,
            checkpoint_flag,
        )

        # The outputs of the 2nd encoder layer and the 7th decoder layer are concatenated along the channel dimension.
        self.patchrecovery2d = PatchRecovery2D(
            img_size, patch_size[1:], 2 * embed_dim, self.number_surface_variables
        )
        self.patchrecovery3d = PatchRecovery3D(
            (number_atmosphere_levels, img_size[0], img_size[1]),
            patch_size,
            2 * embed_dim,
            number_atmosphere_variables,
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): [batch, 4+3+5*13, lat, lon]
        """
        surface = x[
            :, : self.number_constant_variables + self.number_surface_variables, :, :
        ]
        upper_air = x[
            :, self.number_constant_variables + self.number_surface_variables :, :, :
        ].reshape(
            x.shape[0],
            self.number_air_variables,
            self.number_air_levels,
            x.shape[2],
            x.shape[3],
        )
        surface = self.patchembed2d(surface)
        upper_air = self.patchembed3d(upper_air)

        x = torch.concat([surface.unsqueeze(2), upper_air], dim=2)
        B, C, Pl, Lat, Lon = x.shape
        x = x.reshape(B, C, -1).transpose(1, 2)

        output = self.processor(x)

        output = output.transpose(1, 2).reshape(B, -1, Pl, Lat, Lon)
        output_surface = output[:, :, 0, :, :]
        output_upper_air = output[:, :, 1:, :, :]

        output_surface = self.patchrecovery2d(output_surface)
        output_upper_air = self.patchrecovery3d(output_upper_air)
        s = output_upper_air.shape
        output_upper_air = output_upper_air.reshape(s[0], s[1] * s[2], *s[3:])
        return torch.concat([output_surface, output_upper_air], dim=1)
