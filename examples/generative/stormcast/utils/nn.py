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

import torch
from modulus.models import Module
from modulus.models.diffusion import EDMPrecond, StormCastUNet
from modulus.utils.generative import deterministic_sampler


def get_preconditioned_architecture(
    name: str,
    target_channels: int,
    conditional_channels: int = 0,
    spatial_embedding: bool = True,
    hrrr_resolution: tuple = (512, 640),
    attn_resolutions: list = [],
):
    """

    Args:
        name: 'regression' or 'diffusion' to select between either model type
        target_channels: The number of channels in the target
        conditional_channels: The number of channels in the conditioning
        spatial_embedding: whether or not to use the additive spatial embedding in the U-Net
        hrrr_resolution: resolution of HRRR data (U-Net inputs/outputs)
        attn_resolutions: resolution of internal U-Net stages to use self-attention
    Returns:
        EDMPrecond or StormCastUNet: a wrapped torch module net(x+n, sigma, condition, class_labels) -> x
    """
    if name == "diffusion":
        return EDMPrecond(
            img_resolution=hrrr_resolution,
            img_channels=target_channels + conditional_channels,
            img_out_channels=target_channels,
            model_type="SongUNet",
            channel_mult=[1, 2, 2, 2, 2],
            attn_resolutions=attn_resolutions,
            additive_pos_embed=spatial_embedding,
        )

    elif name == "regression":
        return StormCastUNet(
            img_resolution=hrrr_resolution,
            img_in_channels=conditional_channels,
            img_out_channels=target_channels,
            model_type="SongUNet",
            embedding_type="zero",
            channel_mult=[1, 2, 2, 2, 2],
            attn_resolutions=attn_resolutions,
            additive_pos_embed=spatial_embedding,
        )


def diffusion_model_forward(
    model, hrrr_0, diffusion_channel_indices, invariant_tensor, sampler_args={}
):
    """Helper function to run diffusion model sampling"""

    b, c, h, w = hrrr_0[:, diffusion_channel_indices, :, :].shape

    latents = torch.randn(b, c, h, w, device=hrrr_0.device, dtype=hrrr_0.dtype)

    if b > 1 and invariant_tensor.shape[0] != b:
        invariant_tensor = invariant_tensor.expand(b, -1, -1, -1)
    condition = torch.cat((hrrr_0, invariant_tensor), dim=1)

    output_images = deterministic_sampler(
        model, latents=latents, img_lr=condition, **sampler_args
    )

    return output_images


def regression_model_forward(model, hrrr, era5, invariant_tensor):
    """Helper function to run regression model forward pass in inference"""

    x = torch.cat([hrrr, era5, invariant_tensor], dim=1)

    return model(x)


def regression_loss_fn(
    net: Module,
    images,
    condition,
    class_labels=None,
    augment_pipe=None,
    return_model_outputs=False,
):
    """Helper function for training the StormCast regression model, so that it has a similar call signature as
    the EDMLoss and the same training loop can be used to train both regression and diffusion models

    Args:
        net: modulus.models.diffusion.StormCastUNet
        images: Target data, shape [batch_size, target_channels, w, h]
        condition: input to the model, shape=[batch_size, condition_channel, w, h]
        class_labels: unused (applied to match EDMLoss signature)
        augment_pipe: optional data augmentation pipe
        return_model_outputs: If True, will return the generated outputs
    Returns:
        out: loss function with shape [batch_size, target_channels, w, h]
            This should be averaged to get the mean loss for gradient descent.
    """

    y, augment_labels = (
        augment_pipe(images) if augment_pipe is not None else (images, None)
    )

    D_yn = net(x=condition)
    loss = (D_yn - y) ** 2
    if return_model_outputs:
        return loss, D_yn
    else:
        return loss
