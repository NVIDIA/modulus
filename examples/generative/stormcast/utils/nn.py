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
from modulus.models.diffusion import EDMPrecond, UNet


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
        resolution (int): _description_
        target_channels: The number of channels in the target
        conditional_channels: The number of channels in the conditioning
        label_dim: size of label data
        sigma_min:  Defaults to 0.
        sigma_max: Defaults to float("inf").
        sigma_data:  Defaults to 0.5.

    Returns:
        EDMPrecond or RegressionWrapperV2: a wrapped torch module net(x+n, sigma, condition, class_labels) -> x
    """
    if name == "diffusion":
        in_channels = target_channels + conditional_channels
        return EDMPrecond(
            img_resolution=hrrr_resolution,
            img_channels=in_channels,
            out_channels=target_channels,
            model_type="SongUNet",
            channel_mult=[1, 2, 2, 2, 2],
            attn_resolutions=attn_resolutions,
            additive_pos_embed=spatial_embedding,
        )

    elif name == "regression":
        in_channels = conditional_channels
        return UNet(
            img_resolution=hrrr_resolution,
            img_channels=in_channels,
            img_in_channels=in_channels
            - target_channels,  # TODO hack since modulus UNet adds output channels to input channels
            img_out_channels=target_channels,
            model_type="SongUNet",
            embedding_type="zero",
            channel_mult=[1, 2, 2, 2, 2],
            attn_resolutions=attn_resolutions,
            additive_pos_embed=spatial_embedding,
        )


def regression_model_forward(model, hrrr, era5, invariant_tensor):
    """Helper function to run regression model forward pass in inference"""

    condition = torch.cat(
        [hrrr, era5, invariant_tensor.repeat(hrrr.shape[0], 1, 1, 1)], dim=1
    )

    sigma = torch.randn([condition.shape[0], 1, 1, 1], device=condition.device)

    return model(x=condition, img_lr=None, sigma=sigma)


def regression_loss_fn(
    net: Module,
    images,
    condition=None,
    class_labels=None,
    augment_pipe=None,
):
    """Helper function for training the StormCast regression model, so that it has a similar call signature as
    the EDMLoss and the same training loop can be used to train both regression and diffusion models

    Args:
        net:
        x: The latent data (to be denoised). shape [batch_size, target_channels, w, h]
        class_labels: optional, shape [batch_size, label_dim]
        condition: optional, the conditional inputs,
            shape=[batch_size, condition_channel, w, h]
    Returns:
        out: loss function with shape [batch_size, target_channels, w, h]
            This should be averaged to get the mean loss for gradient descent.
    """

    sigma = torch.ones([images.shape[0], 1, 1, 1], device=images.device)
    y, augment_labels = (
        augment_pipe(images) if augment_pipe is not None else (images, None)
    )

    D_yn = net(x=condition, img_lr=None, sigma=sigma)
    loss = (D_yn - y) ** 2
    return loss
