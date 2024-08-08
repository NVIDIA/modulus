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
import tqdm
import nvtx

from modulus.utils.generative import StackedRandomGenerator

def regression_step(
    net: torch.nn.Module,
    img_lr: torch.Tensor,
    latents_shape: torch.Size
) -> torch.Tensor:
    """
    Given a low-res input, performs a regression step to produce ensemble mean.
    This function performs the regression on a single instance and then replicates
    the results across the batch dimension.

    Args:
        net (torch.nn.Module): U-Net model for regression.
        img_lr (torch.Tensor): Low-resolution input.
        latents_shape (torch.Size): Shape of the latent representation. Typically
        (batch_size, out_channels, image_shape_x, image_shape_y).


    Returns:
        torch.Tensor: Predicted output at the next time step.
    """
    # Create a tensor of zeros with the given shape and move it to the appropriate device
    x_hat = torch.zeros(latents_shape, dtype=torch.float64, device=net.device)
    t_hat = torch.tensor(1.0, dtype=torch.float64, device=net.device)

    # Perform regression on a single batch element
    with torch.inference_mode():
        x = net(x_hat[0:1], img_lr, t_hat)

    # If the batch size is greater than 1, repeat the prediction
    if x_hat.shape[0] > 1:
        x  = x.repeat([d if i == 0 else 1 for i, d in enumerate(x_hat.shape)])

    return x

def diffusion_step(
    net: torch.nn.Module,
    sampler_fn: callable,
    seed_batch_size: int,
    img_shape: tuple,
    img_out_channels: int,
    rank_batches: list,
    img_lr: torch.Tensor,
    rank: int,
    device: torch.device,
    hr_mean: torch.Tensor = None
) -> torch.Tensor:

    """
    Generate images using diffusion techniques as described in the relevant paper.

    Args:
        net (torch.nn.Module): The diffusion model network.
        sampler_fn (callable): Function used to sample images from the diffusion model.
        seed_batch_size (int): Number of seeds per batch.
        img_shape (tuple): Shape of the images, (height, width).
        img_out_channels (int): Number of output channels for the image.
        rank_batches (list): List of batches of seeds to process.
        img_lr (torch.Tensor): Low-resolution input image.
        rank (int): Rank of the current process for distributed processing.
        device (torch.device): Device to perform computations.
        mean_hr (torch.Tensor, optional): High-resolution mean tensor, to be used as an additional input. By default None.

    Returns:
        torch.Tensor: Generated images concatenated across batches.
    """

    img_lr = img_lr.to(memory_format=torch.channels_last)

    # Handling of the high-res mean
    additional_args = {}
    if hr_mean:
        additional_args["mean_hr"] =  hr_mean

    # Loop over batches
    all_images = []
    for batch_seeds in tqdm.tqdm(rank_batches, unit="batch", disable=(rank != 0)):
        with nvtx.annotate(f"generate {len(all_images)}", color="rapids"):
            batch_size = len(batch_seeds)
            if batch_size == 0:
                continue

            # Initialize random generator, and generate latents
            rnd = StackedRandomGenerator(device, batch_seeds)
            latents = rnd.randn(
                    [
                        seed_batch_size,
                        img_out_channels,
                        img_shape[1],
                        img_shape[0],
                    ],
                    device=device,
            ).to(memory_format=torch.channels_last)

            with torch.inference_mode():
                images = sampler_fn(
                    net, latents, img_lr, randn_like=rnd.randn_like, **additional_args
                )
            all_images.append(images)
    return torch.cat(all_images)
