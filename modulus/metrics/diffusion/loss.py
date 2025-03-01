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


"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from modulus.utils.patching import RandomPatching2D


class VPLoss:
    """
    Loss function corresponding to the variance preserving (VP) formulation.

    Parameters
    ----------
    beta_d: float, optional
        Coefficient for the diffusion process, by default 19.9.
    beta_min: float, optional
        Minimum bound, by defaults 0.1.
    epsilon_t: float, optional
        Small positive value, by default 1e-5.

    Note:
    -----
    Reference: Song, Y., Sohl-Dickstein, J., Kingma, D.P., Kumar, A., Ermon, S. and
    Poole, B., 2020. Score-based generative modeling through stochastic differential
    equations. arXiv preprint arXiv:2011.13456.

    """

    def __init__(
        self, beta_d: float = 19.9, beta_min: float = 0.1, epsilon_t: float = 1e-5
    ):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(
        self,
        net: torch.nn.Module,
        images: torch.Tensor,
        labels: torch.Tensor,
        augment_pipe: Optional[Callable] = None,
    ):
        """
        Calculate and return the loss corresponding to the variance preserving (VP)
        formulation.

        The method adds random noise to the input images and calculates the loss as the
        square difference between the network's predictions and the input images.
        The noise level is determined by 'sigma', which is computed as a function of
        'epsilon_t' and random values. The calculated loss is weighted based on the
        inverse of 'sigma^2'.

        Parameters:
        ----------
        net: torch.nn.Module
            The neural network model that will make predictions.

        images: torch.Tensor
            Input images to the neural network.

        labels: torch.Tensor
            Ground truth labels for the input images.

        augment_pipe: callable, optional
            An optional data augmentation function that takes images as input and
            returns augmented images. If not provided, no data augmentation is applied.

        Returns:
        -------
        torch.Tensor
            A tensor representing the loss calculated based on the network's
            predictions.
        """
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma**2
        y, augment_labels = (
            augment_pipe(images) if augment_pipe is not None else (images, None)
        )
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(
        self, t: Union[float, torch.Tensor]
    ):  # NOTE: also exists in preconditioning
        """
        Compute the sigma(t) value for a given t based on the VP formulation.

        The function calculates the noise level schedule for the diffusion process based
        on the given parameters `beta_d` and `beta_min`.

        Parameters
        ----------
        t : Union[float, torch.Tensor]
            The timestep or set of timesteps for which to compute sigma(t).

        Returns
        -------
        torch.Tensor
            The computed sigma(t) value(s).
        """
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t**2) + self.beta_min * t).exp() - 1).sqrt()


class VELoss:
    """
    Loss function corresponding to the variance exploding (VE) formulation.

    Parameters
    ----------
    sigma_min : float
        Minimum supported noise level, by default 0.02.
    sigma_max : float
        Maximum supported noise level, by default 100.0.

    Note:
    -----
    Reference: Song, Y., Sohl-Dickstein, J., Kingma, D.P., Kumar, A., Ermon, S. and
    Poole, B., 2020. Score-based generative modeling through stochastic differential
    equations. arXiv preprint arXiv:2011.13456.
    """

    def __init__(self, sigma_min: float = 0.02, sigma_max: float = 100.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, labels, augment_pipe=None):
        """
        Calculate and return the loss corresponding to the variance exploding (VE)
        formulation.

        The method adds random noise to the input images and calculates the loss as the
        square difference between the network's predictions and the input images.
        The noise level is determined by 'sigma', which is computed as a function of
        'sigma_min' and 'sigma_max' and random values. The calculated loss is weighted
        based on the inverse of 'sigma^2'.

        Parameters:
        ----------
        net: torch.nn.Module
            The neural network model that will make predictions.

        images: torch.Tensor
            Input images to the neural network.

        labels: torch.Tensor
            Ground truth labels for the input images.

        augment_pipe: callable, optional
            An optional data augmentation function that takes images as input and
            returns augmented images. If not provided, no data augmentation is applied.

        Returns:
        -------
        torch.Tensor
            A tensor representing the loss calculated based on the network's
            predictions.
        """
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma**2
        y, augment_labels = (
            augment_pipe(images) if augment_pipe is not None else (images, None)
        )
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss


class EDMLoss:
    """
    Loss function proposed in the EDM paper.

    Parameters
    ----------
    P_mean: float, optional
        Mean value for `sigma` computation, by default -1.2.
    P_std: float, optional:
        Standard deviation for `sigma` computation, by default 1.2.
    sigma_data: float, optional
        Standard deviation for data, by default 0.5.

    Note
    ----
    Reference: Karras, T., Aittala, M., Aila, T. and Laine, S., 2022. Elucidating the
    design space of diffusion-based generative models. Advances in Neural Information
    Processing Systems, 35, pp.26565-26577.
    """

    def __init__(
        self, P_mean: float = -1.2, P_std: float = 1.2, sigma_data: float = 0.5
    ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, condition=None, labels=None, augment_pipe=None):
        """
        Calculate and return the loss corresponding to the EDM formulation.

        The method adds random noise to the input images and calculates the loss as the
        square difference between the network's predictions and the input images.
        The noise level is determined by 'sigma', which is computed as a function of
        'P_mean' and 'P_std' random values. The calculated loss is weighted as a
        function of 'sigma' and 'sigma_data'.

        Parameters:
        ----------
        net: torch.nn.Module
            The neural network model that will make predictions.

        images: torch.Tensor
            Input images to the neural network.

        labels: torch.Tensor
            Ground truth labels for the input images.

        augment_pipe: callable, optional
            An optional data augmentation function that takes images as input and
            returns augmented images. If not provided, no data augmentation is applied.

        Returns:
        -------
        torch.Tensor
            A tensor representing the loss calculated based on the network's
            predictions.
        """
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = (
            augment_pipe(images) if augment_pipe is not None else (images, None)
        )
        n = torch.randn_like(y) * sigma
        if condition is not None:
            D_yn = net(
                y + n,
                sigma,
                condition=condition,
                class_labels=labels,
                augment_labels=augment_labels,
            )
        else:
            D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss


class EDMLossSR:
    """
    Variation of the loss function proposed in the EDM paper for Super-Resolution.

    Parameters
    ----------
    P_mean: float, optional
        Mean value for `sigma` computation, by default -1.2.
    P_std: float, optional:
        Standard deviation for `sigma` computation, by default 1.2.
    sigma_data: float, optional
        Standard deviation for data, by default 0.5.

    Note
    ----
    Reference: Mardani, M., Brenowitz, N., Cohen, Y., Pathak, J., Chen, C.Y.,
    Liu, C.C.,Vahdat, A., Kashinath, K., Kautz, J. and Pritchard, M., 2023.
    Generative Residual Diffusion Modeling for Km-scale Atmospheric Downscaling.
    arXiv preprint arXiv:2309.15214.
    """

    def __init__(
        self, P_mean: float = -1.2, P_std: float = 1.2, sigma_data: float = 0.5
    ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, img_clean, img_lr, labels=None, augment_pipe=None):
        """
        Calculate and return the loss corresponding to the EDM formulation.

        The method adds random noise to the input images and calculates the loss as the
        square difference between the network's predictions and the input images.
        The noise level is determined by 'sigma', which is computed as a function of
        'P_mean' and 'P_std' random values. The calculated loss is weighted as a
        function of 'sigma' and 'sigma_data'.

        Parameters:
        ----------
        net: torch.nn.Module
            The neural network model that will make predictions.

        images: torch.Tensor
            Input images to the neural network.

        labels: torch.Tensor
            Ground truth labels for the input images.

        augment_pipe: callable, optional
            An optional data augmentation function that takes images as input and
            returns augmented images. If not provided, no data augmentation is applied.

        Returns:
        -------
        torch.Tensor
            A tensor representing the loss calculated based on the network's
            predictions.
        """
        rnd_normal = torch.randn([img_clean.shape[0], 1, 1, 1], device=img_clean.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

        # augment for conditional generation
        img_tot = torch.cat((img_clean, img_lr), dim=1)
        y_tot, augment_labels = (
            augment_pipe(img_tot) if augment_pipe is not None else (img_tot, None)
        )
        y = y_tot[:, : img_clean.shape[1], :, :]
        y_lr = y_tot[:, img_clean.shape[1] :, :, :]

        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, y_lr, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss


class RegressionLoss:
    """
    Regression loss function for the U-Net for deterministic predictions.
    Note: this loss does not apply any reduction.

    Parameters
    ----------
    sigma_data: float, optional
        Standard deviation for data. Deprecated and ignored. By default 0.5.

    Note
    ----
    Reference: Mardani, M., Brenowitz, N., Cohen, Y., Pathak, J., Chen, C.Y.,
    Liu, C.C.,Vahdat, A., Kashinath, K., Kautz, J. and Pritchard, M., 2023.
    Generative Residual Diffusion Modeling for Km-scale Atmospheric Downscaling.
    arXiv preprint arXiv:2309.15214.
    """

    def __init__(self, sigma_data: float = 0.5):
        self.sigma_data = sigma_data

    def __call__(self, net, img_clean, img_lr, labels=None, augment_pipe=None):
        """
        Calculate and return the regression loss for the U-Net for
        deterministic predictions.

        Parameters:
        ----------
        net: torch.nn.Module
            The neural network model that will make predictions.

        img_clean: torch.Tensor
            Input images (high resolution). Used as ground truth and for data
            augmentation if 'augment_pipe' is provided.

        img_lr: torch.Tensor
            Input images (low resolution). Used as input to the neural network.

        labels: torch.Tensor
            Not used. Only present for compatibility with other loss functions.

        augment_pipe: callable, optional
            An optional data augmentation function that takes concatenated high
            and low resolution images as input and returns augmented images. If
            not provided, no data augmentation is applied.

        Returns:
        -------
        torch.Tensor
            A tensor representing the per-sample element-wise squared
            difference between the network's  predictions and the (possibly
            data-augmented by 'augment_pipe') high resolution images. This
            tensor can then be used to compute the loss by sum, mean, or other
            appropriate reduction function.
        """
        weight = (
            1.0  # (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        )

        img_tot = torch.cat((img_clean, img_lr), dim=1)
        y_tot, augment_labels = (
            augment_pipe(img_tot) if augment_pipe is not None else (img_tot, None)
        )
        y = y_tot[:, : img_clean.shape[1], :, :]
        y_lr = y_tot[:, img_clean.shape[1] :, :, :]

        zero_input = torch.zeros_like(y, device=img_clean.device)
        D_yn = net(
            x=zero_input, img_lr=y_lr, force_fp32=False, augment_labels=augment_labels
        )
        loss = weight * ((D_yn - y) ** 2)

        return loss


class ResidualLoss:
    """
    Mixture loss function for denoising score matching.

    This class implements a loss function that combines deterministic
    regression with denoising score matching for improved super-resolution. It
    uses a pre-trained regression network to compute residuals before applying
    the diffusion process.

    Parameters
    ----------
    regression_net : torch.nn.Module
        Pre-trained regression network used to compute residuals.
    img_shape_y : int
        Height of the input images.
    img_shape_x : int
        Width of the input images.
    P_mean : float, optional
        Mean value for noise level computation, by default 0.0.
    P_std : float, optional
        Standard deviation for noise level computation, by default 1.2.
    sigma_data : float, optional
        Standard deviation for data weighting, by default 0.5.
    hr_mean_conditioning : bool, optional
        Whether to use high-resolution mean for conditioning, by default False.

    Attributes
    ----------
    unet : torch.nn.Module
        The regression network used for computing residuals.
    P_mean : float
        Mean value for noise level computation.
    P_std : float
        Standard deviation for noise level computation.
    sigma_data : float
        Standard deviation for data weighting.
    img_shape_x : int
        Width of the input images.
    img_shape_y : int
        Height of the input images.
    hr_mean_conditioning : bool
        Flag indicating whether to use high-resolution mean for conditioning.

    Note
    ----
    Reference: Mardani, M., Brenowitz, N., Cohen, Y., Pathak, J., Chen, C.Y.,
    Liu, C.C., Vahdat, A., Kashinath, K., Kautz, J. and Pritchard, M., 2023.
    Generative Residual Diffusion Modeling for Km-scale Atmospheric
    Downscaling. arXiv preprint arXiv:2309.15214.
    """

    def __init__(
        self,
        regression_net: torch.nn.Module,
        img_shape_y: int,
        img_shape_x: int,
        P_mean: float = 0.0,
        P_std: float = 1.2,
        sigma_data: float = 0.5,
        hr_mean_conditioning: bool = False,
    ):
        self.unet = regression_net
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.img_shape_x = img_shape_x
        self.img_shape_y = img_shape_y
        self.hr_mean_conditioning = hr_mean_conditioning

    def __call__(
        self,
        net: torch.nn.Module,
        img_clean: Tensor,
        img_lr: Tensor,
        patching: Optional[RandomPatching2D] = None,
        labels: Optional[Tensor] = None,
        lead_time_label: Optional[Tensor] = None,
        augment_pipe: Optional[Callable[[Tensor], Tuple[Tensor, Tensor]]] = None,
    ):
        """
        Calculate and return the loss for denoising score matching.

        This method computes a mixture loss that combines deterministic
        regression with denoising score matching. It first computes residuals
        using the regression network, then applies the diffusion process to
        these residuals.

        Parameters
        ----------
        net : torch.nn.Module
            The neural network model that will make predictions.
        img_clean : Tensor
            High-resolution input images of shape (batch_size, c_hr,
            img_shape_y, img_shape_x).
        img_lr : Tensor
            Low-resolution input images of shape (batch_size, c_lr,
            img_shape_y, img_shape_x).
        patching : Optional[RandomPatching2D], optional
            Patching strategy for processing large images, by default None. See
            :class:`modulus.utils.patching.RandomPatching2D` for details.
        labels : Optional[Tensor], optional
            Ground truth labels for the input images, by default None.
        lead_time_label : Optional[Tensor], optional
            Lead time labels for temporal predictions, by default None.
        augment_pipe : Optional[Callable[[Tensor], Tuple[Tensor, Tensor]]]
            Data augmentation function that takes images as input and returns
            augmented images and labels, by default None.

        Returns
        -------
        Tensor
            A tensor representing the computed loss with shape (batch_size [*
            patch_num], c_hr, patch_shape_y, patch_shape_x).

        Raises
        ------
        ValueError
            If patching is provided but is not an instance of RandomPatching2D.
        """

        # Safety check: enforce patching object
        if patching and not isinstance(patching, RandomPatching2D):
            raise ValueError("patching must be a 'RandomPatching2D' object.")
        # Safety check: enforce shapes
        if (
            img_clean.shape[0] != img_lr.shape[0]
            or img_clean.shape[2:] != img_lr.shape[2:]
        ):
            raise ValueError(
                f"Shape mismatch between img_clean {img_clean.shape} and "
                f"img_lr {img_lr.shape}. "
                f"Batch size, height and width must match."
            )

        rnd_normal = torch.randn([img_clean.shape[0], 1, 1, 1], device=img_clean.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

        # augment for conditional generation
        img_tot = torch.cat((img_clean, img_lr), dim=1)
        y_tot, augment_labels = (
            augment_pipe(img_tot) if augment_pipe is not None else (img_tot, None)
        )
        y = y_tot[:, : img_clean.shape[1], :, :]
        y_lr = y_tot[:, img_clean.shape[1] :, :, :]
        y_lr_res = y_lr
        batch_size = y.shape[0]

        # form residual
        if lead_time_label is not None:
            y_mean = self.unet(
                torch.zeros_like(y, device=img_clean.device),
                y_lr_res,
                sigma,
                labels,
                lead_time_label=lead_time_label,
                augment_labels=augment_labels,
            )
        else:
            y_mean = self.unet(
                torch.zeros_like(y, device=img_clean.device),
                y_lr_res,
                sigma,
                labels,
                augment_labels=augment_labels,
            )

        y = y - y_mean

        if self.hr_mean_conditioning:
            y_lr = torch.cat((y_mean, y_lr), dim=1).contiguous()

        # patchified training
        # conditioning: cat(y_mean, y_lr, input_interp, pos_embd), 4+12+100+4
        if patching:
            # Patched residual
            # (batch_size * patch_num, c_out, patch_shape_y, patch_shape_x)
            y_patched = patching.apply(input=y)
            # Patched conditioning on y_lr and interp(img_lr)
            # (batch_size * patch_num, 2*c_in, patch_shape_y, patch_shape_x)
            y_lr_patched = patching.apply(input=y_lr, additional_input=img_lr)
            # Patched global grid coordinates (pos_embd)
            # (batch_size * patch_num, 2, patch_shape_y, patch_shape_x)
            global_index = patching.global_index(batch_size=batch_size)
            y = y_patched
            y_lr = y_lr_patched
        else:
            global_index = None

        # Noise
        rnd_normal = torch.randn([y.shape[0], 1, 1, 1], device=img_clean.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

        # Input + noise
        latent = y + torch.randn_like(y) * sigma

        if lead_time_label is not None:
            D_yn = net(
                latent,
                y_lr,
                sigma,
                labels,
                global_index=global_index,
                lead_time_label=lead_time_label,
                augment_labels=augment_labels,
            )
        else:
            D_yn = net(
                latent,
                y_lr,
                sigma,
                labels,
                global_index=global_index,
                augment_labels=augment_labels,
            )
        loss = weight * ((D_yn - y) ** 2)

        return loss


class VELoss_dfsr:
    """
    Loss function for dfsr model, modified from class VELoss.

    Parameters
    ----------
    beta_start : float
        Noise level at the initial step of the forward diffusion process, by default 0.0001.
    beta_end : float
        Noise level at the Final step of the forward diffusion process, by default 0.02.
    num_diffusion_timesteps : int
        Total number of forward/backward diffusion steps, by default 1000.


    Note:
    -----
    Reference: Ho J, Jain A, Abbeel P. Denoising diffusion probabilistic models.
    Advances in neural information processing systems. 2020;33:6840-51.
    """

    def __init__(
        self,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        num_diffusion_timesteps: int = 1000,
    ):
        # scheduler for diffusion:
        self.beta_schedule = "linear"
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.num_diffusion_timesteps = num_diffusion_timesteps
        betas = self.get_beta_schedule(
            beta_schedule=self.beta_schedule,
            beta_start=self.beta_start,
            beta_end=self.beta_end,
            num_diffusion_timesteps=self.num_diffusion_timesteps,
        )
        self.betas = torch.from_numpy(betas).float()
        self.num_timesteps = betas.shape[0]

    def get_beta_schedule(
        self, beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps
    ):
        """
        Compute the variance scheduling parameters {beta(0), ..., beta(t), ..., beta(T)}
        based on the VP formulation.

        beta_schedule: str
            Method to construct the sequence of beta(t)'s.
        beta_start: float
            Noise level at the initial step of the forward diffusion process, e.g., beta(0)
        beta_end: float
            Noise level at the final step of the forward diffusion process, e.g., beta(T)
        num_diffusion_timesteps: int
            Total number of forward/backward diffusion steps
        """

        def sigmoid(x):
            return 1 / (np.exp(-x) + 1)

        if beta_schedule == "quad":
            betas = (
                np.linspace(
                    beta_start**0.5,
                    beta_end**0.5,
                    num_diffusion_timesteps,
                    dtype=np.float64,
                )
                ** 2
            )
        elif beta_schedule == "linear":
            betas = np.linspace(
                beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
            )
        elif beta_schedule == "const":
            betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
        elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
            betas = 1.0 / np.linspace(
                num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
            )
        elif beta_schedule == "sigmoid":
            betas = np.linspace(-6, 6, num_diffusion_timesteps)
            betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
        else:
            raise NotImplementedError(beta_schedule)
        if betas.shape != (num_diffusion_timesteps,):
            raise ValueError(
                f"Expected betas to have shape ({num_diffusion_timesteps},), "
                f"but got {betas.shape}"
            )
        return betas

    def __call__(self, net, images, labels, augment_pipe=None):
        """
        Calculate and return the loss corresponding to the variance preserving
        formulation.

        The method adds random noise to the input images and calculates the loss as the
        square difference between the network's predictions and the noise samples added
        to the t-th step of the diffusion process.
        The noise level is determined by 'beta_t' based on the given parameters 'beta_start',
        'beta_end' and the current diffusion timestep t.

        Parameters:
        ----------
        net: torch.nn.Module
            The neural network model that will make predictions.

        images: torch.Tensor
            Input fluid flow data samples to the neural network.

        labels: torch.Tensor
            Ground truth labels for the input fluid flow data samples. Not required for dfsr.

        augment_pipe: callable, optional
            An optional data augmentation function that takes images as input and
            returns augmented images. If not provided, no data augmentation is applied.

        Returns:
        -------
        torch.Tensor
            A tensor representing the loss calculated based on the network's
            predictions.
        """
        t = torch.randint(
            low=0, high=self.num_timesteps, size=(images.size(0) // 2 + 1,)
        ).to(images.device)
        t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[: images.size(0)]
        e = torch.randn_like(images)
        b = self.betas.to(images.device)
        a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
        x = images * a.sqrt() + e * (1.0 - a).sqrt()

        output = net(x, t, labels)
        loss = (e - output).square()

        return loss


class RegressionLossCE:
    """
    A regression loss function for the GEFS-HRRR model with probability
    channels. Adapted from
    :class:`modulus.metrics.diffusion.loss.RegressionLoss` to train
    deterministic regression model. In this version,
    probability channels are evaluated using CrossEntropyLoss instead of
    squared error.
    Note: this loss does not apply any reduction.

    Parameters
    ----------
    P_mean: float, optional
        Mean value for `sigma` computation, by default -1.2.
    P_std: float, optional:
        Standard deviation for `sigma` computation, by default 1.2.
    sigma_data: float, optional
        Standard deviation for data, by default 0.5.
    prob_channels: list, optional
        A index list of output probability channels.

    Note
    ----
    Reference: Mardani, M., Brenowitz, N., Cohen, Y., Pathak, J., Chen, C.Y.,
    Liu, C.C.,Vahdat, A., Kashinath, K., Kautz, J. and Pritchard, M., 2023.
    Generative Residual Diffusion Modeling for Km-scale Atmospheric Downscaling.
    arXiv preprint arXiv:2309.15214.
    """

    def __init__(
        self,
        prob_channels: list = [4, 5, 6, 7, 8],
    ):
        self.entropy = torch.nn.CrossEntropyLoss(reduction="none")
        self.prob_channels = prob_channels

    def __call__(
        self,
        net,
        img_clean,
        img_lr,
        lead_time_label=None,
        labels=None,
        augment_pipe=None,
    ):
        """
        Calculate and return the loss for the U-Net for deterministic
        predictions.

        Parameters:
        ----------
        net: torch.nn.Module
            The neural network model that will make predictions.

        img_clean: torch.Tensor
            Input images (high resolution). Used as ground truth and for data
            augmentation if 'augment_pipe' is provided.

        img_lr: torch.Tensor
            Input images (low resolution). Used as input to the neural network.

        lead_time_label: torch.Tensor
            Lead time labels for input batches.

        labels: torch.Tensor
            Not used. Only present for compatibility with other loss functions.

        augment_pipe: callable, optional
            An optional data augmentation function that takes concatenated high
            and low resolution images as input and returns augmented images. If
            not provided, no data augmentation is applied.

        Returns:
        -------
        torch.Tensor
            A tensor representing the per-sample element-wise loss between the
            network's predictions and the (possibly
            data-augmented by `augment_pipe`) high resolution images. Loss for
            the probability channels is computed using cross entropy, and loss
            for the scalar channels is computed using squared error. Both are
            then concatenated along the channel dimension. This tensor can then
            be used to compute the loss by sum, mean, or other appropriate
            reduction function.
        """
        all_channels = list(range(img_clean.shape[1]))  # [0, 1, 2, ..., 10]
        scalar_channels = [
            item for item in all_channels if item not in self.prob_channels
        ]
        weight = (
            1.0  # (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        )

        img_tot = torch.cat((img_clean, img_lr), dim=1)
        y_tot, augment_labels = (
            augment_pipe(img_tot) if augment_pipe is not None else (img_tot, None)
        )
        y = y_tot[:, : img_clean.shape[1], :, :]
        y_lr = y_tot[:, img_clean.shape[1] :, :, :]

        input = torch.zeros_like(y, device=img_clean.device)

        if lead_time_label is not None:
            D_yn = net(
                input,
                y_lr,
                lead_time_label=lead_time_label,
                augment_labels=augment_labels,
            )
        else:
            D_yn = net(
                input,
                y_lr,
                lead_time_label=lead_time_label,
                augment_labels=augment_labels,
            )
        loss1 = weight * (D_yn[:, scalar_channels] - y[:, scalar_channels]) ** 2
        loss2 = (
            weight
            * self.entropy(D_yn[:, self.prob_channels], y[:, self.prob_channels])[
                :, None
            ]
        )
        loss = torch.cat((loss1, loss2), dim=1)
        return loss
