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

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
from torch_utils import persistence

# ----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".


@persistence.persistent_class
class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, images, labels, augment_pipe=None):
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

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t**2) + self.beta_min * t).exp() - 1).sqrt()


# ----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".


@persistence.persistent_class
class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, labels, augment_pipe=None):
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

# ----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).
# weather
@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, img_clean, img_lr, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([img_clean.shape[0], 1, 1, 1], device=img_clean.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

        # augment for cifar-10 (unconditional)
        # y, augment_labels = augment_pipe(img_clean) if augment_pipe is not None else (img_clean, None)
        # augment for conditional generaiton
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


# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# UNet deterministic
@persistence.persistent_class
class RegressionLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, img_clean, img_lr, labels=None, augment_pipe=None):
        sigma = 0.
        
        img_tot = torch.cat((img_clean, img_lr), dim=1)
        _, augment_labels = augment_pipe(img_tot) if augment_pipe is not None else (img_tot, None)
        
        # Or we could even append Gaussian noise here to make it stochastic. Why waste it with zero vectors?
        zeros = torch.zeros_like(img_clean, device=img_clean.device) 
        img_pred = net(zeros, img_lr, sigma, labels, augment_labels=augment_labels)
        loss = ((img_clean - img_pred) ** 2)
        
        return loss


# ----------------------------------------------------------------------------


# P_mean=-1.2
@persistence.persistent_class
class ResLossv1:
    def __init__(self, P_mean=0.0, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

        # print('dist.get_rank', dist.get_rank)
        # if dist.get_rank == 0:
        #     breakpoint()

        with torch.no_grad():
            # resume_state_dump = '/lustre/fsw/nvresearch/mmardani/output/logs/edm_era5_cwb_12chans_fcn_4x_crop448_zarr_fulldata_unetregression/complex-mayfly_2023.04.25_16.42/output/training-state-006924.pt'
            resume_state_dump = "/lustre/fsw/nvresearch/mmardani/output/logs/edm_era5_cwb_12chans_fcn_4x_crop448_zarr_fulldata_unetregression/mindful-parakeet_2023.04.03_22.54/output/training-state-042650.pt"
            print(
                f'Loading training state from "{resume_state_dump}"...'
            )  # TODO print on rank zero only
            data = torch.load(resume_state_dump, map_location=torch.device("cpu"))
            self.unet = data["net"].cuda()
            # misc.copy_params_and_buffers(src_module=data['net'], dst_module=net, require_all=True)

    def __call__(self, net, img_clean, img_lr, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([img_clean.shape[0], 1, 1, 1], device=img_clean.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2

        # augment for conditional generaiton
        img_tot = torch.cat((img_clean, img_lr), dim=1)
        y_tot, augment_labels = (
            augment_pipe(img_tot) if augment_pipe is not None else (img_tot, None)
        )
        y = y_tot[:, : img_clean.shape[1], :, :]
        y_lr = y_tot[:, img_clean.shape[1] :, :, :]
        # form residual
        y_mean = self.unet(
            torch.zeros_like(y, device=img_clean.device),
            y_lr,
            sigma,
            labels,
            augment_labels=augment_labels,
        )
        y = y - y_mean

        latent = y + torch.randn_like(y) * sigma
        D_yn = net(latent, y_lr, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)

        return loss