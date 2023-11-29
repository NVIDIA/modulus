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

#----------------------------------------------------------------------------
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
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

#----------------------------------------------------------------------------
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
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

# #----------------------------------------------------------------------------
# # Improved loss function proposed in the paper "Elucidating the Design Space
# # of Diffusion-Based Generative Models" (EDM).
# # cifar10
# @persistence.persistent_class
# class EDMLoss:
#     def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
#         self.P_mean = P_mean
#         self.P_std = P_std
#         self.sigma_data = sigma_data

#     def __call__(self, net, images, labels=None, augment_pipe=None):
#         rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
#         sigma = (rnd_normal * self.P_std + self.P_mean).exp()
#         weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
#         y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
#         n = torch.randn_like(y) * sigma
#         D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
#         loss = weight * ((D_yn - y) ** 2)
#         return loss

# #----------------------------------------------------------------------------



#----------------------------------------------------------------------------
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
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        
        #augment for cifar-10 (unconditional)
        #y, augment_labels = augment_pipe(img_clean) if augment_pipe is not None else (img_clean, None)
        #augment for conditional generaiton
        img_tot = torch.cat((img_clean, img_lr), dim=1)
        y_tot, augment_labels = augment_pipe(img_tot) if augment_pipe is not None else (img_tot, None)
        y = y_tot[:,:img_clean.shape[1],:,:]
        y_lr = y_tot[:,img_clean.shape[1]:,:,:]
        
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, y_lr, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------




#----------------------------------------------------------------------------
# UNet deterministic 
@persistence.persistent_class
class RegressionLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, img_clean, img_lr, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([img_clean.shape[0], 1, 1, 1], device=img_clean.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = 1.0  #(sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        
        img_tot = torch.cat((img_clean, img_lr), dim=1)
        y_tot, augment_labels = augment_pipe(img_tot) if augment_pipe is not None else (img_tot, None)
        y = y_tot[:,:img_clean.shape[1],:,:]
        y_lr = y_tot[:,img_clean.shape[1]:,:,:]
        
        input = torch.zeros_like(y, device=img_clean.device)
        D_yn = net(input, y_lr, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        
        return loss

#----------------------------------------------------------------------------


#----------------------------------------------------------------------------
# Mixture loss: denoising score matching + regression 
@persistence.persistent_class
class MixtureLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, img_clean, img_lr, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([img_clean.shape[0], 1, 1, 1], device=img_clean.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()   #in the range [0,2], but signal is in [-4, 7]
        den_weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2   #in the range [5,2000] with high prob. if randn in [-1,+1]
        
        img_tot = torch.cat((img_clean, img_lr), dim=1)
        y_tot, augment_labels = augment_pipe(img_tot) if augment_pipe is not None else (img_tot, None)
        y = y_tot[:,:img_clean.shape[1],:,:]
        y_lr = y_tot[:,img_clean.shape[1]:,:,:]
        
        n = torch.randn_like(y) * sigma
        latent = y + n
        D_yn = net(latent, y_lr, sigma, labels, augment_labels=augment_labels)
        R_yn = net(latent*0.0, y_lr, sigma, labels, augment_labels=augment_labels)   #regression loss, zero stochasticity
        
        
        reg_weight = torch.tensor(5.0).cuda()
        loss = den_weight * ((D_yn - y) ** 2) + reg_weight * ((R_yn - y) ** 2)
        
        # dist.print0('**********************************************')
        # dist.print0('img-cleam-max', torch.max(img_clean))
        # dist.print0('img-cleam-min', torch.min(img_clean))
        # dist.print0('img-cleam-avg', torch.mean(img_clean))
        # dist.print0('img-cleam-var', torch.var(img_clean))
        # dist.print0('sigma', sigma)
        # dist.print0('den_weight', den_weight)
        # dist.print0('reg_weight', reg_weight)
        # dist.print0('loss_den', ((D_yn - y) ** 2).mean())
        # dist.print0('loss_reg', ((R_yn - y) ** 2).mean())
        # dist.print0('mixture-loss', torch.mean(loss))
        
        return loss
    

#P_mean=-1.2
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
            #resume_state_dump = '/lustre/fsw/nvresearch/mmardani/output/logs/edm_era5_cwb_12chans_fcn_4x_crop448_zarr_fulldata_unetregression/complex-mayfly_2023.04.25_16.42/output/training-state-006924.pt'
            resume_state_dump = '/lustre/fsw/nvresearch/mmardani/output/logs/edm_era5_cwb_12chans_fcn_4x_crop448_zarr_fulldata_unetregression/mindful-parakeet_2023.04.03_22.54/output/training-state-042650.pt'
            print(f'Loading training state from "{resume_state_dump}"...')  # TODO print on rank zero only
            data = torch.load(resume_state_dump, map_location=torch.device('cpu'))
            self.unet = data['net'].cuda()
            #misc.copy_params_and_buffers(src_module=data['net'], dst_module=net, require_all=True)
            

    def __call__(self, net, img_clean, img_lr, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([img_clean.shape[0], 1, 1, 1], device=img_clean.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        
        #augment for conditional generaiton
        img_tot = torch.cat((img_clean, img_lr), dim=1)
        y_tot, augment_labels = augment_pipe(img_tot) if augment_pipe is not None else (img_tot, None)
        y = y_tot[:,:img_clean.shape[1],:,:]
        y_lr = y_tot[:,img_clean.shape[1]:,:,:]
        
        # dist.print0('before ....')
        # dist.print0('y.max()', y.max())
        # dist.print0('y.min()', y.min())
        # dist.print0('y.mean()', y.mean())
        
        #form residual
        y_mean = self.unet(torch.zeros_like(y, device=img_clean.device), y_lr, sigma, labels, augment_labels=augment_labels)
        y = y - y_mean
        
        # dist.print0('mean ....')
        # dist.print0('ymean.max()', y_mean.max())
        # dist.print0('ymean.min()', y_mean.min())
        # dist.print0('ymean.mean()', y_mean.mean())
        
        # dist.print0('after ....')
        # dist.print0('y.max()', y.max())
        # dist.print0('y.min()', y.min())
        # dist.print0('y.mean()', y.mean())

        latent = y + torch.randn_like(y) * sigma
        D_yn = net(latent, y_lr, sigma, labels, augment_labels=augment_labels)     
        loss = weight * ((D_yn - y) ** 2)
        
        return loss



#P_mean=2.0
@persistence.persistent_class
class MixtureLossV1:
    def __init__(self, P_mean=2.4, P_std=3.6, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.p = 1.0
        self.reg_weight = 5.0

    def __call__(self, net, img_clean, img_lr, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([img_clean.shape[0], 1, 1, 1], device=img_clean.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        
        #augment for conditional generaiton
        img_tot = torch.cat((img_clean, img_lr), dim=1)
        y_tot, augment_labels = augment_pipe(img_tot) if augment_pipe is not None else (img_tot, None)
        y = y_tot[:,:img_clean.shape[1],:,:]
        y_lr = y_tot[:,img_clean.shape[1]:,:,:]
        
        #dropout loss: p*denoise_loss + (1-p)*regression_loss
        probability_tensor = torch.tensor([1 - self.p]).to(img_clean.device)
        binary_random_variable = torch.bernoulli(probability_tensor).item()
        # dist.print0('binary_random_variable', binary_random_variable)
        
        if binary_random_variable == 0:
            input = y + torch.randn_like(y) * sigma
            weight = weight
        else:
            input = torch.zeros_like(y, device=img_clean.device)
            weight = torch.full((img_clean.shape[0], 1, 1, 1), self.reg_weight, device=img_clean.device)
        
        D_yn = net(input, y_lr, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        
        return loss



#P_mean=-1.2
@persistence.persistent_class
class MixtureLossV2:
    def __init__(self, P_mean=0.0, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.p = 1.0
        self.reg_weight = 5.0

    def __call__(self, net, img_clean, img_lr, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([img_clean.shape[0], 1, 1, 1], device=img_clean.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        
        #augment for conditional generaiton
        img_tot = torch.cat((img_clean, img_lr), dim=1)
        y_tot, augment_labels = augment_pipe(img_tot) if augment_pipe is not None else (img_tot, None)
        y = y_tot[:,:img_clean.shape[1],:,:]
        y_lr = y_tot[:,img_clean.shape[1]:,:,:]
        
        #dropout loss: p*denoise_loss + (1-p)*regression_loss
        probability_tensor = torch.tensor([1 - self.p]).to(img_clean.device)
        binary_random_variable = torch.bernoulli(probability_tensor).item()
        # dist.print0('binary_random_variable', binary_random_variable)
        
        if binary_random_variable == 0:
            input = y + torch.randn_like(y) * sigma
            weight = weight
        else:
            input = torch.zeros_like(y, device=img_clean.device)
            weight = torch.full((img_clean.shape[0], 1, 1, 1), self.reg_weight, device=img_clean.device)
        
        D_yn = net(input, y_lr, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        
        return loss



#P_mean=-0.6
@persistence.persistent_class
class MixtureLossV3:
    def __init__(self, P_mean=0.0, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.p = 0.75
        self.reg_weight = 5.0

    def __call__(self, net, img_clean, img_lr, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([img_clean.shape[0], 1, 1, 1], device=img_clean.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        
        #augment for conditional generaiton
        img_tot = torch.cat((img_clean, img_lr), dim=1)
        y_tot, augment_labels = augment_pipe(img_tot) if augment_pipe is not None else (img_tot, None)
        y = y_tot[:,:img_clean.shape[1],:,:]
        y_lr = y_tot[:,img_clean.shape[1]:,:,:]
        
        #dropout loss: p*denoise_loss + (1-p)*regression_loss
        probability_tensor = torch.tensor([1 - self.p]).to(img_clean.device)
        binary_random_variable = torch.bernoulli(probability_tensor).item()
        # dist.print0('binary_random_variable', binary_random_variable)
        
        if binary_random_variable == 0:
            input = y + torch.randn_like(y) * sigma
            weight = weight
        else:
            input = torch.zeros_like(y, device=img_clean.device)
            weight = torch.full((img_clean.shape[0], 1, 1, 1), self.reg_weight, device=img_clean.device)
        
        D_yn = net(input, y_lr, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        
        return loss




#P_mean=-0.2
@persistence.persistent_class
class MixtureLossV4:
    def __init__(self, P_mean=0.0, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.p = 0.25
        self.reg_weight = 5.0

    def __call__(self, net, img_clean, img_lr, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([img_clean.shape[0], 1, 1, 1], device=img_clean.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        
        #augment for conditional generaiton
        img_tot = torch.cat((img_clean, img_lr), dim=1)
        y_tot, augment_labels = augment_pipe(img_tot) if augment_pipe is not None else (img_tot, None)
        y = y_tot[:,:img_clean.shape[1],:,:]
        y_lr = y_tot[:,img_clean.shape[1]:,:,:]   
        
        #dropout loss: p*denoise_loss + (1-p)*regression_loss
        probability_tensor = torch.tensor([1 - self.p]).to(img_clean.device)
        binary_random_variable = torch.bernoulli(probability_tensor).item()
        # dist.print0('binary_random_variable', binary_random_variable)
        
        if binary_random_variable == 0:
            input = y + torch.randn_like(y) * sigma
            weight = weight
        else:
            input = torch.zeros_like(y, device=img_clean.device)
            weight = torch.full((img_clean.shape[0], 1, 1, 1), self.reg_weight, device=img_clean.device)
        
        D_yn = net(input, y_lr, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        
        return loss

#----------------------------------------------------------------------------


#P_mean=-1.2
@persistence.persistent_class
class MixtureLossV5:
    def __init__(self, P_mean=0.0, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.p = 0.0
        self.reg_weight = 5.0

    def __call__(self, net, img_clean, img_lr, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([img_clean.shape[0], 1, 1, 1], device=img_clean.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        
        #augment for conditional generaiton
        img_tot = torch.cat((img_clean, img_lr), dim=1)
        y_tot, augment_labels = augment_pipe(img_tot) if augment_pipe is not None else (img_tot, None)
        y = y_tot[:,:img_clean.shape[1],:,:]
        y_lr = y_tot[:,img_clean.shape[1]:,:,:]
        
        #dropout loss: p*denoise_loss + (1-p)*regression_loss
        probability_tensor = torch.tensor([1 - self.p]).to(img_clean.device)
        binary_random_variable = torch.bernoulli(probability_tensor).item()
        # dist.print0('binary_random_variable', binary_random_variable)
        
        if binary_random_variable == 0:
            input = y + torch.randn_like(y) * sigma
            weight = weight
        else:
            input = torch.zeros_like(y, device=img_clean.device)
            weight = torch.full((img_clean.shape[0], 1, 1, 1), self.reg_weight, device=img_clean.device)
        
        D_yn = net(input, y_lr, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        
        return loss
    



#old loss from previous codebase
@persistence.persistent_class
class OurLoss:
    def __init__(self, mu=-1.2, sigma=1.2, signal_std=0.5, importance=0):
        self.mu = mu
        self.sigma = sigma
        self.signal_std = signal_std
        self.importance = importance

    def accumulate_gradients(self, net, img_clean, img_lr, labels, augment_pipe=None):
        sampling_sigma = self.sigma / np.sqrt(1 - self.importance)
        noise_std_log = torch.randn([img_clean.shape[0], 1, 1, 1], device=img_clean.device) * sampling_sigma + self.mu
        noise_std = noise_std_log.exp()

        net_kwargs = dnnlib.EasyDict()
        if augment_pipe is not None:
            img_clean, net_kwargs.augment_labels = augment_pipe(img_clean)

        noisy_std = (self.signal_std ** 2 + noise_std ** 2).sqrt()
        img_noisy = img_clean + torch.randn_like(img_clean) * noise_std
        img_denoised = net(img_noisy, img_lr, noise_std.flatten(), labels=labels, **net_kwargs)
        loss = img_clean - img_denoised
        loss = loss / (self.signal_std * noise_std / noisy_std)

        loss = loss.square().mean([1, 2, 3])
        if self.importance != 0:
            loss = loss * (-0.5 * self.importance * (((noise_std_log - self.mu) / self.sigma) ** 2)).exp()
        training_stats.report('Loss/loss', loss)
        loss.mean().backward()