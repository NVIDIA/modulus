import os
from utils.YParams import YParams
import numpy as np
import torch
import glob

from utils.diffusions.generate import edm_sampler
from utils.diffusions.networks import get_preconditioned_architecture
from utils.diffusions.power_ema import sigma_rel_to_gamma, solve_weights
from utils.data_loader_hrrr_era5 import get_dataset


class EDMRunner():

    def __init__(self, params, checkpoint_path, device=None, sampler_args={}):

        if device is None:
            device = torch.device('cuda:0')

        self.sampler_args = sampler_args
        dataset_obj = get_dataset(params, train=True)

        _, hrrr_channels = dataset_obj._get_hrrr_channel_names()
        self.input_channels = hrrr_channels if params.input_channels == 'all' else params.input_channels
        self.input_channel_indices = list(range(len(hrrr_channels)))
        self.diffusion_channels = hrrr_channels if params.diffusion_channels == 'all' else params.diffusion_channels
        self.diffusion_channel_indices = list(range(len(hrrr_channels)))


        '''
        _, kept_hrrr_channels = dataset_train._get_hrrr_channel_names()
    hrrr_channels = kept_hrrr_channels

    diffusion_channels = params.diffusion_channels
    if diffusion_channels == "all":
        diffusion_channels = hrrr_channels
        '''

        invariant_array = dataset_obj._get_invariants()
        self.invariant_tensor = torch.from_numpy(invariant_array).to(device)
        self.invariant_tensor = self.invariant_tensor.unsqueeze(0)

        resolution = 512
        n_target_channels = len(self.diffusion_channel_indices)
        n_input_channels = 2*len(self.input_channel_indices) + len(params.invariants)

        self.net = get_preconditioned_architecture(
            name="ddpmpp-cwb-v0",
            resolution=resolution,
            target_channels=n_target_channels,
            conditional_channels=n_input_channels,
            label_dim=0,
            spatial_embedding=params.spatial_pos_embed,
            attn_resolutions=params.attn_resolutions,
        ).requires_grad_(False)

        assert self.net.sigma_min < self.net.sigma_max
        
        # Load pretrained weights
        chkpt = torch.load(checkpoint_path, weights_only=True)
        self.net.load_state_dict(chkpt["net"], strict=True)
        self.net = self.net.to(device)
        self.params = params

        print("n target channels: ", n_target_channels)


    def run(self, hrrr_0):

        with torch.no_grad():

            ensemble_size, c, h, w = hrrr_0[:, self.diffusion_channel_indices, :, :].shape
            latents = torch.randn(ensemble_size, c, h, w, device=hrrr_0.device, dtype=hrrr_0.dtype)

            if ensemble_size > 1 and self.invariant_tensor.shape[0] != ensemble_size:
                self.invariant_tensor = self.invariant_tensor.expand(ensemble_size, -1, -1, -1)
            condition = torch.cat((hrrr_0, self.invariant_tensor), dim=1)
            output_images = edm_sampler(self.net, latents=latents, condition=condition, **self.sampler_args)
            
        
        return output_images, self.diffusion_channels
