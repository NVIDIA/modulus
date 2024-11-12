import os
import time
import copy
import json
from utils.YParams import YParams
import pickle
import psutil
import numpy as np
import torch
import dnnlib
import glob
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc
from utils.diffusions.generate import edm_sampler
import utils.diffusions.networks
import utils.diffusions.losses
from utils.diffusions.power_ema import sigma_rel_to_gamma, solve_weights
from utils.data_loader_hrrr_era5 import get_data_loader, get_dataset, worker_init
import utils.img_utils
from torchvision import transforms
import matplotlib.pyplot as plt
from networks.swinv2_hrrr import swinv2net


class EDMRunner():

    def __init__(self, params, resume_pkl=None, posthoc_ema_sigma='None', ema=True):

        device = torch.device('cuda:0')

        dataset_obj = get_dataset(params, train=True)
        #hrrr_channels = dataset_obj.hrrr_channels.values.tolist()[:-1]
        base_hrrr_channels, hrrr_channels = dataset_obj._get_hrrr_channel_names()
        self.input_channels = params.input_channels
        if self.input_channels == 'all':
            self.input_channels = hrrr_channels
            self.input_channel_indices = list(range(len(hrrr_channels)))
        else:
            self.input_channel_indices = [hrrr_channels.index(channel) for channel in self.input_channels]
        self.diffusion_channels = params.diffusion_channels
        if self.diffusion_channels == 'all':
            self.diffusion_channels = hrrr_channels
            self.diffusion_channel_indices = list(range(len(hrrr_channels)))
        else:
            self.diffusion_channel_indices = [hrrr_channels.index(channel) for channel in self.diffusion_channels]
        

        if len(params.invariants) > 0:
            #conditional_channels += len(params.invariants)
            invariant_array = dataset_obj._get_invariants()
            self.invariant_tensor = torch.from_numpy(invariant_array).to(device)
            self.invariant_tensor = self.invariant_tensor.unsqueeze(0)
        
        if params.linear_grid:
            dims = params.hrrr_img_size
            grid_x, grid_y = torch.meshgrid(torch.linspace(0, 1, dims[0]), torch.linspace(0, 1, dims[1]))
            grid = torch.stack((grid_x, grid_y), dim=0).to(device)
            grid = grid.unsqueeze(0)
            if len(params.invariants) > 0:
                self.invariant_tensor = torch.cat((self.invariant_tensor, grid), dim=1)
            else:
                self.invariant_tensor = grid        

 
        resolution = params.crop_size if not params.crop_size == None else 512
        n_target_channels = len(self.diffusion_channel_indices)
        if params.pure_diffusion:
            n_input_channels = len(self.input_channel_indices) + 26
        else:
            n_input_channels = len(self.input_channel_indices) if not params.previous_step_conditioning else 2*len(self.input_channel_indices)

        if len(params.invariants) > 0:
            n_input_channels += len(params.invariants)
        
        if params.linear_grid:
            print("Adding linear grid to input channels")
            n_input_channels += 2
        label_dim = 0

        self.net = utils.diffusions.networks.get_preconditioned_architecture(
            name="ddpmpp-cwb-v0",
            resolution=resolution,
            target_channels=n_target_channels,
            conditional_channels=n_input_channels,
            label_dim=0,
            spatial_embedding=params.spatial_pos_embed
        ).requires_grad_(False)
        assert self.net.sigma_min < self.net.sigma_max

        if posthoc_ema_sigma != 'None':
            # Instantiate model weights post-hoc for new sigma_rel value
            target_sigma_rel = float(posthoc_ema_sigma)
            self._load_reweighted_model(resume_pkl, target_sigma_rel)
        else:
            # Backward compatible for models with traditional ema
            #if resume_pkl ends in 'pkl', load the model from the pkl file 
            if resume_pkl.endswith('pkl'):
                with open(resume_pkl, 'rb') as f:
                    data = pickle.load(f)
            elif resume_pkl.endswith('pt'):
                    data = torch.load(resume_pkl, map_location=torch.device('cpu'))
            if ema:
                self.net.load_state_dict(data['ema'].state_dict(), strict=False)
            else:
                print("not using ema, loading net state dict instead")
                self.net.load_state_dict(data['net'].state_dict(), strict=False)

        self.net = self.net.to(device)
        self.params = params


    def _load_reweighted_model(self, path, target_sigma_rel):

        save_path = os.path.join(path, 'posthoc_sigma_%.04f.pkl'%target_sigma_rel)

        if not os.path.isfile(save_path):

            # Get snapshot times and gamma values, stack into 1d arrays and solve for weight values
            print('Computing post-hoc weights for sigma_rel=%.04f, saving to %s'%(target_sigma_rel, save_path))
            snapshots = sorted(glob.glob(os.path.join(path, 'ema-snapshot*.pkl')))
            times = [int(''.join([char for char in snap_path.split('/')[-1] if char.isdigit()])) for snap_path in snapshots]
            with open(snapshots[0], 'rb') as f:
                data = pickle.load(f)
                gamma1, gamma2 = [sigma_rel_to_gamma(x) for x in data['ema_sigma_rel']]

            snap_times = np.array([times]*2).flatten()
            snap_gammas = np.array([gamma1]*len(times) + [gamma2]*len(times))
            snapshot_weights = torch.from_numpy(solve_weights(snap_times, snap_gammas, np.array(times[-1]), np.array([target_sigma_rel])))
            weights1, weights2 = torch.chunk(snapshot_weights, 2, dim=0)

            # Populate network with weighted sum
            self.net = self.net.to(torch.float16)
            for p in self.net.parameters():
                p.data.zero_()
            for time_idx, snap in enumerate(snapshots):
                with open(snap, 'rb') as f:
                    data = pickle.load(f)
                for p_ema1, p_ema2, p_net in zip(data['ema1'].parameters(), data['ema2'].parameters(), self.net.parameters()):
                    p_net += weights1[time_idx]*p_ema1 + weights2[time_idx]*p_ema2
                
            # Save for reuse
            with open(save_path, 'wb') as f:
                    pickle.dump(dict(weights=self.net), f)

        else:

            print('Loading post-hoc weights for sigma_rel=%.04f from %s'%(target_sigma_rel, save_path))
            with open(save_path, 'rb') as f:
                data = pickle.load(f)
            self.net = self.net.to(torch.float16)
            self.net.load_state_dict(data['weights'].state_dict(), strict=False)

        self.net = self.net.to(torch.float32)
        

    def run(self, hrrr_0):

        with torch.no_grad():

            #n = ensemble_size 
            #ensemble_size = hrrr_0.shape[0]
            ensemble_size, c, h, w = hrrr_0[:, self.diffusion_channel_indices, :, :].shape
            latents = torch.randn(ensemble_size, c, h, w, device=hrrr_0.device, dtype=hrrr_0.dtype)
            if len(self.params.invariants) > 0:
                if ensemble_size > 1 and self.invariant_tensor.shape[0] != ensemble_size:
                    self.invariant_tensor = self.invariant_tensor.expand(ensemble_size, -1, -1, -1)
                condition = torch.cat((hrrr_0, self.invariant_tensor), dim=1)
                output_images = edm_sampler(self.net, latents=latents, condition=condition)
            else:
                output_images = edm_sampler(self.net, latents=latents, condition=hrrr_0)
            
        
        return output_images, self.diffusion_channels


def main():

    config_directory = "./config"
    config_file = "hrrr_swin.yaml"
    config = "baseline_v3"

    params = YParams(os.path.join(config_directory, config_file), config)

    edm_runner = EDMRunner(params, resume_pkl=resume_pkl)

