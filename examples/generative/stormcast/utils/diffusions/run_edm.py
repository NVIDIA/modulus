import os
from utils.YParams import YParams
import pickle
import numpy as np
import torch
import glob

from utils.diffusions.generate import edm_sampler
from utils.diffusions.networks import get_preconditioned_architecture
from utils.diffusions.power_ema import sigma_rel_to_gamma, solve_weights
from utils.data_loader_hrrr_era5 import get_dataset


class EDMRunner():

    def __init__(self, params, resume_pkl=None, posthoc_ema_sigma='None', ema=True, device=None, sampler_args={}):

        if device is None:
            device = torch.device('cuda:0')

        self.sampler_args = sampler_args
        dataset_obj = get_dataset(params, train=True)

        _, hrrr_channels = dataset_obj._get_hrrr_channel_names()
        self.input_channels = params.input_channels
        self.input_channels = hrrr_channels
        self.input_channel_indices = list(range(len(hrrr_channels)))
        self.diffusion_channels = params.diffusion_channels
        self.diffusion_channels = hrrr_channels
        self.diffusion_channel_indices = list(range(len(hrrr_channels)))

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

        print("n target channels: ", n_target_channels)


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

            ensemble_size, c, h, w = hrrr_0[:, self.diffusion_channel_indices, :, :].shape
            latents = torch.randn(ensemble_size, c, h, w, device=hrrr_0.device, dtype=hrrr_0.dtype)

            if ensemble_size > 1 and self.invariant_tensor.shape[0] != ensemble_size:
                self.invariant_tensor = self.invariant_tensor.expand(ensemble_size, -1, -1, -1)
            condition = torch.cat((hrrr_0, self.invariant_tensor), dim=1)
            output_images = edm_sampler(self.net, latents=latents, condition=condition, **self.sampler_args)
            
        
        return output_images, self.diffusion_channels
