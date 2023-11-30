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

"""Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
from typing import Optional, List
import pandas
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import cftime
import dnnlib
import cftime
import sys
import netCDF4 as nc

from torch_utils import misc

from training.dataset import Era5Dataset, CWBDataset, CWBERA5DatasetV2, ZarrDataset, _ZarrDataset, denormalize
import training.dataset
#from training.dataset_old import Era5Dataset, CWBDataset, CWBERA5DatasetV2, ZarrDataset

from training.YParams import YParams
import training.time

from einops import rearrange, reduce, repeat
import math   
import matplotlib.pyplot as plt

try:
    from edmss import edm_sampler
except ImportError:
    raise ImportError("Please get the edm_sampler by running: pip install git+https://github.com/mnabian/edmss.git")

from modulus.distributed import DistributedManager
from modulus.launch.logging import PythonLogger, RankZeroLoggingWrapper

def unet_regression(
    net, latents, img_lr, class_labels=None, randn_like=torch.randn_like,
    num_steps=2, sigma_min=0.0, sigma_max=0.0, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=0.0,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0
    
    #conditioning
    x_lr = img_lr

    # Main sampling loop.
    x_hat = latents.to(torch.float64) * t_steps[0]
    t_hat = torch.tensor(1.0).to(torch.float64).cuda()

    x_next = net(x_hat, x_lr, t_hat, class_labels).to(torch.float64)

    return x_next


#----------------------------------------------------------------------------
# Generalized ablation sampler, representing the superset of all sampling
# methods discussed in the paper.

def ablation_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=None, sigma_max=None, rho=7,
    solver='heun', discretization='edm', schedule='linear', scaling='none',
    epsilon_s=1e-3, C_1=0.001, C_2=0.008, M=1000, alpha=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    assert solver in ['euler', 'heun']
    assert discretization in ['vp', 've', 'iddpm', 'edm']
    assert schedule in ['vp', 've', 'linear']
    assert scaling in ['vp', 'none']

    # Helper functions for VP & VE noise level schedules.
    vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
    vp_sigma_deriv = lambda beta_d, beta_min: lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
    vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
    ve_sigma = lambda t: t.sqrt()
    ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
    ve_sigma_inv = lambda sigma: sigma ** 2

    # Select default noise level range based on the specified time step discretization.
    if sigma_min is None:
        vp_def = vp_sigma(beta_d=19.1, beta_min=0.1)(t=epsilon_s)
        sigma_min = {'vp': vp_def, 've': 0.02, 'iddpm': 0.002, 'edm': 0.002}[discretization]
    if sigma_max is None:
        vp_def = vp_sigma(beta_d=19.1, beta_min=0.1)(t=1)
        sigma_max = {'vp': vp_def, 've': 100, 'iddpm': 81, 'edm': 80}[discretization]

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Compute corresponding betas for VP.
    vp_beta_d = 2 * (np.log(sigma_min ** 2 + 1) / epsilon_s - np.log(sigma_max ** 2 + 1)) / (epsilon_s - 1)
    vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d

    # Define time steps in terms of noise level.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    if discretization == 'vp':
        orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
        sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
    elif discretization == 've':
        orig_t_steps = (sigma_max ** 2) * ((sigma_min ** 2 / sigma_max ** 2) ** (step_indices / (num_steps - 1)))
        sigma_steps = ve_sigma(orig_t_steps)
    elif discretization == 'iddpm':
        u = torch.zeros(M + 1, dtype=torch.float64, device=latents.device)
        alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
        for j in torch.arange(M, 0, -1, device=latents.device): # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
        sigma_steps = u_filtered[((len(u_filtered) - 1) / (num_steps - 1) * step_indices).round().to(torch.int64)]
    else:
        assert discretization == 'edm'
        sigma_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

    # Define noise level schedule.
    if schedule == 'vp':
        sigma = vp_sigma(vp_beta_d, vp_beta_min)
        sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
        sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
    elif schedule == 've':
        sigma = ve_sigma
        sigma_deriv = ve_sigma_deriv
        sigma_inv = ve_sigma_inv
    else:
        assert schedule == 'linear'
        sigma = lambda t: t
        sigma_deriv = lambda t: 1
        sigma_inv = lambda sigma: sigma

    # Define scaling schedule.
    if scaling == 'vp':
        s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
        s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
    else:
        assert scaling == 'none'
        s = lambda t: 1
        s_deriv = lambda t: 0

    # Compute final time steps based on the corresponding noise levels.
    t_steps = sigma_inv(net.round_sigma(sigma_steps))
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    t_next = t_steps[0]
    x_next = latents.to(torch.float64) * (sigma(t_next) * s(t_next))
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= sigma(t_cur) <= S_max else 0
        t_hat = sigma_inv(net.round_sigma(sigma(t_cur) + gamma * sigma(t_cur)))
        x_hat = s(t_hat) / s(t_cur) * x_cur + (sigma(t_hat) ** 2 - sigma(t_cur) ** 2).clip(min=0).sqrt() * s(t_hat) * S_noise * randn_like(x_cur)

        # Euler step.
        h = t_next - t_hat
        denoised = net(x_hat / s(t_hat), sigma(t_hat), class_labels).to(torch.float64)
        d_cur = (sigma_deriv(t_hat) / sigma(t_hat) + s_deriv(t_hat) / s(t_hat)) * x_hat - sigma_deriv(t_hat) * s(t_hat) / sigma(t_hat) * denoised
        x_prime = x_hat + alpha * h * d_cur
        t_prime = t_hat + alpha * h

        # Apply 2nd order correction.
        if solver == 'euler' or i == num_steps - 1:
            x_next = x_hat + h * d_cur
        else:
            assert solver == 'heun'
            denoised = net(x_prime / s(t_prime), sigma(t_prime), class_labels).to(torch.float64)
            d_prime = (sigma_deriv(t_prime) / sigma(t_prime) + s_deriv(t_prime) / s(t_prime)) * x_prime - sigma_deriv(t_prime) * s(t_prime) / sigma(t_prime) * denoised
            x_next = x_hat + h * ((1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_prime)

    return x_next

#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

def load_pickle(network_pkl, rank):
    # Load network. 
    with dnnlib.util.open_url(network_pkl, verbose=(rank == 0)) as f:
        return pickle.load(f)['ema']


#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges


def get_config_file(data_type):
    config_root = os.getenv("CONFIG_ROOT", "configs")
    config_file = os.path.join(config_root, data_type + '.yaml')
    return config_file


def get_dataset_and_sampler(data_type, params):

    if data_type == 'cwb':
        dataset = CWBDataset(params, params.test_data_path, train=False, task=opts.task)
        sampler = misc.InfiniteSampler(dataset=dataset, rank=0, num_replicas=1, seed=0)   #rank=0
        if max_times:
            sampler = [i for count, i in enumerate(dataset_sampler) if count < max_times]
    elif data_type == 'era5-cwb-v1':
        filelist = os.listdir(path=params.cwb_data_dir)  
        filelist = [name for name in filelist if "2018" in name]
        dataset = CWBERA5DatasetV2(params, filelist=filelist, train=True, task=opts.task)
        sampler = misc.InfiniteSampler(dataset=dataset, rank=0, num_replicas=1, seed=0)   #rank=0
        if max_times:
            sampler = [i for count, i in enumerate(dataset_sampler) if count < max_times]
    elif data_type == 'era5-cwb-v2':
        dataset = ZarrDataset(params, params.train_data_path, train=True)
        sampler = misc.InfiniteSampler(dataset=dataset, rank=0, num_replicas=1, seed=0)   #rank=0
        if max_times:
            sampler = [i for count, i in enumerate(dataset_sampler) if count < max_times]
    elif data_type == 'era5-cwb-v3':
        dataset = training.dataset.get_zarr_dataset(params, train=None, all_times=True)
        plot_times = [
            training.time.convert_datetime_to_cftime(time)
            for time in params.times
        ]
        all_times = dataset.time()
        time_indices = [all_times.index(t) for t in plot_times]
        sampler = time_indices
    elif data_type == 'netcdf':
        dataset = training.dataset.get_netcdf_dataset(params)
        sampler = list(range(len(dataset)))
    else:
        raise ValueError(data_type)

    return dataset, sampler

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl',  help='Network pickle filename', metavar='PATH|URL',                      type=str, required=True)
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--seeds',                   help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0-63', show_default=True)
@click.option("--max-times", help='maximum number of samples to draw', type=int, default=None)
@click.option('--subdirs',                 help='Create subdirectory for every 1000 seeds',                         is_flag=True)
@click.option('--class', 'class_idx',      help='Class label  [default: random]', metavar='INT',                    type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=64, show_default=True)

@click.option('--steps', 'num_steps',      help='Number of sampling steps', metavar='INT',                          type=click.IntRange(min=1), default=18, show_default=True)
@click.option('--sigma_min',               help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max',               help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True))
@click.option('--rho',                     help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--S_churn', 'S_churn',      help='Stochasticity strength', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min',          help='Stoch. min noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max',          help='Stoch. max noise level', metavar='FLOAT',                          type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise',      help='Stoch. noise inflation', metavar='FLOAT',                          type=float, default=1, show_default=True)

@click.option('--solver',                  help='Ablate ODE solver', metavar='euler|heun',                          type=click.Choice(['euler', 'heun']))
@click.option('--disc', 'discretization',  help='Ablate time step discretization {t_i}', metavar='vp|ve|iddpm|edm', type=click.Choice(['vp', 've', 'iddpm', 'edm']))
@click.option('--schedule',                help='Ablate noise schedule sigma(t)', metavar='vp|ve|linear',           type=click.Choice(['vp', 've', 'linear']))
@click.option('--scaling',                 help='Ablate signal scaling s(t)', metavar='vp|none',                    type=click.Choice(['vp', 'none']))

@click.option('--data_type',     help='String to include the data type', metavar='cwb|era5|cwb-era5')
@click.option('--data_config',   help='String to include the data config', metavar='full_field_val_crop64')
@click.option('--task',          help='String to include the task', metavar='sr|pred',                              type=click.Choice(['sr', 'pred']))

@click.option('--sample_res',          help='String to include the task', metavar='full|patch',                     type=click.Choice(['full', 'patch']))

@click.option('--pretext',          help='String to include the task', metavar='full|patch',                        type=click.Choice(['gen', 'reg', 'res']))
@click.option('--res_edm',          help='if residual based edm used',                                              is_flag=True)

@click.option('--network_reg', 'network_reg_pkl',  help='Network pickle filename', metavar='PATH|URL', type=str)

# def main(data_config, task, data_type, det_batch=None, gen_batch=None):
def main(max_times: Optional[int], seeds: List[int], **kwargs):
    
    opts = dnnlib.EasyDict(kwargs)
    
    # Initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    # Initialize logger.
    logger = PythonLogger("generate")  # General python logger
    logger0 = RankZeroLoggingWrapper(logger, dist)
    logger.file_logging("generate.log")

    det_batch = None
    gen_batch = None

    if gen_batch is None: gen_batch = 1  #max(4096 // net.img_resolution, 1)
    if det_batch is None: det_batch = 1  #max(gen_batch, 64)
    assert det_batch % gen_batch == 0
    
    logger0.info(f'opts.data_config: {opts.data_config}')
        
    # Data
    config_file = get_config_file(opts.data_type)
    logger0.info(f"Config file: {config_file}")
    params = YParams(config_file, config_name=opts.data_config)
    patch_size = params.patch_size
    crop_size_x = params.crop_size_x
    crop_size_y = params.crop_size_y

    root = os.getenv("DATA_ROOT", "")
    params["train_data_path"] = os.path.join(root, params["train_data_path"])
    logger0.info(f"Train data path: {params.train_data_path}")
    dataset, sampler = get_dataset_and_sampler(opts.data_type, params)
    
    with nc.Dataset(opts.outdir.format(rank=dist.rank), "w") as f:
        # add attributes
        f.history = ' '.join(sys.argv)
        f.network_pkl = kwargs["network_pkl"]

        # Load network
        logger.info(f'torch.__version__: {torch.__version__}')
        logger0.info(f'Loading network from "{opts.network_pkl}"...')
        net = load_pickle(opts.network_pkl, dist.rank)
        logger0.info(f'Loading network from "{opts.network_reg_pkl}"...')
        net_reg = load_pickle(opts.network_reg_pkl, dist.rank) if opts.res_edm else None

        # move to device
        num_gpus = dist.world_size
        torch.cuda.set_device(dist.rank)
        device = dist.device
        net = net.to(device)
        net_reg = net_reg.to(device) if net_reg else None

        batch_size = min(len(sampler), det_batch)

        def generate_fn(image_lr):
            """Function to generate an image with

            Args: 
                image_lr: low resolution input. shape: (b, c, h, w)

            Return
                image_hr: high resolution output: shape (b, c, h, w)
            """
            sample_res = opts.sample_res
            class_idx = opts.class_idx
            if sample_res == 'full':
                image_lr_patch = image_lr
            else:
                image_lr_patch = rearrange(image_lr, 'b c (h1 h) (w1 w) -> (b h1 w1) c h w', h1=crop_size_x//patch_size, w1=crop_size_y//patch_size)
                
            sample_seeds = seeds

            logger0.info(f'seeds: {sample_seeds}')
            if net_reg:
                image_mean = generate(
                    net=net_reg, img_lr=image_lr_patch,
                    max_batch_size=image_lr_patch.shape[0], seeds=sample_seeds,
                    pretext='reg', class_idx=class_idx
                )
                image_out = image_mean + generate(
                    net=net, img_lr=image_lr_patch,
                    max_batch_size=image_lr_patch.shape[0], seeds=sample_seeds,
                    pretext='gen', class_idx=class_idx
                )
            else:
                image_out = generate(
                    net=net, img_lr=image_lr_patch,
                    max_batch_size=image_lr_patch.shape[0], seeds=sample_seeds,
                    pretext=opts.pretext, class_idx=class_idx
                )
            
            #reshape: (1*9*9)x3x50x50  --> 1x3x450x450
            if sample_res == 'full':
                image_lr_patch = image_lr
            else:
                image_out = rearrange(image_out, '(b h1 w1) c h w -> b c (h1 h) (w1 w)', h1=crop_size_x//patch_size, w1=crop_size_y//patch_size)

            return image_out
        
        # generate images
        logger0.info('Generating images...')
        generate_and_save(dataset, sampler, f, generate_fn, device, batch_size, logger0)

    # Done.
    if dist.world_size > 1:
        torch.distributed.barrier()
    logger0.info('Done.')


def _get_name(channel_info):
    plev = "" if np.isnan(channel_info['pressure']) else "{:d}".format(int(channel_info['pressure']))
    return channel_info["variable"] + plev


class NetCDFWriter:

    def __init__(self, f, lat, lon, input_channels, output_channels):
        self._f = f

        # create unlimited dimensions
        f.createDimension("time")
        f.createDimension("ensemble")

        assert lat.shape == lon.shape
        ny, nx = lat.shape

        # create lat/lon grid
        f.createDimension("x", nx-2)
        f.createDimension("y", ny-2)

        v = f.createVariable("lat", "f", dimensions=("y", "x"))
        v[:] = lat[1:-1, 1:-1]
        v.standard_name = "latitude"
        v.units = "degrees_north"

        v = f.createVariable("lon", "f", dimensions=("y", "x"))
        v[:] = lon[1:-1, 1:-1]
        v.standard_name = "longitude"
        v.units = "degrees_east"

        # create time dimension
        v = f.createVariable("time", "i8", ("time"))
        v.calendar = "standard"
        v.units = "hours since 1990-01-01 0:0:0"

        self.truth_group = f.createGroup("truth")
        self.prediction_group = f.createGroup("prediction")
        self.input_group = f.createGroup("input")


        for variable in output_channels:
            name = _get_name(variable)
            self.truth_group.createVariable(name, "f", dimensions=("time", "y", "x"))
            self.prediction_group.createVariable(name, "f", dimensions=("ensemble", "time", "y", "x"))

        # setup input data in netCDF
        
        n_grid_inputs = 4 # TODO get this from the model object
        for i in range(n_grid_inputs):
            input_channels.append({"variable": "grid", "pressure": i})

        for variable in input_channels:
            name = _get_name(variable)
            self.input_group.createVariable(name, "f", dimensions=("time", "y", "x"))

    def write_input(self, channel_name, time_index, val):
        self.input_group[channel_name][time_index] = val
    
    def write_truth(self, channel_name, time_index, val):
        self.truth_group[channel_name][time_index] = val

    def write_prediction(self, channel_name, time_index, ensemble_index, val):
        self.prediction_group[channel_name][ensemble_index, time_index] = val
    
    def write_time(self, time_index, time):
        time_v = self._f["time"]
        self._f["time"][time_index] = cftime.date2num(time, time_v.units, time_v.calendar)


def writer_from_input_dataset(f, dataset):
    return NetCDFWriter(f, lat=dataset.latitude(), lon=dataset.longitude(), input_channels=dataset.input_channels(), output_channels=dataset.output_channels())


def generate_and_save(dataset, sampler, f: nc.Dataset, generate_fn, device, batch_size, logger):
    # Instantiate distributed manager.
    dist = DistributedManager()
    device = dist.device

    data_loader = torch.utils.data.DataLoader(dataset=dataset, sampler=sampler, batch_size=batch_size, pin_memory=True)
    time_index = -1
    writer = writer_from_input_dataset(f, dataset)

    for image_tar, image_lr, index in iter(data_loader):
        time_index  += 1
        if dist.rank == 0:
            logger.info(f"starting index: {time_index}")  # TODO print on rank zero
        input_data = image_lr = image_lr.to(device=device).to(torch.float32)
        image_tar = image_tar.to(device=device).to(torch.float32)
        image_out = generate_fn(image_lr)

        #for validation - make 3x450x450 to an ordered sequence of 50x50 patches
        #input; 1x3x450x450 --> (1*9*9)x3x50x50

        # weather sub-plot
        mx, sx = dataset.info()['input_normalization']
        mx = mx[dataset.in_channels]
        image_lr2 = image_lr[0].unsqueeze(0)

        # add zeros for grid embeddings
        padding = image_lr2.shape[1] - len(mx)
        assert padding >= 0

        mx = np.concatenate([mx, np.zeros(padding)])
        # add zeros for grid embeddings
        sx = sx[dataset.in_channels]
        sx = np.concatenate([sx, np.ones(padding)])
        image_lr2 = image_lr2.cpu().numpy()
        image_lr2 = denormalize(image_lr2, mx, sx)

        my, sy = dataset.info()['target_normalization']
        my = my[dataset.out_channels]
        sy = sy[dataset.out_channels]
        image_tar2 = image_tar[0].unsqueeze(0)
        image_tar2 = image_tar2.cpu().numpy()
        image_tar2 = denormalize(image_tar2, my, sy)

        # some runtime assertions
        assert image_tar2.ndim == 4

        for idx in range(image_out.shape[0]):
            image_out2 = image_out[idx].unsqueeze(0)
            assert image_out2.ndim == 4

            # Denormalize the input and outputs
            image_out2 = image_out2.cpu().numpy()
            image_out2 = denormalize(image_out2, my, sy)

            t_index = index[0]
            assert len(index) == 1
            time = dataset.time()[t_index]
            writer.write_time(time_index, time)
            for channel_idx in range(image_out2.shape[1]):
                output_channels = dataset.output_channels()
                info = output_channels[channel_idx]
                channel_name = _get_name(info)
                truth = image_tar2[0, channel_idx]

                writer.write_truth(channel_name, time_index, truth)
                writer.write_prediction(channel_name, time_index, idx, image_out2[0, channel_idx])

            input_channels = dataset.input_channels()
            for channel_idx in range(len(input_channels)):
                info = input_channels[channel_idx]
                channel_name = _get_name(info)
                writer.write_input(channel_name, time_index, image_lr2[0, channel_idx])
            

def generate(net, seeds, class_idx, max_batch_size, img_lr=None, pretext=None, **sampler_kwargs):
    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".
    
    Examples:

    \b
    # Generate 64 images and save them as out/*.png
    python generate.py --outdir=out --seeds=0-63 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl

    \b
    # Generate 1024 images using 2 GPUs
    torchrun --standalone --nproc_per_node=2 generate.py --outdir=out --seeds=0-999 --batch=64 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/edm/pretrained/edm-cifar10-32x32-cond-vp.pkl
    """

    # Instantiate distributed manager.
    dist = DistributedManager()
    device = dist.device

    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.world_size) + 1) * dist.world_size
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.rank :: dist.world_size]

    # Rank 0 goes first.
    if dist.world_size > 1 and dist.rank != 0:
        torch.distributed.barrier()

        # Other ranks follow.
        if dist.world_size > 1 and dist.rank == 0:
            torch.distributed.barrier()
        
    # Loop over batches.
    all_images = []
    for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.rank != 0)):
        if dist.world_size > 1:
            torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        # Pick latents and labels.
        rnd = StackedRandomGenerator(device, batch_seeds)
        #latents = rnd.randn([batch_size, net.img_in_channels, net.img_resolution, net.img_resolution], device=device)
        latents = rnd.randn([max_batch_size, net.img_out_channels, net.img_resolution, net.img_resolution], device=device)
        

        class_labels = None
        if net.label_dim:
            class_labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[batch_size], device=device)]
        if class_idx is not None:
            class_labels[:, :] = 0
            class_labels[:, class_idx] = 1


        # Generate images.
        sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
        have_ablation_kwargs = any(x in sampler_kwargs for x in ['solver', 'discretization', 'schedule', 'scaling'])
        
        if pretext == 'gen':
            if have_ablation_kwargs:
                sampler_fn = ablation_sampler
            else:
                sampler_fn = edm_sampler
        elif pretext == 'reg':
            latents = torch.zeros_like(latents)
            sampler_fn = unet_regression
        
        images = sampler_fn(net, latents, img_lr, class_labels, randn_like=rnd.randn_like, **sampler_kwargs)
        all_images.append(images)
        
    return torch.cat(all_images, dim=0)

#----------------------------------------------------------------------------

    
if __name__ == "__main__":
    
    main()

#----------------------------------------------------------------------------
