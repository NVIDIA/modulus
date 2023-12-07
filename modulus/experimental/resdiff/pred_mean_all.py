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
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import cftime
import dnnlib
from torch_utils import distributed as dist
import cftime
import sys
import netCDF4 as nc

from torch_utils import misc

from training.dataset import denormalize
import training.dataset
from training.YParams import YParams
import training.time


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
    print(config_file)
    return config_file


def get_dataset_and_sampler(data_type, data_config, config_file=None):
    root = os.getenv("DATA_ROOT", "")
    if config_file is None:
        config_file = get_config_file(data_type)
    params = YParams(config_file, config_name=data_config)
    params["train_data_path"] = os.path.join(root, params["train_data_path"])
    print(f"train data path: {params.train_data_path}")
    dataset_train = training.dataset.get_zarr_dataset(params, train=True)
    sampler = dataset_train.time()

    return dataset_train, sampler

#----------------------------------------------------------------------------

@click.command()
@click.option('--network_reg', 'network_reg_pkl',  help='Network pickle filename', metavar='PATH|URL', type=str)
@click.option('--outdir',                  help='Where to save the output images', metavar='DIR',                   type=str, required=True)
@click.option('--seeds',                   help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0-63', show_default=True)
@click.option("--max-times", help='maximum number of samples to draw', type=int, default=None)
@click.option('--subdirs',                 help='Create subdirectory for every 1000 seeds',                         is_flag=True)
@click.option('--class', 'class_idx',      help='Class label  [default: random]', metavar='INT',                    type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--data_type',     help='String to include the data type', metavar='cwb|era5|cwb-era5')
@click.option('--data_config',   help='String to include the data config', metavar='full_field_val_crop64')
@click.option('--task',          help='String to include the task', metavar='sr|pred',                              type=click.Choice(['sr', 'pred']))

@click.option('--sample_res',          help='String to include the task', metavar='full|patch',                     type=click.Choice(['full', 'patch']))



def main(max_times: Optional[int], seeds: List[int], **kwargs):
    
    opts = dnnlib.EasyDict(kwargs)
    
    det_batch = None
    gen_batch = None

    if gen_batch is None: gen_batch = 1  #max(4096 // net.img_resolution, 1)
    if det_batch is None: det_batch = 1  #max(gen_batch, 64)
    assert det_batch % gen_batch == 0
    
    print('opts.data_config', opts.data_config)

    # Data
    config_file = get_config_file(opts.data_type)
    params = YParams(config_file, config_name=opts.data_config)
    patch_size = params.patch_size
    crop_size_x = params.crop_size_x
    crop_size_y = params.crop_size_y

    dataset, sampler = get_dataset_and_sampler(opts.data_type, opts.data_config)
    # dataset = get_dataset_and_sampler(opts.data_type, opts.data_config)
    with nc.Dataset(opts.outdir, "w") as f:
        # add attributes
        f.history = ' '.join(sys.argv)
        # Load network
        dist.print0('Generating images...')
        net_reg = load_pickle(opts.network_reg_pkl)
        device = torch.device('cuda')
        net_reg = net_reg.to(device) if net_reg else None

        batch_size = min(len(dataset), det_batch)

        def generate_fn(image_lr):
            class_idx = opts.class_idx
            sample_seeds = seeds
            image_out = generate(
                net=net_reg, img_lr=image_lr,
                max_batch_size=image_lr.shape[0], seeds=sample_seeds,
                pretext='reg', class_idx=class_idx
            )
            return image_out
        generate_and_save(dataset, sampler, f, generate_fn, device, batch_size)
        # generate_and_save(dataset, None, f, generate_fn, device, batch_size)

    # Done.
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    dist.print0('Done.')


def _get_name(channel_info):
    plev = "" if np.isnan(channel_info['pressure']) else "{:d}".format(int(channel_info['pressure']))
    return channel_info["variable"] + plev


class NetCDFWriter:

    def __init__(self, f, lat, lon, input_channels, output_channels):
        self._f = f

        # create unlimited dimensions
        f.createDimension("time")

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
        self.prediction_mean_group = f.createGroup("prediction_mean")
        self.residual_group = f.createGroup("residual")
        self.input_group = f.createGroup("input")


        for variable in output_channels:
            name = _get_name(variable)
            self.truth_group.createVariable(name, "f", dimensions=("time", "y", "x"))
            self.prediction_mean_group.createVariable(name, "f", dimensions=("time", "y", "x"))
            self.residual_group.createVariable(name, "f", dimensions=("time", "y", "x"))

        n_grid_inputs = 4 
        for i in range(n_grid_inputs):
            input_channels.append({"variable": "grid", "pressure": i})

        for variable in input_channels:
            name = _get_name(variable)
            self.input_group.createVariable(name, "f", dimensions=("time", "y", "x"))

    def write_input(self, channel_name, time_index, val):
        self.input_group[channel_name][time_index] = val
    
    def write_truth(self, channel_name, time_index, val):
        self.truth_group[channel_name][time_index] = val
        
    def write_prediction_mean(self, channel_name, time_index, val):
        self.prediction_mean_group[channel_name][time_index] = val
        
    def write_residual(self, channel_name, time_index, val):
        self.residual_group[channel_name][time_index] = val
    
    def write_time(self, time_index, time):
        time_v = self._f["time"]
        self._f["time"][time_index] = cftime.date2num(time, time_v.units, time_v.calendar)


def writer_from_input_dataset(f, dataset):
    return NetCDFWriter(f, lat=dataset.latitude(), lon=dataset.longitude(), input_channels=dataset.input_channels(), output_channels=dataset.output_channels())
    # return NetCDFWriter(f)

"""
For pred_mean_all.py, we do not use denormalization to revert the range back to the original.
"""
def generate_and_save(dataset, sampler, f: nc.Dataset, generate_fn, device, batch_size):
    data_loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, pin_memory=True)
    time_index = -1
    
    writer = writer_from_input_dataset(f, dataset)

    for image_tar, image_lr, index in iter(data_loader):
        time_index  += 1
        if dist.get_rank() == 0:
            print("starting index", time_index)
        image_lr = image_lr.to(device=device).to(torch.float32)
        image_tar = image_tar.to(device=device).to(torch.float32)
        image_mean = generate_fn(image_lr)

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

        my, sy = dataset.info()['target_normalization']
        my = my[dataset.out_channels]
        sy = sy[dataset.out_channels]
        image_tar2 = image_tar[0].unsqueeze(0)
        image_tar2 = image_tar2.cpu().numpy()

        # write image_mean
        image_mean = image_mean.cpu().numpy()
        residual = image_tar2 - image_mean

        t_index = index[0]
        assert len(index) == 1
        time = dataset.time()[t_index]
        writer.write_time(time_index, time)
        
        for channel_idx in range(image_mean.shape[1]):
            output_channels = dataset.output_channels()
            info = output_channels[channel_idx]
            channel_name = _get_name(info)
            truth = image_tar2[0, channel_idx]

            writer.write_truth(channel_name, time_index, truth)
            writer.write_prediction_mean(channel_name, time_index, image_mean[0, channel_idx])
            writer.write_residual(channel_name, time_index, residual[0, channel_idx])

        input_channels = dataset.input_channels()
        for channel_idx in range(len(input_channels)):
            info = input_channels[channel_idx]
            channel_name = _get_name(info)
            writer.write_input(channel_name, time_index, image_lr2[0, channel_idx])
            

def load_pickle(network_pkl):
    dist.print0(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        print('torch.__version__', torch.__version__)
        return pickle.load(f)['ema']


def generate(net, seeds, class_idx, max_batch_size, img_lr=None, device=torch.device('cuda'), pretext=None, **sampler_kwargs):
    num_batches = ((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]
    # Rank 0 goes first.
    if torch.distributed.is_initialized():
        if dist.get_rank() != 0:
            torch.distributed.barrier()

        # Other ranks follow.
        if dist.get_rank() == 0:
            torch.distributed.barrier()
        
    # Loop over batches.
    all_images = []
    for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue
        class_labels = None
        if class_idx is not None:
            class_labels[:, :] = 0
            class_labels[:, class_idx] = 1

        # Generate images.
        sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
        
        latents = torch.zeros(
            [max_batch_size, net.img_out_channels, net.img_resolution, net.img_resolution], 
            device=device
        )
        sampler_fn = unet_regression
        
        images = sampler_fn(net, latents, img_lr, class_labels)
        all_images.append(images)
        
    return torch.cat(all_images, dim=0)

#----------------------------------------------------------------------------

    
if __name__ == "__main__":
    
    main()

#----------------------------------------------------------------------------
