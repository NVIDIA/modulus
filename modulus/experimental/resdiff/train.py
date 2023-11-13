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

"""Train diffusion-based generative model using the techniques described in the
paper "Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
os.environ["TORCHELASTIC_ENABLE_FILE_TIMER"] = "1"

import os
import re
import json
import click
import torch
import dnnlib
from torch_utils import distributed as dist
from training import training_loop

import warnings
warnings.filterwarnings('ignore', 'Grad strides do not match bucket view strides') # False warning printed by PyTorch 1.12.


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

#----------------------------------------------------------------------------

@click.command()

# Main options.
@click.option('--outdir',        help='Where to save the results', metavar='DIR',                   type=str, required=True)
@click.option('--data',          help='Path to the dataset', metavar='ZIP|DIR',                     type=str, required=True)
@click.option('--cond',          help='Train class-conditional model', metavar='BOOL',              type=bool, default=False, show_default=True)
@click.option('--arch',          help='Network architecture', metavar='ddpmpp|ncsnpp|adm',          type=click.Choice(['ddpmpp', 'ddpmpp-cifar', 'ddpmpp-cwb-v0', 'ddpmpp-cwb-v1', 'ddpmpp-cwb-v2', 'ncsnpp', 'adm', 'ddpmpp-cwb-v0-regression']), default='ddpmpp', show_default=True)
@click.option('--precond',       help='Preconditioning & loss function', metavar='vp|ve|edm',       type=click.Choice(['vp', 've', 'edm', 'unetregression', 'mixture', 'mixturev1', 'mixturev2', 'mixturev3', 'mixturev4', 'mixturev5', 'reslossv1']), default='edm', show_default=True)

# Hyperparameters.
@click.option('--duration',      help='Training duration', metavar='MIMG',                          type=click.FloatRange(min=0, min_open=True), default=200, show_default=True)
@click.option('--batch',         help='Total batch size', metavar='INT',                            type=click.IntRange(min=1), default=512, show_default=True)
@click.option('--batch-gpu',     help='Limit batch size per GPU', metavar='INT',                    type=click.IntRange(min=1))
@click.option('--cbase',         help='Channel multiplier  [default: varies]', metavar='INT',       type=int)
@click.option('--cres',          help='Channels per resolution  [default: varies]', metavar='LIST', type=parse_int_list)
@click.option('--lr',            help='Learning rate', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=10e-4, show_default=True)
@click.option('--ema',           help='EMA half-life', metavar='MIMG',                              type=click.FloatRange(min=0), default=0.5, show_default=True)
@click.option('--dropout',       help='Dropout probability', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0.13, show_default=True)
@click.option('--augment',       help='Augment probability', metavar='FLOAT',                       type=click.FloatRange(min=0, max=1), default=0.12, show_default=True)
@click.option('--xflip',         help='Enable dataset x-flips', metavar='BOOL',                     type=bool, default=False, show_default=True)

# Performance-related.
@click.option('--fp16',          help='Enable mixed-precision training', metavar='BOOL',            type=bool, default=False, show_default=True)
@click.option('--ls',            help='Loss scaling', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=1, show_default=True)
@click.option('--bench',         help='Enable cuDNN benchmarking', metavar='BOOL',                  type=bool, default=True, show_default=True)
@click.option('--cache',         help='Cache dataset in CPU memory', metavar='BOOL',                type=bool, default=True, show_default=True)
@click.option('--workers',       help='DataLoader worker processes', metavar='INT',                 type=click.IntRange(min=1), default=1, show_default=True)

# I/O-related.
@click.option('--desc',          help='String to include in result dir name', metavar='STR',        type=str)
@click.option('--nosubdir',      help='Do not create a subdirectory for results',                   is_flag=True)
@click.option('--tick',          help='How often to print progress', metavar='KIMG',                type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--snap',          help='How often to save snapshots', metavar='TICKS',               type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--dump',          help='How often to dump state', metavar='TICKS',                   type=click.IntRange(min=1), default=500, show_default=True)
@click.option('--seed',          help='Random seed  [default: random]', metavar='INT',              type=int)
@click.option('--transfer',      help='Transfer learning from network pickle', metavar='PKL|URL',   type=str)
@click.option('--resume',        help='Resume from previous training state', metavar='PT',          type=str)
@click.option('-n', '--dry-run', help='Print training options and exit',                            is_flag=True)

# Weather-related.
@click.option('--data_config',   help='String to include the data config', metavar='STR',           type=str)
@click.option('--task',          help='String to include the task', metavar='STR',                  type=str)
@click.option('--data_type',     help='String to include the data type', metavar='STR',             type=str)

#regression
@click.option('--ckpt_unet',        help='Checkpoint for the UNet to predict the mean', metavar='PT',          type=str)



def main(**kwargs):
    """Train diffusion-based generative model using the techniques described in the
    paper "Elucidating the Design Space of Diffusion-Based Generative Models".

    Examples:

    \b
    # Train DDPM++ model for class-conditional CIFAR-10 using 8 GPUs
    torchrun --standalone --nproc_per_node=8 train.py --outdir=training-runs \\
        --data=datasets/cifar10-32x32.zip --cond=1 --arch=ddpmpp
    """
    opts = dnnlib.EasyDict(kwargs)
    torch.multiprocessing.set_start_method('spawn', force=True)
    dist.init()

    # Initialize config dict.
    c = dnnlib.EasyDict()
    #c.dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=opts.data, use_labels=opts.cond, xflip=opts.xflip, cache=opts.cache)  #cifar10
    #c.dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.Era5Dataset', path=opts.data, xflip=opts.xflip, cache=opts.cache)  #era5   #use_labels=opts.cond,
    c.dataset_kwargs = dnnlib.EasyDict(path=opts.data, xflip=False, cache=True, use_labels=False)
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=opts.workers, prefetch_factor=2)
    c.network_kwargs = dnnlib.EasyDict()
    c.loss_kwargs = dnnlib.EasyDict()
    c.optimizer_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', lr=opts.lr, betas=[0.9,0.999], eps=1e-8)
    
    
    dataset_name = opts.data_type

    # # Validate dataset options.
    # try:
    #     dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs)
    #     dataset_name = dataset_obj.name
    #     c.dataset_kwargs.resolution = dataset_obj.resolution # be explicit about dataset resolution
    #     c.dataset_kwargs.max_size = len(dataset_obj) # be explicit about dataset size
    #     if opts.cond and not dataset_obj.has_labels:
    #         raise click.ClickException('--cond=True requires labels specified in dataset.json')
    #     del dataset_obj # conserve memory
    # except IOError as err:
    #     raise click.ClickException(f'--data: {err}')

    # Network architecture.
    if opts.arch == 'ddpmpp-cwb-v2':
        c.network_kwargs.update(model_type='SongUNet', embedding_type='positional', encoder_type='standard', decoder_type='standard')  #, attn_resolutions=[28]
        c.network_kwargs.update(channel_mult_noise=1, resample_filter=[1,1], model_channels=128, channel_mult=[1,2,2,4,4,8], attn_resolutions=[14])   #era5-cwb, larger run, 448x448 
    
    elif opts.arch == 'ddpmpp-cwb-v1':
        c.network_kwargs.update(model_type='SongUNet', embedding_type='positional', encoder_type='standard', decoder_type='standard')  #, attn_resolutions=[28]
        c.network_kwargs.update(channel_mult_noise=1, resample_filter=[1,1], model_channels=128, channel_mult=[1,2,2,4,4], attn_resolutions=[28])   #era5-cwb, 448x448
        
    elif opts.arch == 'ddpmpp-cwb-v0-regression':
        c.network_kwargs.update(model_type='SongUNet', embedding_type='zero', encoder_type='standard', decoder_type='standard')  #, attn_resolutions=[28]
        c.network_kwargs.update(channel_mult_noise=1, resample_filter=[1,1], model_channels=128, channel_mult=[1,2,2,2,2], attn_resolutions=[28])   #era5-cwb, 448x448
        
    elif opts.arch == 'ddpmpp-cwb-v0':
        c.network_kwargs.update(model_type='SongUNet', embedding_type='positional', encoder_type='standard', decoder_type='standard')  #, attn_resolutions=[28]
        c.network_kwargs.update(channel_mult_noise=1, resample_filter=[1,1], model_channels=128, channel_mult=[1,2,2,2,2], attn_resolutions=[28])   #era5-cwb, 448x448
        
    elif opts.arch == 'ddpmpp-cifar':
        c.network_kwargs.update(model_type='SongUNet', embedding_type='positional', encoder_type='standard', decoder_type='standard')  #, attn_resolutions=[28]
        c.network_kwargs.update(channel_mult_noise=1, resample_filter=[1,1], model_channels=128, channel_mult=[2,2,2])    #cifar-10, 32x32

    elif opts.arch == 'ncsnpp':
        c.network_kwargs.update(model_type='SongUNet', embedding_type='fourier', encoder_type='residual', decoder_type='standard')
        c.network_kwargs.update(channel_mult_noise=2, resample_filter=[1,3,3,1], model_channels=128, channel_mult=[2,2,2])
        
    else:
        assert opts.arch == 'adm'
        c.network_kwargs.update(model_type='DhariwalUNet', model_channels=192, channel_mult=[1,2,3,4])

    # Preconditioning & loss function.
    if opts.precond == 'vp':
        c.network_kwargs.class_name = 'training.networks.VPPrecond'
        c.loss_kwargs.class_name = 'training.loss.VPLoss'
    elif opts.precond == 've':
        c.network_kwargs.class_name = 'training.networks.VEPrecond'
        c.loss_kwargs.class_name = 'training.loss.VELoss'
    elif opts.precond == 'edm':
        c.network_kwargs.class_name = 'training.networks.EDMPrecond'
        c.loss_kwargs.class_name = 'training.loss.EDMLoss'
    elif opts.precond == 'unetregression':
        c.network_kwargs.class_name = 'training.networks.UNet'
        c.loss_kwargs.class_name = 'training.loss.RegressionLoss'
    elif opts.precond == 'mixture':
        c.network_kwargs.class_name = 'training.networks.EDMPrecond'
        c.loss_kwargs.class_name = 'training.loss.MixtureLoss'
    elif opts.precond == 'mixturev1':
        c.network_kwargs.class_name = 'training.networks.EDMPrecond'
        c.loss_kwargs.class_name = 'training.loss.MixtureLossV1'
    elif opts.precond == 'mixturev2':
        c.network_kwargs.class_name = 'training.networks.EDMPrecond'
        c.loss_kwargs.class_name = 'training.loss.MixtureLossV2'
    elif opts.precond == 'mixturev3':
        c.network_kwargs.class_name = 'training.networks.EDMPrecond'
        c.loss_kwargs.class_name = 'training.loss.MixtureLossV3'
    elif opts.precond == 'mixturev4':
        c.network_kwargs.class_name = 'training.networks.EDMPrecond'
        c.loss_kwargs.class_name = 'training.loss.MixtureLossV4'
    elif opts.precond == 'mixturev5':
        c.network_kwargs.class_name = 'training.networks.EDMPrecond'
        c.loss_kwargs.class_name = 'training.loss.MixtureLossV5'
    elif opts.precond == 'reslossv1':
        c.network_kwargs.class_name = 'training.networks.EDMPrecond'
        c.loss_kwargs.class_name = 'training.loss.ResLossv1'
        
        
    # Network options.
    if opts.cbase is not None:
        c.network_kwargs.model_channels = opts.cbase
    if opts.cres is not None:
        c.network_kwargs.channel_mult = opts.cres
    if opts.augment:
        c.augment_kwargs = dnnlib.EasyDict(class_name='training.augment.AugmentPipe', p=opts.augment)
        c.augment_kwargs.update(xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1)
        c.network_kwargs.augment_dim = 9
    c.network_kwargs.update(dropout=opts.dropout, use_fp16=opts.fp16)

    # Training options.
    c.total_kimg = max(int(opts.duration * 1000), 1)
    c.ema_halflife_kimg = int(opts.ema * 1000)
    c.update(batch_size=opts.batch, batch_gpu=opts.batch_gpu)
    c.update(loss_scaling=opts.ls, cudnn_benchmark=opts.bench)
    c.update(kimg_per_tick=opts.tick, snapshot_ticks=opts.snap, state_dump_ticks=opts.dump)

    # Random seed.
    if opts.seed is not None:
        c.seed = opts.seed
    else:
        seed = torch.randint(1 << 31, size=[], device=torch.device('cuda'))
        torch.distributed.broadcast(seed, src=0)
        c.seed = int(seed)

    #output dir
    #c.ckpt_dir = opts.outdir 
    
    #check if resume.txt exists
    resume_path = os.path.join(opts.outdir,'resume.txt')
    #print('resume_path', resume_path)
    if os.path.exists(resume_path):
        with open(resume_path, "r") as f:
            opts.resume = f.read()
            #print('opts.resume', opts.resume)
            f.close()
            
    dist.print0('opts.resume', opts.resume)

    # Transfer learning and resume.
    if opts.transfer is not None:
        if opts.resume is not None:
            raise click.ClickException('--transfer and --resume cannot be specified at the same time')
        c.resume_pkl = opts.transfer
        c.ema_rampup_ratio = None
    elif opts.resume is not None:
        print('gets into elif opts.resume is not None ...')
        match = re.fullmatch(r'training-state-(\d+).pt', os.path.basename(opts.resume))
        print('match', match)
        print('match.group(1)', match.group(1))
        # if not match or not os.path.isfile(opts.resume):
        #      raise click.ClickException('--resume must point to training-state-*.pt from a previous training run')
        c.resume_pkl = os.path.join(os.path.dirname(opts.resume), f'network-snapshot-{match.group(1)}.pkl')
        c.resume_kimg = int(match.group(1))
        c.resume_state_dump = opts.resume
        dist.print0('c.resume_pkl', c.resume_pkl)
        dist.print0('c.resume_kimg', c.resume_kimg)
        dist.print0('c.resume_state_dump', c.resume_state_dump)
        # import pdb; pdb.set_trace()


    # Description string.
    cond_str = 'cond' if c.dataset_kwargs.use_labels else 'uncond'
    dtype_str = 'fp16' if c.network_kwargs.use_fp16 else 'fp32'
    desc = f'{dataset_name:s}-{cond_str:s}-{opts.arch:s}-{opts.precond:s}-gpus{dist.get_world_size():d}-batch{c.batch_size:d}-{dtype_str:s}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'
        

    c.run_dir = opts.outdir
    
    # # Pick output directory.
    # if dist.get_rank() != 0:
    #     c.run_dir = None
    #     print('c.run_dir', c.run_dir)
    # elif opts.nosubdir:
    #     c.run_dir = opts.outdir
    #     print('c.run_dir', c.run_dir)
    # else:
    #     prev_run_dirs = []
    #     if os.path.isdir(opts.outdir):
    #         prev_run_dirs = [x for x in os.listdir(opts.outdir) if os.path.isdir(os.path.join(opts.outdir, x))]
    #     prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    #     prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    #     cur_run_id = max(prev_run_ids, default=-1) + 1
    #     c.run_dir = os.path.join(opts.outdir, f'{cur_run_id:05d}-{desc}')
    #     assert not os.path.exists(c.run_dir)
    
    # Weather data
    c.data_type = opts.data_type
    c.data_config = opts.data_config
    c.task = opts.task

    # Print options.
    dist.print0()
    dist.print0('Training options:')
    dist.print0(json.dumps(c, indent=2))
    dist.print0()
    dist.print0(f'Output directory:        {c.run_dir}')
    dist.print0(f'Dataset path:            {c.dataset_kwargs.path}')
    dist.print0(f'Class-conditional:       {c.dataset_kwargs.use_labels}')
    dist.print0(f'Network architecture:    {opts.arch}')
    dist.print0(f'Preconditioning & loss:  {opts.precond}')
    dist.print0(f'Number of GPUs:          {dist.get_world_size()}')
    dist.print0(f'Batch size:              {c.batch_size}')
    dist.print0(f'Mixed-precision:         {c.network_kwargs.use_fp16}')
    dist.print0()

    # Dry run?
    if opts.dry_run:
        dist.print0('Dry run; exiting.')
        return

    # Create output directory.
    dist.print0('Creating output directory...')
    if dist.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c, f, indent=2)
        dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)
    

    # Train.
    training_loop.training_loop(**c)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
