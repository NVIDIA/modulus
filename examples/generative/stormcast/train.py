# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Train diffusion-based generative model using the techniques described in the
paper "Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import json
import torch
import wandb
from utils.misc import EasyDict
from utils import distributed as dist
#from modulus.distributed import DistributedManager
from utils.diffusions import training_loop
import glob
import argparse

def main(**kwargs):
    """Train regression or diffusion models for use in the StormCast (https://arxiv.org/abs/2408.10958) ML-based weather model
    """

    parser = argparse.ArgumentParser(
        description="Train regression or diffusion models for use in StormCast"
    )

    # Main options.
    parser.add_argument('--outdir',        help='Where to save the results', metavar='DIR',                   type=str, required=True)
    parser.add_argument('--precond',       help='Preconditioning & loss function', metavar='vp|ve|edm',       type=str, default='edm')
    parser.add_argument('--config_file',   help='Path to config file', metavar='FILE',                        type=str, required=True)
    parser.add_argument('--config_name',   help='Name of config to use', metavar='NAME',                      type=str, required=True)
    parser.add_argument('--log_to_wandb',  help='Log to wandb',                                               default=False, action='store_true')
    parser.add_argument('--run_id',        help='run id', metavar='INT',                                      type=int, default=None)

    # Hyperparameters.
    parser.add_argument('--batch',         help='Total batch size', metavar='INT',                            type=int, default=512)
    parser.add_argument('--batch-gpu',     help='Limit batch size per GPU', metavar='INT',                    type=int)
    parser.add_argument('--cbase',         help='Channel multiplier  [default: varies]', metavar='INT',       type=int)
    parser.add_argument('--lr',            help='Learning rate', metavar='FLOAT',                             type=float, default=4e-4)
    parser.add_argument('--dropout',       help='Dropout probability', metavar='FLOAT',                       type=float, default=0.13)
    parser.add_argument('--augment',       help='Augment probability', metavar='FLOAT',                       type=float, default=0.12)
    parser.add_argument('--xflip',         help='Enable dataset x-flips', metavar='BOOL',                     type=bool, default=False)

    # Performance-related.
    parser.add_argument('--fp16',          help='Enable mixed-precision training', metavar='BOOL',            type=bool, default=False)
    parser.add_argument('--ls',            help='Loss scaling', metavar='FLOAT',                              type=float, default=1)
    parser.add_argument('--bench',         help='Enable cuDNN benchmarking', metavar='BOOL',                  type=bool, default=True)
    parser.add_argument('--cache',         help='Cache dataset in CPU memory', metavar='BOOL',                type=bool, default=True)
    parser.add_argument('--workers',       help='DataLoader worker processes', metavar='INT',                 type=int, default=1)

    # I/O-related.
    parser.add_argument('--desc',          help='String to include in result dir name', metavar='STR',        type=str)
    parser.add_argument('--nosubdir',      help='Do not create a subdirectory for results',                   action='store_true')
    parser.add_argument('--tick',          help='How often to print progress', metavar='KIMG',                type=int, default=50)
    parser.add_argument('--snap',          help='How often to save snapshots', metavar='TICKS',               type=int, default=10)
    parser.add_argument('--dump',          help='How often to dump state', metavar='TICKS',                   type=int, default=10)
    parser.add_argument('--seed',          help='Random seed  [default: random]', metavar='INT',              type=int)
    parser.add_argument('--transfer',      help='Transfer learning from network pickle', metavar='PKL|URL',   type=str)
    parser.add_argument('--resume',        help='Resume from previous training state', metavar='PT',          type=str)
    parser.add_argument('-n', '--dry-run', help='Print training options and exit',                            action='store_true')


    opts = parser.parse_args()
    torch.multiprocessing.set_start_method('spawn')
    dist.init()
    #DistributedManager.initialize()
    #dist = DistributedManager()


    # Initialize config dict.
    c = EasyDict()

    # Training options.
    c.optimizer_kwargs = EasyDict(lr=opts.lr, betas=[0.9,0.999], eps=1e-8)
    c.update(loss_scaling=opts.ls, cudnn_benchmark=opts.bench)
    c.update(kimg_per_tick=opts.tick, snapshot_ticks=opts.snap, state_dump_ticks=opts.dump)

    # Random seed.
    if opts.seed is not None:
        c.seed = opts.seed
    else:
        seed = torch.randint(1 << 31, size=[], device=torch.device('cuda'))
        torch.distributed.broadcast(seed, src=0)
        c.seed = int(seed)
    

    # Description string.
    desc = f'hrrr-gpus{dist.get_world_size():d}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'
    
    desc = opts.config_name + '-' + desc

    # Pick output directory.
    cur_run_id = opts.run_id if opts.run_id is not None else 0
    c.run_dir = os.path.join(opts.outdir, f'{cur_run_id}-{desc}')

    #if run_dir exists, then resume training
    if os.path.exists(c.run_dir):
        training_states = sorted(glob.glob(os.path.join(c.run_dir, 'training-state-*.pt'))) 
        if training_states:
            print('Resuming training from previous run_dir: ' + c.run_dir)
            last_training_state = sorted(glob.glob(os.path.join(c.run_dir, 'training-state-*.pt')))[-1]
            last_network_snapshot = sorted(glob.glob(os.path.join(c.run_dir, 'network-snapshot-*.pkl')))[-1]
            last_kimg = int(re.fullmatch(r'network-snapshot-(\d+).pkl', os.path.basename(last_network_snapshot)).group(1))
            c.resume_pkl = last_network_snapshot
            c.resume_kimg = last_kimg
            c.resume_state_dump = last_training_state
            print('Resuming training from previous network-snapshot-*.pkl file: ' + last_network_snapshot)
            print('Resuming training from previous training-state-*.pt file: ' + last_training_state)
        
        
    # Transfer learning and resume. If a resume or transfer file is specified, it takes precedence over the existing run_dir.
    if opts.transfer is not None:
        if opts.resume is not None:
            raise ValueError('--transfer and --resume cannot be specified at the same time')
        c.resume_pkl = opts.transfer
        c.ema_rampup_ratio = None
    elif opts.resume is not None:
        match = re.fullmatch(r'training-state-(\d+).pt', os.path.basename(opts.resume))
        if not match or not os.path.isfile(opts.resume):
            raise ValueError('--resume must point to training-state-*.pt from a previous training run')
        c.resume_pkl = os.path.join(os.path.dirname(opts.resume), f'network-snapshot-{match.group(1)}.pkl')
        c.resume_kimg = int(match.group(1))
        c.resume_state_dump = opts.resume

    # Print options.
    if opts.dry_run:
        dist.print0('Dry run; exiting.')
        return


    

    # Create output directory.
    dist.print0('Creating output directory...')
    if dist.get_rank() == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, 'training_options.json'), 'wt') as f:
            json.dump(c, f, indent=2)
        #dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

        if opts.log_to_wandb:
            entity, project = 'nv-research-climate', 'hrrr'
            entity = entity
            wandb_project = project
            wandb_name = opts.config_name + '_' + desc
            wandb_group = opts.config_name + '_' + str(cur_run_id)
            os.makedirs(os.path.join(c.run_dir, "wandb"), exist_ok=True)
            wandb.init(dir=os.path.join(c.run_dir, "wandb"),
                config=c, name=wandb_name, group=wandb_group, project=wandb_project, entity=entity, 
                resume=opts.resume, mode='online')
    
    #config options
    c.config_file = opts.config_file
    c.config_name = opts.config_name
    c.log_to_wandb = opts.log_to_wandb
    
    # Train.
    training_loop.training_loop(**c)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
