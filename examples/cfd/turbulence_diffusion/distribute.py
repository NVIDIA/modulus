# ignore_header_test
# coding=utf-8
#
# SPDX-FileCopyrightText: Copyright (c) 2024 - Edmund Ross
# SPDX-License-Identifier: MIT
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import params
import util

import torch
import torch.distributed as dist
import torch.multiprocessing as mult

from itertools import chain


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12359'

    # Initialise the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def spawn_processes(fn):
    """Setup hyperparameters, initialise experiment"""
    args = params.get_args()
    experiment_path = util.initialise_experiment(args)

    n_gpus = torch.cuda.device_count()
    print(f'[{args.experiment_name}] Found {n_gpus} GPUs. Spawning processes...')

    # Spawn the processes
    mult.spawn(fn,
               args=(n_gpus, args, experiment_path),
               nprocs=n_gpus,
               join=True)


def cleanup():
    """Clean up distributed session"""
    dist.destroy_process_group()


def load(experiment_path, args_model, ddp_diffusion, optimizer=None, ema=None):
    """Load checkpoint from disk"""
    model_path = util.to_path(experiment_path, 'checkpoints', args_model)

    checkpoint = torch.load(model_path)
    epoch_start = checkpoint['epoch'] + 1
    ddp_diffusion.load_state_dict(checkpoint['diffusion'])

    if ema is not None:
        ema.load_state_dict(checkpoint['ema'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    # Make sure everyone is loaded before proceeding
    dist.barrier()
    return epoch_start


def save(experiment_path, epoch, ddp_diffusion, optimizer, ema):
    """Save checkpoint to disk"""
    torch.save({
        'epoch': epoch,
        'diffusion': ddp_diffusion.state_dict(),
        'optimizer': optimizer.state_dict(),
        'ema': ema.state_dict(),
    }, util.to_path(experiment_path, 'checkpoints', f'model_{epoch}.pt'))


def interleave_arrays(*arrays):
    """Collect data from the GPUs, and interleave into a single array, respecting the order"""
    interleaved = list(chain.from_iterable(zip(*arrays)))
    return interleaved
