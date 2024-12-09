# ignore_header_test
# coding=utf-8
#
# SPDX-FileCopyrightText: Copyright (c) 2024 - Edmund Ross
# SPDX-License-Identifier: Apache-2.0
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
