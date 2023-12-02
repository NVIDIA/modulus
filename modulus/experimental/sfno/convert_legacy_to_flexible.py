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

import argparse
import os
import tempfile
from functools import partial

import pynvml
import torch

from modulus.experimental.sfno.networks.model_package import (
    _load_static_data,
    MODEL_PACKAGE_CHECKPOINT_PATH,
    save_model_package,
    LocalPackage,
)
from modulus.experimental.sfno.utils import logging_utils

import torch.distributed as dist

from modulus.experimental.sfno.networks.models import get_model

# distributed computing stuff
from modulus.experimental.sfno.utils import comm
from modulus.experimental.sfno.utils.trainer import Trainer
from modulus.experimental.sfno.utils.YParams import ParamsBase


class CheckpointSaver(Trainer):
    """
    Inferencer class holding all the necessary information to perform inference. Design is similar to Trainer, however only keeping the necessary information.
    """

    def __init__(self, params, world_rank):
        self.params = None
        self.world_rank = world_rank

        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        else:
            self.device = torch.device("cpu")

        # resuming needs is set to False so loading checkpoints does not attempt to set the optimizer state
        params["resuming"] = False

        self.params = params
        self.model = get_model(params).to(self.device)
        self.preprocessor = self.model.preprocessor
        self.iters = None
        self.optimizer = None
        self.epoch = None
        self.scheduler = None

        # print model
        if self.world_rank == 0:
            print(self.model)


def get_params(path):
    config = os.path.join(path, "config.json")
    return ParamsBase.from_json(config)


def save_checkpoint(path, output_path, rank, world_size, store_path):
    package = LocalPackage(path)
    params = get_params(path)
    store = dist.FileStore(store_path, world_size)
    # setup distributed
    dist.init_process_group(
        store=store, backend="nccl", rank=rank, world_size=world_size
    )
    # adjust checkpoint_path to be inside of ``path``. The checkpoint may not be in
    # the same location it was during training.
    checkpoint_template = os.path.basename(params.checkpoint_path)
    checkpoint_path = os.path.join(path, "training_checkpoints", checkpoint_template)
    params.log_to_wandb = False
    with torch.cuda.device(dist.get_rank() % torch.cuda.device_count()):
        _load_static_data(package, params)
        model_parallel_sizes = params.get("model_parallel_sizes", [1])
        model_parallel_names = params.get("model_parallel_names", ["model"])
        params.model_parallel_size = comm.init_model_parallel_info(
            sizes=model_parallel_sizes, names=model_parallel_names
        )
        saver = CheckpointSaver(params, world_rank=comm.get_world_rank())
        saver.restore_checkpoint(
            checkpoint_path, checkpoint_mode=params["load_checkpoint"]
        )
        output_checkpoint_path = os.path.join(
            output_path, MODEL_PACKAGE_CHECKPOINT_PATH
        )
        if rank == 0:
            os.makedirs(os.path.dirname(output_checkpoint_path), exist_ok=True)
        dist.barrier()
        saver.save_checkpoint(output_checkpoint_path, checkpoint_mode="flexible")

        params.experiment_dir = output_path
        if rank == 0:
            save_model_package(params)


help_str = """Convert legacy (checkpoint files per-rank) model packages to a single flexible
model package. 

This script should be run as a normal python script from an interactive session
with era5_wind installed.  Under the hood, it infers the number of needed ranks
to loads the original checkpoint and then spawns the needed processes.

Example::

    python3 convert_legacy_to_flexible.py /path_to_run_dir/sfno_linear_73chq_sc2_layers8_edim960_wstgl2/ngpu256_sp4/ package/
"""

if __name__ == "__main__":
    logging_utils.config_logger()
    parser = argparse.ArgumentParser(usage=help_str)
    parser.add_argument(
        "experiment_root",
        help="for example: sfno_linear_73chq_sc2_layers8_edim960_wstgl2/ngpu256_sp4",
    )
    parser.add_argument("output", help="Where to save the collected checkpoint.")
    args = parser.parse_args()
    f = tempfile.mktemp()
    params = get_params(args.experiment_root)
    nproc = len(params.model_parallel_sizes)
    torch.multiprocessing.spawn(
        partial(save_checkpoint, args.experiment_root, args.output),
        args=(nproc, f),
        nprocs=nproc,
    )
