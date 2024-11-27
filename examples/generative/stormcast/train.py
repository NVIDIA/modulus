# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
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

"""Train diffusion-based generative model using the techniques described in the
paper "Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import json
import torch
import wandb
import glob
import argparse
from modulus.distributed import DistributedManager

from utils.misc import EasyDict, print0
from utils.diffusions import training_loop


def main(**kwargs):
    """Train regression or diffusion models for use in the StormCast (https://arxiv.org/abs/2408.10958) ML-based weather model"""

    parser = argparse.ArgumentParser(
        description="Train regression or diffusion models for use in StormCast"
    )

    # Main options.
    parser.add_argument(
        "--outdir",
        help="Where to save the results",
        metavar="DIR",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--config_file",
        help="Path to config file",
        metavar="FILE",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--config_name",
        help="Name of config to use",
        metavar="NAME",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--log_to_wandb", help="Log to wandb", default=False, action="store_true"
    )
    parser.add_argument(
        "--run_id", help="run id", metavar="INT", type=int, default=None
    )

    # Performance-related.
    parser.add_argument(
        "--bench",
        help="Enable cuDNN benchmarking",
        metavar="BOOL",
        type=bool,
        default=True,
    )

    # I/O-related.
    parser.add_argument(
        "--desc", help="String to include in result dir name", metavar="STR", type=str
    )
    parser.add_argument(
        "--dump", help="How often to dump state", metavar="TICKS", type=int, default=10
    )
    parser.add_argument(
        "--seed", help="Random seed  [default: random]", metavar="INT", type=int
    )
    parser.add_argument(
        "--resume", help="Resume from previous training state", metavar="PT", type=str
    )
    parser.add_argument(
        "-n", "--dry-run", help="Print training options and exit", action="store_true"
    )

    # Initialize
    opts = parser.parse_args()
    DistributedManager.initialize()
    dist = DistributedManager()

    # Initialize config dict.
    c = EasyDict()

    # Training options.
    c.optimizer_kwargs = EasyDict(betas=[0.9, 0.999], eps=1e-8)
    c.update(cudnn_benchmark=opts.bench, state_dump_ticks=opts.dump)

    # Random seed.
    if opts.seed is not None:
        c.seed = opts.seed
    else:
        seed = torch.randint(1 << 31, size=[], device=torch.device("cuda"))
        torch.distributed.broadcast(seed, src=0)
        c.seed = int(seed)

    # Description string.
    desc = f"hrrr-gpus{dist.world_size:d}"
    if opts.desc is not None:
        desc += f"-{opts.desc}"

    desc = opts.config_name + "-" + desc

    # Pick output directory.
    cur_run_id = opts.run_id if opts.run_id is not None else 0
    c.run_dir = os.path.join(opts.outdir, f"{cur_run_id}-{desc}")

    # if run_dir exists, then resume training
    if os.path.exists(c.run_dir):
        training_states = sorted(
            glob.glob(os.path.join(c.run_dir, "training-state-*.pt"))
        )
        if training_states:
            print0("Resuming training from previous run_dir: " + c.run_dir)
            last_training_state = sorted(
                glob.glob(os.path.join(c.run_dir, "training-state-*.pt"))
            )[-1]
            last_kimg = int(
                re.fullmatch(
                    r"training-state-(\d+).pt", os.path.basename(last_training_state)
                ).group(1)
            )
            c.resume_kimg = last_kimg
            c.resume_state_dump = last_training_state
            print0(
                "Resuming training from previous training-state-*.pt file: "
                + last_training_state
            )

    if opts.resume is not None:
        match = re.fullmatch(r"training-state-(\d+).pt", os.path.basename(opts.resume))
        if not match or not os.path.isfile(opts.resume):
            raise ValueError(
                "--resume must point to training-state-*.pt from a previous training run"
            )
        c.resume_kimg = int(match.group(1))
        c.resume_state_dump = opts.resume

    # Print options.
    if opts.dry_run:
        print0("Dry run; exiting.")
        return

    # Create output directory.
    print0("Creating output directory...")
    if dist.rank == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, "training_options.json"), "wt") as f:
            json.dump(c, f, indent=2)
        # Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=True)

        if opts.log_to_wandb:
            entity, project = "nv-research-climate", "hrrr"
            entity = entity
            wandb_project = project
            wandb_name = opts.config_name + "_" + desc
            wandb_group = opts.config_name + "_" + str(cur_run_id)
            os.makedirs(os.path.join(c.run_dir, "wandb"), exist_ok=True)
            wandb.init(
                dir=os.path.join(c.run_dir, "wandb"),
                config=c,
                name=wandb_name,
                group=wandb_group,
                project=wandb_project,
                entity=entity,
                resume=opts.resume,
                mode="online",
            )

    # config options
    c.config_file = opts.config_file
    c.config_name = opts.config_name
    c.log_to_wandb = opts.log_to_wandb

    # Train.
    training_loop.training_loop(**c)


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
