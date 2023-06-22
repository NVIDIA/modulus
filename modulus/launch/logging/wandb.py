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

"""Weights and Biases Routines and Utilities"""

import logging
import os
import wandb
from typing import Literal
from pathlib import Path
from datetime import datetime
from wandb import AlertLevel
from modulus.distributed import DistributedManager
from .utils import create_ddp_group_tag

DEFAULT_WANDB_CONFIG = "~/.netrc"
logger = logging.getLogger(__name__)

_WANDB_INITIALIZED = False


def initialize_wandb(
    project: str,
    entity: str,
    name: str = "train",
    group: str = None,
    sync_tensorboard: bool = False,
    save_code: bool = False,
    resume: str = None,
    config=None,
    mode: Literal["offline", "online", "disabled"] = "offline",
    results_dir: str = None,
):
    """Function to initialize wandb client with the weights and biases server.

    Parameters
    ----------
    project : str
        Name of the project to sync data with
    entity : str,
        Name of the wanbd entity
    sync_tensorboard : bool, optional
        sync tensorboard summary writer with wandb, by default False
    save_code : bool, optional
        Whether to push a copy of the code to wandb dashboard, by default False
    name : str, optional
        Name of the task running, by default "train"
    group : str, optional
        Group name of the task running. Good to set for ddp runs, by default None
    resume: str, optional
        Sets the resuming behavior. Options: "allow", "must", "never", "auto" or None,
        by default None.
    config : optional
        a dictionary-like object for saving inputs , like hyperparameters.
        If dict, argparse or absl.flags, it will load the key value pairs into the
        wandb.config object. If str, it will look for a yaml file by that name,
        by default None.
    mode: str, optional
        Can be "offline", "online" or "disabled", by default "offline"
    results_dir : str, optional
        Output directory of the experiment, by default "/<run directory>/wandb"
    """

    # Set default value here for Hydra
    if results_dir is None:
        results_dir = str(Path("./wandb").absolute())

    wandb_dir = results_dir
    if DistributedManager.is_initialized() and DistributedManager().distributed:
        if group is None:
            group = create_ddp_group_tag()
        start_time = datetime.now().astimezone()
        time_string = start_time.strftime("%m/%d/%y_%H:%M:%S")
        wandb_name = f"{name}_Process_{DistributedManager().rank}_{time_string}"
    else:
        start_time = datetime.now().astimezone()
        time_string = start_time.strftime("%m/%d/%y_%H:%M:%S")
        wandb_name = f"{name}_{time_string}"

    if not os.path.exists(wandb_dir):
        os.makedirs(wandb_dir)

    wandb.init(
        project=project,
        entity=entity,
        sync_tensorboard=sync_tensorboard,
        name=wandb_name,
        resume=resume,
        config=config,
        mode=mode,
        dir=wandb_dir,
        group=group,
        save_code=save_code,
    )


def alert(title, text, duration=300, level=0, is_master=True):
    """Send alert."""
    alert_levels = {0: AlertLevel.INFO, 1: AlertLevel.WARN, 2: AlertLevel.ERROR}
    if is_wandb_initialized() and is_master:
        wandb.alert(
            title=title, text=text, level=alert_levels[level], wait_duration=duration
        )


def is_wandb_initialized():
    """Check if wandb has been initialized."""
    global _WANDB_INITIALIZED
    return _WANDB_INITIALIZED
