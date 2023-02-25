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

import glob
import re
import torch

from typing import Union, List, NewType
from pathlib import Path
from torch.optim.lr_scheduler import _LRScheduler
from torch.cuda.amp import GradScaler

from modulus.distributed import DistributedManager
from modulus.utils.capture import _StaticCapture
from modulus.launch.logging import PythonLogger

optimizer = NewType("optimizer", torch.optim)
scheduler = NewType("scheduler", _LRScheduler)
scaler = NewType("scaler", GradScaler)

checkpoint_logging = PythonLogger("checkpoint")


def _get_checkpoint_filename(
    path: str, index: Union[int, None] = None, saving: bool = False
) -> str:
    """Gets the file name /path of checkpoint

    This function has three different ways of providing a checkout filename:
    - If supplied an index this will return the checkpoint name using that index.
    - If index is None and saving is false, this will get the checkpoint with the
    largest index (latest save).
    - If index is None and saving is true, it will return the next valid index file name
    which is calculated by indexing the largest checkpoint index found by one.

    Parameters
    ----------
    path : str
         Path to checkpoints
    index : Union[int, None], optional
        Checkpoint index, by default None
    saving : bool, optional
        Get filename for saving a new checkpoint, by default False

    Returns
    -------
    str
        Checkpoint file name
    """
    # Get model parallel rank so all processes in the first model parallel group
    # can save their checkpoint. In the case without model parallelism, model_parallel_rank
    # should be the same as the process rank itself and only rank 0 saves
    manager = DistributedManager()
    model_parallel_rank = (
        manager.group_rank("model_parallel") if manager.distributed else 0
    )

    # Input file name
    checkpoint_filename = str(Path(path) / f"checkpoint.{model_parallel_rank}")
    # If epoch is provided load that file
    if not index is None:
        checkpoint_filename = checkpoint_filename + f".{index}"
        checkpoint_filename += ".pth"
    # Otherwise try loading the latest epoch or rolling checkpoint
    else:
        file_names = []
        for fname in glob.glob(checkpoint_filename + "*.pth", recursive=False):
            file_names.append(Path(fname).name)

        if len(file_names) > 0:
            # If checkpoint from a null index save exists load that
            # This is the most likely line to error since it will fail with invalid checkpoint names
            file_idx = [
                int(re.sub(f"^checkpoint.{model_parallel_rank}.|.pth", "", fname))
                for fname in file_names
            ]
            file_idx.sort()
            # If we are saving index by 1 to get the next free file name
            if saving:
                checkpoint_filename = checkpoint_filename + f".{file_idx[-1]+1}"
            else:
                checkpoint_filename = checkpoint_filename + f".{file_idx[-1]}"
            checkpoint_filename += ".pth"
        else:
            checkpoint_filename += ".0.pth"

    return checkpoint_filename


def save_checkpoint(
    path: str,
    model: Union[torch.nn.Module, List[torch.nn.Module], None] = None,
    optimizer: Union[optimizer, None] = None,
    scheduler: Union[scheduler, None] = None,
    scaler: Union[scaler, None] = None,
    epoch: Union[int, None] = None,
) -> None:
    """Training checkpoint saving utility

    This will save a training checkpoint in the provided path following the file naming
    convention "checkpoint.{model parallel id}.{epoch/index}.pth". The load checkpoint
    method in Modulus core can then be used to read this file.

    Parameters
    ----------
    path : str
        Path to save the training checkpoint
    model : Union[torch.nn.Module, List[torch.nn.Module], None], optional
        A single or list of PyTorch models, by default None
    optimizer : Union[optimizer, None], optional
        Optimizer, by default None
    scheduler : Union[scheduler, None], optional
        Learning rate scheduler, by default None
    scaler : Union[scaler, None], optional
        AMP grad scaler. Will attempt to save on in static capture if none provided, by
        default None
    epoch : Union[int, None], optional
        Epoch checkpoint to load. If none this will save the checkpoint in the next valid
        index, by default None
    """
    # Create checkpoint directory if it does not exist
    if not Path(path).is_dir():
        checkpoint_logging.warning(
            f"Output directory {path} does not exist, will " "attempt to create"
        )
        Path(path).mkdir(parents=True, exist_ok=True)

    checkpoint_dict = {}
    # Model state dict
    if isinstance(model, list):
        # If a list of models is provided, add them all into a global dictionary
        model_state_dict = {}
        for i, model0 in enumerate(model):
            model_state_dict[i] = model0.state_dict()
        checkpoint_dict["model_state_dict"] = model_state_dict
    elif model:
        model_state_dict = model.state_dict()
        checkpoint_dict["model_state_dict"] = model_state_dict

    # Optimizer state dict
    if optimizer:
        checkpoint_dict["optimizer_state_dict"] = optimizer.state_dict()

    # Scheduler state dict
    if scheduler:
        checkpoint_dict["scheduler_state_dict"] = scheduler.state_dict()

    # Scheduler state dict
    if scaler:
        checkpoint_dict["scaler_state_dict"] = scaler.state_dict()
    # Static capture is being used, save its grad scaler
    elif _StaticCapture.scaler_singleton:
        checkpoint_dict[
            "scaler_state_dict"
        ] = _StaticCapture.scaler_singleton.state_dict()

    # Output file name
    output_filename = _get_checkpoint_filename(path, epoch, True)
    if epoch:
        checkpoint_dict["epoch"] = epoch

    # Save checkpoint to memory
    checkpoint_logging.info(f"Saving checkpoint: {output_filename}")
    torch.save(
        checkpoint_dict,
        output_filename,
    )


def load_checkpoint(
    path: str,
    model: Union[torch.nn.Module, List[torch.nn.Module], None] = None,
    optimizer: Union[optimizer, None] = None,
    scheduler: Union[scheduler, None] = None,
    scaler: Union[scaler, None] = None,
    epoch: Union[int, None] = None,
    device: Union[str, torch.device] = "cpu",
) -> int:
    """Checkpoint loading utility

    This loader is designed to be used with the save checkpoint utility in Modulus Launch.
    Given a path, this method will try to find a checkpoint and load state dictionaries into
    the provided training objects.

    Parameters
    ----------
    path : str
        Path to training checkpoint
    model : Union[torch.nn.Module, List[torch.nn.Module], None], optional
        A single or list of PyTorch models, by default None
    optimizer : Union[optimizer, None], optional
        Optimizer, by default None
    scheduler : Union[scheduler, None], optional
        Learning rate scheduler, by default None
    scaler : Union[scaler, None], optional
        AMP grad scaler, by default None
    epoch : Union[int, None], optional
        Epoch checkpoint to load. If none is provided this will attempt to load the
        checkpoint with the largest index, by default None
    device : Union[str, torch.device], optional
        Target device, by default "cpu"

    Returns
    -------
    int
        Loaded epoch
    """
    # Check if checkpoint directory exists
    if not Path(path).is_dir():
        checkpoint_logging.warning(
            f"Provided checkpoint directory {path} does not exist, skipping load"
        )
        return 0

    checkpoint_filename = _get_checkpoint_filename(path, epoch)

    if not Path(checkpoint_filename).is_file():
        checkpoint_logging.warning(
            f"Could not find valid checkpoint file, skipping load"
        )
        return 0

    checkpoint_dict = torch.load(checkpoint_filename, map_location=device)
    checkpoint_logging.success(
        f"Loaded checkpoint file {checkpoint_filename} to device {device}"
    )
    # Model state dict
    if "model_state_dict" in checkpoint_dict:
        if isinstance(model, list):
            # If a list of models is provided, add them all into a global dictionary
            model_state_dict = checkpoint_dict["model_state_dict"]
            for i, model0 in enumerate(model):
                model0.load_state_dict(model_state_dict[i])
                checkpoint_logging.success(f"Loaded model {i} state dictionary")
        elif model:
            model.load_state_dict(checkpoint_dict["model_state_dict"])
            checkpoint_logging.success("Loaded model state dictionary")

    # Optimizer state dict
    if optimizer and "optimizer_state_dict" in checkpoint_dict:
        optimizer.load_state_dict(checkpoint_dict["optimizer_state_dict"])
        checkpoint_logging.success("Loaded optimizer state dictionary")

    # Scheduler state dict
    if scheduler and "scheduler_state_dict" in checkpoint_dict:
        scheduler.load_state_dict(checkpoint_dict["scheduler_state_dict"])
        checkpoint_logging.success("Loaded scheduler state dictionary")

    # Scheduler state dict
    if "scaler_state_dict" in checkpoint_dict:
        if scaler:
            scaler.load_state_dict(checkpoint_dict["scaler_state_dict"])
            checkpoint_logging.success("Loaded grad scaler state dictionary")
        else:
            # Load into static capture for initialization
            _StaticCapture.scaler_dict = checkpoint_dict["scaler_state_dict"]

    epoch = 0
    if "epoch" in checkpoint_dict:
        epoch = checkpoint_dict["epoch"]
    return epoch
