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


from abc import ABC, abstractmethod
import functools
import logging
import os
from typing import Any, Mapping, Optional

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from termcolor import colored

from torch import nn

import torch
import wandb

from physicsnemo.distributed import DistributedManager

logger = logging.getLogger("lmgn")


class TermColorFormatter(logging.Formatter):
    """Custom logging formatter that colors the log output based on log level."""

    def __init__(
        self,
        fmt: Optional[str] = None,
        datefmt: Optional[str] = None,
        style: str = "%",
        validate: bool = True,
        log_colors: Optional[Mapping[str, str]] = None,
        *,
        defaults=None,
    ):
        super().__init__(fmt, datefmt, style, validate, defaults=defaults)
        self.log_colors = log_colors if log_colors is not None else {}

    def format(self, record):
        log_message = super().format(record)
        color = self.log_colors.get(record.levelname, "white")
        return colored(log_message, color)


def init_python_logging(
    config: DictConfig, rank: int = 0, base_filename: str = "train"
) -> None:
    """Initializes Python logging."""

    pylog_cfg = OmegaConf.select(config, "logging.python")
    if pylog_cfg is None:
        return

    # Set up Python loggers.
    pylog_cfg.output = config.output
    pylog_cfg.rank = rank
    pylog_cfg.base_filename = base_filename
    # Enable logging only on rank 0, if requested.
    if pylog_cfg.rank0_only and pylog_cfg.rank != 0:
        pylog_cfg.handlers = {}
        for l in pylog_cfg.loggers.values():
            l.handlers = []
    # Configure logging.
    logging.config.dictConfig(OmegaConf.to_container(pylog_cfg, resolve=True))


def get_gpu_info() -> str:
    """Returns information about available GPUs."""

    if not torch.cuda.is_available():
        return "\nCUDA is not available."

    res = f"\n\nPyTorch CUDA Version: {torch.version.cuda}\nAvailable GPUs:"
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        props = torch.cuda.get_device_properties(i)
        total_memory = props.total_memory / (1024**3)
        res += (
            f"\n{torch.device(i)}: {name} ("
            f"{total_memory:.0f} GiB, "
            f"sm_{props.major}{props.minor})"
        )

    res += f"\nCurrent device: {torch.cuda.current_device()}\n"
    return res


def rank0(func):
    """Decorator that allows the function to be executed only in rank 0 process."""

    @functools.wraps(func)
    def rank0_only(*args, **kwargs):
        if DistributedManager().rank == 0:
            func(*args, **kwargs)

    return rank0_only


class ExperimentLogger(ABC):
    """Provides unified interface to a logger."""

    @abstractmethod
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        pass

    @abstractmethod
    def log_image(self, tag: str, value, step: int) -> None:
        pass

    @abstractmethod
    def log(self, data: Mapping[str, Any], step: int) -> None:
        pass

    @abstractmethod
    def watch_model(self, model: nn.Module) -> None:
        pass


class WandBLogger(ExperimentLogger):
    """Wrapper for Weights & Biases logger."""

    def __init__(self, **kwargs) -> None:
        if DistributedManager().rank != 0:
            return

        if wandb_key := kwargs.pop("wandb_key", None) is not None:
            logger.warning("Passing W&B key via config is not recommended.")
            wandb.login(key=wandb_key)

        # If wandb_id is not provided to resume the experiment,
        # create new id if wandb_id.txt does not exist,
        # otherwise - load id from the file.
        if wandb_id := kwargs.pop("id", None) is None:
            wandb_id_file = os.path.join(kwargs["dir"], "wandb_id.txt")
            if not os.path.exists(wandb_id_file):
                wandb_id = wandb.util.generate_id()
                with open(wandb_id_file, "w", encoding="utf-8") as f:
                    f.write(wandb_id)
                logger.info(f"Starting new wandb run: {wandb_id}")
            else:
                with open(wandb_id_file, encoding="utf-8") as f:
                    wandb_id = f.read()
                logger.info(f"Resuming wandb run: {wandb_id}")
        resume = kwargs.pop("resume", "allow")

        self.watch = kwargs.pop("watch_model", False)

        wandb.init(**kwargs, id=wandb_id, resume=resume)

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        wandb.log({tag: value}, step=step)

    def log_image(self, tag: str, value, step: int) -> None:
        wandb.log({tag: wandb.Image(value)}, step=step)

    def log(self, data: Mapping[str, Any], step: int) -> None:
        wandb.log(data, step=step)

    def watch_model(self, model: nn.Module):
        if self.watch:
            wandb.watch(model)


class CompositeLogger(ExperimentLogger):
    """Wraps a list of loggers providing unified interface."""

    loggers: dict[str, ExperimentLogger] = None

    def __init__(self, config: DictConfig) -> None:
        if DistributedManager().rank != 0:
            self.loggers = {}
            return
        # Instantiate loggers only when running on rank 0.
        self.loggers = instantiate(config.loggers)

    @rank0
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        for l in self.loggers.values():
            l.log_scalar(tag, value, step)

    @rank0
    def log_image(self, tag: str, value: float, step: int) -> None:
        for l in self.loggers.values():
            l.log_image(tag, value, step)

    @rank0
    def log(self, data: Mapping[str, Any], step: int) -> None:
        for l in self.loggers.values():
            l.log(data, step)

    @rank0
    def watch_model(self, model: nn.Module) -> None:
        for l in self.loggers.values():
            l.watch_model(model)
