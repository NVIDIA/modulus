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

import torch
import os
import hydra
import wandb
import matplotlib.pyplot as plt
import time
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import SequentialLR

# Add eval to OmegaConf TODO: Remove when OmegaConf is updated
OmegaConf.register_new_resolver("eval", eval)

try:
    from apex import optimizers
except:
    raise ImportError(
        "training requires apex package for optimizer."
        + "See https://github.com/nvidia/apex for install details."
    )

from modulus import Module
from modulus.distributed import DistributedManager
from modulus.utils import StaticCaptureTraining, StaticCaptureEvaluateNoGrad
from modulus.launch.logging import (
    LaunchLogger,
    PythonLogger,
    initialize_mlflow,
    RankZeroLoggingWrapper,
)
from modulus.launch.utils import load_checkpoint, save_checkpoint

from datapipe import ERA5Datapipe

def loss_func(x, y, p=2.0):
    yv = y.reshape(x.size()[0], -1)
    xv = x.reshape(x.size()[0], -1)
    diff_norms = torch.linalg.norm(xv - yv, ord=p, dim=1)
    y_norms = torch.linalg.norm(yv, ord=p, dim=1)

    return torch.mean(diff_norms / y_norms)

@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    # Resolve config so that all values are concrete
    OmegaConf.resolve(cfg)

    # Initialize distributed environment for training
    DistributedManager.initialize()
    dist = DistributedManager()

    # Initialize loggers
    initialize_mlflow(
        experiment_name="Modulus-Launch-Dev",
        experiment_desc="Modulus launch development",
        run_name=f"{cfg.model.name}-trainng",
        run_desc="ERA5 Training",
        user_name="Modulus User",
        mode="offline",
    )
    LaunchLogger.initialize(use_mlflow=True)  # Modulus launch logger
    logger = PythonLogger("main")  # General python logger
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)  # Rank 0 logger

    # Initialize model
    model = Module.instantiate(
        {
            "__name__": cfg.model.name,
            "__args__": cfg.model.args,
        }
    )
    model = model.to(dist.device)

    # Distributed learning
    if dist.world_size > 1:
        ddps = torch.cuda.Stream()
        with torch.cuda.stream(ddps):
            model = DistributedDataParallel(
                model,
                device_ids=[dist.local_rank],
                output_device=dist.device,
                broadcast_buffers=dist.broadcast_buffers,
                find_unused_parameters=dist.find_unused_parameters,
            )
        torch.cuda.current_stream().wait_stream(ddps)
        torch.device.synchronize()

    # Initialize optimizer
    OptimizerClass = getattr(optimizers, cfg.training.optimizer.name)
    optimizer = OptimizerClass(model.parameters(), **cfg.training.optimizer.args)

    # Initialize scheduler
    schedulers = []
    milestones = []
    for scheduler_cfg in cfg.training.schedulers:
        SchedulerClass = getattr(torch.optim.lr_scheduler, scheduler_cfg.name)
        schedulers.append(SchedulerClass(optimizer, **scheduler_cfg.args))
        if not milestones:
            milestones.append(scheduler_cfg.num_iterations)
        else:
            milestones.append(milestones[-1] + scheduler_cfg.num_iterations)
    milestones.pop(-1)
    scheduler = SequentialLR(
        optimizer,
        schedulers=schedulers,
        milestones=milestones,
    )

    # Attempt to load latest checkpoint if one exists
    if dist.world_size > 1:
        torch.distributed.barrier()
    loaded_epoch = load_checkpoint(
        "./checkpoints",
        models=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=dist.device,
    )

    # Initialize filesytem
    if cfg.filesystem.type == "local":
        fs = None
    elif cfg.filesystem.type == "s3":
        fs = fsspec.filesystem(
            cfg.filesystem.type,
            key=cfg.filesystem.key,
            secret=os.environ["AWS_SECRET_ACCESS_KEY"],
            client_kwargs={
                "endpoint_url": cfg.filesystem.endpoint_url,
                "region_name": cfg.filesystem.region_name,
            },
        )

    # Initialize datapipes
    train_datapipe = ERA5Datapipe(
        static_variables=cfg.datapipe.static_variables,
        input_variables=cfg.datapipe.input_variables,
        base_path=cfg.dataset.base_path,
        fs=fs,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        device=dist.device,
        process_rank=dist.rank,
        world_size=dist.world_size,
    )

    # Unroll network
    def unroll(model, static_variables, input_variables, nr_input_steps):
        # Get number of steps to unroll
        steps = input_variables.shape[1] - nr_input_steps

        # Create first input
        static_i = static_variables[:, :nr_input_steps]
        input_i = input_variables[:, :nr_input_steps]

        # Unroll
        predict_variables = []
        for i in range(steps):
            # Get prediction for the first step
            predict_i = model(torch.cat([static_i, input_i], dim=2).flatten(1, 2))
            predict_variables.append(predict_i)

            # Create new inputs
            static_i = static_variables[:, i : nr_input_steps + i]
            input_i = torch.cat([input_i[:, 1:], predict_i.unsqueeze(1)], dim=1)

        # Stack predictions
        predict_variables = torch.stack(predict_variables, dim=1)

        return predict_variables

    # Evaluation forward pass
    @StaticCaptureEvaluateNoGrad(model=model, logger=logger, use_graphs=False)
    def eval_forward(model, static_variables, input_variables, nr_input_steps):
        return unroll(model, static_variables, input_variables, nr_input_steps)

    # Training forward pass
    @StaticCaptureTraining(model=model, optim=optimizer, logger=logger)
    def train_step_forward(model, static_variables, input_variables, nr_input_steps):
        # Forward pass
        predict_variables = unroll(model, static_variables, input_variables, nr_input_steps)

        # Compute loss
        loss = loss_func(predict_variables, input_variables[:, nr_input_steps:])

        return loss

    # Main training loop
    max_epoch = 80
    for epoch in range(max(1, loaded_epoch + 1), max_epoch + 1):
        # Wrap epoch in launch logger for console / WandB logs
        with LaunchLogger(
            "train", epoch=epoch, num_mini_batch=len(train_datapipe), epoch_alert_freq=10
        ) as log:

            # Training loop
            for j, data in enumerate(train_datapipe):
                # Get input and static variables from datapipe
                input_variables, static_variables = data[0]["input"], data[0]["static"]

                # Perform remapping
                # TODO: Add remapping

                # Perform training step
                loss = train_step_forward(model, static_variables, input_variables, cfg.training.nr_input_steps)
                log.log_minibatch({"loss": loss.detach()})

            # Log learning rate
            log.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})

        # Perform validation
        if dist.rank == 0:

            # Wrap validation in launch logger for console / WandB logs
            with LaunchLogger("valid", epoch=epoch) as log:
                pass

                ## Validation loop
                #error = validation_step(
                #    eval_step_forward, model, validation_datapipe, epoch=epoch
                #)
                #log.log_epoch({"Validation error": error})

        # Sync after each epoch
        if dist.world_size > 1:
            torch.distributed.barrier()

        # Step scheduler
        scheduler.step()

        # Save checkpoint
        if (epoch % 5 == 0 or epoch == 1) and dist.rank == 0:
            # Use Modulus Launch checkpoint
            save_checkpoint(
                "./checkpoints",
                models=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
            )

    # Finish training
    if dist.rank == 0:
        logger.info("Finished training!")


if __name__ == "__main__":
    main()
