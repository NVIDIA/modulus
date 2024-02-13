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

import torch
import os
import hydra
import wandb
import matplotlib.pyplot as plt

from torch.nn.parallel import DistributedDataParallel
from omegaconf import DictConfig

from modulus.models.swinvrnn import SwinRNN
from modulus.datapipes.climate import WeatherBenchDatapipe
from modulus.distributed import DistributedManager
from modulus.utils import StaticCaptureTraining, StaticCaptureEvaluateNoGrad

from modulus.launch.logging import LaunchLogger, PythonLogger, initialize_mlflow
from modulus.launch.utils import load_checkpoint, save_checkpoint

try:
    from apex import optimizers
except:
    raise ImportError(
        "FCN training requires apex package for optimizer."
        + "See https://github.com/nvidia/apex for install details."
    )


def loss_func(x, y, p=2.0):
    yv = y.reshape(x.size()[0], -1)
    xv = x.reshape(x.size()[0], -1)
    diff_norms = torch.linalg.norm(xv - yv, ord=p, dim=1)
    y_norms = torch.linalg.norm(yv, ord=p, dim=1)

    return torch.mean(diff_norms / y_norms)


@torch.no_grad()
def validation_step(eval_step, swinrnn_model, datapipe, channels=[0, 1], epoch=0):
    loss_epoch = 0
    num_examples = 0  # Number of validation examples
    # Dealing with DDP wrapper
    if hasattr(swinrnn_model, "module"):
        swinrnn_model = swinrnn_model.module
    swinrnn_model.eval()
    for i, data in enumerate(datapipe):
        invar = torch.permute(data[0]["invar"], (0, 2, 1, 3, 4)).detach()
        outvar = data[0]["outvar"].cpu().detach()
        predvar = torch.zeros_like(outvar)

        for t in range(outvar.shape[1]):
            output = eval_step(swinrnn_model, invar)
            invar[:, :, :-1, :, :] = invar[:, :, 1:, :, :]
            invar[:, :, -1, :, :] = output
            predvar[:, t] = output.detach().cpu()

        num_elements = torch.prod(torch.Tensor(list(predvar.shape[1:])))
        loss_epoch += torch.sum(torch.pow(predvar - outvar, 2)) / num_elements

        num_examples += predvar.shape[0]

    swinrnn_model.train()
    return loss_epoch / num_examples


@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist = DistributedManager()

    # Initialize loggers
    # initialize_wandb(
    #     project="Modulus-Launch-Dev",
    #     entity="Modulus",
    #     name="FourCastNet-Training",
    #     group="FCN-DDP-Group",
    # )
    initialize_mlflow(
        experiment_name="Modulus-Launch-Dev",
        experiment_desc="Modulus launch development",
        run_name="SwinRNN-Training",
        run_desc="SwinRNN WeatherBench Training",
        user_name="Modulus User",
        mode="offline",
    )
    LaunchLogger.initialize(use_mlflow=cfg.use_mlflow)  # Modulus launch logger
    logger = PythonLogger("main")  # General python logger

    no_channals_swinrnn = 4 + 5
    datapipe = WeatherBenchDatapipe(
        data_dir="/data/train/",
        channels=[i for i in range(no_channals_swinrnn)],
        constants_channels=[0, 1],
        num_samples_per_year=cfg.num_samples_per_year_train,
        batch_size=16,
        patch_size=(1, 1),
        num_workers=8,
        device=dist.device,
        process_rank=dist.rank,
        world_size=dist.world_size,
    )
    logger.success(f"Loaded datapipe of size {len(datapipe)}")

    if dist.rank == 0:
        logger.file_logging()
        validation_datapipe = WeatherBenchDatapipe(
            data_dir="/data/train",
            channels=[i for i in range(no_channals_swinrnn)],
            constants_channels=[0, 1],
            num_steps=1,
            num_samples_per_year=4,
            batch_size=16,
            patch_size=(1, 1),
            device=dist.device,
            num_workers=8,
            shuffle=False,
        )
        logger.success(f"Loaded validaton datapipe of size {len(validation_datapipe)}")

    swinrnn_model = SwinRNN(
        img_size=(6, 32, 64),
        in_chans=71,
        out_chans=71,
        embed_dim=768,
        patch_size=(6, 1, 1),
        window_size=8,
    ).to(dist.device)

    if dist.rank == 0 and wandb.run is not None:
        wandb.watch(
            swinrnn_model, log="all", log_freq=1000, log_graph=(True)
        )  # currently does not work with scripted modules. This will be fixed in the next release of W&B SDK.
    # Distributed learning
    if dist.world_size > 1:
        ddps = torch.cuda.Stream()
        with torch.cuda.stream(ddps):
            swinrnn_model = DistributedDataParallel(
                swinrnn_model,
                device_ids=[dist.local_rank],
                output_device=dist.device,
                broadcast_buffers=dist.broadcast_buffers,
                find_unused_parameters=dist.find_unused_parameters,
            )
        torch.cuda.current_stream().wait_stream(ddps)

    # Initialize optimizer and scheduler
    optimizer = optimizers.FusedAdam(
        swinrnn_model.parameters(), betas=(0.9, 0.999), lr=0.0001, weight_decay=0.000003
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # Attempt to load latest checkpoint if one exists
    loaded_epoch = load_checkpoint(
        "./checkpoints",
        models=swinrnn_model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=dist.device,
    )

    @StaticCaptureEvaluateNoGrad(model=swinrnn_model, logger=logger, use_graphs=False)
    def eval_step_forward(my_model, invar):
        return my_model(invar)[0]

    @StaticCaptureTraining(model=swinrnn_model, optim=optimizer, logger=logger)
    def train_step_forward(my_model, invar, outvar):
        # Multi-step prediction
        loss = 0
        # Multi-step not supported
        for t in range(outvar.shape[1]):
            outpred = my_model(invar)[0]
            invar = outpred
            loss += loss_func(outpred, outvar[:, t])
        return loss

    # Main training loop
    max_epoch = cfg.max_epoch
    for epoch in range(max(1, loaded_epoch + 1), max_epoch + 1):
        # Wrap epoch in launch logger for console / WandB logs
        with LaunchLogger(
            "train", epoch=epoch, num_mini_batch=len(datapipe), epoch_alert_freq=10
        ) as log:
            # === Training step ===
            for j, data in enumerate(datapipe):
                invar = torch.permute(data[0]["invar"], (0, 2, 1, 3, 4))
                outvar = data[0]["outvar"]
                loss = train_step_forward(swinrnn_model, invar, outvar)

                log.log_minibatch({"loss": loss.detach()})
            log.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})

        if dist.rank == 0:
            # Wrap validation in launch logger for console / WandB logs
            with LaunchLogger("valid", epoch=epoch) as log:
                # === Validation step ===
                error = validation_step(
                    eval_step_forward, swinrnn_model, validation_datapipe, epoch=epoch
                )
                log.log_epoch({"Validation error": error})

        if dist.world_size > 1:
            torch.distributed.barrier()

        scheduler.step()

        if (epoch % 5 == 0 or epoch == 1) and dist.rank == 0:
            # Use Modulus Launch checkpoint
            save_checkpoint(
                "./checkpoints",
                models=swinrnn_model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
            )

    if dist.rank == 0:
        logger.info("Finished training!")


if __name__ == "__main__":
    main()
