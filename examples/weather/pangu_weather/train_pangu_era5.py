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
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from torch.nn.parallel import DistributedDataParallel
from torch.cuda import amp
from torch.cuda.amp import GradScaler

from omegaconf import DictConfig, OmegaConf

from tqdm import tqdm

from modulus.models.pangu import Pangu
from modulus.datapipes.climate import ERA5HDF5Datapipe
from modulus.distributed import DistributedManager
from modulus.utils import StaticCaptureTraining, StaticCaptureEvaluateNoGrad

from modulus.launch.logging import (
    RankZeroLoggingWrapper,
    PythonLogger,
)
from modulus.launch.utils import load_checkpoint, save_checkpoint

try:
    from apex import optimizers
except:
    raise ImportError(
        "Pangu-Weather training requires apex package for optimizer."
        + "See https://github.com/nvidia/apex for install details."
    )

OmegaConf.register_new_resolver("lambda_lr", lambda x, y: (lambda epoch: x / y))
torch._dynamo.config.optimize_ddp = False


@torch.jit.script
def loss_func(x: torch.Tensor, y: torch.Tensor, weights: torch.Tensor):
    return torch.mean(
        weights * torch.nn.functional.smooth_l1_loss(x, y, reduction="none", beta=0.5)
    ) / torch.mean(weights)


@torch.no_grad()
def validation_step(
    eval_step,
    pangu_model,
    datapipe,
    surface_mask,
    weights,
    channels,
    epoch,
):

    num_channels = len(channels)
    num_steps = datapipe.num_steps
    loss_epoch = torch.zeros((num_channels, num_steps), device="cpu")
    num_examples = 0  # Number of validation examples

    # Dealing with DDP wrapper
    if hasattr(pangu_model, "module"):
        pangu_model = pangu_model.module

    pangu_model.eval()

    # Loop over datapipe
    for di, data in enumerate(datapipe):
        # Get input data
        invar = data[0]["invar"].detach()
        cos_zenith = data[0]["cos_zenith"].detach().squeeze(dim=2)
        cos_zenith = torch.clamp(cos_zenith, min=0.0) - 1.0 / torch.pi
        outvar = data[0]["outvar"].detach()
        sm = surface_mask.repeat(invar.shape[0], 1, 1, 1)

        num_examples += outvar.shape[0]

        # If first batch then create buffer for outputs
        if di == 0:
            outpred = torch.zeros_like(outvar, device="cpu").pin_memory()

        for t in range(outvar.shape[1]):
            out, loss = eval_step(
                pangu_model, invar, cos_zenith[:, t : t + 1], sm, outvar[:, t], weights
            )
            invar = out.clone()
            out = out.detach().cpu()
            loss = loss.detach().cpu()

            # Normalize
            out = out * datapipe.sd + datapipe.mu
            loss = loss * datapipe.sd[0, :, 0, 0]
            loss_epoch[:, t] += loss[channels]

            # If first batch then save out to buffer
            if di == 0:
                outpred[:, t].copy_(out, non_blocking=True)

        # If first batch plot images
        if (di == 0) and (epoch % 10 == 0):
            os.makedirs("./images", exist_ok=True)
            num_plots = max(4, num_steps)

            outvar = outvar.cpu() * datapipe.sd + datapipe.mu
            for i, ch in enumerate(channels):
                fig, ax = plt.subplots(
                    nrows=3, ncols=num_plots, figsize=(4 * num_plots, 2 * 3 + 1)
                )

                for j in range(num_plots):
                    op = outpred[0, j, ch]
                    ov = outvar[0, j, ch]
                    vmin = ov.min().item()
                    vmax = ov.max().item()
                    pred = ax[0, j].imshow(op, vmin=vmin, vmax=vmax)
                    ax[0, j].set_title(f"Channel {ch} Step {j} Prediction")
                    plt.colorbar(
                        pred, ax=ax[0, j], shrink=0.75, orientation="horizontal"
                    )

                    truth = ax[1, j].imshow(ov, vmin=vmin, vmax=vmax)
                    ax[1, j].set_title(f"Channel {ch} Step {j} Truth")
                    plt.colorbar(
                        truth, ax=ax[1, j], shrink=0.75, orientation="horizontal"
                    )

                    diff = ax[2, j].imshow((op - ov) / ov.abs().mean())
                    ax[2, j].set_title(f"Channel {ch} Step {j} Relative Error")
                    plt.colorbar(
                        diff, ax=ax[2, j], shrink=0.75, orientation="horizontal"
                    )

                plt.tight_layout()
                plt.savefig(
                    f"./images/diff_channel_{ch}_epoch_{epoch}.png",
                    dpi=600,
                    bbox_inches="tight",
                )
                plt.clf()

    loss_epoch = torch.sqrt(loss_epoch / num_examples).numpy()

    # Save losses
    csv_file_name = "validation_rmse_loss.csv"
    try:
        # See if there is an existing file.
        df = pd.read_csv(csv_file_name, index_col=0)
    except FileNotFoundError:
        # Create a new dataframe otherwise.
        df = pd.DataFrame(columns=["epoch", "channel_id", "step", "loss"])

    dd = []
    for i, ch in enumerate(channels):
        for j in range(datapipe.num_steps):
            dd.append([epoch, ch, j, loss_epoch[i, j]])

    df = pd.concat([df, pd.DataFrame(dd, columns=df.columns)], ignore_index=True)
    df.to_csv(csv_file_name)

    pangu_model.train()


@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist = DistributedManager()

    # Initialize loggers

    logger = PythonLogger("main")  # General python logger
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)
    rank_zero_logger.file_logging()

    # print ranks and devices
    logger.info(f"Rank: {dist.rank}, Device: {dist.device}")

    number_channels_pangu = (
        cfg.pangu.number_surface_variables
        + cfg.pangu.number_atmosphere_variables * cfg.pangu.number_atmosphere_levels
    )
    img_size = OmegaConf.to_object(cfg.pangu.img_size)

    mask_dir = cfg.train.mask_dir
    if cfg.train.get("mask_dtype", "float32") == "float32":
        mask_dtype = np.float32
    elif cfg.train.get("mask_dtype", "float32") == "float16":
        mask_dtype = np.float16
    else:
        mask_dtype = np.float32

    land_mask = torch.from_numpy(
        np.load(os.path.join(mask_dir, "land_mask.npy")).astype(mask_dtype)
    )
    soil_type = torch.from_numpy(
        np.load(os.path.join(mask_dir, "soil_type.npy")).astype(mask_dtype)
    )
    topography = torch.from_numpy(
        np.load(os.path.join(mask_dir, "topography.npy")).astype(mask_dtype)
    )
    topography = (topography - topography.mean()) / topography.std()
    surface_mask = (
        torch.stack([land_mask, soil_type, topography], dim=0)
        .to(dist.device)
        .unsqueeze(0)
    )
    logger.success(f"Rank {dist.rank}: Loaded suface constant mask from {mask_dir}")

    pangu_model = Pangu(
        img_size=img_size,
        patch_size=OmegaConf.to_object(cfg.pangu.patch_size),
        embed_dim=cfg.pangu.embed_dim,
        num_heads=OmegaConf.to_object(cfg.pangu.num_heads),
        window_size=OmegaConf.to_object(cfg.pangu.window_size),
        number_constant_variables=cfg.pangu.number_constant_variables
        + int(cfg.train.use_cosine_zenith),
        number_surface_variables=cfg.pangu.number_surface_variables,
        number_atmosphere_levels=cfg.pangu.number_atmosphere_levels,
        number_atmosphere_variables=cfg.pangu.number_atmosphere_variables,
        number_up_sampled_blocks=cfg.pangu.number_up_sampled_blocks,
        number_down_sampled_blocks=cfg.pangu.number_down_sampled_blocks,
        checkpoint_flag=cfg.pangu.checkpoint_flag,
    ).to(dist.device)

    # pangu_model.compile()

    weights = (
        torch.abs(torch.cos(torch.linspace(90, -90, img_size[0]) * torch.pi / 180.0))
        .unsqueeze(1)
        .repeat(1, img_size[1])
        .to(dist.device)
    )

    # Distributed learning
    if dist.world_size > 1:
        ddps = torch.cuda.Stream()
        with torch.cuda.stream(ddps):
            pangu_model = DistributedDataParallel(
                pangu_model,
                device_ids=[dist.local_rank],
                output_device=dist.device,
                broadcast_buffers=dist.broadcast_buffers,
                find_unused_parameters=dist.find_unused_parameters,
            )
        torch.cuda.current_stream().wait_stream(ddps)

    # Initialize optimizer and scheduler
    optimizer = optimizers.FusedAdam(
        pangu_model.parameters(),
        betas=(0.9, 0.999),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    # Load Validation Datapipe
    if dist.rank == 0:
        validation_datapipe = ERA5HDF5Datapipe(
            data_dir=cfg.val.data_dir,
            stats_dir=cfg.val.stats_dir,
            channels=[i for i in range(number_channels_pangu)],
            num_steps=cfg.val.num_rollout_steps,
            num_samples_per_year=cfg.val.num_samples_per_year,
            batch_size=cfg.val.batch_size,
            device=dist.device,
            num_workers=cfg.val.num_workers,
            shuffle=False,
            use_cos_zenith=cfg.train.use_cosine_zenith,
            cos_zenith_args={
                "dt": 6.0,
                "start_year": 1980,
                "latlon_bounds": ((90, -90), (0, 360)),
            },
            latlon_resolution=img_size,
        )
        logger.success(
            f"Rank {dist.rank}: Loaded validaton datapipe of size {len(validation_datapipe)}"
        )

    @StaticCaptureEvaluateNoGrad(model=pangu_model, logger=logger, use_graphs=False)
    def eval_step_forward(my_model, invar, cos_zenith, surface_mask, outvar, weights):
        # Multi-step prediction
        invar = torch.concat([surface_mask, cos_zenith, invar], dim=1)
        outpred = my_model(invar)
        loss = torch.sum(
            weights * (outpred - outvar) ** 2, dim=(0, -2, -1)
        ) / torch.sum(weights)
        return outpred, loss

    @StaticCaptureTraining(
        model=pangu_model,
        optim=optimizer,
        logger=logger,
        use_graphs=cfg.train.enable_graphs,
        use_amp=cfg.train.enable_amp,
    )
    def train_step_forward(my_model, invar, cos_zenith, surface_mask, outvar, weights):
        # Multi-step prediction
        loss = 0
        batch_size = outvar.shape[0]
        for b in range(batch_size):
            invar_ = invar[b : b + 1]
            cos_zenith_ = cos_zenith[b : b + 1]
            for t in range(outvar.shape[1]):
                invar_ = torch.concat(
                    [surface_mask, cos_zenith_[:, t : t + 1], invar_], dim=1
                )
                outpred = my_model(invar_)
                loss += loss_func(outpred, outvar[b : b + 1, t], weights) / batch_size
                invar_ = outpred

        return loss

    # Main training loop

    # Attempt to load latest checkpoint if one exists
    loaded_epoch = load_checkpoint(
        cfg.train.checkpoint_dir,
        models=pangu_model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        device=dist.device,
    )

    rank_zero_logger.info("Rank {dist.rank}: Training started.")
    global_epoch = 0
    for stage in cfg.train.stages:
        if loaded_epoch > global_epoch:
            if loaded_epoch >= global_epoch + stage.num_epochs:
                # Skip stage
                global_epoch += stage.num_epochs
                continue
            else:
                num_epochs = stage.num_epochs - (loaded_epoch - global_epoch)

            global_epoch = loaded_epoch
        else:
            num_epochs = stage.num_epochs

        rank_zero_logger.info(
            f"Rank {dist.rank}: Starting stage {stage.name} at epoch # {loaded_epoch}."
        )

        # Load datapipe for this stage
        train_datapipe = ERA5HDF5Datapipe(
            data_dir=cfg.train.data_dir,
            stats_dir=cfg.train.stats_dir,
            channels=[i for i in range(number_channels_pangu)],
            num_samples_per_year=cfg.train.num_samples_per_year,
            use_cos_zenith=cfg.train.use_cosine_zenith,
            cos_zenith_args={
                "dt": 6.0,
                "start_year": 1980,
                "latlon_bounds": ((90, -90), (0, 360)),
            },
            num_steps=stage.num_rollout_steps,
            latlon_resolution=img_size,
            batch_size=stage.batch_size,
            num_workers=cfg.train.num_workers,
            device=dist.device,
            process_rank=dist.rank,
            world_size=dist.world_size,
        )
        logger.success(
            f"Rank {dist.rank}: Loaded datapipe of size {len(train_datapipe)}"
        )

        # Initialize scheduler
        SchedulerClass = getattr(torch.optim.lr_scheduler, stage.lr_scheduler_name)
        scheduler = SchedulerClass(optimizer, **stage.args)

        # Set scheduler to current step
        scheduler.step(stage.num_epochs - num_epochs)

        # Get current step for checking if max iterations is reached
        current_step = len(train_datapipe) * (stage.num_epochs - num_epochs)

        for epoch in range(num_epochs):
            logger.info(f"Rank {dist.rank}: Starting Epoch {global_epoch}.")
            loss_agg = 0.0
            for j, data in tqdm(enumerate(train_datapipe), disable=(dist.rank != 0)):
                if current_step > stage.max_iterations:
                    break

                invar = data[0]["invar"]
                outvar = data[0]["outvar"]
                cos_zenith = data[0]["cos_zenith"].squeeze(dim=2)
                cos_zenith = torch.clamp(cos_zenith, min=0.0) - 1.0 / torch.pi

                loss_agg += train_step_forward(
                    pangu_model, invar, cos_zenith, surface_mask, outvar, weights
                )

                current_step += 1

                if (
                    current_step % int(len(train_datapipe) // 5) == 0
                ) and dist.rank == 0:
                    tqdm.write(
                        f"Epoch: {global_epoch} \t iteration: {current_step} "
                        + f"\t loss: {loss_agg / int(len(train_datapipe) // 5)}"
                    )
                    loss_agg = 0.0

            # Step scheduler
            scheduler.step()

            # Perform validation
            if dist.rank == 0:
                del invar, cos_zenith, outvar
                torch.cuda.empty_cache()

                # Use Modulus Launch checkpoint
                save_checkpoint(
                    cfg.train.checkpoint_dir,
                    models=pangu_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=None,
                    epoch=global_epoch + 1,
                )

                validation_step(
                    eval_step_forward,
                    pangu_model,
                    validation_datapipe,
                    surface_mask,
                    weights,
                    cfg.val.channels,
                    global_epoch + 1,
                )
            global_epoch += 1
            torch.cuda.empty_cache()
            torch.distributed.barrier()

    if dist.rank == 0:
        logger.info("Rank {dist.rank}: Finished training!")


if __name__ == "__main__":
    main()
