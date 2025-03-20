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

"""Main training loop."""

import os
import time
import numpy as np
import torch
import psutil
from physicsnemo.models import Module
from physicsnemo.distributed import DistributedManager
from physicsnemo.metrics.diffusion import EDMLoss
from physicsnemo.utils.generative import InfiniteSampler

from physicsnemo.launch.utils import save_checkpoint, load_checkpoint
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper
from utils.nn import (
    regression_model_forward,
    diffusion_model_forward,
    regression_loss_fn,
    get_preconditioned_architecture,
)
from utils.data_loader_hrrr_era5 import HrrrEra5Dataset, worker_init
import matplotlib.pyplot as plt
import wandb
from utils.spectrum import ps1d_plots
from torch.nn.utils import clip_grad_norm_


logger = PythonLogger("train")


def training_loop(cfg):

    # Initialize.
    start_time = time.time()
    dist = DistributedManager()
    device = dist.device
    logger0 = RankZeroLoggingWrapper(logger, dist)

    # Shorthand for config items
    batch_size = cfg.training.batch_size
    local_batch_size = batch_size // dist.world_size
    use_regression_net = cfg.model.use_regression_net
    previous_step_conditioning = cfg.model.previous_step_conditioning
    resume_checkpoint = cfg.training.resume_checkpoint
    log_to_wandb = cfg.training.log_to_wandb

    loss_type = cfg.training.loss
    if loss_type == "regression":
        train_regression_unet = True
        net_name = "regression"
    elif loss_type == "edm":
        train_regression_unet = False
        net_name = "diffusion"

    # Seed and Performance settings
    np.random.seed((cfg.training.seed * dist.world_size + dist.rank) % (1 << 31))
    torch.manual_seed(cfg.training.seed)
    torch.backends.cudnn.benchmark = cfg.training.cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    total_train_steps = cfg.training.total_train_steps

    # Load dataset.
    logger0.info("Loading dataset...")

    dataset_train = HrrrEra5Dataset(cfg.dataset, train=True)
    dataset_valid = HrrrEra5Dataset(cfg.dataset, train=False)

    _, hrrr_channels = dataset_train._get_hrrr_channel_names()
    diffusion_channels = (
        hrrr_channels
        if cfg.dataset.diffusion_channels == "all"
        else cfg.dataset.diffusion_channels
    )
    input_channels = (
        hrrr_channels
        if cfg.dataset.input_channels == "all"
        else cfg.dataset.input_channels
    )
    input_channel_indices = [hrrr_channels.index(channel) for channel in input_channels]
    diffusion_channel_indices = [
        hrrr_channels.index(channel) for channel in diffusion_channels
    ]

    sampler = InfiniteSampler(
        dataset=dataset_train,
        rank=dist.rank,
        num_replicas=dist.world_size,
        seed=cfg.training.seed,
    )
    valid_sampler = InfiniteSampler(
        dataset=dataset_valid,
        rank=dist.rank,
        num_replicas=dist.world_size,
        seed=cfg.training.seed,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=local_batch_size,
        num_workers=cfg.training.num_data_workers,
        sampler=sampler,
        worker_init_fn=worker_init,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )

    valid_data_loader = torch.utils.data.DataLoader(
        dataset=dataset_valid,
        batch_size=local_batch_size,
        num_workers=cfg.training.num_data_workers,
        sampler=valid_sampler,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )

    dataset_iterator = iter(data_loader)
    valid_dataset_iterator = iter(valid_data_loader)

    # load pretrained regression net if training diffusion
    if use_regression_net:
        regression_net = Module.from_checkpoint(cfg.model.regression_weights)
        regression_net = regression_net.to(device)

    # Construct network
    logger0.info("Constructing network...")
    target_channels = len(diffusion_channels)
    if train_regression_unet:
        conditional_channels = (
            len(input_channels) + 26
        )  # 26 is the number of era5 channels
    else:
        conditional_channels = (
            len(input_channels)
            if not previous_step_conditioning
            else 2 * len(input_channels)
        )

    conditional_channels += len(cfg.dataset.invariants)
    invariant_array = dataset_train._get_invariants()
    invariant_tensor = torch.from_numpy(invariant_array).to(device)
    invariant_tensor = invariant_tensor.unsqueeze(0)
    invariant_tensor = invariant_tensor.repeat(local_batch_size, 1, 1, 1)

    logger0.info(f"hrrr_channels {hrrr_channels}")
    logger0.info(f"target_channels for diffusion {target_channels}")
    logger0.info(f"conditional_channels for diffusion {conditional_channels}")

    net = get_preconditioned_architecture(
        name=net_name,
        hrrr_resolution=tuple(cfg.dataset.hrrr_img_size),
        target_channels=target_channels,
        conditional_channels=conditional_channels,
        spatial_embedding=cfg.model.spatial_pos_embed,
        attn_resolutions=list(cfg.model.attn_resolutions),
    )

    net.train().requires_grad_(True).to(device)

    # Setup optimizer.
    logger0.info("Setting up optimizer...")
    if cfg.training.loss == "regression":
        loss_fn = regression_loss_fn
    elif cfg.training.loss == "edm":
        loss_fn = EDMLoss(P_mean=cfg.model.P_mean)
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.training.lr)
    augment_pipe = None
    ddp = torch.nn.parallel.DistributedDataParallel(
        net, device_ids=[device], broadcast_buffers=False
    )

    # Resume training from previous snapshot.
    total_steps = 0
    if resume_checkpoint is not None:
        logger0.info(f'Resuming training state from "{resume_checkpoint}"...')

        total_steps = load_checkpoint(
            path=os.path.join(cfg.training.rundir, "checkpoints"),
            models=net,
            optimizer=optimizer,
        )

    # Train.
    logger0.info(
        f"Training up to {total_train_steps} steps starting from step {total_steps}..."
    )
    stats_jsonl = None
    wandb_logs = {}
    done = total_steps >= total_train_steps

    train_start = time.time()
    avg_train_loss = 0
    train_steps = 0
    while not done:
        # Accumulate gradients.
        optimizer.zero_grad(set_to_none=True)
        batch = next(dataset_iterator)
        hrrr_0 = batch["hrrr"][0].to(device).to(torch.float32)
        hrrr_1 = batch["hrrr"][1].to(device).to(torch.float32)

        if use_regression_net:
            era5 = batch["era5"][0].to(device).to(torch.float32)

            with torch.no_grad():
                reg_out = regression_model_forward(
                    regression_net, hrrr_0, era5, invariant_tensor
                )
                hrrr_0 = torch.cat(
                    (
                        hrrr_0[:, input_channel_indices, :, :],
                        reg_out[:, input_channel_indices, :, :],
                    ),
                    dim=1,
                )
                hrrr_1 = hrrr_1 - reg_out
                del reg_out

        elif train_regression_unet:
            assert diffusion_channel_indices == input_channel_indices

            era5 = batch["era5"][0].to(device).to(torch.float32)

            hrrr_0 = torch.cat((hrrr_0[:, input_channel_indices, :, :], era5), dim=1)

        hrrr_1 = hrrr_1[
            :, diffusion_channel_indices, :, :
        ]  # targets of the diffusion model

        hrrr_0 = torch.cat((hrrr_0, invariant_tensor), dim=1)

        loss = loss_fn(
            net=ddp, images=hrrr_1, condition=hrrr_0, augment_pipe=augment_pipe
        )
        channelwise_loss = loss.mean(dim=(0, 2, 3))
        channelwise_loss_dict = {
            f"ChLoss/{diffusion_channels[i]}": channelwise_loss[i].item()
            for i in range(target_channels)
        }
        if log_to_wandb:
            wandb_logs["channelwise_loss"] = channelwise_loss_dict

        loss_value = loss.sum() / target_channels
        loss_value.backward()

        if cfg.training.clip_grad_norm > 0:
            clip_grad_norm_(net.parameters(), cfg.training.clip_grad_norm)

        # Update weights.
        for g in optimizer.param_groups:
            g["lr"] = cfg.training.lr * min(
                total_steps / max(cfg.training.lr_rampup_steps, 1e-8), 1
            )
            if log_to_wandb:
                wandb_logs["lr"] = g["lr"]
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(
                    param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad
                )

        optimizer.step()

        if dist.world_size > 1:
            torch.distributed.barrier()
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)

        avg_train_loss += loss.mean().cpu().item()
        train_steps += 1
        if log_to_wandb:
            wandb_logs["loss"] = loss.mean().cpu().item() / train_steps

        # Perform maintenance tasks once per tick.
        total_steps += 1
        done = total_steps >= total_train_steps

        # Perform validation step
        if total_steps % cfg.training.validation_freq == 0:
            valid_start = time.time()
            batch = next(valid_dataset_iterator)

            with torch.no_grad():

                hrrr_0, hrrr_1 = batch["hrrr"]
                hrrr_0 = hrrr_0.to(torch.float32).to(device)
                hrrr_1 = hrrr_1.to(torch.float32).to(device)

                if use_regression_net:
                    with torch.no_grad():
                        era5 = batch["era5"][0].to(device).to(torch.float32)
                        reg_out = regression_model_forward(
                            regression_net, hrrr_0, era5, invariant_tensor
                        )
                        hrrr_0 = torch.cat(
                            (
                                hrrr_0[:, input_channel_indices, :, :],
                                reg_out[:, input_channel_indices, :, :],
                            ),
                            dim=1,
                        )

                        loss_target = hrrr_1 - reg_out
                        output_images = diffusion_model_forward(
                            net,
                            hrrr_0,
                            diffusion_channel_indices,
                            invariant_tensor,
                            sampler_args=dict(cfg.sampler.args),
                        )

                        valid_loss = loss_fn(
                            net=ddp,
                            images=loss_target[:, diffusion_channel_indices],
                            condition=torch.cat((hrrr_0, invariant_tensor), dim=1),
                            augment_pipe=augment_pipe,
                        )
                        output_images += reg_out[:, diffusion_channel_indices, :, :]
                        del reg_out

                elif train_regression_unet:
                    assert (
                        use_regression_net == False
                    ), "use_regression_net must be False when training regression unet"
                    assert (
                        input_channel_indices == diffusion_channel_indices
                    ), "input_channel_indices must be equal to diffusion_channel_indices when training regression unet"
                    condition = torch.cat(
                        (
                            hrrr_0[:, input_channel_indices, :, :],
                            era5[:],
                            invariant_tensor,
                        ),
                        dim=1,
                    )
                    valid_loss, output_images = loss_fn(
                        net=ddp,
                        images=hrrr_1[:, diffusion_channel_indices, :, :],
                        condition=condition,
                        augment_pipe=augment_pipe,
                        return_model_outputs=True,
                    )

                    if log_to_wandb:
                        channelwise_valid_loss = valid_loss.mean(dim=[0, 2, 3])
                        channelwise_valid_loss_dict = {
                            f"ChLoss_valid/{diffusion_channels[i]}": channelwise_valid_loss[
                                i
                            ].item()
                            for i in range(target_channels)
                        }
                        wandb_logs[
                            "channelwise_valid_loss"
                        ] = channelwise_valid_loss_dict

                hrrr_1 = hrrr_1[:, diffusion_channel_indices, :, :]

                if dist.world_size > 1:
                    torch.distributed.barrier()
                    torch.distributed.all_reduce(
                        valid_loss, op=torch.distributed.ReduceOp.AVG
                    )
                val_loss = valid_loss.mean().cpu().item()
                if log_to_wandb:
                    wandb_logs["valid_loss"] = val_loss

            # Save plots locally (and optionally to wandb)
            if dist.rank == 0:

                for i in range(output_images.shape[0]):
                    image = output_images[i].cpu().numpy()
                    fields = ["u10m", "v10m", "t2m", "refc", "q1", "q5", "q10"]

                    # Compute spectral metrics
                    figs, spec_ratios = ps1d_plots(
                        output_images[i], hrrr_1[i], fields, diffusion_channels
                    )

                    for f_ in fields:
                        f_index = diffusion_channels.index(f_)
                        image_dir = os.path.join(cfg.training.rundir, "images", f_)
                        generated = image[f_index]
                        truth = hrrr_1[i, f_index].cpu().numpy()

                        fig, (a, b) = plt.subplots(1, 2)
                        im = a.imshow(generated)
                        a.set_title("generated, {}.png".format(f_))
                        plt.colorbar(im, fraction=0.046, pad=0.04)
                        im = b.imshow(truth)
                        b.set_title("truth")
                        plt.colorbar(im, fraction=0.046, pad=0.04)
                        os.makedirs(image_dir, exist_ok=True)
                        plt.savefig(
                            os.path.join(image_dir, f"{total_steps}_{i}_{f_}.png")
                        )
                        plt.close("all")

                        specfig = "PS1D_" + f_
                        figs[specfig].savefig(
                            os.path.join(image_dir, f"{total_steps}{i}{f_}_spec.png")
                        )
                        plt.close(figs[specfig])
                        if log_to_wandb:
                            # Save plots as wandb Images
                            for figname, plot in figs.items():
                                wandb_logs[figname] = wandb.Image(plot)
                            wandb_logs.update({f"generated_{f_}": wandb.Image(fig)})

                if log_to_wandb:
                    wandb_logs.update(spec_ratios)
                    wandb.log(wandb_logs, step=total_steps)

            valid_time = time.time() - valid_start

        # Print training stats
        current_time = time.time()
        if total_steps % cfg.training.print_progress_freq == 0:
            fields = []
            fields += [f"steps {total_steps:<5d}"]
            fields += [f"samples {total_steps*batch_size}"]
            fields += [f"tot_time {current_time - start_time: .2f}"]
            fields += [
                f"step_time {(current_time - train_start - valid_time) / train_steps : .2f}"
            ]
            fields += [f"valid_time {valid_time: .2f}"]
            fields += [
                f"cpumem {psutil.Process(os.getpid()).memory_info().rss / 2**30:<6.2f}"
            ]
            fields += [
                f"gpumem {torch.cuda.max_memory_allocated(device) / 2**30:<6.2f}"
            ]
            fields += [f"train_loss {avg_train_loss/train_steps:<6.3f}"]
            fields += [f"val_loss {val_loss:<6.3f}"]
            logger0.info(" ".join(fields))

            # Reset counters
            train_steps = 0
            train_start = time.time()
            avg_train_loss = 0
            torch.cuda.reset_peak_memory_stats()

        # Save full dump of the training state.
        if (
            (done or total_steps % cfg.training.checkpoint_freq == 0)
            and total_steps != 0
            and dist.rank == 0
        ):

            save_checkpoint(
                path=os.path.join(cfg.training.rundir, "checkpoints"),
                models=net,
                optimizer=optimizer,
                epoch=total_steps,
            )

    # Done.
    torch.distributed.barrier()
    logger0.info("\nExiting...")
