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
from modulus.models import Module
from modulus.distributed import DistributedManager
from modulus.metrics.diffusion import EDMLoss
from modulus.utils.generative import InfiniteSampler

from modulus.launch.utils import save_checkpoint, load_checkpoint
from modulus.launch.logging import PythonLogger, RankZeroLoggingWrapper
from utils.nn import (
    diffusion_model_forward,
    regression_loss_fn,
    get_preconditioned_architecture,
    build_network_condition_and_target,
)
from utils.plots import validation_plot
from datasets import dataset_classes
from datasets.dataset import worker_init
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

    dataset_cls = dataset_classes[cfg.dataset.name]

    dataset_train = dataset_cls(cfg.dataset, train=True)
    dataset_valid = dataset_cls(cfg.dataset, train=False)

    background_channels = dataset_train.background_channels()
    state_channels = dataset_train.state_channels()

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
    else:
        regression_net = None

    # Construct network
    logger0.info("Constructing network...")
    if train_regression_unet:
        num_condition_channels = len(background_channels) + len(state_channels)
    else:
        num_condition_channels = (
            len(state_channels)
            if not previous_step_conditioning
            else 2 * len(state_channels)
        )

    invariant_array = dataset_train.get_invariants()
    if invariant_array is not None:
        num_condition_channels += invariant_array.shape[0]
        invariant_tensor = torch.from_numpy(invariant_array).to(device)
        invariant_tensor = invariant_tensor.unsqueeze(0)
        invariant_tensor = invariant_tensor.repeat(local_batch_size, 1, 1, 1)

    logger0.info(f"background_channels {background_channels}")
    logger0.info(f"state_channels {state_channels}")
    logger0.info(f"num_condition_channels {num_condition_channels}")

    net = get_preconditioned_architecture(
        name=net_name,
        img_resolution=dataset_train.image_shape(),
        target_channels=len(state_channels),
        conditional_channels=num_condition_channels,
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
            path=os.path.join(cfg.training.rundir, f"checkpoints_{net_name}"),
            models=net,
            optimizer=optimizer,
        )

    # Train.
    logger0.info(
        f"Training up to {total_train_steps} steps starting from step {total_steps}..."
    )
    wandb_logs = {}
    done = total_steps >= total_train_steps

    train_start = time.time()
    avg_train_loss = 0
    train_steps = 0
    while not done:
        # Format input batch
        batch = next(dataset_iterator)
        background = batch["background"].to(device=device, dtype=torch.float32)
        state = [s.to(device=device, dtype=torch.float32) for s in batch["state"]]
        (condition, target, reg_out) = build_network_condition_and_target(
            background,
            state,
            invariant_tensor,
            regression_net=regression_net,
            train_regression_unet=train_regression_unet,
        )

        # Accumulate gradients.
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(
            net=ddp, images=target, condition=condition, augment_pipe=augment_pipe
        )
        if log_to_wandb:
            channelwise_loss = loss.mean(dim=(0, 2, 3))
            channelwise_loss_dict = {
                f"ChLoss/{ch}": channelwise_loss[i].item()
                for (i, ch) in enumerate(state_channels)
            }
            wandb_logs["channelwise_loss"] = channelwise_loss_dict

        loss_value = loss.sum() / len(state_channels)
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
                background = batch["background"].to(device=device, dtype=torch.float32)
                state = [
                    s.to(device=device, dtype=torch.float32) for s in batch["state"]
                ]
                (condition, target, reg_out) = build_network_condition_and_target(
                    background,
                    state,
                    invariant_tensor,
                    regression_net=regression_net,
                    train_regression_unet=train_regression_unet,
                )

                if use_regression_net:
                    output_images = (
                        diffusion_model_forward(
                            net,
                            condition,
                            state[1].shape,
                            sampler_args=dict(cfg.sampler.args),
                        )
                        + reg_out
                    )
                    del reg_out

                    valid_loss = loss_fn(
                        net=net,
                        images=target,
                        condition=condition,
                        augment_pipe=augment_pipe,
                    )
                elif train_regression_unet:
                    valid_loss, output_images = loss_fn(
                        net=net,
                        images=target,
                        condition=condition,
                        augment_pipe=augment_pipe,
                        return_model_outputs=True,
                    )

                    if log_to_wandb:
                        channelwise_valid_loss = valid_loss.mean(dim=[0, 2, 3])
                        channelwise_valid_loss_dict = {
                            f"ChLoss_valid/{state_channels[i]}": channelwise_valid_loss[
                                i
                            ].item()
                            for i in range(state_channels)
                        }
                        wandb_logs[
                            "channelwise_valid_loss"
                        ] = channelwise_valid_loss_dict

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
                    fields = cfg.training.validation_plot_variables

                    # Compute spectral metrics
                    figs, spec_ratios = ps1d_plots(
                        output_images[i], state[1][i], fields, state_channels
                    )

                    for f_ in fields:
                        f_index = state_channels.index(f_)
                        image_dir = os.path.join(cfg.training.rundir, "images", f_)
                        os.makedirs(image_dir, exist_ok=True)

                        generated = image[f_index]
                        truth = state[1][i, f_index].cpu().numpy()

                        fig = validation_plot(generated, truth, f_)
                        fig.savefig(
                            os.path.join(image_dir, f"{total_steps}_{i}_{f_}.png")
                        )

                        specfig = "PS1D_" + f_
                        figs[specfig].savefig(
                            os.path.join(image_dir, f"{total_steps}_{i}_{f_}_spec.png")
                        )
                        if log_to_wandb:
                            # Save plots as wandb Images
                            for figname, plot in figs.items():
                                wandb_logs[figname] = wandb.Image(plot)
                            wandb_logs.update({f"generated_{f_}": wandb.Image(fig)})

                    plt.close("all")

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
                path=os.path.join(cfg.training.rundir, f"checkpoints_{net_name}"),
                models=net,
                optimizer=optimizer,
                epoch=total_steps,
            )

    # Done.
    torch.distributed.barrier()
    logger0.info("\nExiting...")
