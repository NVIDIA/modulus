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

import copy
import json
import os
import sys
import time
import wandb as wb

import numpy as np
import psutil
import torch

from hydra.utils import to_absolute_path
from torch.nn.parallel import DistributedDataParallel
from . import training_stats

sys.path.append("../")
from modulus import Module
from modulus.distributed import DistributedManager
from modulus.launch.logging import (
    PythonLogger,
    RankZeroLoggingWrapper,
    initialize_wandb,
)
from modulus.utils.generative import (
    construct_class_by_name,
    ddp_sync,
    format_time,
)

# ----------------------------------------------------------------------------


def training_loop(
    dataset,
    dataset_iterator,
    validation_dataset,
    validation_dataset_iterator,
    *,
    task,
    run_dir=".",  # Output directory.
    network_kwargs={},  # Options for model and preconditioning.
    loss_kwargs={},  # Options for loss function.
    optimizer_kwargs={},  # Options for optimizer.
    augment_kwargs=None,  # Options for augmentation pipeline, None = disable.
    seed=0,  # Global random seed.
    batch_size_global=512,  # Total batch size for one training iteration.
    batch_size_gpu=None,  # Limit batch size per GPU, None = no limit.
    total_kimg=200000,  # Training duration, measured in thousands of training images.
    ema_halflife_kimg=500,  # Half-life of the exponential moving average (EMA) of model weights.
    ema_rampup_ratio=0.05,  # EMA ramp-up coefficient, None = no rampup.
    lr_rampup_kimg=10000,  # Learning rate ramp-up duration.
    loss_scaling=1,  # Loss scaling factor for reducing FP16 under/overflows.
    kimg_per_tick=50,  # Interval of progress prints.
    state_dump_ticks=500,  # How often to dump training state, None = disable.
    cudnn_benchmark=True,  # Enable torch.backends.cudnn.benchmark?
    patch_shape_x=448,
    patch_shape_y=448,
    patch_num=1,
    wandb_mode="disabled",
    wandb_project="Modulus-Generative",
    wandb_entity="CorrDiff-DDP-Group",
    wandb_name="CorrDiff",
    wandb_group="CorrDiff-Group",
    fp_optimizations="fp32",  # The floating point optimization mode
    regression_checkpoint_path=None,
    hr_mean_conditioning=False,
    N_grid_channels=4,
    gridtype="sinusoidal",
    in_channel=12,
    grad_clip_threshold=None,
    lr_decay=0.8,
    valid_dump_ticks=5000,
    num_validation_evals=10,
):
    """CorrDiff training loop"""

    # Instantiate distributed manager.
    dist = DistributedManager()
    device = dist.device

    # Initialize logger.
    logger = PythonLogger(name="training_loop")  # General python logger
    logger0 = RankZeroLoggingWrapper(logger, dist)
    logger.file_logging(file_name=f"logs/training_loop_{dist.rank}.log")

    # wandb logger
    if dist.rank == 0:
        initialize_wandb(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_name,
            mode=wandb_mode,
            group=wandb_group,
            save_code=True,
        )
        # log code
        wb.run.log_code(
            to_absolute_path("."),
            exclude_fn=lambda path: ("outputs" in path) and (os.getcwd() not in path),
        )

    enable_amp = fp_optimizations.startswith("amp")
    amp_dtype = torch.float16 if (fp_optimizations == "amp-fp16") else torch.bfloat16

    # Initialize.
    start_time = time.time()

    np.random.seed((seed * dist.world_size + dist.rank) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch size per GPU.
    batch_gpu_total = batch_size_global // dist.world_size
    logger0.info(f"batch_size_gpu: {batch_size_gpu}")
    if batch_size_gpu is None or batch_size_gpu > batch_gpu_total:
        batch_size_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_size_gpu
    if batch_size_global != batch_size_gpu * num_accumulation_rounds * dist.world_size:
        raise ValueError(
            "batch_size_global must be equal to batch_size_gpu * num_accumulation_rounds * dist.world_size"
        )

    img_in_channels = (
        len(dataset.input_channels()) + N_grid_channels
    )  # noise + low-res input
    (img_shape_y, img_shape_x) = dataset.image_shape()
    img_out_channels = len(dataset.output_channels())
    if hr_mean_conditioning:
        img_in_channels += img_out_channels

    # interpolate global channel if patch-based model is used
    if img_shape_x != patch_shape_x:
        img_in_channels += in_channel

    # Construct network.
    logger0.info("Constructing network...")
    interface_kwargs = dict(
        img_resolution=[img_shape_y, img_shape_x],
        img_channels=img_out_channels,
        img_in_channels=img_in_channels,
        img_out_channels=img_out_channels,
        label_dim=0,
        gridtype=gridtype,
        N_grid_channels=N_grid_channels,
    )  # weather
    merged_args = {**network_kwargs, **interface_kwargs}
    net = construct_class_by_name(**merged_args)  # subclass of torch.nn.Module
    net.train().requires_grad_(True).to(device)

    # Setup optimizer.
    logger0.info("Setting up optimizer...")
    if task == "diffusion":
        if regression_checkpoint_path is None:
            raise FileNotFoundError(
                "Need to specify regression_checkpoint_path for training the diffusion model"
            )
        net_reg = Module.from_checkpoint(regression_checkpoint_path)
        net_reg.eval().requires_grad_(False).to(device)

        if loss_kwargs["class_name"] == "modulus.metrics.diffusion.ResLoss":
            interface_kwargs = dict(
                regression_net=net_reg,
                img_shape_x=img_shape_x,
                img_shape_y=img_shape_y,
                patch_shape_x=patch_shape_x,
                patch_shape_y=patch_shape_y,
                patch_num=patch_num,
                hr_mean_conditioning=hr_mean_conditioning,
            )
        else:
            interface_kwargs = {}
        logger0.success("Loaded the pre-trained regression network")
    else:
        interface_kwargs = {}
    loss_fn = construct_class_by_name(**loss_kwargs, **interface_kwargs)
    optimizer = construct_class_by_name(
        params=net.parameters(), **optimizer_kwargs
    )  # subclass of torch.optim.Optimizer
    augment_pipe = (
        construct_class_by_name(**augment_kwargs)
        if augment_kwargs is not None
        else None
    )
    if dist.world_size > 1:
        ddp = DistributedDataParallel(
            net,
            device_ids=[dist.local_rank],
            broadcast_buffers=True,
            output_device=dist.device,
            find_unused_parameters=dist.find_unused_parameters,
        )
    else:
        ddp = net
    ema = copy.deepcopy(net).eval().requires_grad_(False)

    # Resume training from previous snapshot.
    max_index = -1
    max_index_file = " "
    for filename in os.listdir(run_dir):
        if filename.startswith(f"training-state-{task}-") and filename.endswith(
            ".mdlus"
        ):
            index_str = filename.split("-")[-1].split(".")[0]
            try:
                index = int(index_str)
                if index > max_index:
                    max_index = index
                    max_index_file = filename
                    max_index_file_optimizer = f"optimizer-state-{task}-{index_str}.pt"
            except ValueError:
                continue

    try:
        net.load(os.path.join(run_dir, max_index_file))
        # load state directly to each gpu to reduce memory usage
        map_location = {"cuda:%d" % 0: "cuda:%d" % int(dist.local_rank)}
        optimizer_state_dict = torch.load(
            os.path.join(run_dir, max_index_file_optimizer), map_location=map_location
        )
        optimizer.load_state_dict(optimizer_state_dict["optimizer_state_dict"])
        cur_nimg = max_index * 1000
        logger0.success(f"Loaded network and optimizer states with index {max_index}")
    except FileNotFoundError:
        cur_nimg = 0
        logger0.warning("Could not load network and optimizer states")

    # Train.
    logger0.info(f"Training for {total_kimg} kimg...")
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    stats_jsonl = None
    while True:
        # Accumulate gradients.
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0
        for round_idx in range(num_accumulation_rounds):
            with ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                # Fetch training data: weather
                img_clean, img_lr, labels = next(dataset_iterator)

                # Normalization: weather (normalized already in the dataset)
                img_clean = (
                    img_clean.to(device).to(torch.float32).contiguous()
                )  # [-4.5, +4.5]
                img_lr = img_lr.to(device).to(torch.float32).contiguous()
                labels = labels.to(device).contiguous()

                with torch.autocast("cuda", dtype=amp_dtype, enabled=enable_amp):
                    loss = loss_fn(
                        net=ddp,
                        img_clean=img_clean,
                        img_lr=img_lr,
                        labels=labels,
                        augment_pipe=augment_pipe,
                    )
                training_stats.report("Loss/loss", loss)
                loss = loss.sum().mul(loss_scaling / batch_size_gpu)
                loss_accum += loss / num_accumulation_rounds
                loss.backward()

        loss_sum = torch.tensor([loss_accum], device=device)
        if dist.world_size > 1:
            torch.distributed.all_reduce(loss_sum, op=torch.distributed.ReduceOp.SUM)
        average_loss = loss_sum / dist.world_size
        if dist.rank == 0:
            wb.log({"training loss": average_loss}, step=cur_nimg)

        # Update weights.
        for g in optimizer.param_groups:
            g["lr"] = optimizer_kwargs["lr"] * min(
                cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1
            )  # TODO better handling (potential bug)
            g["lr"] *= lr_decay ** ((cur_nimg - lr_rampup_kimg * 1000) // 5e6)
            if dist.rank == 0:
                wb.log({"lr": g["lr"]}, step=cur_nimg)
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(
                    param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad
                )
        if grad_clip_threshold:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                net.parameters(), grad_clip_threshold
            )
        optimizer.step()
        if validation_dataset_iterator is not None:
            valid_loss_accum = 0
            if cur_tick % valid_dump_ticks == 0:
                with torch.no_grad():
                    for _ in range(num_validation_evals):
                        img_clean_valid, img_lr_valid, labels_valid = next(
                            validation_dataset_iterator
                        )

                        img_clean_valid = (
                            img_clean_valid.to(device).to(torch.float32).contiguous()
                        )  # [-4.5, +4.5]
                        img_lr_valid = (
                            img_lr_valid.to(device).to(torch.float32).contiguous()
                        )
                        labels_valid = labels_valid.to(device).contiguous()
                        loss_valid = loss_fn(
                            net=ddp,
                            img_clean=img_clean_valid,
                            img_lr=img_lr_valid,
                            labels=labels_valid,
                            augment_pipe=augment_pipe,
                        )
                        training_stats.report("Loss/validation loss", loss_valid)
                        loss_valid = loss_valid.sum().mul(
                            loss_scaling / batch_size_gpu
                        )
                        valid_loss_accum += loss_valid / num_validation_evals
                    valid_loss_sum = torch.tensor([valid_loss_accum], device=device)
                    if dist.world_size > 1:
                        torch.distributed.all_reduce(
                            valid_loss_sum, op=torch.distributed.ReduceOp.SUM
                        )
                    average_valid_loss = valid_loss_sum / dist.world_size
                    if dist.rank == 0:
                        wb.log({"validation loss": average_valid_loss}, step=cur_nimg)

        # Update EMA.
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size_global / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        # Perform maintenance tasks once per tick.
        cur_nimg += batch_size_global
        done = cur_nimg >= total_kimg * 1000
        if (
            (not done)
            # and (cur_tick != 0)
            and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000)
        ):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [
            f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"
        ]
        fields += [
            f"time {format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"
        ]
        fields += [
            f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"
        ]
        fields += [
            f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"
        ]
        fields += [
            f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"
        ]
        fields += [
            f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"
        ]
        fields += [
            f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"
        ]
        fields += [
            f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"
        ]
        torch.cuda.reset_peak_memory_stats()
        logger0.info(" ".join(fields))

        # Save full dump of the training state.
        if (
            (state_dump_ticks is not None)
            and (done or cur_tick % state_dump_ticks == 0)
            and dist.rank == 0
        ):
            filename = f"training-state-{task}-{cur_nimg//1000:06d}.mdlus"
            net.save(os.path.join(run_dir, filename), verbose=True)
            logger0.info(f"Saved model in the {run_dir} directory")

            filename = f"optimizer-state-{task}-{cur_nimg//1000:06d}.pt"
            torch.save(
                {"optimizer_state_dict": optimizer.state_dict()},
                os.path.join(run_dir, filename),
            )
            logger0.info(f"Saved optimizer state in the {run_dir} directory")

        # Update logs.
        training_stats.default_collector.update()
        if dist.rank == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, "stats.jsonl"), "at")
            stats_jsonl.write(
                json.dumps(
                    dict(
                        training_stats.default_collector.as_dict(),
                        timestamp=time.time(),
                    )
                )
                + "\n"
            )
            stats_jsonl.flush()

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    logger0.info("Exiting...")
