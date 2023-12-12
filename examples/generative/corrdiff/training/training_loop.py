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

"""Main training loop."""

import copy
import json
import os
import pickle  # ruff: noqa: E402
import sys
import time

import numpy as np
import psutil
import torch
from torch.nn.parallel import DistributedDataParallel
from . import training_stats

from modulus.distributed import DistributedManager
from modulus.launch.logging import PythonLogger, RankZeroLoggingWrapper
from modulus.utils.generative import (
    InfiniteSampler,
    check_ddp_consistency,
    construct_class_by_name,
    copy_params_and_buffers,
    ddp_sync,
    format_time,
    open_url,
)

from .dataset import get_zarr_dataset

# ----------------------------------------------------------------------------


def training_loop(
    run_dir=".",  # Output directory.
    dataset_kwargs={},  # Options for training set.
    data_loader_kwargs={},  # Options for torch.utils.data.DataLoader.
    network_kwargs={},  # Options for model and preconditioning.
    loss_kwargs={},  # Options for loss function.
    optimizer_kwargs={},  # Options for optimizer.
    augment_kwargs=None,  # Options for augmentation pipeline, None = disable.
    seed=0,  # Global random seed.
    batch_size=512,  # Total batch size for one training iteration.
    batch_gpu=None,  # Limit batch size per GPU, None = no limit.
    total_kimg=200000,  # Training duration, measured in thousands of training images.
    ema_halflife_kimg=500,  # Half-life of the exponential moving average (EMA) of model weights.
    ema_rampup_ratio=0.05,  # EMA ramp-up coefficient, None = no rampup.
    lr_rampup_kimg=10000,  # Learning rate ramp-up duration.
    loss_scaling=1,  # Loss scaling factor for reducing FP16 under/overflows.
    kimg_per_tick=50,  # Interval of progress prints.
    snapshot_ticks=50,  # How often to save network snapshots, None = disable.
    state_dump_ticks=500,  # How often to dump training state, None = disable.
    resume_pkl=None,  # Start from the given network snapshot, None = random initialization.
    resume_state_dump=None,  # Start from the given training state, None = reset training state.
    resume_kimg=0,  # Start from the given training progress.
    cudnn_benchmark=True,  # Enable torch.backends.cudnn.benchmark?
    train_data_path=None,
    crop_size_x=448,
    crop_size_y=448,
    n_history=0,
    in_channels=[0, 1, 2, 3, 4, 9, 10, 11, 12, 17, 18, 19],
    out_channels=[0, 17, 18, 19],
    img_shape_x=448,
    img_shape_y=448,
    roll=False,
    add_grid=True,
    ds_factor=4,
    min_path=None,
    max_path=None,
    global_means_path=None,
    global_stds_path=None,
    gridtype="sinusoidal",
    N_grid_channels=4,
    normalization="v1",
):
    """CorrDiff training loop"""

    # Instantiate distributed manager.
    dist = DistributedManager()
    device = dist.device

    # Initialize logger.
    logger = PythonLogger(name="training_loop")  # General python logger
    logger0 = RankZeroLoggingWrapper(logger, dist)
    logger.file_logging(file_name="training_loop.log")

    # Initialize.
    start_time = time.time()

    np.random.seed((seed * dist.world_size + dist.rank) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.world_size
    logger0.info(f"batch_gpu: {batch_gpu}")
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    if batch_size != batch_gpu * num_accumulation_rounds * dist.world_size:
        raise ValueError(
            "batch_size must be equal to batch_gpu * num_accumulation_rounds * dist.world_size"
        )

    # Load dataset: weather
    logger0.info("Loading dataset...")
    dataset_obj = get_zarr_dataset(
        dataset=train_data_path,
        in_channels=in_channels,
        out_channels=out_channels,
        img_shape_x=img_shape_x,
        img_shape_y=img_shape_y,
        crop_size_x=crop_size_x,
        crop_size_y=crop_size_y,
        roll=roll,
        add_grid=add_grid,
        ds_factor=ds_factor,
        train=True,
        all_times=False,  # TODO check if this should be False
        n_history=n_history,
        min_path=min_path,
        max_path=max_path,
        global_means_path=global_means_path,
        global_stds_path=global_stds_path,
        normalization=normalization,
        gridtype=gridtype,
        N_grid_channels=N_grid_channels,
    )
    worker_init_fn = None

    dataset_sampler = InfiniteSampler(
        dataset=dataset_obj, rank=dist.rank, num_replicas=dist.world_size, seed=seed
    )
    dataset_iterator = iter(
        torch.utils.data.DataLoader(
            dataset=dataset_obj,
            sampler=dataset_sampler,
            batch_size=batch_gpu,
            worker_init_fn=worker_init_fn,
            **data_loader_kwargs,
        )
    )

    img_in_channels = len(in_channels)  # noise + low-res input
    if add_grid:
        img_in_channels = img_in_channels + N_grid_channels

    img_out_channels = len(out_channels)

    # Construct network.
    logger0.info("Constructing network...")
    interface_kwargs = dict(
        img_resolution=crop_size_x,
        img_channels=img_out_channels,
        img_in_channels=img_in_channels,
        img_out_channels=img_out_channels,
        label_dim=0,
    )  # weather
    net = construct_class_by_name(
        **network_kwargs, **interface_kwargs
    )  # subclass of torch.nn.Module
    net.train().requires_grad_(True).to(device)

    # Setup optimizer.
    logger0.info("Setting up optimizer...")
    loss_fn = construct_class_by_name(**loss_kwargs)  # training.loss.(VP|VE|EDM)Loss
    optimizer = construct_class_by_name(
        params=net.parameters(), **optimizer_kwargs
    )  # subclass of torch.optim.Optimizer
    augment_pipe = (
        construct_class_by_name(**augment_kwargs)
        if augment_kwargs is not None
        else None
    )  # training.augment.AugmentPipe
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

    # Import autoresume module
    SUBMIT_SCRIPTS = "/lustre/fsw/adlr/adlr-others/gpeled/adlr-utils/release/cluster-interface/latest"
    sys.path.append(SUBMIT_SCRIPTS)
    # sync autoresums across gpus ...
    AutoResume = None
    try:
        from userlib.auto_resume import AutoResume

        AutoResume.init()
    except ImportError:
        logger0.warning("AutoResume not imported")

    # Resume training from previous snapshot.
    if resume_pkl is not None:
        logger0.info(f'Loading network weights from "{resume_pkl}"...')
        if dist.rank != 0:
            torch.distributed.barrier()  # rank 0 goes first
        with open_url(resume_pkl, verbose=(dist.rank == 0)) as f:
            # ruff: noqa: S301  # TODO remove exception
            data = pickle.load(f)
        if dist.rank == 0:
            torch.distributed.barrier()  # other ranks follow
        copy_params_and_buffers(
            src_module=data["ema"], dst_module=net, require_all=False
        )
        copy_params_and_buffers(
            src_module=data["ema"], dst_module=ema, require_all=False
        )
        del data  # conserve memory
    if resume_state_dump:
        logger0.info(f'Loading training state from "{resume_state_dump}"...')
        data = torch.load(resume_state_dump, map_location=torch.device("cpu"))
        copy_params_and_buffers(
            src_module=data["net"], dst_module=net, require_all=True
        )
        optimizer.load_state_dict(data["optimizer_state"])
        del data  # conserve memory

    # Train.
    logger0.info(f"Training for {total_kimg} kimg...")
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    stats_jsonl = None
    while True:
        # Accumulate gradients.
        optimizer.zero_grad(set_to_none=True)
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

                loss = loss_fn(
                    net=ddp,
                    img_clean=img_clean,
                    img_lr=img_lr,
                    labels=labels,
                    augment_pipe=augment_pipe,
                )
                training_stats.report("Loss/loss", loss)
                loss.sum().mul(loss_scaling / batch_gpu_total).backward()

        # Update weights.
        for g in optimizer.param_groups:
            g["lr"] = optimizer_kwargs["lr"] * min(
                cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1
            )
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(
                    param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad
                )
        optimizer.step()

        # Update EMA.
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        # Perform maintenance tasks once per tick.
        cur_nimg += batch_size
        done = cur_nimg >= total_kimg * 1000
        if (
            (not done)
            and (cur_tick != 0)
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

        ckpt_dir = run_dir

        logger0.info(
            f"AutoResume.termination_requested(): {AutoResume.termination_requested()}"
        )
        logger0.info(f"AutoResume: {AutoResume}")

        if AutoResume.termination_requested():
            AutoResume.request_resume()
            logger0.info("Training terminated. Returning")
            done = True
            with open(os.path.join(ckpt_dir, "resume.txt"), "w") as f:
                f.write(
                    os.path.join(ckpt_dir, f"training-state-{cur_nimg//1000:06d}.pt")
                )
                logger0.info(
                    os.path.join(ckpt_dir, f"training-state-{cur_nimg//1000:06d}.pt")
                )
                f.close()
                # return 0

        # Check for abort.
        logger0.info(f"dist.should_stop(): {dist.should_stop()}")
        logger0.info(f"done: {done}")

        # Save network snapshot.
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            data = dict(
                ema=ema,
                loss_fn=loss_fn,
                augment_pipe=augment_pipe,
                dataset_kwargs=dict(dataset_kwargs),
            )
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value  # conserve memory
            if dist.rank == 0:
                with open(
                    os.path.join(run_dir, f"network-snapshot-{cur_nimg//1000:06d}.pkl"),
                    "wb",
                ) as f:
                    pickle.dump(data, f)
            del data  # conserve memory

        # Save full dump of the training state.
        if (
            (state_dump_ticks is not None)
            and (done or cur_tick % state_dump_ticks == 0)
            and dist.rank == 0
        ):
            torch.save(
                dict(net=net, optimizer_state=optimizer.state_dict()),
                os.path.join(run_dir, f"training-state-{cur_nimg//1000:06d}.pt"),
            )

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


# ----------------------------------------------------------------------------
