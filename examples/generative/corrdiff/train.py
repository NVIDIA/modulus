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

import os, time, psutil, hydra, torch
import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
from modulus import Module
from modulus.models.diffusion import UNet, EDMPrecondSRV2
from modulus.distributed import DistributedManager
from modulus.launch.logging import PythonLogger, RankZeroLoggingWrapper
from modulus.metrics.diffusion import RegressionLoss, ResLoss
from modulus.launch.logging import PythonLogger, RankZeroLoggingWrapper
from modulus.launch.utils import load_checkpoint, save_checkpoint
from datasets.dataset import init_train_valid_datasets_from_config
from train_helpers import (
    set_patch_shape,
    set_seed,
    configure_cuda_for_consistent_precision,
    compute_num_accumulation_rounds,
    handle_and_clip_gradients,
    parse_model_args,
)


# Train the CorrDiff model using the configurations in "conf/config_training.yaml"
@hydra.main(version_base="1.2", config_path="conf", config_name="config_training")
def main(cfg: DictConfig) -> None:

    # Initialize distributed environment for training
    DistributedManager.initialize()
    dist = DistributedManager()

    # Initialize loggers
    if dist.rank == 0:
        writer = SummaryWriter(log_dir="tensorboard")
    logger = PythonLogger("main")  # General python logger
    logger0 = RankZeroLoggingWrapper(logger, dist)  # Rank 0 logger

    # Resolve and parse configs
    OmegaConf.resolve(cfg)
    dataset_cfg = OmegaConf.to_container(cfg.dataset)  # TODO needs better handling
    if hasattr(cfg, "validation_dataset"):
        validation_dataset_cfg = OmegaConf.to_container(cfg.validation_dataset)
    else:
        validation_dataset_cfg = None
    fp_optimizations = cfg.training.perf.fp_optimizations
    fp16 = fp_optimizations == "fp16"
    enable_amp = fp_optimizations.startswith("amp")
    amp_dtype = torch.float16 if (fp_optimizations == "amp-fp16") else torch.bfloat16
    logger.info(f"Saving the outputs in {os.getcwd()}")

    # Set seeds and configure CUDA and cuDNN settings to ensure consistent precision
    set_seed(dist.rank)
    configure_cuda_for_consistent_precision()

    # Instantiate the dataset
    data_loader_kwargs = {
        "pin_memory": True,
        "num_workers": cfg.training.perf.dataloader_workers,
        "prefetch_factor": 2,
    }
    (
        dataset,
        dataset_iterator,
        validation_dataset,
        validation_dataset_iterator,
    ) = init_train_valid_datasets_from_config(
        dataset_cfg,
        data_loader_kwargs,
        batch_size=cfg.training.hp.batch_size_per_gpu,
        seed=0,
        validation_dataset_cfg=validation_dataset_cfg,
    )

    # Parse image configuration & update model args
    dataset_channels = len(dataset.input_channels())
    img_in_channels = dataset_channels
    img_shape = dataset.image_shape()
    img_out_channels = len(dataset.output_channels())
    if cfg.model.hr_mean_conditioning:
        img_in_channels += img_out_channels

    # Parse the patch shape
    if hasattr(cfg, "training.hp.patch_shape_x"):
        patch_shape_x = cfg.training.hp.patch_shape_x
    else:
        patch_shape_x = None
    if hasattr(cfg, "training.hp.patch_shape_y"):
        patch_shape_y = cfg.training.hp.patch_shape_y
    else:
        patch_shape_y = None
    patch_shape = (patch_shape_y, patch_shape_x)
    patch_shape, img_shape = set_patch_shape(img_shape, patch_shape)
    if patch_shape != img_shape:
        logger0.info("Patch-based training enabled")
    else:
        logger0.info("Patch-based training disabled")
    # interpolate global channel if patch-based model is used
    if img_shape[1] != patch_shape[1]:
        img_in_channels += dataset_channels

    # Instantiate the model and move to device.
    additional_model_args = {
        "img_out_channels": img_out_channels,
        "img_resolution": list(img_shape),
        "use_fp16": fp16,
    }
    if cfg.model.name == "regression":
        model = UNet(
            img_channels=4,
            N_grid_channels=4,
            embedding_type="zero",
            img_in_channels=img_in_channels + 4,
            **additional_model_args,
        )
    elif cfg.model.name == "diffusion":
        model = EDMPrecondSRV2(
            img_channels=4,
            gridtype="sinusoidal",
            N_grid_channels=4,
            img_in_channels=img_in_channels + 4,
            **additional_model_args,
        )
    elif cfg.model.name == "patched_diffusion":
        model = EDMPrecondSRV2(
            img_channels=4,
            gridtype="learnable",
            N_grid_channels=100,
            img_in_channels=img_in_channels + 100,
            **additional_model_args,
        )
    else:
        raise ValueError("Invalid model")
    model.train().requires_grad_(True).to(dist.device)

    # Enable distributed data parallel if applicable
    if dist.world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[dist.local_rank],
            broadcast_buffers=True,
            output_device=dist.device,
            find_unused_parameters=dist.find_unused_parameters,
        )

    # Load the regression checkpoint if applicable
    if hasattr(cfg.training.io, "regression_checkpoint_path"):
        regression_checkpoint_path = to_absolute_path(
            cfg.training.io.regression_checkpoint_path
        )
        if not os.path.exists(regression_checkpoint_path):
            raise FileNotFoundError(
                "Expected a this regression checkpoint but not found: {regression_checkpoint_path}"
            )
        regression_net = Module.from_checkpoint(regression_checkpoint_path)
        regression_net.eval().requires_grad_(False).to(dist.device)
        logger0.success("Loaded the pre-trained regression model")

    # Instantiate the loss function
    patch_num = getattr(cfg.training, "patch_num", 1)
    if cfg.model.name in ("diffusion", "patched_diffusion"):
        loss_fn = ResLoss(
            regression_net=regression_net,
            img_shape_x=img_shape[1],
            img_shape_y=img_shape[0],
            patch_shape_x=patch_shape[1],
            patch_shape_y=patch_shape[0],
            patch_num=patch_num,
            hr_mean_conditioning=cfg.model.hr_mean_conditioning,
        )
    elif cfg.model.name == "regression":
        loss_fn = RegressionLoss()

    # Instantiate the optimizer
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=cfg.training.hp.lr, betas=[0.9, 0.999], eps=1e-8
    )

    # Record the current time to measure the duration of subsequent operations.
    start_time = time.time()

    # Compute the number of required gradient accumulation rounds
    # It is automatically used if batch_size_per_gpu * dist.world_size < total_batch_size
    batch_gpu_total, num_accumulation_rounds = compute_num_accumulation_rounds(
        cfg.training.hp.total_batch_size,
        cfg.training.hp.batch_size_per_gpu,
        dist.world_size,
    )
    logger0.info(f"Using {num_accumulation_rounds} gradient accumulation rounds")

    ## Resume training from previous checkpoints if exists
    if dist.world_size > 1:
        torch.distributed.barrier()
    cur_nimg = load_checkpoint(
        path="checkpoints", models=model, optimizer=optimizer, device=dist.device
    )

    ############################################################################
    #                            MAIN TRAINING LOOP                            #
    ############################################################################

    logger0.info(f"Training for {cfg.training.hp.training_duration} images...")
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    while True:
        # Compute & accumulate gradients
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0
        for _ in range(num_accumulation_rounds):
            img_clean, img_lr, labels = next(dataset_iterator)
            img_clean = img_clean.to(dist.device).to(torch.float32).contiguous()
            img_lr = img_lr.to(dist.device).to(torch.float32).contiguous()
            labels = labels.to(dist.device).contiguous()
            with torch.autocast("cuda", dtype=amp_dtype, enabled=enable_amp):
                loss = loss_fn(
                    net=model,
                    img_clean=img_clean,
                    img_lr=img_lr,
                    labels=labels,
                    augment_pipe=None,
                )
            loss = loss.sum() / batch_gpu_total
            loss_accum += loss / num_accumulation_rounds
            loss.backward()

        loss_sum = torch.tensor([loss_accum], device=dist.device)
        if dist.world_size > 1:
            torch.distributed.all_reduce(loss_sum, op=torch.distributed.ReduceOp.SUM)
        average_loss = (loss_sum / dist.world_size).cpu().item()
        if dist.rank == 0:
            writer.add_scalar("training_loss", average_loss, cur_nimg)

        # Update weights.
        for g in optimizer.param_groups:
            lr_rampup = 10000000  # ramp up the learning rate within 10M images
            g["lr"] = cfg.training.hp.lr * min(cur_nimg / lr_rampup, 1)
            g["lr"] *= cfg.training.hp.lr_decay ** ((cur_nimg - lr_rampup) // 5e6)
            current_lr = g["lr"]
            if dist.rank == 0:
                writer.add_scalar("learning_rate", current_lr, cur_nimg)
        handle_and_clip_gradients(
            model, grad_clip_threshold=cfg.training.hp.grad_clip_threshold
        )
        optimizer.step()

        # Validation
        if validation_dataset_iterator is not None:
            valid_loss_accum = 0
            if cur_tick % cfg.training.io.validation_freq == 0:
                with torch.no_grad():
                    for _ in range(cfg.training.io.validation_steps):
                        img_clean_valid, img_lr_valid, labels_valid = next(
                            validation_dataset_iterator
                        )

                        img_clean_valid = (
                            img_clean_valid.to(dist.device)
                            .to(torch.float32)
                            .contiguous()
                        )
                        img_lr_valid = (
                            img_lr_valid.to(dist.device).to(torch.float32).contiguous()
                        )
                        labels_valid = labels_valid.to(dist.device).contiguous()
                        loss_valid = loss_fn(
                            net=model,
                            img_clean=img_clean_valid,
                            img_lr=img_lr_valid,
                            labels=labels_valid,
                            augment_pipe=None,
                        )
                        loss_valid = (loss_valid.sum() / batch_gpu_total).cpu().item()
                        valid_loss_accum += (
                            loss_valid / cfg.training.io.validation_steps
                        )
                    valid_loss_sum = torch.tensor(
                        [valid_loss_accum], device=dist.device
                    )
                    if dist.world_size > 1:
                        torch.distributed.all_reduce(
                            valid_loss_sum, op=torch.distributed.ReduceOp.SUM
                        )
                    average_valid_loss = valid_loss_sum / dist.world_size
                    if dist.rank == 0:
                        writer.add_scalar(
                            "validation_loss", average_valid_loss, cur_nimg
                        )

        cur_nimg += cfg.training.hp.total_batch_size
        done = cur_nimg >= cfg.training.hp.training_duration
        if (not done) and (
            cur_nimg < tick_start_nimg + cfg.training.io.print_progress_freq
        ):  # TODO revert
            continue

        # Print stats
        tick_end_time = time.time()
        if dist.rank == 0:
            fields = []
            fields += [f"tick {cur_tick:<5d}"]
            fields += [f"samples {(cur_nimg):<9.1f}"]
            fields += [f"training_loss {average_loss:<7.2f}"]
            fields += [f"learning_rate {current_lr:<7.8f}"]
            fields += [f"total_sec {(tick_end_time - start_time):<7.1f}"]
            fields += [f"sec_per_tick {(tick_end_time - tick_start_time):<7.1f}"]
            fields += [
                f"sec_per_sample {((tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg)):<7.2f}"
            ]
            fields += [
                f"cpu_mem_gb {(psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"
            ]
            fields += [
                f"peak_gpu_mem_gb {(torch.cuda.max_memory_allocated(dist.device) / 2**30):<6.2f}"
            ]
            fields += [
                f"peak_gpu_mem_reserved_gb {(torch.cuda.max_memory_reserved(dist.device) / 2**30):<6.2f}"
            ]
            logger0.info(" ".join(fields))
        torch.cuda.reset_peak_memory_stats()

        # Save checkpoints
        if (
            (cfg.training.io.save_checkpoint_freq is not None)
            and (done or cur_tick % cfg.training.io.save_checkpoint_freq == 0)
            and dist.rank == 0
        ):
            if dist.world_size > 1:
                torch.distributed.barrier()
            save_checkpoint(
                path="checkpoints", models=model, optimizer=optimizer, epoch=cur_nimg
            )

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        if done:
            break

    # Done.
    logger0.info("Training Completed.")


if __name__ == "__main__":
    main()
