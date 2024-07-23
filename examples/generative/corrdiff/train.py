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

import os, json, time, psutil  # TODO organize imports
import hydra
from hydra.utils import to_absolute_path
import torch
import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf

from torch.nn.parallel import DistributedDataParallel
#from . import training_stats

from modulus.utils.generative import (
    construct_class_by_name,
    ddp_sync,
    format_time,
)

from modulus import Module
from modulus.distributed import DistributedManager
from modulus.launch.logging import PythonLogger, RankZeroLoggingWrapper
from modulus.utils.generative import EasyDict
from modulus.metrics.diffusion import RegressionLoss, ResLoss

from modulus.launch.logging import (
    LaunchLogger,
    PythonLogger,
    RankZeroLoggingWrapper,
    initialize_mlflow,
)

from datasets.dataset import init_train_valid_datasets_from_config
from utils import set_patch_shape

@hydra.main(version_base="1.2", config_path="conf", config_name="config_training")
def main(cfg: DictConfig) -> None:
    """Train diffusion-based generative model using the techniques described in the
    paper "Elucidating the Design Space of Diffusion-Based Generative Models".
    """

    # Resolve and dump configs
    OmegaConf.resolve(cfg)
    #os.makedirs(cfg.outdir, exist_ok=True)
    #OmegaConf.save(cfg, os.path.join(cfg.outdir, "config.yaml"))

    # Parse regression checkpoint options
    regression_checkpoint_path = getattr(cfg, "regression_checkpoint_path", None)
    if hasattr(cfg, "training.io.regression_checkpoint_path"):
        regression_checkpoint_path = to_absolute_path(cfg.training.io.regression_checkpoint_path)
    else:
        regression_checkpoint_path = None

    # Parse performance options
    fp_optimizations = cfg.training.perf.fp_optimizations
    fp16 = fp_optimizations == "fp16"
    enable_amp = fp_optimizations.startswith("amp")
    amp_dtype = torch.float16 if (fp_optimizations == "amp-fp16") else torch.bfloat16

    # Parse dataset options
    dataset_cfg = OmegaConf.to_container(cfg.dataset)  # TODO needs better handling
    validation_dataset_cfg = (
        OmegaConf.to_container(cfg.validation_dataset)
        if hasattr(cfg, "validation_dataset")
        else None
    )

    # Initialize distributed environment for training
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device

    # Set seeds for NumPy and PyTorch to ensure reproducibility in distributed settings, and configure
    # cuDNN and CUDA to disable TF32 and reduced precision settings for consistent precision.
    np.random.seed(dist.rank % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cfg.training.perf.cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Initialize loggers
    initialize_mlflow(
        experiment_name=cfg.experiment_name,
        experiment_desc=cfg.experiment_desc,
        run_name=f"{cfg.model.name}-trainng",
        run_desc=cfg.experiment_desc,
        user_name="Modulus User",
        mode="offline",
    )
    LaunchLogger.initialize(use_mlflow=True)  # Modulus launch logger
    logger = PythonLogger("main")  # General python logger
    logger0 = RankZeroLoggingWrapper(logger, dist) # Rank 0 logger

    # inform about the output
    logger.info(
        f"Checkpoints, logs, configs, and stats will be written in this directory: {os.getcwd()}"
    )

    # Instantiate the dataset
    data_loader_kwargs = {
        'pin_memory': True, 'num_workers': cfg.training.perf.dataloader_workers, 'prefetch_factor': 2
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
    dataset_channels =  len(dataset.input_channels())
    img_in_channels = (
        dataset_channels + cfg.model.args.N_grid_channels
    )  # noise + low-res input
    img_shape = dataset.image_shape()
    img_out_channels = len(dataset.output_channels())
    if cfg.model.hr_mean_conditioning:
        img_in_channels += img_out_channels

    # Parse the patch shape
    if not hasattr(cfg.training.hp.patch_shape_x):
        cfg.training.hp.patch_shape_x = None
    if not hasattr(cfg.training.hp.patch_shape_y):
        cfg.training.hp.patch_shape_y = None
    patch_shape = (cfg.patch_shape_y, cfg.patch_shape_x)
    patch_shape , img_shape = set_patch_shape(img_shape, patch_shape)
    if patch_shape != img_shape:
        logger0.info("Patch-based training enabled")
    else:
        logger0.info("Patch-based training disabled")
    # interpolate global channel if patch-based model is used
    if img_shape[1] != patch_shape[1]:
        img_in_channels += dataset_channels

    # Instantiate the model
    cfg.model.args.img_in_channels = img_in_channels
    cfg.model.args.img_out_channels = img_out_channels
    cfg.model.args.img_resolution = list(img_shape)
    cfg.model.args.use_fp1 = fp16
    model = Module.instantiate(
        {
            "__name__": cfg.model.name,
            "__args__": {
                k: tuple(v) if isinstance(v, ListConfig) else v
                for k, v in cfg.model.args.items()
            },
        }
    )
    model.train().requires_grad_(True).to(device)

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
    if hasattr(cfg.training.io, 'regression_checkpoint_path'):
        if not os.path.exists(cfg.regression_checkpoint_path):
            raise FileNotFoundError("The specified regression_checkpoint_path does not exist for training the diffusion model")
        regression_net = Module.from_checkpoint(regression_checkpoint_path)
        regression_net.eval().requires_grad_(False).to(device)
        logger0.success("Loaded the pre-trained regression model")    


    # Instantiate the loss function
    patch_num = getattr(cfg.training, 'patch_num', 1)
    if cfg.model.name=="EDMPrecondSR":
        loss_fn = ResLoss(regression_net=regression_net, img_shape_x=img_shape[1], img_shape_y=img_shape[0], patch_shape_x=patch_shape[1], patch_shape_y=patch_shape[0], patch_num=patch_num, hr_mean_conditioning=cfg.model.hr_mean_conditioning)
    elif cfg.model.name=="UNetWrapper":  # TODO this breaks old checkpoints
        loss_fn = RegressionLoss()

    # Instantiate the optimizer
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=cfg.training.io.lr, betas=[0.9, 0.999], eps=1e-8
    )
    
    # Record the current time to measure the duration of subsequent operations.
    start_time = time.time()

    # Calculate the total batch size per GPU in a distributed setting, log the batch size per GPU, ensure it's within valid limits,
    # determine the number of accumulation rounds, and validate that the global batch size matches the expected value.
    batch_gpu_total = cfg.training.hp.total_batch_size // dist.world_size
    logger0.info(f"batch_size_gpu: {batch_size_gpu}")
    if batch_size_gpu is None or batch_size_gpu > batch_gpu_total:
        batch_size_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_size_gpu
    if cfg.training.hp.total_batch_size != batch_size_gpu * num_accumulation_rounds * dist.world_size:
        raise ValueError(
            "total_batch_size must be equal to batch_size_gpu * num_accumulation_rounds * dist.world_size"
        )

    # Resume training from previous snapshot.
    max_index = -1
    max_index_file = " "
    for filename in os.listdir(cfg.training.io.outdir):
        if filename.startswith(f"training-state-") and filename.endswith(
            ".mdlus"
        ):
            index_str = filename.split("-")[-1].split(".")[0]
            try:
                index = int(index_str)
                if index > max_index:
                    max_index = index
                    max_index_file = filename
                    max_index_file_optimizer = f"optimizer-state-{index_str}.pt"
            except ValueError:
                continue

    try:
        model.load(os.path.join(cfg.training.io.outdir, max_index_file))
        # load state directly to each gpu to reduce memory usage
        map_location = {"cuda:%d" % 0: "cuda:%d" % int(dist.local_rank)}
        optimizer_state_dict = torch.load(
            os.path.join(cfg.training.io.outdir, max_index_file_optimizer), map_location=map_location
        )
        optimizer.load_state_dict(optimizer_state_dict["optimizer_state_dict"])
        cur_nimg = max_index * 1000
        logger0.success(f"Loaded network and optimizer states with index {max_index}")
    except FileNotFoundError:
        cur_nimg = 0
        logger0.warning("Could not load network and optimizer states")

    ########################################################
    #                  MAIN TRAINING LOOP                  #
    ########################################################

    logger0.info(f"Training for {cfg.training.hp.training_duration} kimgs...")
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    stats_jsonl = None
    while True:
        # Accumulate gradients.
        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0

        ##### Continue refactoring from here

        for round_idx in range(num_accumulation_rounds):
            with ddp_sync(model, (round_idx == num_accumulation_rounds - 1)):
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
                        augment_pipe=None,
                    )
                training_stats.report("Loss/loss", loss)
                loss = loss.sum() / batch_gpu_total
                loss_accum += loss / num_accumulation_rounds
                loss.backward()

        loss_sum = torch.tensor([loss_accum], device=device)
        if dist.world_size > 1:
            torch.distributed.all_reduce(loss_sum, op=torch.distributed.ReduceOp.SUM)
        average_loss = loss_sum / dist.world_size

        # Update weights.
        for g in optimizer.param_groups:
            g["lr"] = optimizer_kwargs["lr"] * min(
                cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1
            )  # TODO better handling (potential bug)
            g["lr"] *= cfg.training.io.lr_decay ** ((cur_nimg - lr_rampup_kimg * 1000) // 5e6)
        for param in model.parameters():
            if param.grad is not None:
                torch.nan_to_num(
                    param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad
                )
        if cfg.training.hp.grad_clip_threshold:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.training.hp.grad_clip_threshold
            )
        optimizer.step()
        if validation_dataset_iterator is not None:
            valid_loss_accum = 0
            if cur_tick % cfg.training.io.validation_freq * 1000 == 0:
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
                            augment_pipe=None,
                        )
                        training_stats.report("Loss/validation loss", loss_valid)
                        loss_valid = loss_valid.sum() / batch_gpu_total
                        valid_loss_accum += loss_valid / num_validation_evals
                    valid_loss_sum = torch.tensor([valid_loss_accum], device=device)
                    if dist.world_size > 1:
                        torch.distributed.all_reduce(
                            valid_loss_sum, op=torch.distributed.ReduceOp.SUM
                        )
                    average_valid_loss = valid_loss_sum / dist.world_size

        # Perform maintenance tasks once per tick.
        cur_nimg += cfg.training.hp.total_batch_size
        done = cur_nimg >= cfg.training.hp.training_duration * 1000
        if (
            (not done)
            # and (cur_tick != 0)
            and (cur_nimg < tick_start_nimg + print_progress_freq * 1000)
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
            (cfg.training.io.save_checkpoint_freq is not None)
            and (done or cur_tick % cfg.training.io.save_checkpoint_freq == 0)
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
    logger0.info("Training Completed.")

if __name__ == "__main__":
    main()
