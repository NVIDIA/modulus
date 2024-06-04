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

"""Train diffusion-based generative model using the techniques described in the
paper "Elucidating the Design Space of Diffusion-Based Generative Models"."""
import sys; sys.path.append('/code/modulus')
import json
import os
import shutil

# ruff: noqa: E402
os.environ["TORCHELASTIC_ENABLE_FILE_TIMER"] = "1"

import hydra
from hydra.utils import to_absolute_path
from hydra.core.hydra_config import HydraConfig
import torch
from omegaconf import OmegaConf, DictConfig, ListConfig

import modulus
from modulus.distributed import DistributedManager
from modulus.launch.logging import PythonLogger, RankZeroLoggingWrapper
from modulus.utils.generative import EasyDict, parse_int_list

from training import training_loop
from datasets.dataset import init_train_valid_datasets_from_config


@hydra.main(version_base="1.2", config_path="conf", config_name="config_train_base")
def main(cfg: DictConfig) -> None:
    """Train diffusion-based generative model using the techniques described in the
    paper "Elucidating the Design Space of Diffusion-Based Generative Models".
    """

    # Sanity check
    if not hasattr(cfg, "task"):
        raise ValueError(
            """Need to specify the task. Make sure the right config file is used. Run training using python train.py --config-name=<your_yaml_file>.
            For example, for regression training, run python train.py --config-name=config_train_regression.
            And for diffusion training, run python train.py --config-name=config_train_diffusion."""
        )

    # Dump the configs
    os.makedirs(cfg.outdir, exist_ok=True)
    OmegaConf.save(cfg, os.path.join(cfg.outdir, "config.yaml"))

    # Parse options
    regression_checkpoint_path = getattr(cfg, "regression_checkpoint_path", None)
    if regression_checkpoint_path:
        regression_checkpoint_path = to_absolute_path(regression_checkpoint_path)
    task = getattr(cfg, "task")
    outdir = getattr(cfg, "outdir", "./output")
    arch = getattr(cfg, "arch", "ddpmpp-cwb-v0-regression")
    precond = getattr(cfg, "precond", "unetregression")

    # parse hyperparameters
    duration = getattr(cfg, "duration", 200)
    batch_size_global = getattr(cfg, "batch_size_global", 256)
    batch_size_gpu = getattr(cfg, "batch_size_gpu", 2)
    cbase = getattr(cfg, "cbase", 1)
    # cres = parse_int_list(getattr(cfg, "cres", None))
    lr = getattr(cfg, "lr", 0.0002)
    ema = getattr(cfg, "ema", 0.5)
    dropout = getattr(cfg, "dropout", 0.13)
    augment = getattr(cfg, "augment", 0.0)

    # Parse performance options
    if hasattr(cfg, "fp_optimizations"):
        fp_optimizations = cfg.fp_optimizations
        fp16 = fp_optimizations == "fp16"
    else:
        # look for legacy "fp16" parameter
        fp16 = getattr(cfg, "fp16", False)
        fp_optimizations = "fp16" if fp16 else "fp32"
    ls = getattr(cfg, "ls", 1)
    bench = getattr(cfg, "bench", True)
    workers = getattr(cfg, "workers", 4)
    songunet_checkpoint_level = getattr(cfg, "songunet_checkpoint_level", 0)

    # Parse I/O-related options
    wandb_mode = getattr(cfg, "wandb_mode", "disabled")
    wandb_project = getattr(cfg, "wandb_project", "Modulus-Generative")
    wandb_entity = getattr(cfg, "wandb_entity", "CorrDiff-DDP-Group")
    tick = getattr(cfg, "tick", 1)
    dump = getattr(cfg, "dump", 500)
    validation_dump = getattr(cfg, "validation_dump", 500)
    validation_steps = getattr(cfg, "validation_steps", 10)
    seed = getattr(cfg, "seed", 0)
    transfer = getattr(cfg, "transfer")
    dry_run = getattr(cfg, "dry_run", False)

    # Parse weather data options
    c = EasyDict()
    c.task = task
    c.fp_optimizations = fp_optimizations
    c.wandb_mode = wandb_mode
    c.wandb_project = wandb_project
    c.wandb_entity = wandb_entity
    c.wandb_name = HydraConfig.get().job.name
    c.patch_shape_x = getattr(cfg, "patch_shape_x", None)
    c.patch_shape_y = getattr(cfg, "patch_shape_y", None)
    c.patch_num = getattr(cfg, "patch_num", 1)
    c.grad_clip_threshold = getattr(cfg, "grad_clip_threshold", None)
    c.lr_decay = getattr(cfg, "lr_decay", 0.8)
    c.N_grid_channels = getattr(cfg, "N_grid_channels")
    c.gridtype = getattr(cfg, "gridtype")
    dataset_cfg = OmegaConf.to_container(cfg.dataset)
    validation_dataset_cfg = (
        OmegaConf.to_container(cfg.validation_dataset)
        if hasattr(cfg, "validation_dataset")
        else None
    )
    data_loader_kwargs = EasyDict(
        pin_memory=True, num_workers=workers, prefetch_factor=2
    )
    c.hr_mean_conditioning = getattr(cfg, "hr_mean_conditioning", False)
    c.in_channel = len(dataset_cfg["in_channels"])
    # Initialize distributed manager.
    DistributedManager.initialize()
    dist = DistributedManager()

    # Initialize logger.
    os.makedirs("logs", exist_ok=True)
    logger = PythonLogger(name="train")  # General python logger
    logger0 = RankZeroLoggingWrapper(logger, dist)
    logger.file_logging(file_name=f"logs/train_{dist.rank}.log")

    # Save a copy of the Modulus source code
    # if dist.rank == 0:
    #    shutil.copytree(
    #        os.path.dirname(modulus.__file__), "modulus", dirs_exist_ok=True
    #    )

    # inform about the output
    logger.info(
        f"Checkpoints, logs, configs, and stats will be written in this directory: {os.getcwd()}"
    )

    # Initialize config dict.
    c.network_kwargs = EasyDict()
    c.loss_kwargs = EasyDict()
    c.optimizer_kwargs = EasyDict(
        class_name="torch.optim.Adam", lr=lr, betas=[0.9, 0.999], eps=1e-8
    )

    # Network architecture.
    valid_archs = {
        "ddpmpp-cwb",
        "ddpmpp-cwb-v0-regression",
        "ncsnpp",
        "adm",
    }
    if arch not in valid_archs:
        raise ValueError(
            f"Invalid network architecture {arch}; " f"valid choices are {valid_archs}"
        )

    if arch == "ddpmpp-cwb":
        c.network_kwargs.update(
            model_type="SongUNetPosEmbd",
            embedding_type="positional",
            encoder_type="standard",
            decoder_type="standard",
            checkpoint_level=songunet_checkpoint_level,
        )  # , attn_resolutions=[28]
        c.network_kwargs.update(
            channel_mult_noise=1,
            resample_filter=[1, 1],
            model_channels=128,
            channel_mult=[1, 2, 2, 2, 2],
            attn_resolutions=[28],
        )  # era5-cwb, 448x448

    elif arch == "ddpmpp-cwb-v0-regression":
        c.network_kwargs.update(
            model_type="SongUNetPosEmbd",
            embedding_type="zero",
            encoder_type="standard",
            decoder_type="standard",
            checkpoint_level=songunet_checkpoint_level,
        )  # , attn_resolutions=[28]
        c.network_kwargs.update(
            channel_mult_noise=1,
            resample_filter=[1, 1],
            model_channels=128,
            channel_mult=[1, 2, 2, 2, 2],
            attn_resolutions=[28],
        )  # era5-cwb, 448x448

    elif arch == "ncsnpp":
        c.network_kwargs.update(
            model_type="SongUNetPosEmbd",
            embedding_type="fourier",
            encoder_type="residual",
            decoder_type="standard",
            checkpoint_level=songunet_checkpoint_level,
        )
        c.network_kwargs.update(
            channel_mult_noise=2,
            resample_filter=[1, 3, 3, 1],
            model_channels=128,
            channel_mult=[2, 2, 2],
        )

    else:
        c.network_kwargs.update(
            model_type="DhariwalUNet", model_channels=192, channel_mult=[1, 2, 3, 4]
        )

    # Preconditioning & loss function.
    if precond == "edmv2" or precond == "edm":
        c.network_kwargs.class_name = "modulus.models.diffusion.EDMPrecondSRV2"
        c.loss_kwargs.class_name = "modulus.metrics.diffusion.EDMLossSR"
    elif precond == "edmv1":
        c.network_kwargs.class_name = "modulus.models.diffusion.EDMPrecondSR"
        c.loss_kwargs.class_name = "modulus.metrics.diffusion.EDMLossSR"
    elif precond == "unetregression":
        c.network_kwargs.class_name = "modulus.models.diffusion.UNet"
        c.loss_kwargs.class_name = "modulus.metrics.diffusion.RegressionLoss"
    elif precond == "resloss":
        c.network_kwargs.class_name = "modulus.models.diffusion.EDMPrecondSR"
        c.loss_kwargs.class_name = "modulus.metrics.diffusion.ResLoss"

    # Network options.
    if cbase is not None:
        c.network_kwargs.model_channels = cbase
    # if cres is not None:
    #    c.network_kwargs.channel_mult = cres
    if augment:
        if augment < 0 or augment > 1:
            raise ValueError("Augment probability should be within [0,1].")
        # import augmentation pipe
        try:
            from edmss import AugmentPipe
        except ImportError:
            raise ImportError(
                "Please get the augmentation pipe  by running: pip install git+https://github.com/mnabian/edmss.git"
            )
        c.augment_kwargs = EasyDict(class_name="edmss.AugmentPipe", p=augment)
        c.augment_kwargs.update(
            xflip=1e8, yflip=1, scale=1, rotate_frac=1, aniso=1, translate_frac=1
        )
        c.network_kwargs.augment_dim = 9
    c.network_kwargs.update(dropout=dropout, use_fp16=fp16)

    # Training options.
    c.total_kimg = max(int(duration * 1000), 1)
    c.ema_halflife_kimg = int(ema * 1000)
    c.update(batch_size_gpu=batch_size_gpu, batch_size_global=batch_size_global)
    c.update(loss_scaling=ls, cudnn_benchmark=bench)
    c.update(
        kimg_per_tick=tick,
        state_dump_ticks=dump,
        valid_dump_ticks=validation_dump,
        num_validation_evals=validation_steps,
    )
    if regression_checkpoint_path:
        c.regression_checkpoint_path = regression_checkpoint_path

    # Random seed.
    if seed is None:
        seed = torch.randint(1 << 31, size=[], device=dist.device)
        if dist.distributed:
            torch.distributed.broadcast(seed, src=0)
        seed = int(seed)

    # Transfer learning and resume.
    if transfer is not None:
        c.resume_pkl = transfer
        c.ema_rampup_ratio = None

    c.run_dir = outdir

    # Print options.
    for key in list(c.keys()):
        val = c[key]
        if isinstance(val, (ListConfig, DictConfig)):
            c[key] = OmegaConf.to_container(val, resolve=True)
    logger0.info("Training options:")
    logger0.info(json.dumps(c, indent=2))
    logger0.info(f"Output directory:        {c.run_dir}")
    if "data_path" in dataset_cfg:
        logger0.info(f"Dataset path:            {dataset_cfg['data_path']}")
    logger0.info(f"Network architecture:    {arch}")
    logger0.info(f"Preconditioning & loss:  {precond}")
    logger0.info(f"Number of GPUs:          {dist.world_size}")
    logger0.info(f"Batch size:              {c.batch_size_global}")
    logger0.info(f"Mixed-precision:         {c.fp_optimizations}")

    # Dry run?
    if dry_run:
        logger0.info("Dry run; exiting.")
        return

    # Create output directory.
    logger0.info("Creating output directory...")
    if dist.rank == 0:
        os.makedirs(c.run_dir, exist_ok=True)
        with open(os.path.join(c.run_dir, "training_options.json"), "wt") as f:
            json.dump(c, f, indent=2)

    (
        dataset,
        dataset_iter,
        valid_dataset,
        valid_dataset_iter,
    ) = init_train_valid_datasets_from_config(
        dataset_cfg,
        data_loader_kwargs,
        batch_size=batch_size_gpu,
        seed=seed,
        validation_dataset_cfg=validation_dataset_cfg,
    )

    (img_shape_y, img_shape_x) = dataset.image_shape()
    if (c.patch_shape_x is None) or (c.patch_shape_x > img_shape_x):
        c.patch_shape_x = img_shape_x
    if (c.patch_shape_y is None) or (c.patch_shape_y > img_shape_y):
        c.patch_shape_y = img_shape_y
    if c.patch_shape_x != img_shape_x or c.patch_shape_y != img_shape_y:
        if c.patch_shape_x != c.patch_shape_y:
            raise NotImplementedError("Rectangular patch not supported yet")
        if c.patch_shape_x % 32 != 0 or c.patch_shape_y % 32 != 0:
            raise ValueError("Patch shape needs to be a multiple of 32")
        logger0.info("Patch-based training enabled")
    else:
        logger0.info("Patch-based training disabled")

    # Train.
    training_loop.training_loop(
        dataset, dataset_iter, valid_dataset, valid_dataset_iter, **c
    )


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
