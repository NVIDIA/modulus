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

"""Train diffusion-based generative model using the techniques described in the
paper "Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os

# ruff: noqa: E402
os.environ["TORCHELASTIC_ENABLE_FILE_TIMER"] = "1"

import json
import re
import warnings

import hydra
import torch
from omegaconf import OmegaConf, DictConfig, ListConfig
from training import training_loop

from modulus.distributed import DistributedManager
from modulus.launch.logging import PythonLogger, RankZeroLoggingWrapper
from modulus.utils.generative import EasyDict, parse_int_list

warnings.filterwarnings(
    "ignore", "Grad strides do not match bucket view strides"
)  # False warning printed by PyTorch 1.12.


@hydra.main(version_base="1.2", config_path="conf", config_name="config_train")
def main(cfg: DictConfig) -> None:
    """Train diffusion-based generative model using the techniques described in the
    paper "Elucidating the Design Space of Diffusion-Based Generative Models".
    """

    # Parse options
    outdir = getattr(cfg, "outdir", "./output")
    data = getattr(cfg, "data", "./data")
    arch = getattr(cfg, "arch", "ddpmpp-cwb-v0-regression")
    precond = getattr(cfg, "precond", "unetregression")

    # parse hyperparameters
    duration = getattr(cfg, "duration", 200)
    batch = getattr(cfg, "batch", 256)
    batch_gpu = getattr(cfg, "batch_gpu", 2)
    cbase = getattr(cfg, "cbase", 1)
    cres = parse_int_list(getattr(cfg, "cres", None))
    lr = getattr(cfg, "lr", 0.0002)
    ema = getattr(cfg, "ema", 0.5)
    dropout = getattr(cfg, "dropout", 0.13)
    augment = getattr(cfg, "augment", 0.0)

    # Parse performance options
    fp16 = getattr(cfg, "fp16", False)
    ls = getattr(cfg, "ls", 1)
    bench = getattr(cfg, "bench", True)
    workers = getattr(cfg, "workers", 4)

    # Parse I/O-related options
    desc = getattr(cfg, "desc")
    tick = getattr(cfg, "tick", 1)
    snap = getattr(cfg, "snap", 1)
    dump = getattr(cfg, "dump", 500)
    seed = getattr(cfg, "seed", 0)
    transfer = getattr(cfg, "transfer")
    resume = getattr(cfg, "resume")
    dry_run = getattr(cfg, "dry_run", False)

    # Parse weather data options
    c = EasyDict()
    c.train_data_path = getattr(cfg, "train_data_path")
    c.crop_size_x = getattr(cfg, "crop_size_x", 448)
    c.crop_size_y = getattr(cfg, "crop_size_y", 448)
    c.n_history = getattr(cfg, "n_history", 0)
    c.in_channels = getattr(
        cfg, "in_channels", [0, 1, 2, 3, 4, 9, 10, 11, 12, 17, 18, 19]
    )
    c.out_channels = getattr(cfg, "out_channels", [0, 17, 18, 19])
    c.img_shape_x = getattr(cfg, "img_shape_x", 448)
    c.img_shape_y = getattr(cfg, "img_shape_y", 448)
    c.roll = getattr(cfg, "roll", False)
    c.add_grid = getattr(cfg, "add_grid", True)
    c.ds_factor = getattr(cfg, "ds_factor", 1)
    c.min_path = getattr(cfg, "min_path", None)
    c.max_path = getattr(cfg, "max_path", None)
    c.global_means_path = getattr(cfg, "global_means_path", None)
    c.global_stds_path = getattr(cfg, "global_stds_path", None)
    c.gridtype = getattr(cfg, "gridtype", "sinusoidal")
    c.N_grid_channels = getattr(cfg, "N_grid_channels", 4)
    c.normalization = getattr(cfg, "normalization", "v2")

    # Initialize distributed manager.
    DistributedManager.initialize()
    dist = DistributedManager()

    # Initialize logger.
    logger = PythonLogger(name="train")  # General python logger
    logger0 = RankZeroLoggingWrapper(logger, dist)
    logger.file_logging(file_name="train.log")

    # Initialize config dict.
    c.dataset_kwargs = EasyDict(path=data, xflip=False, cache=True, use_labels=False)
    c.data_loader_kwargs = EasyDict(
        pin_memory=True, num_workers=workers, prefetch_factor=2
    )
    c.network_kwargs = EasyDict()
    c.loss_kwargs = EasyDict()
    c.optimizer_kwargs = EasyDict(
        class_name="torch.optim.Adam", lr=lr, betas=[0.9, 0.999], eps=1e-8
    )

    # Network architecture.
    valid_archs = {
        "ddpmpp-cwb-v2",
        "ddpmpp-cwb-v1",
        "ddpmpp-cwb-v0-regression",
        "ddpmpp-cwb-v0",
        "ddpmpp-cifar",
        "ncsnpp",
        "adm",
    }
    if arch not in valid_archs:
        raise ValueError(
            f"Invalid network architecture {arch}; " f"valid choices are {valid_archs}"
        )

    if arch == "ddpmpp-cwb-v2":
        c.network_kwargs.update(
            model_type="SongUNet",
            embedding_type="positional",
            encoder_type="standard",
            decoder_type="standard",
        )  # , attn_resolutions=[28]
        c.network_kwargs.update(
            channel_mult_noise=1,
            resample_filter=[1, 1],
            model_channels=128,
            channel_mult=[1, 2, 2, 4, 4, 8],
            attn_resolutions=[14],
        )  # era5-cwb, larger run, 448x448

    elif arch == "ddpmpp-cwb-v1":
        c.network_kwargs.update(
            model_type="SongUNet",
            embedding_type="positional",
            encoder_type="standard",
            decoder_type="standard",
        )  # , attn_resolutions=[28]
        c.network_kwargs.update(
            channel_mult_noise=1,
            resample_filter=[1, 1],
            model_channels=128,
            channel_mult=[1, 2, 2, 4, 4],
            attn_resolutions=[28],
        )  # era5-cwb, 448x448

    elif arch == "ddpmpp-cwb-v0-regression":
        c.network_kwargs.update(
            model_type="SongUNet",
            embedding_type="zero",
            encoder_type="standard",
            decoder_type="standard",
        )  # , attn_resolutions=[28]
        c.network_kwargs.update(
            channel_mult_noise=1,
            resample_filter=[1, 1],
            model_channels=128,
            channel_mult=[1, 2, 2, 2, 2],
            attn_resolutions=[28],
        )  # era5-cwb, 448x448

    elif arch == "ddpmpp-cwb-v0":
        c.network_kwargs.update(
            model_type="SongUNet",
            embedding_type="positional",
            encoder_type="standard",
            decoder_type="standard",
        )  # , attn_resolutions=[28]
        c.network_kwargs.update(
            channel_mult_noise=1,
            resample_filter=[1, 1],
            model_channels=128,
            channel_mult=[1, 2, 2, 2, 2],
            attn_resolutions=[28],
        )  # era5-cwb, 448x448

    elif arch == "ddpmpp-cifar":
        c.network_kwargs.update(
            model_type="SongUNet",
            embedding_type="positional",
            encoder_type="standard",
            decoder_type="standard",
        )  # , attn_resolutions=[28]
        c.network_kwargs.update(
            channel_mult_noise=1,
            resample_filter=[1, 1],
            model_channels=128,
            channel_mult=[2, 2, 2],
        )  # cifar-10, 32x32

    elif arch == "ncsnpp":
        c.network_kwargs.update(
            model_type="SongUNet",
            embedding_type="fourier",
            encoder_type="residual",
            decoder_type="standard",
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
    if precond == "edm":
        c.network_kwargs.class_name = "training.networks.EDMPrecondSR"
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
    if cres is not None:
        c.network_kwargs.channel_mult = cres
    if augment > 0:
        raise NotImplementedError("Augmentation is not implemented")
    c.network_kwargs.update(dropout=dropout, use_fp16=fp16)

    # Training options.
    c.total_kimg = max(int(duration * 1000), 1)
    c.ema_halflife_kimg = int(ema * 1000)
    c.update(batch_size=batch, batch_gpu=batch_gpu)
    c.update(loss_scaling=ls, cudnn_benchmark=bench)
    c.update(kimg_per_tick=tick, snapshot_ticks=snap, state_dump_ticks=dump)

    # Random seed.
    if seed is not None:
        c.seed = seed
    else:
        seed = torch.randint(1 << 31, size=[], device=dist.device)
        if dist.distributed:
            torch.distributed.broadcast(seed, src=0)
        c.seed = int(seed)

    # check if resume.txt exists
    resume_path = os.path.join(outdir, "resume.txt")
    if os.path.exists(resume_path):
        with open(resume_path, "r") as f:
            resume = f.read()
            f.close()

    logger0.info(f"resume: { resume}")

    # Transfer learning and resume.
    if transfer is not None:
        if resume is not None:
            raise ValueError("transfer and resume cannot be specified at the same time")
        c.resume_pkl = transfer
        c.ema_rampup_ratio = None
    elif resume is not None:
        logger.info("gets into elif resume is not None ...")
        match = re.fullmatch(r"training-state-(\d+).pt", os.path.basename(resume))
        logger.info("match", match)
        logger.info("match.group(1)", match.group(1))
        c.resume_pkl = os.path.join(
            os.path.dirname(resume), f"network-snapshot-{match.group(1)}.pkl"
        )
        c.resume_kimg = int(match.group(1))
        c.resume_state_dump = resume
        logger0.info(f"c.resume_pkl: {c.resume_pkl}")
        logger0.info(f"c.resume_kimg: {c.resume_kimg}")
        logger0.info(f"c.resume_state_dump: {c.resume_state_dump}")

    # Description string.
    cond_str = "cond" if c.dataset_kwargs.use_labels else "uncond"
    dtype_str = "fp16" if c.network_kwargs.use_fp16 else "fp32"
    desc = f"{cond_str:s}-{arch:s}-{precond:s}-gpus{dist.world_size:d}-batch{c.batch_size:d}-{dtype_str:s}"
    if desc is not None:
        desc += f"-{desc}"

    c.run_dir = outdir

    # Print options.
    for key in list(c.keys()):
        val = c[key]
        if isinstance(val, (ListConfig, DictConfig)):
            c[key] = OmegaConf.to_container(val, resolve=True)
    logger0.info("Training options:")
    logger0.info(json.dumps(c, indent=2))
    logger0.info(f"Output directory:        {c.run_dir}")
    logger0.info(f"Dataset path:            {c.dataset_kwargs.path}")
    logger0.info(f"Class-conditional:       {c.dataset_kwargs.use_labels}")
    logger0.info(f"Network architecture:    {arch}")
    logger0.info(f"Preconditioning & loss:  {precond}")
    logger0.info(f"Number of GPUs:          {dist.world_size}")
    logger0.info(f"Batch size:              {c.batch_size}")
    logger0.info(f"Mixed-precision:         {c.network_kwargs.use_fp16}")

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
        # dnnlib.util.Logger(
        #     file_name=os.path.join(c.run_dir, "log.txt"),
        #     file_mode="a",
        #     should_flush=True,
        # )

    # Train.
    training_loop.training_loop(**c)


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
