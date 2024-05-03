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

from contextlib import nullcontext
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel
import numpy as np
import time
import wandb
import torch.cuda.profiler as profiler
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, LambdaLR

# import modules
import os

from modulus.models.graphcast.graph_cast_net import GraphCastNet
from modulus.utils.graphcast.loss import CellAreaWeightedLossFunction
from modulus.launch.logging import (
    PythonLogger,
    initialize_wandb,
    RankZeroLoggingWrapper,
)
from modulus.launch.utils import load_checkpoint, save_checkpoint

from train_utils import count_trainable_params
from loss.utils import grid_cell_area
from train_base import BaseTrainer
from validation import Validation
from modulus.datapipes.climate import ERA5HDF5Datapipe, SyntheticWeatherDataLoader
from modulus.distributed import DistributedManager

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig


class GraphCastTrainer(BaseTrainer):
    """GraphCast Trainer"""

    def __init__(self, cfg: DictConfig, dist, rank_zero_logger):
        super().__init__()
        self.dist = dist
        self.dtype = torch.bfloat16 if cfg.full_bf16 else torch.float32
        self.enable_scaler = False
        self.amp = cfg.amp
        self.amp_dtype = None
        self.pyt_profiler = cfg.pyt_profiler
        self.grad_clip_norm = cfg.grad_clip_norm
        static_dataset_path = (
            to_absolute_path(cfg.static_dataset_path)
            if cfg.static_dataset_path
            else None
        )

        if cfg.full_bf16:
            assert torch.cuda.is_bf16_supported()
            rank_zero_logger.info(f"Using {str(self.dtype)} dtype")
            if cfg.amp:
                raise ValueError(
                    "Full bfloat16 training is enabled, switch off amp in config"
                )

        if cfg.amp:
            rank_zero_logger.info(f"Using config amp with dtype {cfg.amp_dtype}")
            if cfg.amp_dtype == "float16" or cfg.amp_dtype == "fp16":
                self.amp_dtype = torch.float16
                self.enable_scaler = True
            elif self.amp_dtype == "bfloat16" or self.amp_dtype == "bf16":
                self.amp_dtype = torch.bfloat16
            else:
                raise ValueError("Invalid dtype for config amp")

        # instantiate the model
        self.model = GraphCastNet(
            meshgraph_path=to_absolute_path(cfg.icospheres_path),
            static_dataset_path=static_dataset_path,
            input_dim_grid_nodes=cfg.num_channels,
            input_dim_mesh_nodes=3,
            input_dim_edges=4,
            output_dim_grid_nodes=cfg.num_channels,
            processor_layers=cfg.processor_layers,
            hidden_dim=cfg.hidden_dim,
            do_concat_trick=cfg.concat_trick,
            use_cugraphops_encoder=cfg.cugraphops_encoder,
            use_cugraphops_processor=cfg.cugraphops_processor,
            use_cugraphops_decoder=cfg.cugraphops_decoder,
            recompute_activation=cfg.recompute_activation,
        )

        # set gradient checkpointing
        if cfg.force_single_checkpoint:
            self.model.set_checkpoint_model(True)
        if cfg.checkpoint_encoder:
            self.model.set_checkpoint_encoder(True)
        if cfg.checkpoint_processor:
            self.model.set_checkpoint_processor(cfg.segments)
        if cfg.checkpoint_decoder:
            self.model.set_checkpoint_decoder(True)

        # JIT compile the model, and specify the device and dtype
        if cfg.jit:
            torch.jit.script(self.model).to(dtype=self.dtype).to(device=dist.device)
            rank_zero_logger.success("JIT compiled the model")
        else:
            self.model = self.model.to(dtype=self.dtype).to(device=dist.device)
        if cfg.watch_model and not cfg.jit and dist.rank == 0:
            wandb.watch(self.model)

        # distributed data parallel for multi-node training
        if dist.world_size > 1:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[dist.local_rank],
                output_device=dist.device,
                broadcast_buffers=dist.broadcast_buffers,
                find_unused_parameters=dist.find_unused_parameters,
                gradient_as_bucket_view=True,
                static_graph=True,
            )
        rank_zero_logger.info(
            f"Model parameter count is {count_trainable_params(self.model)}"
        )

        # instantiate the training datapipe
        DataPipe = (
            SyntheticWeatherDataLoader if cfg.synthetic_dataset else ERA5HDF5Datapipe
        )
        self.datapipe = DataPipe(
            data_dir=to_absolute_path(os.path.join(cfg.dataset_path, "train")),
            stats_dir=to_absolute_path(os.path.join(cfg.dataset_path, "stats")),
            channels=[i for i in range(cfg.num_channels)],
            num_samples_per_year=cfg.num_samples_per_year_train,
            num_steps=1,
            batch_size=1,
            num_workers=cfg.num_workers,
            device=dist.device,
            process_rank=dist.rank,
            world_size=dist.world_size,
        )
        rank_zero_logger.success(
            f"Loaded training datapipe of size {len(self.datapipe)}"
        )

        # instantiate the validation
        if dist.rank == 0 and not cfg.synthetic_dataset:
            self.validation = Validation(cfg, self.model, self.dtype, self.dist)
        else:
            self.validation = None

        # enable train mode
        self.model.train()

        # get area
        if hasattr(self.model, "module"):
            self.area = grid_cell_area(
                self.model.module.lat_lon_grid[:, :, 0], unit="deg"
            )
        else:
            self.area = grid_cell_area(self.model.lat_lon_grid[:, :, 0], unit="deg")
        self.area = self.area.to(dtype=self.dtype).to(device=dist.device)

        # instantiate loss, optimizer, and scheduler
        self.criterion = CellAreaWeightedLossFunction(self.area)
        try:
            self.optimizer = apex.optimizers.FusedAdam(
                self.model.parameters(), lr=cfg.lr, betas=(0.9, 0.95), weight_decay=0.1
            )
            rank_zero_logger.info("Using FusedAdam optimizer")
        except:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        scheduler1 = LinearLR(
            self.optimizer,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=cfg.num_iters_step1,
        )
        scheduler2 = CosineAnnealingLR(
            self.optimizer, T_max=cfg.num_iters_step2, eta_min=0.0
        )
        scheduler3 = LambdaLR(
            self.optimizer, lr_lambda=lambda epoch: (cfg.lr_step3 / cfg.lr)
        )
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[scheduler1, scheduler2, scheduler3],
            milestones=[cfg.num_iters_step1, cfg.num_iters_step1 + cfg.num_iters_step2],
        )
        self.scaler = GradScaler(enabled=self.enable_scaler)

        # load checkpoint
        if dist.world_size > 1:
            torch.distributed.barrier()
        self.iter_init = load_checkpoint(
            to_absolute_path(cfg.ckpt_path),
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=dist.device,
        )


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:

    # Optionally import apex
    if cfg.use_apex:
        try:
            import apex
        except:
            raise ImportError("Apex is not installed.")

    if cfg.cugraphops_encoder or cfg.cugraphops_processor or cfg.cugraphops_decoder:
        try:
            import pylibcugraphops
        except:
            raise ImportError(
                "pylibcugraphops is not installed. Refer the Dockerfile for instructions"
                + "on how to install this package."
            )

    # initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    # initialize loggers
    initialize_wandb(
        project="Modulus-Launch",
        entity="Modulus",
        name="GraphCast-Training",
        group="GraphCast-DDP-Group",
    )  # Wandb logger
    logger = PythonLogger("main")  # General python logger
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)  # Rank 0 logger
    rank_zero_logger.file_logging()

    # specify the datapipe
    if cfg.synthetic_dataset:
        DataPipe = SyntheticWeatherDataLoader
        cfg.static_dataset_path = None
        rank_zero_logger.warning("Using Dummy dataset. Ignoring static dataset.")
    else:
        DataPipe = ERA5HDF5Datapipe

    # initialize trainer
    trainer = GraphCastTrainer(cfg, dist, rank_zero_logger)
    start = time.time()
    rank_zero_logger.info("Training started...")
    loss_agg, iter, tagged_iter, num_rollout_steps = 0, trainer.iter_init, 1, 1
    terminate_training, finetune, update_dataloader = False, False, False

    with torch.autograd.profiler.emit_nvtx() if cfg.profile else nullcontext():
        # training loop
        while True:
            assert (
                iter < cfg.num_iters_step1 + cfg.num_iters_step2 + cfg.num_iters_step3
            ), "Training is already finished!"
            for i, data in enumerate(trainer.datapipe):
                # profiling
                if cfg.profile and iter == cfg.profile_range[0]:
                    rank_zero_logger.info("Starting profile", "green")
                    profiler.start()
                if cfg.profile and iter == cfg.profile_range[1]:
                    rank_zero_logger.info("Ending profile", "green")
                    profiler.stop()
                torch.cuda.nvtx.range_push("Training iteration")

                if iter >= cfg.num_iters_step1 + cfg.num_iters_step2 and not finetune:
                    finetune = True
                    if cfg.force_single_checkpoint_finetune:
                        if hasattr(trainer.model, "module"):
                            trainer.model.module.set_checkpoint_model(True)
                        else:
                            trainer.model.set_checkpoint_model(True)
                    if cfg.checkpoint_encoder_finetune:
                        if hasattr(trainer.model, "module"):
                            trainer.model.module.set_checkpoint_encoder(True)
                        else:
                            trainer.model.set_checkpoint_encoder(True)
                    if cfg.checkpoint_processor_finetune:
                        if hasattr(trainer.model, "module"):
                            trainer.model.module.set_checkpoint_processor(cfg.segments)
                        else:
                            trainer.model.set_checkpoint_encoder(True)
                    if cfg.checkpoint_decoder_finetune:
                        if hasattr(trainer.model, "module"):
                            trainer.model.module.set_checkpoint_decoder(True)
                        else:
                            trainer.model.set_checkpoint_encoder(True)
                if (
                    finetune
                    and (iter - (cfg.num_iters_step1 + cfg.num_iters_step2))
                    % cfg.step_change_freq
                    == 0
                    and iter != tagged_iter
                ):
                    update_dataloader = True
                    tagged_iter = iter

                # update the dataloader for finetuning
                if update_dataloader:
                    num_rollout_steps = (
                        iter - (cfg.num_iters_step1 + cfg.num_iters_step2)
                    ) // cfg.step_change_freq + 2
                    trainer.datapipe = DataPipe(
                        data_dir=os.path.join(cfg.dataset_path, "train"),
                        stats_dir=os.path.join(cfg.dataset_path, "stats"),
                        channels=[i for i in range(cfg.num_channels)],
                        num_steps=num_rollout_steps,
                        batch_size=1,
                        num_workers=cfg.num_workers,
                        device=dist.device,
                        process_rank=dist.rank,
                        world_size=dist.world_size,
                    )
                    update_dataloader = False
                    rank_zero_logger.info(
                        f"Switching to {num_rollout_steps}-step rollout!"
                    )
                    break

                # prepare the data
                # TODO modify for history > 0
                data_x = data[0]["invar"]
                data_y = data[0]["outvar"]
                # move to device & dtype
                data_x = data_x.to(dtype=trainer.dtype)
                grid_nfeat = data_x
                y = data_y.to(dtype=trainer.dtype).to(device=dist.device)

                # training step
                loss = trainer.train(grid_nfeat, y)
                if dist.rank == 0:
                    loss_agg += loss.detach().cpu()

                # validation
                if trainer.validation and iter % cfg.val_freq == 0:
                    # free up GPU memory
                    del data_x, y
                    torch.cuda.empty_cache()
                    error = trainer.validation.step(
                        channels=list(np.arange(cfg.num_channels_val)), iter=iter
                    )
                    logger.log(f"iteration {iter}, Validation MSE: {error:.04f}")

                # distributed barrier
                if dist.world_size > 1:
                    torch.distributed.barrier()

                # print logs and save checkpoint
                if dist.rank == 0 and iter % cfg.save_freq == 0:
                    save_checkpoint(
                        to_absolute_path(cfg.ckpt_path),
                        models=trainer.model,
                        optimizer=trainer.optimizer,
                        scheduler=trainer.scheduler,
                        scaler=trainer.scaler,
                        epoch=iter,
                    )
                    logger.info(f"Saved model on rank {dist.rank}")
                    logger.log(
                        f"iteration: {iter}, loss: {loss_agg/cfg.save_freq:10.3e}, \
                            time per iter: {(time.time()-start)/cfg.save_freq:10.3e}"
                    )
                    wandb.log(
                        {
                            "loss": loss_agg / cfg.save_freq,
                            "learning_rate": trainer.scheduler.get_last_lr()[0],
                        },
                        step=iter,
                    )
                    loss_agg = 0
                    start = time.time()
                iter += 1

                torch.cuda.nvtx.range_pop()

                # wrap up & terminate if training is finished
                if (
                    iter
                    >= cfg.num_iters_step1 + cfg.num_iters_step2 + cfg.num_iters_step3
                ):
                    if dist.rank == 0:
                        del data_x, y
                        torch.cuda.empty_cache()
                        error = trainer.validation.step(
                            channels=list(np.arange(cfg.num_channels_val)), iter=iter
                        )
                        logger.log(f"iteration {iter}, Validation MSE: {error:.04f}")
                        save_checkpoint(
                            to_absolute_path(cfg.ckpt_path),
                            trainer.model,
                            trainer.optimizer,
                            trainer.scheduler,
                            trainer.scaler,
                            iter,
                        )
                        logger.info(f"Saved model on rank {dist.rank}")
                        logger.log(
                            f"iteration: {iter}, loss: {loss_agg/cfg.save_freq:10.3e}, \
                                time per iter: {(time.time()-start)/cfg.save_freq:10.3e}"
                        )
                    terminate_training = True
                    break
            if terminate_training:
                rank_zero_logger.info("Finished training!")
                break


if __name__ == "__main__":
    main()
