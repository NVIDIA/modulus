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

import time

import hydra
from hydra.utils import to_absolute_path
import torch
import wandb
import time

from dgl.dataloading import GraphDataLoader

from omegaconf import DictConfig
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel

from modulus.datapipes.gnn.lagrangian_dataset import LagrangianDataset
from modulus.distributed.manager import DistributedManager
from modulus.launch.logging import (
    PythonLogger,
    RankZeroLoggingWrapper,
    initialize_wandb,
)
from modulus.launch.utils import load_checkpoint, save_checkpoint
from modulus.models.meshgraphnet import MeshGraphNet


class MGNTrainer:
    def __init__(self, cfg: DictConfig, rank_zero_logger: RankZeroLoggingWrapper):
        assert DistributedManager.is_initialized()
        self.dist = DistributedManager()
        self.amp = cfg.amp
        self.radius = cfg.radius
        self.dt = cfg.dt
        self.dim = cfg.dim
        self.gravity = torch.zeros(self.dim, device=self.dist.device)
        self.gravity[-1] = -9.8

        # MGN with recompute_activation currently supports only SiLU activation function.
        mlp_act = cfg.activation
        if cfg.recompute_activation and cfg.activation.lower() != "silu":
            raise ValueError(
                f"recompute_activation only supports SiLU activation function, "
                f"but got {cfg.activation}. Please either set activation='silu' "
                f"or disable recompute_activation."
            )

        # instantiate dataset
        self.dataset = LagrangianDataset(
            name="Water",
            data_dir=to_absolute_path(cfg.data_dir),
            split="train",
            num_samples=cfg.num_training_samples,
            num_steps=cfg.num_training_time_steps,
            radius=cfg.radius,
            dt=cfg.dt,
        )
        self.dataset.set_normalizer_device(device=self.dist.device)
        self.time_integrator = self.dataset.time_integrator

        # instantiate dataloader
        self.dataloader = GraphDataLoader(
            self.dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            use_ddp=self.dist.world_size > 1,
            num_workers=cfg.num_dataloader_workers,
        )

        # instantiate the model
        self.model = MeshGraphNet(
            cfg.num_input_features,
            cfg.num_edge_features,
            cfg.num_output_features,
            cfg.processor_size,
            mlp_activation_fn=mlp_act,
            do_concat_trick=cfg.do_concat_trick,
            num_processor_checkpoint_segments=cfg.num_processor_checkpoint_segments,
            recompute_activation=cfg.recompute_activation,
            # aggregation="mean",
        )
        if cfg.jit:
            if not self.model.meta.jit:
                raise ValueError("MeshGraphNet is not yet JIT-compatible.")
            self.model = torch.jit.script(self.model).to(self.dist.device)
        else:
            self.model = self.model.to(self.dist.device)
        if cfg.watch_model and not cfg.jit and self.dist.rank == 0:
            wandb.watch(self.model)

        # distributed data parallel for multi-node training
        if self.dist.world_size > 1:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.dist.local_rank],
                output_device=self.dist.device,
                broadcast_buffers=self.dist.broadcast_buffers,
                find_unused_parameters=self.dist.find_unused_parameters,
            )

        # enable train mode
        self.model.train()

        # instantiate loss, optimizer, and scheduler
        # self.criterion = self.l2loss
        self.criterion = torch.nn.MSELoss()

        self.optimizer = None
        try:
            if cfg.use_apex:
                from apex.optimizers import FusedAdam

                self.optimizer = FusedAdam(self.model.parameters(), lr=cfg.lr)
        except ImportError:
            rank_zero_logger.warning(
                "NVIDIA Apex (https://github.com/nvidia/apex) is not installed, "
                "FusedAdam optimizer will not be used."
            )
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=cfg.lr, weight_decay=1e-5
            )
        rank_zero_logger.info(f"Using {self.optimizer.__class__.__name__} optimizer")

        num_iteration = (
            cfg.epochs
            * cfg.num_training_samples
            * cfg.num_training_time_steps
            // cfg.batch_size
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_iteration, eta_min=cfg.lr_min
        )
        self.scaler = GradScaler()

        # load checkpoint
        if self.dist.world_size > 1:
            torch.distributed.barrier()
        self.epoch_init = load_checkpoint(
            to_absolute_path(cfg.ckpt_path),
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=self.dist.device,
        )

    def train(self, graph):
        graph = graph.to(self.dist.device)
        self.optimizer.zero_grad()
        loss_pos, loss_vel, loss_acc = self.forward(graph)
        loss = loss_acc + loss_vel + loss_pos
        self.backward(loss_acc)
        self.scheduler.step()
        return loss, [loss_pos.item(), loss_vel.item(), loss_acc.item()]

    def forward(self, graph):
        # forward pass
        with autocast(enabled=self.amp):
            # predict the acceleration
            pred_acc = self.model(graph.ndata["x"], graph.edata["x"], graph)
            loss_acc = self.criterion(pred_acc, graph.ndata["y"][..., 2 * self.dim :])

            # use the integrator to get the position
            pred_pos, pred_vel = self.time_integrator(
                position=graph.ndata["x"][..., : self.dim],
                velocity=graph.ndata["x"][..., 1 * self.dim : 2 * self.dim],
                acceleration=pred_acc,
                dt=self.dt,
            )
            loss_pos = self.criterion(pred_pos, graph.ndata["y"][..., : self.dim])
            pred_vel = self.dataset.normalize_velocity(pred_vel)
            loss_vel = self.criterion(
                pred_vel, graph.ndata["y"][..., self.dim : 2 * self.dim]
            )

            return loss_pos, loss_vel, loss_acc

    def backward(self, loss):
        # backward pass
        if self.amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

    def l2loss(self, input, target, p=2, eps=1e-5):
        input = input.flatten(start_dim=1)
        target = target.flatten(start_dim=1)
        l2loss = torch.norm(input - target, dim=1, p=p) / (
            torch.norm(target, dim=1, p=p) + eps
        )
        l2loss = torch.mean(l2loss)
        return l2loss


@hydra.main(version_base="1.3", config_path="conf", config_name="config_2d")
def main(cfg: DictConfig) -> None:
    # initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    # Initialize loggers.
    wandb.login(key=cfg.wandb_key)
    initialize_wandb(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=cfg.wandb_name,
        mode=cfg.wandb_mode,
    )  # Wandb logger
    logger = PythonLogger("main")  # General python logger
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)  # Rank 0 logger
    rank_zero_logger.file_logging()

    trainer = MGNTrainer(cfg, rank_zero_logger)
    start = time.time()
    rank_zero_logger.info("Training started...")
    for epoch in range(trainer.epoch_init, cfg.epochs):
        loss_list = []
        loss_pos_list = []
        loss_vel_list = []
        loss_acc_list = []
        for graph in trainer.dataloader:
            loss, losses = trainer.train(graph)
            loss_list.append(loss.item())
            loss_pos_list.append(losses[0])
            loss_vel_list.append(losses[1])
            loss_acc_list.append(losses[2])
        mean_loss = sum(loss_list) / len(loss_list)
        mean_loss_pos = sum(loss_pos_list) / len(loss_pos_list)
        mean_loss_vel = sum(loss_vel_list) / len(loss_vel_list)
        mean_loss_acc = sum(loss_acc_list) / len(loss_acc_list)
        rank_zero_logger.info(
            f"epoch: {epoch}, loss: {mean_loss:10.3e}, "
            f"position loss: {mean_loss_pos:10.3e}, "
            f"velocity loss: {mean_loss_vel:10.3e}, "
            f"acceleration loss: {mean_loss_acc:10.3e}, "
            f"time per epoch: {(time.time()-start):10.3e}"
        )
        losses = {
            "loss": mean_loss,
            "loss_pos": mean_loss_pos,
            "loss_vel": mean_loss_vel,
            "loss_acc": mean_loss_acc,
        }
        wandb.log(losses)

        # save checkpoint
        if dist.world_size > 1:
            torch.distributed.barrier()
        if dist.rank == 0:
            save_checkpoint(
                to_absolute_path(cfg.ckpt_path),
                models=trainer.model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                scaler=trainer.scaler,
                epoch=epoch,
            )
            logger.info(f"Saved model on rank {dist.rank}")
        start = time.time()
    rank_zero_logger.info("Training completed!")


if __name__ == "__main__":
    main()
