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

from dgl.dataloading import GraphDataLoader

from omegaconf import DictConfig

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel

import wandb

# from modulus.datapipes.gnn.ahmed_body_dataset import AhmedBodyDataset
from ahmed_body_dataset import AhmedBodyDataset
from modulus.distributed.manager import DistributedManager
from modulus.launch.logging import (
    PythonLogger,
    RankZeroLoggingWrapper,
    initialize_wandb,
)
from modulus.launch.utils import load_checkpoint, save_checkpoint
# from modulus.models.meshgraphnet import MeshGraphNet
from models import BiStrideMeshGraph

from multi_mesh_save_and_load import cal_multi_mesh_all,load_multi_mesh_batch


class MGNTrainer:
    def __init__(self, cfg: DictConfig, rank_zero_logger: RankZeroLoggingWrapper):
        assert DistributedManager.is_initialized()
        self.dist = DistributedManager()
        self.rank_zero_logger = rank_zero_logger

        self.amp = cfg.amp
        # MGN with recompute_activation currently supports only SiLU activation function.
        mlp_act = "relu"
        if cfg.recompute_activation:
            rank_zero_logger.info(
                "Setting MLP activation to SiLU required by recompute_activation."
            )
            mlp_act = "silu"

        ### Training data
        # instantiate dataset
        rank_zero_logger.info("Loading the training dataset...")
        self.dataset = AhmedBodyDataset(
            name="ahmed_body_train",
            data_dir=to_absolute_path(cfg.data_dir),
            split="train",
            num_samples=cfg.num_training_samples,
            num_workers=cfg.num_dataset_workers,
        ) # a list of DGL.graph


        # Generate multi mesh graphs
        cal_multi_mesh_all(self.dataset,to_absolute_path(cfg.multi_mesh_data_dir),"train",cfg.mesh_layer)

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

        ### Validation data
        rank_zero_logger.info("Loading the validation dataset...")
        self.validation_dataset = AhmedBodyDataset(
            name="ahmed_body_validation",
            data_dir=to_absolute_path(cfg.data_dir),
            split="val",
            num_samples=cfg.num_validation_samples,
            num_workers=cfg.num_dataset_workers,
        )

        # Generate multi mesh graphs
        cal_multi_mesh_all(self.validation_dataset,to_absolute_path(cfg.multi_mesh_data_dir),"val",cfg.mesh_layer)

        # instantiate dataloader
        self.validation_dataloader = GraphDataLoader(
            self.validation_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
            use_ddp=False,
            num_workers=cfg.num_dataloader_workers,
        )

        ### Testing data
        rank_zero_logger.info("Loading the test dataset...")
        self.test_dataset = AhmedBodyDataset(
            name="ahmed_body_testing",
            data_dir=to_absolute_path(cfg.data_dir),
            split="test",
            num_samples=cfg.num_test_samples,
            num_workers=cfg.num_dataset_workers,
        )

        # Generate multi mesh graphs
        cal_multi_mesh_all(self.test_dataset,to_absolute_path(cfg.multi_mesh_data_dir),"test",cfg.mesh_layer)

        # instantiate dataloader
        self.test_dataloader = GraphDataLoader(
            self.test_dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
            use_ddp=False,
            num_workers=cfg.num_dataloader_workers,
        )

        # instantiate the model
        self.model = BiStrideMeshGraph(
            cfg.input_dim_nodes,
            cfg.input_dim_edges,
            cfg.output_dim,
            aggregation=cfg.aggregation,
            num_mesh_level = cfg.mesh_layer,
            hidden_dim_node_encoder=cfg.hidden_dim_node_encoder,
            hidden_dim_edge_encoder=cfg.hidden_dim_edge_encoder,
            hidden_dim_node_decoder=cfg.hidden_dim_node_decoder,
            # mlp_activation_fn=mlp_act,
            do_concat_trick=cfg.do_concat_trick,
            num_processor_checkpoint_segments=cfg.num_processor_checkpoint_segments,
            recompute_activation=cfg.recompute_activation,
        )


        if cfg.jit:
            if not self.model.meta.jit:
                raise ValueError("MeshGraphNet is not yet JIT-compatible.")
            self.model = torch.jit.script(self.model).to(self.dist.device)
        else:
            self.model = self.model.to(self.dist.device)

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

        # instantiate optimizer, and scheduler
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
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        rank_zero_logger.info(f"Using {self.optimizer.__class__.__name__} optimizer")

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda epoch: cfg.lr_decay_rate**epoch
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

    def train(self, graph,mesh_dicts):
        self.optimizer.zero_grad()
        loss = self.forward(graph,mesh_dicts)
        self.backward(loss)
        self.scheduler.step()
        return loss

    def forward(self, graph,mesh_dicts):
        # forward pass
        with autocast(enabled=self.amp):
            # pred = self.model(graph.ndata["x"], graph.edata["x"], graph)
            pred = self.model(mesh_dicts,graph)
            diff_norm = torch.norm(
                torch.flatten(pred) - torch.flatten(graph.ndata["y"]), p=2
            )
            y_norm = torch.norm(torch.flatten(graph.ndata["y"]), p=2)
            loss = diff_norm / y_norm
            return loss

    def backward(self, loss):
        # backward pass
        if self.amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        lr = self.get_lr()
        wandb.log({"lr": lr})

    def get_lr(self):
        # get the learning rate
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]

    @torch.no_grad()
    def validation(self,cfg):
        error = 0
        for (graph,graph_id) in self.validation_dataloader:
            graph = graph.to(self.dist.device)
            mesh_dicts = load_multi_mesh_batch(graph_id,to_absolute_path(cfg.multi_mesh_data_dir),"val",cfg.mesh_layer)
            pred = self.model(mesh_dicts, graph)
            pred, gt = self.dataset.denormalize(
                pred, graph.ndata["y"], self.dist.device
            )
            error += (
                torch.mean(torch.norm(pred - gt, p=2) / torch.norm(gt, p=2))
                .cpu()
                .numpy()
            )
        error = error / len(self.validation_dataloader) * 100
        wandb.log({"val_error (%)": error})
        self.rank_zero_logger.info(f"Denormalized validation error (%): {error}")

    @torch.no_grad()
    def testing(self,cfg):
        error = 0
        for (graph,graph_id) in self.test_dataloader:
            graph = graph.to(self.dist.device)
            mesh_dicts = load_multi_mesh_batch(graph_id,to_absolute_path(cfg.multi_mesh_data_dir),"test",cfg.mesh_layer)
            pred = self.model(mesh_dicts, graph)
            pred, gt = self.dataset.denormalize(
                pred, graph.ndata["y"], self.dist.device
            )
            error += (
                torch.mean(torch.norm(pred - gt, p=2) / torch.norm(gt, p=2))
                .cpu()
                .numpy()
            )
        error = error / len(self.test_dataloader) * 100
        wandb.log({"test_error (%)": error})
        self.rank_zero_logger.info(f"Denormalized testing error (%): {error}")


@hydra.main(version_base="1.3", config_path="conf", config_name="config_BSMS_6")
def main(cfg: DictConfig) -> None:
    # initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    # augment ckpt path 
    cfg.ckpt_path = cfg.ckpt_path + "_layer_" + str(cfg.mesh_layer)

    # initialize loggers
    initialize_wandb(
        project="Aero",
        entity="Modulus",
        name="Aero-Training",
        group="Aero-DDP-Group",
        mode=cfg.wandb_mode,
    )  # Wandb logger

    logger = PythonLogger("main")  # General python logger
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)  # Rank 0 logger
    rank_zero_logger.file_logging()

    trainer = MGNTrainer(cfg, rank_zero_logger)
    start = time.time()
    rank_zero_logger.info("Training started...")

    for epoch in range(trainer.epoch_init, cfg.epochs):
        loss_agg = 0
        for (graphs,graph_ids) in trainer.dataloader:
            graphs = graphs.to(dist.device)
            # indexing its/their precomputed multi-level mesh graphs
            mesh_dicts = load_multi_mesh_batch(graph_ids,to_absolute_path(cfg.multi_mesh_data_dir),"train",cfg.mesh_layer)
            loss = trainer.train(graphs,mesh_dicts)
            loss_agg += loss.detach().cpu().numpy()
        loss_agg /= len(trainer.dataloader)
        rank_zero_logger.info(
            f"epoch: {epoch}, loss: {loss_agg:10.3e}, lr: {trainer.get_lr()}, "
            f"time per epoch: {(time.time()-start):10.3e}"
        )
        wandb.log({"loss": loss_agg})

        # validation
        if dist.rank == 0:
            trainer.validation(cfg)
            trainer.testing(cfg)

        # save checkpoint
        if dist.world_size > 1:
            torch.distributed.barrier()
        if dist.rank == 0 and (epoch + 1) % cfg.checkpoint_save_freq == 0:
            save_checkpoint(
                to_absolute_path(cfg.ckpt_path),
                models=trainer.model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                scaler=trainer.scaler,
                epoch=epoch,
            )
            rank_zero_logger.info(f"Saved model on rank {dist.rank}")
        start = time.time()
    rank_zero_logger.info("Training completed!")


if __name__ == "__main__":
    main()
