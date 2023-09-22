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

import os
import time

import torch
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel
import wandb as wb

from modulus.models.meshgraphnet import MeshGraphNet
from modulus.datapipes.gnn.ahmed_body_dataset import AhmedBodyDataset
from modulus.distributed.manager import DistributedManager

from modulus.launch.logging import (
    PythonLogger,
    initialize_wandb,
    RankZeroLoggingWrapper,
)
from modulus.launch.utils import load_checkpoint, save_checkpoint
from constants import Constants

try:
    from dgl.dataloading import GraphDataLoader
except:
    raise ImportError(
        "Ahmed Body example requires the DGL library. Install the "
        + "desired CUDA version at: \n https://www.dgl.ai/pages/start.html"
    )

try:
    import apex
except ImportError:
    pass

# Instantiate constants
C = Constants()


class MGNTrainer:
    def __init__(self, wb, dist, rank_zero_logger):
        self.dist = dist
        self.wb = wb
        self.rank_zero_logger = rank_zero_logger

        # instantiate dataset
        rank_zero_logger.info("Loading the training dataset...")
        self.dataset = AhmedBodyDataset(
            name="ahmed_body_train",
            data_dir=C.data_dir,
            split="train",
            num_samples=C.num_training_samples,
        )

        # instantiate validation dataset
        rank_zero_logger.info("Loading the validation dataset...")
        self.validation_dataset = AhmedBodyDataset(
            name="ahmed_body_validation",
            data_dir=C.data_dir,
            split="validation",
            num_samples=C.num_validation_samples,
        )

        # instantiate dataloader
        self.dataloader = GraphDataLoader(
            self.dataset,
            batch_size=C.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            use_ddp=dist.world_size > 1,
        )

        # instantiate validation dataloader
        self.validation_dataloader = GraphDataLoader(
            self.validation_dataset,
            batch_size=C.batch_size,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
            use_ddp=False,
        )

        # instantiate the model
        self.model = MeshGraphNet(
            C.input_dim_nodes,
            C.input_dim_edges,
            C.output_dim,
            aggregation=C.aggregation,
            hidden_dim_node_encoder=C.hidden_dim_node_encoder,
            hidden_dim_edge_encoder=C.hidden_dim_edge_encoder,
            hidden_dim_node_decoder=C.hidden_dim_node_decoder,
        )
        if C.jit:
            self.model = torch.jit.script(self.model).to(dist.device)
        else:
            self.model = self.model.to(dist.device)

        # distributed data parallel for multi-node training
        if dist.world_size > 1:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[dist.local_rank],
                output_device=dist.device,
                broadcast_buffers=dist.broadcast_buffers,
                find_unused_parameters=dist.find_unused_parameters,
            )

        # enable train mode
        self.model.train()

        # instantiate optimizer, and scheduler
        try:
            self.optimizer = apex.optimizers.FusedAdam(self.model.parameters(), lr=C.lr)
            rank_zero_logger.info("Using FusedAdam optimizer")
        except:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=C.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda epoch: C.lr_decay_rate**epoch
        )
        self.scaler = GradScaler()

        # load checkpoint
        if dist.world_size > 1:
            torch.distributed.barrier()
        self.epoch_init = load_checkpoint(
            os.path.join(C.ckpt_path, C.ckpt_name),
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=dist.device,
        )

    def train(self, graph):
        self.optimizer.zero_grad()
        loss = self.forward(graph)
        self.backward(loss)
        self.scheduler.step()
        return loss

    def forward(self, graph):
        # forward pass
        with autocast(enabled=C.amp):
            pred = self.model(graph.ndata["x"], graph.edata["x"], graph)
            diff_norm = torch.norm(
                torch.flatten(pred) - torch.flatten(graph.ndata["y"]), p=2
            )
            y_norm = torch.norm(torch.flatten(graph.ndata["y"]), p=2)
            loss = diff_norm / y_norm
            return loss

    def backward(self, loss):
        # backward pass
        if C.amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()
        lr = self.get_lr()
        self.wb.log({"lr": lr})

    def get_lr(self):
        # get the learning rate
        for param_group in self.optimizer.param_groups:
            return param_group["lr"]

    @torch.no_grad()
    def validation(self):
        error = 0
        for graph in self.validation_dataloader:
            graph = graph.to(self.dist.device)
            pred = self.model(graph.ndata["x"], graph.edata["x"], graph)
            pred, gt = self.dataset.denormalize(
                pred, graph.ndata["y"], self.dist.device
            )
            error += (
                torch.mean(torch.norm(pred - gt, p=2) / torch.norm(gt, p=2))
                .cpu()
                .numpy()
            )
        error = error / len(self.validation_dataloader) * 100
        self.wb.log({"val_error (%)": error})
        self.rank_zero_logger.info(f"Denormalized validation error (%): {error}")


if __name__ == "__main__":
    # initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    # save constants to JSON file
    if dist.rank == 0:
        os.makedirs(C.ckpt_path, exist_ok=True)
        with open(
            os.path.join(C.ckpt_path, C.ckpt_name.replace(".pt", ".json")), "w"
        ) as json_file:
            json_file.write(C.json(indent=4))

    # initialize loggers
    initialize_wandb(
        project="Aero",
        entity="Modulus",
        name="Aero-Training",
        group="Aero-DDP-Group",
        mode=C.wandb_mode,
    )  # Wandb logger

    logger = PythonLogger("main")  # General python logger
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)  # Rank 0 logger
    logger.file_logging()

    trainer = MGNTrainer(wb, dist, rank_zero_logger)
    start = time.time()
    rank_zero_logger.info("Training started...")

    for epoch in range(trainer.epoch_init, C.epochs):
        loss_agg = 0
        for graph in trainer.dataloader:
            graph = graph.to(dist.device)
            loss = trainer.train(graph)
            loss_agg += loss.detach().cpu().numpy()
        loss_agg /= len(trainer.dataloader)
        rank_zero_logger.info(
            f"epoch: {epoch}, loss: {loss_agg:10.3e}, lr: {trainer.get_lr()}, time per epoch: {(time.time()-start):10.3e}"
        )
        wb.log({"loss": loss_agg})

        # validation
        if dist.rank == 0:
            trainer.validation()

        # save checkpoint
        if dist.world_size > 1:
            torch.distributed.barrier()
        if dist.rank == 0:
            save_checkpoint(
                os.path.join(C.ckpt_path, C.ckpt_name),
                models=trainer.model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                scaler=trainer.scaler,
                epoch=epoch,
            )
            logger.info(f"Saved model on rank {dist.rank}")
        start = time.time()
    rank_zero_logger.info("Training completed!")
