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

from functools import partial
import logging
import time
from typing import Mapping

import hydra
from hydra.utils import instantiate, to_absolute_path

from dgl.dataloading import GraphDataLoader

from omegaconf import DictConfig

import torch
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel

from modulus.distributed.manager import DistributedManager
from modulus.launch.utils import load_checkpoint, save_checkpoint

from loggers import CompositeLogger, ExperimentLogger, init_python_logging


logger = logging.getLogger("agnet")

# Experiment logger will be set later during initialization.
elogger: ExperimentLogger = None


class RRMSELoss(torch.nn.Module):
    """Relative RMSE loss."""

    def forward(self, pred: Tensor, target: Tensor):
        return (torch.norm(pred - target, p=2) / torch.norm(target, p=2)).mean()


class MGNTrainer:
    def __init__(self, cfg: DictConfig):
        assert DistributedManager.is_initialized()
        self.dist = DistributedManager()

        # instantiate training dataset
        logger.info("Loading the training dataset...")
        self.dataset = instantiate(cfg.data.train)
        logger.info(f"Using {len(self.dataset)} training samples.")

        # instantiate validation dataset
        logger.info("Loading the validation dataset...")
        self.validation_dataset = instantiate(cfg.data.val)
        logger.info(f"Using {len(self.validation_dataset)} validation samples.")

        logger.info("Creating the dataloaders...")
        # instantiate training dataloader
        self.dataloader = GraphDataLoader(
            self.dataset,
            **cfg.train.dataloader,
            use_ddp=self.dist.world_size > 1,
        )

        # instantiate validation dataloader
        self.validation_dataloader = GraphDataLoader(
            self.validation_dataset,
            **cfg.val.dataloader,
        )

        logger.info("Creating the model...")
        # instantiate the model
        self.model = instantiate(cfg.model)

        if cfg.compile.enabled:
            self.model = torch.compile(self.model, **cfg.compile.args).to(
                self.dist.device
            )
        else:
            self.model = self.model.to(self.dist.device)

        # distributed data parallel for multi-node training
        if self.dist.distributed:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.dist.local_rank],
                output_device=self.dist.device,
                broadcast_buffers=self.dist.broadcast_buffers,
                find_unused_parameters=self.dist.find_unused_parameters,
            )
        # Set the original model getter to simplify access.
        assert not hasattr(self.model, "model")
        type(self.model).model = (
            (lambda m: m.module) if self.dist.distributed else (lambda m: m)
        )

        # enable train mode
        self.model.train()

        # instantiate losses.
        self.loss = instantiate(cfg.loss)

        # instantiate optimizer, and scheduler
        self.optimizer = instantiate(cfg.optimizer, self.model.parameters())
        self.scheduler = instantiate(cfg.lr_scheduler, self.optimizer)

        self.scaler = instantiate(cfg.amp.scaler)
        self.autocast = partial(
            torch.cuda.amp.autocast,
            enabled=cfg.amp.enabled,
            dtype=hydra.utils.get_object(cfg.amp.autocast.dtype),
        )

        # load checkpoint
        self.epoch_init = load_checkpoint(
            to_absolute_path(cfg.resume_dir),
            models=self.model.model(),
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=self.dist.device,
        )
        if self.dist.world_size > 1:
            torch.distributed.barrier()

        self.visualizers = instantiate(cfg.visualizers)

    def train(self, batch: Mapping[str, Tensor]):
        self.optimizer.zero_grad()
        loss = self.forward(batch)
        self.backward(loss)
        self.scheduler.step()
        return loss

    def forward(self, batch):
        # forward pass
        graph = batch["graph"]
        with self.autocast():
            pred = self.model(graph.ndata["x"], graph.edata["x"], graph)
            # Graph data (e.g. p and WSS) loss.
            ndata_loss = self.loss.ndata(pred, graph.ndata["y"])
            loss = ndata_loss
            return loss

    def backward(self, loss):
        # backward pass.
        # If AMP is disabled, the scaler will fall back to the default behavior.
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

    @torch.no_grad()
    def validation(self, epoch: int):
        error = 0
        for batch in self.validation_dataloader:
            batch = {k: v.to(self.dist.device) for k, v in batch_as_dict(batch).items()}
            graph = batch["graph"]
            pred = self.model(graph.ndata["x"], graph.edata["x"], graph)
            pred, gt = self.dataset.denormalize(
                pred, graph.ndata["y"], self.dist.device
            )
            error += self.loss.ndata(pred, gt)

        # Visualize last batch.
        for vis in self.visualizers.values():
            vis(graph, pred, gt, epoch, elogger)

        return error / len(self.validation_dataloader)


def batch_as_dict(batch):
    return batch if isinstance(batch, Mapping) else {"graph": batch}


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    init_python_logging(cfg)

    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True

    # initialize loggers
    global elogger
    elogger = CompositeLogger(cfg)

    trainer = MGNTrainer(cfg)
    start = time.time()
    logger.info("Training started...")

    for epoch in range(trainer.epoch_init + 1, cfg.train.epochs + 1):
        loss_agg = 0
        for batch in trainer.dataloader:
            batch = {k: v.to(dist.device) for k, v in batch_as_dict(batch).items()}
            loss = trainer.train(batch)
            loss_agg += loss.detach().cpu().numpy()
        loss_agg /= len(trainer.dataloader)

        cur_lr = trainer.scheduler.get_last_lr()[0]
        logger.info(
            f"epoch: {epoch:5,}, loss: {loss_agg:.5f}, "
            f"lr: {cur_lr:.7f}, "
            f"time per epoch: {(time.time() - start):5.2f}"
        )
        elogger.log_scalar("loss", loss_agg, epoch)
        elogger.log_scalar("lr", cur_lr, epoch)

        # validation
        # TODO(akamenev): redundant restriction, val should run on all ranks.
        if dist.rank == 0:
            val_error_pct = trainer.validation(epoch) * 100
            elogger.log_scalar("val_error (%)", val_error_pct, epoch)
            logger.info(f"Denormalized validation error (%): {val_error_pct:4.2f}")

        # save checkpoint
        if dist.world_size > 1:
            torch.distributed.barrier()
        if dist.rank == 0 and epoch % cfg.train.checkpoint_save_freq == 0:
            save_checkpoint(
                cfg.output,
                models=trainer.model.model(),
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                scaler=trainer.scaler,
                epoch=epoch,
            )
            logger.info(f"Saved model on rank {dist.rank}")
        start = time.time()
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
