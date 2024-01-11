import torch
import numpy as np
from dgl.dataloading import GraphDataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel
import time, os
import wandb as wb

from modulus.models.mesh_reduced.temporal_model import Sequence_Model
from modulus.models.mesh_reduced.mesh_reduced import Mesh_Reduced
from modulus.datapipes.gnn.vortex_shedding_re300_1000_dataset import VortexSheddingRe300To1000Dataset, LatentDataset
from modulus.distributed.manager import DistributedManager
from modulus.launch.logging import (
    PythonLogger,
    initialize_wandb,
    RankZeroLoggingWrapper,
)
from modulus.launch.utils import load_checkpoint, save_checkpoint
from constants import Constants

C = Constants()
class Sequence_Trainer:
    def __init__(self, wb, dist, 
                 produce_latents = True,  
                 Encoder = None, 
		         position_mesh = None, 
		         position_pivotal = None, 
                 rank_zero_logger = None):
        self.dist = dist
        dataset_train = LatentDataset(
            split="train",
            produce_latents = produce_latents,
		    Encoder = Encoder, 
		    position_mesh = position_mesh, 
		    position_pivotal = position_pivotal,
            dist = dist
        )

        self.dataloader = GraphDataLoader(
            dataset_train,
            batch_size=C.sequence_batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            use_ddp=dist.world_size > 1,
        )
        self.model = Sequence_Model(C.sequence_dim, C.sequence_content_dim, dist)

        if C.jit:
            self.model = torch.jit.script(self.model).to(dist.device)
        else:
            self.model = self.model.to(dist.device)
        if C.watch_model and not C.jit and dist.rank == 0:
            wb.watch(self.model)
        # enable train mode
        self.model.train()

        # instantiate loss, optimizer, and scheduler
        self.criterion = torch.nn.MSELoss()
        # instantiate loss, optimizer, and scheduler
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=C.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda epoch: C.lr_decay_rate**epoch
        )
        self.scaler = GradScaler()

        # load checkpoint
        if dist.world_size > 1:
            torch.distributed.barrier()
        self.epoch_init = load_checkpoint(
            os.path.join(C.ckpt_sequence_path, C.ckpt_sequence_name),
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=dist.device,
        )

    def forward(self, z, context = None):
        with autocast(enabled=C.amp):
            prediction  = self.model(z, context)
            loss = self.criterion(z[:,1:], prediction[:,:-1])
            relative_error = torch.sqrt(loss/self.criterion(z[:,1:], z[:,1:]*0.0)).detach()
            return loss, relative_error
        
    def train(self, z, context):
        z = z.to(self.dist.device)
        context = context.to(self.dist.device)
        self.optimizer.zero_grad()
        loss, relative_error = self.forward(z, context)
        self.backward(loss)
        self.scheduler.step()
        return loss, relative_error

    def backward(self, loss):
        # backward pass
        if C.amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()



if __name__ == "__main__":
    # initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    # save constants to JSON file
    if dist.rank == 0:
        os.makedirs(C.ckpt_sequence_path, exist_ok=True)
        with open(
            os.path.join(C.ckpt_sequence_path, C.ckpt_sequence_name.replace(".pt", ".json")), "w"
        ) as json_file:
            json_file.write(C.json(indent=4))

    # initialize loggers
    initialize_wandb(
        project="Modulus-Launch",
        entity="Modulus",
        name="Vortex_Shedding-Training",
        group="Vortex_Shedding-DDP-Group",
        mode=C.wandb_mode,
    )  # Wandb logger
    logger = PythonLogger("main")  # General python logger
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)  # Rank 0 logger
    logger.file_logging()

    position_mesh = torch.from_numpy(np.loadtxt(C.mesh_dir)).to(dist.device)
    position_pivotal = torch.from_numpy(np.loadtxt(C.pivotal_dir)).to(dist.device)
    #Load Graph Encoder
    Encoder = Mesh_Reduced(C.num_input_features, C.num_edge_features, C.num_output_features)
    Encoder = Encoder.to(dist.device)
    _ = load_checkpoint(
            os.path.join(C.ckpt_path, C.ckpt_name),
            models=Encoder,
            scaler =GradScaler(),
            device=dist.device
        )




    trainer = Sequence_Trainer(wb, dist, produce_latents=True, Encoder = Encoder, 
		         position_mesh = position_mesh, 
		         position_pivotal = position_pivotal,
                 rank_zero_logger = rank_zero_logger)
    start = time.time()
    rank_zero_logger.info("Training started...")
    
    
    for epoch in range(trainer.epoch_init, C.epochs):
        n_batch = 0.0
        loss_total = 0.0
        for lc in trainer.dataloader:
            loss,relative_error = trainer.train(lc[0], lc[1])
            loss_total = loss_total + loss
            n_batch = n_batch + 1
        avg_loss = loss_total/n_batch
        rank_zero_logger.info(
            f"epoch: {epoch}, loss: {avg_loss:10.3e}, relative_error: {relative_error:10.3e},time per epoch: {(time.time()-start):10.3e}"
        )
        wb.log({"loss": loss.detach().cpu()})

        # save checkpoint
        if dist.world_size > 1:
            torch.distributed.barrier()
        if dist.rank == 0 and epoch%5000 == 0:
            save_checkpoint(
                os.path.join(C.ckpt_sequence_path, C.ckpt_sequence_name),
                models=trainer.model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                scaler=trainer.scaler,
                epoch=epoch,
            )
            logger.info(f"Saved model on rank {dist.rank}")
        start = time.time()
    rank_zero_logger.info("Training completed!")