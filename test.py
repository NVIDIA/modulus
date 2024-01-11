import torch
import numpy as np
from dgl.dataloading import GraphDataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel
import time, os
import wandb as wb

from modulus.models.mesh_reduced.mesh_reduced import Mesh_Reduced
from modulus.datapipes.gnn.vortex_shedding_re300_1000_dataset import VortexSheddingRe300To1000Dataset
from modulus.distributed.manager import DistributedManager
from modulus.launch.logging import (
    PythonLogger,
    initialize_wandb,
    RankZeroLoggingWrapper,
)
from modulus.launch.utils import load_checkpoint, save_checkpoint
from constants import Constants
from tqdm import tqdm
from train import Mesh_ReducedTrainer

C = Constants()




if __name__ == "__main__":
    # initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    

    logger = PythonLogger("main")  # General python logger
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)  # Rank 0 logger
    logger.file_logging()


    trainer = Mesh_ReducedTrainer(wb, dist, rank_zero_logger)
    start = time.time()
    rank_zero_logger.info("Training started...")
    position_mesh = torch.from_numpy(np.loadtxt(C.mesh_dir)).to(dist.device)
    position_pivotal = torch.from_numpy(np.loadtxt(C.pivotal_dir)).to(dist.device)
    loss_total = 0
  
    for graph in tqdm(trainer.dataloader_test):
        loss = trainer.train(graph,position_mesh,position_pivotal)
        loss_total = loss_total + loss
    rank_zero_logger.info(
            f"loss: {loss:10.3e}, time per epoch: {(time.time()-start):10.3e}"
        )
        

      
    rank_zero_logger.info("Training completed!")