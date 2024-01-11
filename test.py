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
    rank_zero_logger.info("Testing started...")
    position_mesh = torch.from_numpy(np.loadtxt(C.mesh_dir)).to(dist.device)
    position_pivotal = torch.from_numpy(np.loadtxt(C.pivotal_dir)).to(dist.device)
    loss_total = 0
    relative_error_total = 0
  
    for graph in tqdm(trainer.dataloader_test):
        loss, relative_error, relative_error_s = trainer.test(graph,position_mesh,position_pivotal)
        loss_total = loss_total + loss
        relative_error_total = relative_error_total + relative_error
    n = len(trainer.dataloader_test)
    avg_relative_error = relative_error_total/n
    avg_loss = loss_total/n
    rank_zero_logger.info(
            f"avg_loss: {avg_loss:10.3e}, avg_relative_error: {avg_relative_error:10.3e},time per epoch: {(time.time()-start):10.3e}"
        )
    print(relative_error_s)
        

      
    rank_zero_logger.info("Testing completed!")