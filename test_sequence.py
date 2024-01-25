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
from train_sequence import Sequence_Trainer

C = Constants()

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




    trainer = Sequence_Trainer(wb, dist, produce_latents=False, Encoder = Encoder, 
		         position_mesh = position_mesh, 
		         position_pivotal = position_pivotal,
                 rank_zero_logger = rank_zero_logger)
    trainer.model.eval()
    start = time.time()
    rank_zero_logger.info("Testing started...")
    for graph in trainer.dataloader_graph_test:
        g = graph.to(dist.device)
        
        break
    ground_trueth = trainer.dataset_graph_test.solution_states


    
    i = 0
    relative_error_sum_u = 0
    relative_error_sum_v = 0
    relative_error_sum_p = 0
    
    for lc in trainer.dataloader_test:
        ground = ground_trueth[i].to(dist.device)
       
        graph.ndata["x"]
        samples,relative_error_u, relative_error_v, relative_error_p  = trainer.sample(lc[0][:,0:2], lc[1], ground, lc[0], Encoder, g, position_mesh, position_pivotal)
        relative_error_sum_u = relative_error_sum_u + relative_error_u
        relative_error_sum_v = relative_error_sum_v + relative_error_v
        relative_error_sum_p = relative_error_sum_p + relative_error_p
        i = i+1
    relative_error_mean_u = relative_error_sum_u/i
    relative_error_mean_v = relative_error_sum_v/i
    relative_error_mean_p = relative_error_sum_p/i
     
    #avg_loss = loss_total/n_batch
    rank_zero_logger.info(
            f"relative_error_mean_u: {relative_error_mean_u:10.3e},relative_error_mean_v: {relative_error_mean_v:10.3e},relative_error_mean_p: {relative_error_mean_p:10.3e},\\\
            time cost: {(time.time()-start):10.3e}"
        )
    # wb.log({"loss": loss.detach().cpu()})

     
   
    rank_zero_logger.info("Sampling completed!")