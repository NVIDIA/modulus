import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from tqdm import trange
import numpy as np
import time, os 



import hydra 
from hydra.utils import to_absolute_path 
from omegaconf import DictConfig

from modulus.models.topodiff import Diffusion
from modulus.models.topodiff import UNetEncoder
from modulus.launch.logging import (
    PythonLogger, 
    initialize_wandb
)
from utils import load_data_topodiff 

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None: 
    
    logger = PythonLogger("main") # General Python Logger 
    logger.log("Start running")
    logger.info("end running ")
    
    train_img = np.load(cfg.path_data  + "Floating/training.npy").astype(np.float64)
    train_labels = np.load(cfg.path_data  + "Floating/training_labels.npy").astype(np.float64)

    valid_img = np.load(cfg.path_data + "Floating/validation.npy").astype(np.float64)
    valid_labels = np.load(cfg.path_data  + "Floating/validation_labels.npy").astype(np.float64)
    
    device = torch.device('cuda:0')
    
    
    classifier = UNetEncoder(in_channels = 1, out_channels=2).to(device)
    
    diffusion = Diffusion(n_steps=cfg.diffusion_steps,device=device)
    print(cfg.model_path + "hello")
    
    batch_size = cfg.batch_size


    lr = cfg.lr
    optimizer = AdamW(classifier.parameters(), lr=lr)
    scheduler = LinearLR(optimizer, start_factor=1, end_factor=0.001, total_iters=cfg.iterations)
    
    for i in range(cfg.iterations):
        # get random batch from training data
        idx = np.random.choice(len(train_img), batch_size, replace=False)
        batch = torch.tensor(train_img[idx]).float().unsqueeze(1).to(device)*2-1
        batch_labels = torch.tensor(train_labels[idx]).long().to(device)
        
        t = torch.randint(0, cfg.diffusion_steps, (batch.shape[0], )).to(device)
        batch = diffusion.q_sample(batch, t)
        logits = classifier(batch,time_steps=t)
        
        loss = F.cross_entropy(logits,batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()        
       
        if i % 100 == 0: 
            print("epoch: %d, loss: %.5f" % (i, loss.item()))
            
    torch.save(classifier.state_dict(), cfg.model_path + "regressor.pt")
    print("job done!")


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------