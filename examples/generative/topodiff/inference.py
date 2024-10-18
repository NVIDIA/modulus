import torch 
from torch.optim import AdamW
from tqdm import trange
import numpy as np
import time, os 


import hydra 
from hydra.utils import to_absolute_path 
from omegaconf import DictConfig

from modulus.models.topodiff import TopoDiff, Diffusion
from modulus.models.topodiff import UNetEncoder
from modulus.launch.logging import (
    PythonLogger, 
    initialize_wandb
)
from utils import load_data_topodiff 

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None: 
    
    logger = PythonLogger("main") # General Python Logger 
    logger.log("Job start!")
    
    topologies = np.load(cfg.path_data + "Test_InDistro/topologies.npy").astype(np.float64)
    constraints = np.load(cfg.path_data + "Test_InDistro/constraints.npy", allow_pickle=True)
    stress = np.load(cfg.path_data + "Test_InDistro/vonmises.npy", allow_pickle=True)
    strain = np.load(cfg.path_data + "Test_InDistro/strain_energy.npy", allow_pickle=True)
    load_imgs = np.load(cfg.path_data + "/Test_InDistro/load_ims.npy")
    
    
    
    device = torch.device('cuda:0')
    model = TopoDiff(64, 6, 1, model_channels=128, attn_resolutions=[16,8]).to(device)
    
    diffusion = Diffusion(n_steps=1000,device=device)
    batch_size = cfg.batch_size
    data = load_data_topodiff(
        topologies, constraints, stress, strain, load_imgs, batch_size= batch_size,deterministic=False
    )
    
    _, cons = next(data)
    
    cons = cons.float().to(device)
    
    n_steps = 1000
    batch_size = 32     
    
    xt = torch.randn(batch_size, 1, 64, 64).to(device)

    result = []
    with torch.no_grad():
        for i in reversed(range(n_steps)):
            if i > 1: 
                z = torch.randn_like(xt).to(device)
            else:
                z = torch.zeros_like(xt).to(device)
            t = torch.tensor([i] * batch_size, device = device) 
            noisy = diffusion.p_sample(model,xt, t, cons)

            xt = 1 / diffusion.alphas[i].sqrt() * (xt - noisy * (1 -  diffusion.alphas[i])/(1 - diffusion.alpha_bars[i]).sqrt()) + + diffusion.betas[i].sqrt() * z 
            
        result = xt.cpu().numpy()
    
    np.save(cfg.generation_path + 'results.npy', result)    
    
    
if __name__ == "__main__": 
    main()
        