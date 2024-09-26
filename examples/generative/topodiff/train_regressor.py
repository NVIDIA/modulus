import torch 
import torch.nn as nn
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
    
    
    topologies = np.load(cfg.path_data + "Compliance/Training/topologies.npy").astype(np.float64)
    constraints = np.load(cfg.path_data + "Compliance/Training/constraints.npy", allow_pickle=True)
    stress = np.load(cfg.path_data+ "Compliance/Training/vonmises.npy", allow_pickle=True)
    strain = np.load(cfg.path_data + "Compliance/Training/strain_energy.npy", allow_pickle=True)
    load_imgs = np.load(cfg.path_data + "Compliance/Training/load_imgs.npy")
    bc_imgs = np.load(cfg.path_data + "Compliance/Training/bc_imgs.npy").astype(np.float64)
    Compliance = np.load(cfg.path_data + "Compliance/Training/compliance.npy").astype(np.float64)


    vfs = []
    for i in range(len(topologies)):
        vfs.append(constraints[i]['VOL_FRAC'])
    vfs = np.array(vfs)    
    
    image_size = topologies.shape[-1]
    
    device = torch.device('cuda:0')
    
    in_channels = 1 + 3 + 2 + 2
    regressor = UNetEncoder(in_channels = in_channels, out_channels=1).to(device)
    
    diffusion = Diffusion(n_steps=cfg.diffusion_steps,device=device)
    print(cfg.model_path + "hello")
    
    
    
    
    batch_size = cfg.batch_size


    lr = cfg.lr
    optimizer = AdamW(regressor.parameters(), lr=lr)
    scheduler = LinearLR(optimizer, start_factor=1, end_factor=0.001, total_iters=cfg.iterations)
    
    vf_in = torch.zeros([batch_size,1,image_size,image_size],dtype=torch.float32,device=device)
    loss_fn = nn.MSELoss()
    for i in range(cfg.iterations):
    
        # get random batch from training data
        idx = np.random.choice(len(topologies), batch_size, replace=False)
        batch = torch.tensor(topologies[idx]).float().unsqueeze(1).to(device)*2-1
        batch_stress = torch.tensor(stress[idx]).float().unsqueeze(1).to(device)
        batch_strain = torch.tensor(strain[idx]).float().unsqueeze(1).to(device)
        batch_load_imgs = torch.tensor(load_imgs[idx]).float().to(device).permute(0,3,1,2)
        batch_bc_imgs = torch.tensor(bc_imgs[idx]).float().to(device).permute(0,3,1,2)
        batch_vf = torch.tensor(vfs[idx]).float().to(device)
        batch_labels = torch.tensor(Compliance[idx]).float().to(device).unsqueeze(1)

        vf_in = vf_in*0
        vf_in = vf_in + batch_vf.unsqueeze(1).unsqueeze(1).unsqueeze(1)

        t = torch.randint(0, cfg.diffusion_steps, (batch.shape[0], )).to(device)
        batch = diffusion.q_sample(batch, t)
       
        batch = torch.cat((batch,batch_stress,batch_strain,batch_load_imgs,vf_in,batch_bc_imgs),dim=1)

        logits = regressor(batch,time_steps=t)

        loss = loss_fn(logits,batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    
        if i % 100 == 0: 
            print("epoch: %d, loss: %.5f" % (i, loss.item()))
            
    torch.save(regressor.state_dict(), cfg.model_path + "regressor.pt")
    print("job done!")


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------