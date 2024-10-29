import torch 
from torch.optim import AdamW
import torch.nn.functional as F
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt


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
    model = TopoDiff(64, 6, 1, model_channels=128, attn_resolutions=[16,8])
    model.load_state_dict(torch.load(cfg.model_path_diffusion))
    model.to(device)
    
    classifier = UNetEncoder(in_channels = 1, out_channels=2)
    classifier.load_state_dict(torch.load(cfg.model_path_classifier))
    classifier.to(device)
    
    diffusion = Diffusion(n_steps=1000,device=device)
    batch_size = cfg.batch_size
    data = load_data_topodiff(
        topologies, constraints, stress, strain, load_imgs, batch_size= batch_size,deterministic=False
    )
    
    _, cons = next(data)
    
    cons = cons.float().to(device)
    
    n_steps = 1000
    #batch_size = 32     
    
    xt = torch.randn(batch_size, 1, 64, 64).to(device)
    floating_labels = torch.tensor([0]*batch_size).long().to(device)
    
    for i in reversed(range(n_steps)): 
        with torch.no_grad():
            if i > 1: 
                z = torch.rand_like(xt).to(device)
            else: 
                z = torch.zeros_like(xt).to(device)
            t = torch.tensor([i] * batch_size, device = device) 
            noisy = diffusion.p_sample(model,xt, t, cons)
        with torch.enable_grad():
            xt.requires_grad_(True)
            logits = classifier(xt,time_steps=t)
            loss = F.cross_entropy(logits,floating_labels)
        
            grad = torch.autograd.grad(loss, xt)[0]
        
        xt = 1 / diffusion.alphas[i].sqrt() * (xt - noisy * (1 -  diffusion.alphas[i])/(1 - diffusion.alpha_bars[i]).sqrt()) 
        
        if i >  0: 
            z = torch.zeros_like(xt).to(device)
            xt = xt + diffusion.betas[i].sqrt() * (z * 0.8 + 0.2 * grad.float())
    
    result = xt.cpu().detach().numpy()


    """
    result = []
    with torch.no_grad():
        for i in reversed(range(n_steps)):
            if i > 1: 
                z = torch.randn_like(xt).to(device)
            else:
                z = torch.zeros_like(xt).to(device)
            t = torch.tensor([i] * batch_size, device = device) 
            noisy = diffusion.p_sample(model,xt, t, cons)
            logits = classifier(batch_size,time_steps=t)
        
            loss = F.cross_entropy(logits,floating_labels)
            
            

            xt = 1 / diffusion.alphas[i].sqrt() * (xt - noisy * (1 -  diffusion.alphas[i])/(1 - diffusion.alpha_bars[i]).sqrt()) + + diffusion.betas[i].sqrt() * z 
            
        result = xt.cpu().detach().numpy()
    """
    np.save(cfg.generation_path + 'results.npy', result)    
    
    # plot images for the generated samples
    fig, axes = plt.subplots(2,4, figsize=(12,6),dpi=300)

    for i in range(2): 
        for j in range(4): 
            img = np.zeros((64,64))
            img[result[i*4 + j ][0] > 0] = 1
            axes[i,j].imshow(img, cmap='gray')
            axes[i,j].axis('off')
        

    plt.xticks([])  # Remove x-axis ticks
    plt.yticks([])  # Remove y-axis ticks
    plt.gca().xaxis.set_visible(False)  # Optionally hide x-axis
    plt.gca().yaxis.set_visible(False)  # Optionally hide y-axis
    
    plt.savefig(cfg.generation_path + 'grid.png', bbox_inches='tight', pad_inches=0)

    
    
if __name__ == "__main__": 
    main()
        