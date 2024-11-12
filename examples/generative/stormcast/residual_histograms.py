from utils.YParams import YParams
from importlib import reload
import torch
import torch.nn as nn
import utils
from utils.data_loader_hrrr_era5 import get_data_loader
import matplotlib.pyplot as plt
import numpy as np
from networks.swinv2_hrrr import swinv2net
import os
import utils.diffusions.networks
from utils.diffusions.networks import RegressionWrapper
import pickle
reload(utils.data_loader_hrrr_era5)

def lognormal_pdf(x, mu, sigma):
    """
    Compute the lognormal PDF value for a given x, mu, and sigma.

    Parameters:
    - x: The value at which to compute the PDF.
    - mu: The mean of the logarithm of the variable.
    - sigma: The standard deviation of the logarithm of the variable.

    Returns:
    - The PDF value of the lognormal distribution at x.
    """
    return (1 / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(-((np.log(x) - mu) ** 2) / (2 * sigma ** 2))


def get_pretrained_regression_net(basepath, device, nettype):

    if nettype == "swin":

        from utils.simple_load_yaml import simple_load_yaml

        hyperparams = simple_load_yaml(os.path.join(basepath, 'hyperparams.yaml'))
        hyperparams.era5_img_size = hyperparams.hrrr_img_size #TODO fix this issue in the original hyperparam dump
        hyperparams.mask_ratio = 0
        hyperparams.nonzero_mask = False
        swin = swinv2net(hyperparams).to(device)
        checkpoint_path = os.path.join(basepath, 'training_checkpoints', 'ckpt.tar')

        # load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        for key in list(checkpoint['model_state'].keys()):
            if 'module.' in key:
                checkpoint['model_state'][key.replace('module.', '')] = checkpoint['model_state'].pop(key)

        swin.load_state_dict(checkpoint['model_state'])

        return swin
    
    elif nettype == "unet":

        params = YParams(os.path.join("./config", 'hrrr_swin.yaml'), 'regression_a2a_v3_1_exclude_w')
        net_name = "song-unet-regression"
        resolution = 512
        target_channels = len(['u10m', 'v10m', 't2m', 'msl', 'u1', 'u2', 'u3', 'u4', 'u5', 'u6', 'u7', 'u8', 'u9', 'u10', 'u11', 'u13', 'u15', 'u20', 'u25', 'u30', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'v13', 'v15', 'v20', 'v25', 'v30', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't13', 't15', 't20', 't25', 't30', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10', 'q11', 'q13', 'q15', 'q20', 'q25', 'q30', 'z1', 'z2', 'z3', 'z4', 'z5', 'z6', 'z7', 'z8', 'z9', 'z10', 'z11', 'z13', 'z15', 'z20', 'z25', 'z30', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p13', 'p15', 'p20', 'refc']) 
        conditional_channels = target_channels + len(params.invariants) + 26

        net = utils.diffusions.networks.get_preconditioned_architecture(
        name=net_name,
        resolution=resolution,
        target_channels=target_channels,
        conditional_channels=conditional_channels,
        label_dim=0,
        spatial_embedding=params.spatial_pos_embed,
        attn_resolutions=params.attn_resolutions,
    )

        resume_pkl = "/pscratch/sd/j/jpathak/hrrr_experiments_eos/0-regression_a2a_v3_1_exclude_w-hrrr-gpus32/network-snapshot-012280.pkl"

        with open(resume_pkl, 'rb') as f:
            data = pickle.load(f)
        net.load_state_dict(data['net'].state_dict(), strict=True)

        return net.to(device)



params = YParams('./config/hrrr_swin.yaml', 'regression_a2a_v3_1_exclude_w')
params.local_batch_size = 4
params["boundary_padding_pixels"] = 0
params["invariants"] = ["orog", "lsm"]
loader, dataset, sampler = get_data_loader(params, distributed=False, train=True)
_, hrrr_channels = dataset._get_hrrr_channel_names()
device = 'cuda:0'

invariant_array = dataset._get_invariants()
invariant_tensor = torch.from_numpy(invariant_array).to(device).repeat(params.local_batch_size, 1, 1, 1)



#create h5 file to store the residuals
import h5py
import os
import numpy as np

nettype = "unet"
generate_residuals = True
log_scale = True
plot_samples = True
h5_path = "./residuals.h5"
n_samples = 1024

if generate_residuals:
    
    regression_net = get_pretrained_regression_net(params.regression_model_basepath, device, nettype=nettype)

    n_batches = n_samples // params.local_batch_size

    #with h5py.File(h5_path, 'w') as f:

    #    f.create_dataset('residuals', (n_samples ,len(hrrr_channels), 512, 640), dtype='f4')


    for i, batch in enumerate(loader):

        if i > n_batches:
            break

        hrrr_0 = batch['hrrr'][0].to(device).to(torch.float32)
        hrrr_1 = batch['hrrr'][1].to(device).to(torch.float32)

        era5 = batch['era5'][0].to(device).to(torch.float32)

        with torch.no_grad():

            if nettype == "swin":
                hrrr_0 = regression_net(hrrr_0, era5, mask=None)
            elif nettype == "unet":
                latents = torch.zeros_like(hrrr_1, device=hrrr_1.device)
                rnd_normal = torch.randn([latents.shape[0], 1, 1, 1], device=latents.device)
                sigma = rnd_normal #this isn't used by the code
                condition = torch.cat((hrrr_0, era5, invariant_tensor), dim=1).to(device)
                hrrr_0 = regression_net(x=latents, sigma=sigma, condition=condition)

            hrrr_1 = hrrr_1 - hrrr_0
        
        if plot_samples:

            if i == 0:
                for ch in range(len(hrrr_channels)):
                    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                    chname = hrrr_channels[ch]
                    ax[0].imshow(hrrr_0[0, ch].cpu().numpy())
                    ax[0].set_title(chname + ' prediction')
                    ax[1].imshow(hrrr_1[0, ch].cpu().numpy())
                    ax[1].set_title(hrrr_channels[ch] + ' residual')
                    plt.savefig(f"./residual_samples_unet/{chname}.png")
                
        

        hrrr_1 = hrrr_1.cpu().numpy()

        with h5py.File(h5_path, 'a') as f:

            f['residuals'][i*params.local_batch_size:(i+1)*params.local_batch_size] = hrrr_1
            print(f"Batch {i} done")
            f.flush()

        del hrrr_0, hrrr_1, era5


#plot the channelwise histograms of the residuals

#make directory to store the histograms
if not os.path.exists('./residual_histograms_unet'):
    os.makedirs('./residual_histograms_unet')

for i, channel in enumerate(hrrr_channels):


    with h5py.File(h5_path, 'r') as f:

        residuals = f['residuals'][:, hrrr_channels.index(channel)]

    
    fig, ax = plt.subplots(3, 1, figsize=(15, 10))

    ax[0].hist(residuals.flatten(), bins=100, log=False, density=True, histtype='step')
    ax[0].set_title(channel + ' residuals linear scaling')
    ax[0].set_xlabel('Residual')
    ax[0].set_ylabel('Density')

    log_scaled_residuals = np.sign(residuals) * np.log(1 + np.abs(residuals))
    ax[1].hist(log_scaled_residuals.flatten(), bins=100, log=False, density=True, histtype='step')
    ax[1].set_title(channel + ' residuals log scaling')
    ax[1].set_xlabel('sign(residual) * log(1 + |residual|)')
    ax[1].set_ylabel('Density')
    x = np.logspace(-8, 1, 1000, base=np.e)
    y = lognormal_pdf(x, -1.2, 1.2)
    ax[1].plot(x, y, label='lognormal PDF, mu = -1.2, sigma = 1.2', color='r')
    y = lognormal_pdf(x, -1.2, 2.0)
    ax[1].plot(x, y, label='lognormal PDF, mu = -1.2, sigma = 2.0', color='b')
    y = lognormal_pdf(x, -1.2, 2.2)
    ax[1].plot(x, y, label='lognormal PDF, mu = -1.2, sigma = 2.2', color='g')
    y = lognormal_pdf(x, 0, 1.2)
    ax[1].plot(x, y, label='lognormal PDF, mu = 0, sigma = 1.2', color='m')
    #legend
    ax[1].legend()
    #set x limit
    ax[1].set_xlim(-2, 2)

    x = np.logspace(-8, 1, 1000)
    y = lognormal_pdf(x, -1.2, 1.2)
    ax[2].plot(x, y, label='lognormal PDF, mu = -1.2, sigma = 1.2', color='r')
    y = lognormal_pdf(x, -1.2, 2.0)
    ax[2].plot(x, y, label='lognormal PDF, mu = -1.2, sigma = 2.0', color='b')
    y = lognormal_pdf(x, -1.2, 2.2)
    ax[2].plot(x, y, label='lognormal PDF, mu = -1.2, sigma = 2.2', color='g')
    y = lognormal_pdf(x, 0.0, 1.2)
    ax[2].plot(x, y, label='lognormal PDF, mu = 0.0, sigma = 1.2', color='m')
    #legend
    ax[2].legend()
    ax[2].set_title('lognormal PDFs')
    ax[2].set_xscale('log')
    

    #set a suptitle
    fig.suptitle(f"Residuals computed w.r.t. the regression {nettype} model for {channel} channel using {n_samples} samples")
    plt.tight_layout()

    plt.savefig(f'./residual_histograms_unet/{channel}_residuals.png')








    
    