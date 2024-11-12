from utils.YParams import YParams
from importlib import reload
import utils
from utils.data_loader_hrrr_era5 import get_data_loader
import matplotlib.pyplot as plt
import numpy as np
import os
reload(utils.data_loader_hrrr_era5)

params = YParams('./config/hrrr_swin.yaml', 'regression_a2a_v3_1_exclude_w')
params.local_batch_size = 1
params["boundary_padding_pixels"] = 0
params["invariants"] = ["orog", "lsm"]
loader, dataset, sampler = get_data_loader(params, distributed=False, train=True)

_, hrrr_channels = dataset._get_hrrr_channel_names()

print(hrrr_channels)

#for i, data in enumerate(loader):
#
#    if i > 0:
#        break
#    
#
#    for i, ch in enumerate(hrrr_channels):
#        if params.task == 'forecast':
#            fig, ax = plt.subplots(1, 2)
#            im = ax[0].imshow(data['hrrr'][0][0, i, :, :], origin='lower', cmap="RdBu_r")
#            fig.colorbar(im, ax=ax[0])
#            im = ax[1].imshow(data['hrrr'][1][0, i, :, :], origin='lower', cmap="RdBu_r")
#            fig.colorbar(im, ax=ax[1])
#            plt.savefig(f'./test_images/{ch}.png')
#        elif params.task == 'downscale':
#            fig, ax = plt.subplots(1, 1)
#            im = ax.imshow(data['hrrr'][0, i, :, :], origin='lower', cmap="RdBu_r")
#            fig.colorbar(im, ax=ax)
#            plt.savefig(f'./test_images/{ch}.png')
#


    
    