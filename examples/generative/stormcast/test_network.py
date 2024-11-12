import torch
import numpy as np
from networks.swinv2_hrrr import swinv2net
from utils.YParams import YParams


params = YParams('./config/hrrr_swin.yaml', 'boundary_test')
params['hrrr_img_size'] =  [512, 896]
params['era5_img_size'] =  [640, 1024]
#params['era5_img_size'] =  [512, 896]
params['boundary_padding_pixels'] =  128
params['n_hrrr_channels'] =  3
params['n_era5_channels'] =  3 
params['embed_dim'] =  96


model = swinv2net(params).to(device = 'cuda:0')

x = torch.randn(1, 3, 512, 896).to(device = 'cuda:0')
y = torch.randn(1, 3, 640, 1024).to(device = 'cuda:0')

z = model(x, y)
