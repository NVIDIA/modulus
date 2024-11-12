import torch
from utils.data_loader_hrmip import get_data_loader
import numpy as np
import random
import h5py
from utils.YParams import YParams
from networks.afnonet import AFNONet, PrecipNet
import matplotlib.pyplot as plt
import time

config_name = 'afno_backbone_ec3p_r2i1p1f1_p4_e768_depth12_lr1em3'
params = YParams('config/hrmip.yaml', config_name)

params.batch_size = 1
params.num_data_workers = 0
dataloader, dataset, sampler = get_data_loader(params, params.data_path, distributed=False, split = "train")
valid_dataloader, dataset_valid  = get_data_loader(params, params.data_path, distributed=False, split = "valid")
params.img_shape_x = dataset_valid.img_shape_x
params.img_shape_y = dataset_valid.img_shape_y
params['N_in_channels'] = dataset_valid.n_channels
params['N_out_channels'] = dataset_valid.n_channels


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
params.device = device
model = AFNONet(params).to(device) 
def loop(num=1000):
    for i, samp in enumerate(dataloader):
        if i >= num:
            break

iters = 0
t0 = time.time()
with torch.no_grad():
    loop()
t = time.time() - t0
print("time = {}".format(t))
