import torch
from networks.afnonet_decoder import AFNONet, PrecipNet
from utils.YParams import YParams
from torchinfo import summary

params = YParams('./config/AFNO.yaml', 'afno_backbone_26var_lamb_embed1536_dpr03_dt4')
params.device = 'cpu'
params['N_in_channels'] = len(params['in_channels'])
params['N_out_channels'] = len(params['out_channels'])
model = AFNONet(params)
summary(model, input_size=(1,26,720,1440))
