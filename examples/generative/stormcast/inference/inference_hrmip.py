#BSD 3-Clause License
#
#Copyright (c) 2022, FourCastNet authors
#All rights reserved.
#
#Redistribution and use in source and binary forms, with or without
#modification, are permitted provided that the following conditions are met:
#
#1. Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
#2. Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
#3. Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
#THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#The code was authored by the following people:
#
#Jaideep Pathak - NVIDIA Corporation
#Shashank Subramanian - NERSC, Lawrence Berkeley National Laboratory
#Peter Harrington - NERSC, Lawrence Berkeley National Laboratory
#Sanjeev Raja - NERSC, Lawrence Berkeley National Laboratory 
#Ashesh Chattopadhyay - Rice University 
#Morteza Mardani - NVIDIA Corporation 
#Thorsten Kurth - NVIDIA Corporation 
#David Hall - NVIDIA Corporation 
#Zongyi Li - California Institute of Technology, NVIDIA Corporation 
#Kamyar Azizzadenesheli - Purdue University 
#Pedram Hassanzadeh - Rice University 
#Karthik Kashinath - NVIDIA Corporation 
#Animashree Anandkumar - California Institute of Technology, NVIDIA Corporation

import os
import sys
import time
import numpy as np
import argparse
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')
from numpy.core.numeric import False_
import h5py
import torch
import torchvision
from torchvision.utils import save_image
import torch.nn as nn
import torch.cuda.amp as amp
import torch.distributed as dist
from collections import OrderedDict
from torch.nn.parallel import DistributedDataParallel
import logging
from utils import logging_utils
from utils.weighted_acc_rmse import weighted_rmse_torch_channels, weighted_acc_torch_channels, unweighted_acc_torch_channels, weighted_acc_masked_torch_channels
logging_utils.config_logger()
from utils.YParams import YParams
from utils.data_loader_hrmip import get_data_loader
from networks.afnonet import AFNONet
#from networks.afnonet_decoder import AFNONet, PrecipNet
import wandb
import matplotlib.pyplot as plt
import glob
from datetime import datetime
from skimage.transform import downscale_local_mean


fld = "z500" # diff flds have diff decor times and hence differnt ics
if fld == "z500" or fld == "2m_temperature" or fld == "t850":
    DECORRELATION_TIME = 33#36 # 9 days (36) for z500, 2 (8 steps) days for u10, v10
else:
    DECORRELATION_TIME = 8 # 9 days (36) for z500, 2 (8 steps) days for u10, v10
idxes = {"u10":32, "z500":28, "2m_temperature":31, "v10":33, "t850":None}

def downscale(img, scale):
    new_img = downscale_local_mean(img, (1, 1, scale[0], scale[1]))
    return new_img

def gaussian_perturb(x, level=0.01, device=0):
    noise = level * torch.randn(x.shape).to(device, dtype=torch.float)
    return (x + noise)

def load_model(model, params, checkpoint_file):
    model.zero_grad()
    checkpoint_fname = checkpoint_file
    checkpoint = torch.load(checkpoint_fname)
    try:
        new_state_dict = OrderedDict()
        for key, val in checkpoint['model_state'].items():
            name = key[7:]
            if name != 'ged':
                new_state_dict[name] = val  
        model.load_state_dict(new_state_dict)
    except:
        model.load_state_dict(checkpoint['model_state'])
    model.eval()
    return model

def downsample(x, scale=0.125):
    return torch.nn.functional.interpolate(x, scale_factor=scale, mode='bilinear')

def concat(pl, sl):
    ''' pl is (nt, n_var, n_l, ix, iy)
        sl is (nt, n_var, ix, iy)
    '''
    pl_shape = pl.shape
    sl_shape = sl.shape
    pllist = []
    for i in range(pl_shape[1]): # for each variable 
        pllist.append(pl[:, i, ...])
    pl = np.concatenate(pllist, axis=1)
    tensor = np.concatenate([pl, sl], axis=1)
    return tensor


def setup(params, year=None):
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    #get data loader
    valid_data_loader, valid_dataset = get_data_loader(params, params.data_path_inf, dist.is_initialized(), split="inference")
    img_shape_x = valid_dataset.img_shape_x
    img_shape_y = valid_dataset.img_shape_y
    params.img_shape_x = img_shape_x
    params.img_shape_y = img_shape_y
    if params.log_to_screen:
        logging.info('Loading trained model checkpoint from {}'.format(params['best_checkpoint_path']))

    n_in_channels = params.n_channels
    n_out_channels = params.n_channels
    in_channels = np.array(list(range(params.n_channels)))
    out_channels = np.array(list(range(params.n_channels)))
    params['N_in_channels'] = n_in_channels
    params['N_out_channels'] = n_out_channels

    params.means = np.load(params.global_means_path)[0, out_channels] # needed to standardize wind data
    params.stds = np.load(params.global_stds_path)[0, out_channels]
    params.device = device

    # load the model
    if params.nettype == 'afno':
      model = AFNONet(params).to(device) 
    else:
      raise Exception("not implemented")

    checkpoint_file  = params['best_checkpoint_path']
    model = load_model(model, params, checkpoint_file)
    model = model.to(device)

    # load the validation data
    if year:
        params.test_years_range = [year, year+1]
    years = list(range(params.test_years_range[0], params.test_years_range[1] + 1))
    files_paths = [params.data_path_inf + "/{}.h5".format(yr) for yr in years]
    files_paths.sort()
    # which year
    yr = 0
    if params.log_to_screen:
        logging.info('Loading inference data')
        logging.info('Inference data from {}'.format(files_paths[yr]))

    pl = h5py.File(files_paths[yr], 'r')['pl']
    sl = h5py.File(files_paths[yr], 'r')['sl']
    valid_data_full = [pl, sl]

    return valid_data_full, model

def autoregressive_inference(params, ic, valid_data_full, model): 
    ic = int(ic) 
    #initialize global variables
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    exp_dir = params['experiment_dir'] 
    dt = int(params.dt)
    prediction_length = int(params.prediction_length/dt)
    n_history = params.n_history
    img_shape_x = params.img_shape_x
    img_shape_y = params.img_shape_y
    in_channels = np.array(list(range(params.n_channels)))
    out_channels = np.array(list(range(params.n_channels)))
    n_in_channels = len(in_channels)
    n_out_channels = len(out_channels)
    means = params.means
    stds = params.stds

    #initialize memory for image sequences and RMSE/ACC
    valid_loss = torch.zeros((prediction_length, n_out_channels)).to(device, dtype=torch.float)
    acc = torch.zeros((prediction_length, n_out_channels)).to(device, dtype=torch.float)

    # compute metrics in a coarse resolution too if params.interp is nonzero
    valid_loss_coarse = torch.zeros((prediction_length, n_out_channels)).to(device, dtype=torch.float)
    acc_coarse = torch.zeros((prediction_length, n_out_channels)).to(device, dtype=torch.float)
    acc_coarse_unweighted = torch.zeros((prediction_length, n_out_channels)).to(device, dtype=torch.float)
    
    acc_unweighted = torch.zeros((prediction_length, n_out_channels)).to(device, dtype=torch.float)
    seq_real = torch.zeros((prediction_length, n_in_channels, img_shape_x, img_shape_y)).to(device, dtype=torch.float)
    seq_pred = torch.zeros((prediction_length, n_in_channels, img_shape_x, img_shape_y)).to(device, dtype=torch.float)

    acc_land = torch.zeros((prediction_length, n_out_channels)).to(device, dtype=torch.float)
    acc_sea = torch.zeros((prediction_length, n_out_channels)).to(device, dtype=torch.float)
    if params.masked_acc:
      maskarray = torch.as_tensor(np.load(params.maskpath)[0:img_shape_x]).to(device, dtype=torch.float)

    pl = valid_data_full[0][ic:(ic+prediction_length*dt+n_history*dt):dt] #extract valid data from first year
    sl = valid_data_full[1][ic:(ic+prediction_length*dt+n_history*dt):dt] #extract valid data from first year
#    n_pixels = pl.shape[3]
#    pl = pl[:,:,:,0:n_pixels-1]
#    sl = sl[:,:,0:n_pixels-1]

    valid_data = concat(pl, sl)
    if params.interp_factor_x != 1 or params.interp_factor_y != 1:
        valid_data = downscale(valid_data, scale = (params.interp_factor_x, params.interp_factor_y))
    # standardize
    valid_data = (valid_data - means)/stds
    valid_data = torch.as_tensor(valid_data).to(device, dtype=torch.float)

    #load time means
    if not params.use_daily_climatology:
      m = torch.as_tensor((np.load(params.time_means_path)[0][out_channels] - means)/stds)[:, 0:img_shape_x] # climatology
      m = torch.unsqueeze(m, 0)
    else:
      # use daily clim like weyn et al. (different from rasp)
      dc_path = params.dc_path
      with h5py.File(dc_path, 'r') as f:
        dc = f['time_means_daily'][ic:ic+prediction_length*dt:dt] # 1460,21,721,1440
      m = torch.as_tensor((dc[:,out_channels,0:img_shape_x,:] - means)/stds) 

    m = m.to(device, dtype=torch.float)
    if params.interp > 0:
        m_coarse = downsample(m, scale=params.interp)

    std = torch.as_tensor(stds[:,0,0]).to(device, dtype=torch.float)

    orography = params.orography
    orography_path = params.orography_path
    if orography:
      orog = torch.as_tensor(np.expand_dims(np.expand_dims(h5py.File(orography_path, 'r')['orog'][0:720], axis = 0), axis = 0)).to(device, dtype = torch.float)
      logging.info("orography loaded; shape:{}".format(orog.shape))

    #autoregressive inference
    if params.log_to_screen:
      logging.info('Begin autoregressive inference')
    
    with torch.no_grad():
      for i in range(valid_data.shape[0]): 
        if i==0: #start of sequence
          first = valid_data[0:n_history+1]
          future = valid_data[n_history+1]
          for h in range(n_history+1):
            seq_real[h] = first[h*n_in_channels : (h+1)*n_in_channels][0:n_out_channels] #extract history from 1st 
            seq_pred[h] = seq_real[h]
          if params.perturb:
            first = gaussian_perturb(first, level=params.n_level, device=device) # perturb the ic
          if orography:
            future_pred = model(torch.cat((first, orog), axis=1))
          else:
            future_pred = model(first)
        else:
          if i < prediction_length-1:
            future = valid_data[n_history+i+1]
          if orography:
            future_pred = model(torch.cat((future_pred, orog), axis=1)) #autoregressive step
          else:
            future_pred = model(future_pred) #autoregressive step

        if i < prediction_length-1: #not on the last step
          seq_pred[n_history+i+1] = future_pred
          seq_real[n_history+i+1] = future
          history_stack = seq_pred[i+1:i+2+n_history]

        future_pred = history_stack
      
        #Compute metrics 
        if params.use_daily_climatology:
            clim = m[i:i+1]
            if params.interp > 0:
                clim_coarse = m_coarse[i:i+1]
        else:
            clim = m
            if params.interp > 0:
                clim_coarse = m_coarse

        pred = torch.unsqueeze(seq_pred[i], 0)
        tar = torch.unsqueeze(seq_real[i], 0)
        valid_loss[i] = weighted_rmse_torch_channels(pred, tar) * std
        acc[i] = weighted_acc_torch_channels(pred-clim, tar-clim)
        acc_unweighted[i] = unweighted_acc_torch_channels(pred-clim, tar-clim)

        if params.masked_acc:
          acc_land[i] = weighted_acc_masked_torch_channels(pred-clim, tar-clim, maskarray)
          acc_sea[i] = weighted_acc_masked_torch_channels(pred-clim, tar-clim, 1-maskarray)

        if params.interp > 0:
            pred = downsample(pred, scale=params.interp)
            tar = downsample(tar, scale=params.interp)
            valid_loss_coarse[i] = weighted_rmse_torch_channels(pred, tar) * std
            acc_coarse[i] = weighted_acc_torch_channels(pred-clim_coarse, tar-clim_coarse)
            acc_coarse_unweighted[i] = unweighted_acc_torch_channels(pred-clim_coarse, tar-clim_coarse)

        if params.log_to_screen:
          idx = idxes[fld] 
          logging.info('Predicted timestep {} of {}. {} RMS Error: {}, ACC: {}'.format(i, prediction_length, fld, valid_loss[i, idx], acc[i, idx]))
          if params.interp > 0:
            logging.info('[COARSE] Predicted timestep {} of {}. {} RMS Error: {}, ACC: {}'.format(i, prediction_length, fld, valid_loss_coarse[i, idx],
                        acc_coarse[i, idx]))

    seq_real = seq_real.cpu().numpy()
    seq_pred = seq_pred.cpu().numpy()
    valid_loss = valid_loss.cpu().numpy()
    acc = acc.cpu().numpy()
    acc_unweighted = acc_unweighted.cpu().numpy()
    acc_coarse = acc_coarse.cpu().numpy()
    acc_coarse_unweighted = acc_coarse_unweighted.cpu().numpy()
    valid_loss_coarse = valid_loss_coarse.cpu().numpy()
    acc_land = acc_land.cpu().numpy()
    acc_sea = acc_sea.cpu().numpy()

    return (np.expand_dims(seq_real[n_history:], 0), np.expand_dims(seq_pred[n_history:], 0), np.expand_dims(valid_loss,0), np.expand_dims(acc, 0),
           np.expand_dims(acc_unweighted, 0), np.expand_dims(valid_loss_coarse, 0), np.expand_dims(acc_coarse, 0),
           np.expand_dims(acc_coarse_unweighted, 0),
           np.expand_dims(acc_land, 0),
           np.expand_dims(acc_sea, 0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_num", default='00', type=str)
    parser.add_argument("--yaml_config", default='./config/hrmip.yaml', type=str)
    parser.add_argument("--config", default='full_field', type=str)
    parser.add_argument("--use_daily_climatology", action='store_true')
    parser.add_argument("--vis", action='store_true')
    parser.add_argument("--save", action='store_true')
    parser.add_argument("--override_dir", default=None, type = str, help = 'Path to store inference outputs; must also set --weights arg')
    parser.add_argument("--interp", default=0, type=float)
    parser.add_argument("--weights", default=None, type=str, help = 'Path to model weights, for use with override_dir option')
    parser.add_argument("--year", default=None, type=int, help = 'Inference year')
    
    args = parser.parse_args()
    params = YParams(os.path.abspath(args.yaml_config), args.config)
    params['world_size'] = 1
    params['interp'] = args.interp
    params['use_daily_climatology'] = args.use_daily_climatology
    params['global_batch_size'] = params.batch_size

    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True
    vis = args.vis

    # Set up directory
    if args.override_dir is not None:
      assert args.weights is not None, 'Must set --weights argument if using --override_dir'
      expDir = args.override_dir
    else:
      assert args.weights is None, 'Cannot use --weights argument without also using --override_dir'
      expDir = os.path.join(params.exp_dir, args.config, str(args.run_num))

    if not os.path.isdir(expDir):
      os.makedirs(expDir)

    params['experiment_dir'] = os.path.abspath(expDir)
    params['best_checkpoint_path'] = args.weights if args.override_dir is not None else os.path.join(expDir, 'training_checkpoints/best_ckpt.tar')
    params['resuming'] = False
    params['local_rank'] = 0

    logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, 'inference_out.log'))
    logging_utils.log_versions()
    params.log()

    n_ics = params['n_initial_conditions']

    n_samples_per_year = 1460

    if params["ics_type"] == 'default':
        num_samples = n_samples_per_year-params.prediction_length
        stop = num_samples
        ics = np.arange(0, stop, DECORRELATION_TIME)
        if vis: # visualization for just the first ic (or any ic)
            ics = [0]
        n_ics = len(ics)
    elif params["ics_type"] == "datetime":
        date_strings = params["date_strings"]
        ics = []
        if params.perturb: #for perturbations use a single date and create n_ics perturbations
            n_ics = params["n_perturbations"]
            date = date_strings[0]
            date_obj = datetime.strptime(date,'%Y-%m-%d %H:%M:%S') 
            day_of_year = date_obj.timetuple().tm_yday - 1
            hour_of_day = date_obj.timetuple().tm_hour
            hours_since_jan_01_epoch = 24*day_of_year + hour_of_day
            for ii in range(n_ics):
                ics.append(int(hours_since_jan_01_epoch/6))
        else:
            for date in date_strings:
                date_obj = datetime.strptime(date,'%Y-%m-%d %H:%M:%S') 
                day_of_year = date_obj.timetuple().tm_yday - 1
                hour_of_day = date_obj.timetuple().tm_hour
                hours_since_jan_01_epoch = 24*day_of_year + hour_of_day
                ics.append(int(hours_since_jan_01_epoch/6))
        n_ics = len(ics)

    logging.info("Inference for {} initial conditions".format(n_ics))
    try:
      autoregressive_inference_filetag = params["inference_file_tag"]
    except:
      autoregressive_inference_filetag = ""

    if params.interp > 0:
        autoregressive_inference_filetag = "_coarse"

    autoregressive_inference_filetag += "_" + fld + ""
    if vis:
        autoregressive_inference_filetag += "_vis"
    if args.save:
        autoregressive_inference_filetag += "_save_33steps"

    # get data and models
    valid_data_full, model = setup(params, args.year)

    #initialize lists for image sequences and RMSE/ACC
    valid_loss = []
    valid_loss_coarse = []
    acc_unweighted = []
    acc = []
    acc_coarse = []
    acc_coarse_unweighted = []
    seq_pred = []
    seq_real = []
    acc_land = []
    acc_sea = []

    if args.save:
        pred_save = []
        save_times = [4, 12, 20, 28]

    #run autoregressive inference for multiple initial conditions
    for i, ic in enumerate(ics):
      logging.info("Initial condition {} of {}".format(i+1, n_ics))
      sr, sp, vl, a, au, vc, ac, acu, accland, accsea = autoregressive_inference(params, ic, valid_data_full, model)

      if i ==0 or len(valid_loss) == 0:
        seq_real = sr
        seq_pred = sp
        if args.save:
            pred_save = sp[0:1,save_times]
        valid_loss = vl
        valid_loss_coarse = vc
        acc = a
        acc_coarse = ac
        acc_coarse_unweighted = acu
        acc_unweighted = au
        acc_land = accland
        acc_sea = accsea
      else:
#        seq_real = np.concatenate((seq_real, sr), 0)
#        seq_pred = np.concatenate((seq_pred, sp), 0)
        if args.save:
            pred_save = np.concatenate((pred_save, sp[0:1,save_times]), 0)
        valid_loss = np.concatenate((valid_loss, vl), 0)
        valid_loss_coarse = np.concatenate((valid_loss_coarse, vc), 0)
        acc = np.concatenate((acc, a), 0)
        acc_coarse = np.concatenate((acc_coarse, ac), 0)
        acc_coarse_unweighted = np.concatenate((acc_coarse_unweighted, acu), 0)
        acc_unweighted = np.concatenate((acc_unweighted, au), 0)
        acc_land = np.concatenate((acc_land, accland), 0)
        acc_sea = np.concatenate((acc_sea, accsea), 0)

    prediction_length = seq_real[0].shape[0]
    n_out_channels = seq_real[0].shape[1]
    img_shape_x = seq_real[0].shape[2]
    img_shape_y = seq_real[0].shape[3]


    #save predictions and loss
    if params.log_to_screen:
      logging.info("Saving files at {}".format(os.path.join(params['experiment_dir'], 'autoregressive_predictions' + autoregressive_inference_filetag + '.h5')))
    with h5py.File(os.path.join(params['experiment_dir'], 'autoregressive_predictions'+ autoregressive_inference_filetag +'.h5'), 'a') as f:


      if args.save:
        try:
            f.create_dataset("pred", data = pred_save, shape = (n_ics, pred_save.shape[1], n_out_channels, img_shape_x, img_shape_y), dtype = np.float32)
        except:
            del f["pred"]
            f.create_dataset("pred", data = pred_save, shape = (n_ics, pred_save.shape[1], n_out_channels, img_shape_x, img_shape_y), dtype = np.float32)
            f["pred"][...]= pred_save
            
      if vis:
        try:
            f.create_dataset("ground_truth", data = seq_real, shape = (n_ics, prediction_length, n_out_channels, img_shape_x, img_shape_y), dtype = np.float32)
        except: 
            del f["ground_truth"]
            f.create_dataset("ground_truth", data = seq_real, shape = (n_ics, prediction_length, n_out_channels, img_shape_x, img_shape_y), dtype = np.float32)
            f["ground_truth"][...] = seq_real

        try:
            f.create_dataset("predicted", data = seq_pred, shape = (n_ics, prediction_length, n_out_channels, img_shape_x, img_shape_y), dtype = np.float32)
        except:
            del f["predicted"]
            f.create_dataset("predicted", data = seq_pred, shape = (n_ics, prediction_length, n_out_channels, img_shape_x, img_shape_y), dtype = np.float32)
            f["predicted"][...]= seq_pred



      try:
        f.create_dataset("rmse", data = valid_loss, shape = (n_ics, prediction_length, n_out_channels), dtype =np.float32)
      except:
        del f["rmse"]
        f.create_dataset("rmse", data = valid_loss, shape = (n_ics, prediction_length, n_out_channels), dtype =np.float32)
        f["rmse"][...] = valid_loss

      try:
        f.create_dataset("acc", data = acc, shape = (n_ics, prediction_length, n_out_channels), dtype =np.float32)
      except:
        del f["acc"]
        f.create_dataset("acc", data = acc, shape = (n_ics, prediction_length, n_out_channels), dtype =np.float32)
        f["acc"][...] = acc   

        
      f.close()
