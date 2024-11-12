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
import time
import numpy as np
import argparse
import sys
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
from utils.weighted_acc_rmse import weighted_rmse_torch_channels, weighted_acc_torch_channels, unlog_tp_torch, top_quantiles_error_torch
logging_utils.config_logger()
from utils.YParams import YParams
from utils.data_loader_multifiles import get_data_loader
from networks.afnonet import AFNONet, PrecipNet 
import wandb
import matplotlib.pyplot as plt
import glob
from datetime import datetime
DECORRELATION_TIME = 8 # 2 days for preicp


def gaussian_perturb(x, level=0.01, device=0):
    noise = level * torch.randn(x.shape).to(device, dtype=torch.float)
    return (x + noise)

def load_model(model, params, checkpoint_file):
    model.zero_grad()
    checkpoint = torch.load(checkpoint_file)
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


def setup(params):
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    #get data loader
    valid_data_loader, valid_dataset = get_data_loader(params, params.inf_data_path, dist.is_initialized(), train=False)
    img_shape_x = valid_dataset.img_shape_x
    img_shape_y = valid_dataset.img_shape_y
    params.img_shape_x = img_shape_x
    params.img_shape_y = img_shape_y
    if params.log_to_screen:
        logging.info('Loading trained model checkpoint from {}'.format(params['best_checkpoint_path']))

    in_channels = np.array(params.in_channels)
    out_channels = np.array(params.in_channels) # same as in for the wind model
    n_in_channels = len(in_channels)
    n_out_channels = len(out_channels)
    params['N_in_channels'] = n_in_channels
    params['N_out_channels'] = n_out_channels
    params.means = np.load(params.global_means_path)[0, out_channels] # needed to standardize wind data
    params.stds = np.load(params.global_stds_path)[0, out_channels]
    # load the models
    # load wind model
    if params.nettype_wind == 'afno':
        model_wind = AFNONet(params).to(device) 
    checkpoint_file  = params['model_wind_path']
    model_wind = load_model(model_wind, params, checkpoint_file)
    model_wind = model_wind.to(device)

    out_channels = np.array(params.out_channels) # change out channels for precip model
    n_out_channels = len(out_channels)
    params['N_out_channels'] = n_out_channels
    if params.nettype == 'afno':
        model = AFNONet(params).to(device) 
    model = PrecipNet(params, backbone=model).to(device)
    checkpoint_file  = params['best_checkpoint_path']
    model = load_model(model, params, checkpoint_file)
    model = model.to(device)

    # load the validation data
    files_paths = glob.glob(params.inf_data_path + "/*.h5")
    files_paths.sort()
    if params.log_to_screen:
        logging.info('Loading validation data')
        logging.info('Validation data from {}'.format(files_paths[0]))
    valid_data_full = h5py.File(files_paths[0], 'r')['fields']
    # precip paths
    path = params.precip + '/out_of_sample'
    precip_paths = glob.glob(path + "/*.h5")
    precip_paths.sort()
    if params.log_to_screen:
        logging.info('Loading validation precip data')
        logging.info('Validation data from {}'.format(precip_paths[0]))
    valid_data_tp_full = h5py.File(precip_paths[0], 'r')['tp']

    return valid_data_full, valid_data_tp_full, model_wind, model

def autoregressive_inference(params, ic, valid_data_full, valid_data_tp_full, model_wind, model):
    ic = int(ic) 
    #initialize global variables
    device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
    exp_dir = params['experiment_dir'] 
    dt = int(params.dt)
    prediction_length = int(params.prediction_length/dt)
    n_history = params.n_history
    img_shape_x = params.img_shape_x
    img_shape_y = params.img_shape_y
    in_channels = np.array(params.in_channels)
    out_channels = np.array(params.out_channels)
    n_in_channels = len(in_channels)
    n_out_channels = len(out_channels)
    means = params.means
    stds = params.stds

    n_pert = params.n_pert

    #initialize memory for image sequences and RMSE/ACC
    valid_loss = torch.zeros((prediction_length, n_out_channels)).to(device, dtype=torch.float)
    acc = torch.zeros((prediction_length, n_out_channels)).to(device, dtype=torch.float)
    acc_unweighted = torch.zeros((prediction_length, n_out_channels)).to(device, dtype=torch.float)
    tqe = torch.zeros((prediction_length, n_out_channels)).to(device, dtype=torch.float)
    # wind seqs
    seq_real = torch.zeros((prediction_length+n_history, n_in_channels, img_shape_x, img_shape_y)).to(device, dtype=torch.float)
    seq_pred = torch.zeros((prediction_length+n_history, n_in_channels, img_shape_x, img_shape_y)).to(device, dtype=torch.float)
    # precip sequences
    seq_real_tp = torch.zeros((prediction_length, n_out_channels, img_shape_x, img_shape_y)).to(device, dtype=torch.float)
    seq_pred_tp = torch.zeros((prediction_length, n_out_channels, img_shape_x, img_shape_y)).to(device, dtype=torch.float)

    valid_data = valid_data_full[ic:(ic+prediction_length*dt+n_history*dt):dt, in_channels, 0:720] #extract valid data from first year
    # standardize
    valid_data = (valid_data - means)/stds
    valid_data = torch.as_tensor(valid_data).to(device, dtype=torch.float)
    len_ic = prediction_length*dt
    valid_data_tp = valid_data_tp_full[ic:(ic+prediction_length*dt):dt, 0:720].reshape(len_ic,n_out_channels,720,img_shape_y) #extract valid data from first year
    # log normalize
    eps = params.precip_eps
    valid_data_tp = np.log1p(valid_data_tp/eps)
    valid_data_tp = torch.as_tensor(valid_data_tp).to(device, dtype=torch.float)
    #load time means
    m = torch.as_tensor(np.load(params.time_means_path_tp)[0][out_channels])[:, 0:img_shape_x] # climatology
    m = torch.unsqueeze(m, 0)
    m = m.to(device)

    if params.log_to_screen:
      logging.info('Begin autoregressive+tp inference')

    for pert in range(n_pert):
      logging.info('Running ensemble {}/{}'.format(pert+1, n_pert))
      with torch.no_grad():
        for i in range(valid_data.shape[0]): 
          if i==0: #start of sequence
            first = valid_data[0:n_history+1]
            first_tp = valid_data_tp[0:1]
            future = valid_data[n_history+1]
            future_tp = valid_data_tp[1]
            for h in range(n_history+1):
              seq_real[h] = first[h*n_in_channels:(h+1)*n_in_channels][0:n_in_channels] #extract history from 1st 
              seq_pred[h] = seq_real[h]
            seq_real_tp[0] = unlog_tp_torch(first_tp)
            seq_pred_tp[0] = unlog_tp_torch(first_tp)
            first = gaussian_perturb(first, level=params.n_level, device=device) # perturb the ic
            future_pred = model_wind(first)
            future_pred_tp = model(future_pred)
          else:
            if i < prediction_length-1:
              future = valid_data[n_history+i+1]
              future_tp = valid_data_tp[i+1]
            future_pred = model_wind(future_pred) #autoregressive step
            future_pred_tp = model(future_pred) # tp diagnosis

          if i < prediction_length-1: #not on the last step
            seq_pred_tp[n_history+i+1] += unlog_tp_torch(torch.squeeze(future_pred_tp,0))  # add up predictions and average later
            seq_real_tp[n_history+i+1] += unlog_tp_torch(future_tp)
      
    #Compute metrics 
    for i in range(valid_data.shape[0]): 
      if i>0:
        # avg images
        seq_pred_tp[i] /= n_pert
        seq_real_tp[i] /= n_pert
      pred = torch.unsqueeze(seq_pred_tp[i], 0)
      tar = torch.unsqueeze(seq_real_tp[i], 0)
      valid_loss[i] = weighted_rmse_torch_channels(pred, tar)
      acc[i] = weighted_acc_torch_channels(pred-m, tar-m)
      tqe[i] = top_quantiles_error_torch(pred, tar)

      if params.log_to_screen:
        logging.info('Timestep {} of {}. TP RMS Error: {}, ACC: {}'.format((i), prediction_length, valid_loss[i,0], acc[i,0]))
      
    seq_real_tp = seq_real_tp.cpu().numpy()
    seq_pred_tp = seq_pred_tp.cpu().numpy()
    valid_loss = valid_loss.cpu().numpy()
    acc = acc.cpu().numpy()
    acc_unweighted = acc_unweighted.cpu().numpy()
    tqe = tqe.cpu().numpy()
    return np.expand_dims(valid_loss, 0), \
           np.expand_dims(acc, 0), \
           np.expand_dims(tqe, 0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_num", default='00', type=str)
    parser.add_argument("--yaml_config", default='./config/AFNO.yaml', type=str)
    parser.add_argument("--config", default='full_field', type=str)
    parser.add_argument("--n_level", default = 0.3, type = float)
    parser.add_argument("--n_pert", default=100, type=int)
    parser.add_argument("--override_dir", default=None, type = str, help = 'Path to store inference outputs; must also set --weights arg')
    parser.add_argument("--weights", default=None, type=str, help = 'Path to model weights, for use with override_dir option')

    args = parser.parse_args()
    params = YParams(os.path.abspath(args.yaml_config), args.config)
    params['world_size'] = 1
    if 'WORLD_SIZE' in os.environ:
      params['world_size'] = int(os.environ['WORLD_SIZE'])
    world_rank = 0
    local_rank = 0
    if params['world_size'] > 1:
      local_rank = int(os.environ["LOCAL_RANK"])
      dist.init_process_group(backend='nccl', init_method='env://')
      args.gpu = local_rank
      world_rank = dist.get_rank()
      world_size = dist.get_world_size()
      params['global_batch_size'] = params.batch_size
#      params['batch_size'] = int(params.batch_size//params['world_size'])
    torch.cuda.set_device(local_rank)
    torch.backends.cudnn.benchmark = True

    # Set up directory
    if args.override_dir is not None:
      assert args.weights is not None, 'Must set --weights argument if using --override_dir'
      expDir = args.override_dir
    else: 
      assert args.weights is None, 'Cannot use --weights argument without also using --override_dir'
      expDir = os.path.join(params.exp_dir, args.config, str(args.run_num))

    if  world_rank==0:
      if not os.path.isdir(expDir):
        os.makedirs(expDir)

    params['experiment_dir'] = os.path.abspath(expDir)
    params['best_checkpoint_path'] = args.weights if args.override_dir is not None else os.path.join(expDir, 'training_checkpoints/best_ckpt.tar')
    args.resuming = False
    params['resuming'] = args.resuming
    params['local_rank'] = local_rank
    # this will be the wandb name
    params['name'] = args.config + '_' + str(args.run_num)
    params['group'] = args.config
    if world_rank==0:
      logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, 'inference_out.log'))
      logging_utils.log_versions()
      params.log()
    params['log_to_wandb'] = (world_rank==0) and params['log_to_wandb']
    params['log_to_screen'] = (world_rank==0) and params['log_to_screen']

    n_ics = params['n_initial_conditions']

    if params["ics_type"] == 'default':
        num_samples = 1460 - params.prediction_length
        stop = num_samples
        ics = np.arange(0, stop, DECORRELATION_TIME)
        n_ics = len(ics)
        logging.info("Inference for {} initial conditions".format(n_ics))
    elif params["ics_type"] == "datetime":
        date_strings = params["date_strings"]
        ics = []
        for date in date_strings:
            date_obj = datetime.strptime(date,'%Y-%m-%d %H:%M:%S') 
            day_of_year = date_obj.timetuple().tm_yday - 1
            hour_of_day = date_obj.timetuple().tm_hour
            hours_since_jan_01_epoch = 24*day_of_year + hour_of_day
            ics.append(int(hours_since_jan_01_epoch/6))

    try:
        autoregressive_inference_filetag = params["inference_file_tag"]
    except:
        autoregressive_inference_filetag = ""

    n_level = args.n_level
    params.n_level = n_level
    params.n_pert = args.n_pert
    logging.info("Doing level = {}".format(n_level))
    autoregressive_inference_filetag += "_" + str(params.n_level) + "_" + str(params.n_pert) + "ens_tp"

    # get data and models
    valid_data_full, valid_data_tp_full, model_wind, model = setup(params)

    #initialize lists for image sequences and RMSE/ACC
    valid_loss = np.zeros
    acc = []
    tqe = []

    # run autoregressive inference for multiple initial conditions
    # parallelize over initial conditions
    if world_size > 1:
      tot_ics = len(ics)
      ics_per_proc = n_ics//world_size
      ics = ics[ics_per_proc*world_rank:ics_per_proc*(world_rank+1)] if world_rank < world_size - 1 else ics[(world_size - 1)*ics_per_proc:]
      n_ics = len(ics)
      logging.info('Rank %d running ics %s'%(world_rank, str(ics)))

    for i, ic in enumerate(ics):
      t1 = time.time()
      logging.info("Initial condition {} of {}".format(i+1, n_ics))
      vl, a, tq = autoregressive_inference(params, ic, valid_data_full, valid_data_tp_full, model_wind, model)
      if i == 0:
        valid_loss = vl
        acc = a
        tqe = tq
      else:
        valid_loss = np.concatenate((valid_loss, vl), 0)
        acc = np.concatenate((acc, a), 0)
        tqe = np.concatenate((tqe, tq), 0)
      t2 = time.time()-t1
      logging.info("Time for inference for ic {} = {}".format(i, t2))

    prediction_length = acc[0].shape[0]
    n_out_channels = acc[0].shape[1]

    #save predictions and loss
       
    #save predictions and loss
    h5name = os.path.join(params['experiment_dir'], 'ens_autoregressive_predictions'+ autoregressive_inference_filetag +'.h5')
    if dist.is_initialized(): 
      if params.log_to_screen:
        logging.info("Saving files at {}".format(h5name))
        logging.info("array shapes: %s"%str((tot_ics, prediction_length, n_out_channels)))

      dist.barrier()
      from mpi4py import MPI
      with h5py.File(h5name, 'a', driver='mpio', comm=MPI.COMM_WORLD) as f:
        if "rmse" in f.keys() or "acc" in f.keys():
          del f["acc"]
          del f["rmse"]
        f.create_dataset("rmse", shape = (tot_ics, prediction_length, n_out_channels), dtype =np.float32)
        f.create_dataset("acc",  shape = (tot_ics, prediction_length, n_out_channels), dtype =np.float32)
        start = world_rank*ics_per_proc
        f["rmse"][start:start+n_ics] = valid_loss
        f["acc"][start:start+n_ics] = acc
      dist.barrier()
    else:
      if params.log_to_screen:
        logging.info("Saving files at {}".format(os.path.join(params['experiment_dir'], 'ens_autoregressive_predictions' + autoregressive_inference_filetag + '.h5')))
      with h5py.File(os.path.join(params['experiment_dir'], 'ens_autoregressive_predictions'+ autoregressive_inference_filetag +'.h5'), 'a') as f:
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
        try:
            f.create_dataset("tqe", data = tqe, shape = (n_ics, prediction_length, n_out_channels), dtype =np.float32)
        except:
            del f["tqe"]
            f.create_dataset("tqe", data = tqe, shape = (n_ics, prediction_length, n_out_channels), dtype =np.float32)
            f["tqe"][...] = tqe
