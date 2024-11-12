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
import h5py
import torch
import cProfile
import re
import torchvision
from torchvision.utils import save_image
import torch.nn as nn
import torch.cuda.amp as amp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import logging
from utils import logging_utils
logging_utils.config_logger()
from utils.YParams import YParams
from utils.data_loader_multifiles import get_data_loader
from networks.afnonet import AFNONet, PrecipNet
from networks.swinv2 import swinv2net
from utils.img_utils import vis
import wandb
from utils.weighted_acc_rmse import weighted_acc, weighted_rmse, weighted_rmse_torch, unlog_tp_torch
from apex import optimizers
from utils.darcy_loss import LpLoss, weighted_rmse_loss
import matplotlib.pyplot as plt
from collections import OrderedDict
import pickle
DECORRELATION_TIME = 36 # 9 days
import json
from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap as ruamelDict
from typing import Callable, Any

def ckpt_identity(layer: Callable, *args: Any, **kwargs: Any) -> Any:
    """Identity function for when activation checkpointing is not needed"""
    return layer(*args)

def set_seed(params, world_size):
    seed = params.seed
    if seed is None:
        seed = np.random.randint(10000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if world_size > 0:
        torch.cuda.manual_seed_all(seed)

class Trainer():
  def count_parameters(self):
    return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

  def __init__(self, params, args):
    self.sweep_id = args.sweep_id
    self.root_dir = params['exp_dir'] 
    self.config = args.config
    params.activation_ckpt = True

    params['enable_amp'] = args.enable_amp
    params['epsilon_factor'] = args.epsilon_factor

    self.world_size = 1
    if 'WORLD_SIZE' in os.environ:
      self.world_size = int(os.environ['WORLD_SIZE'])

    self.local_rank = 0
    self.world_rank = 0
    if self.world_size > 1:
      dist.init_process_group(backend='nccl',
                              init_method='env://')
      self.world_rank = dist.get_rank()
      self.local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(self.local_rank)
    torch.backends.cudnn.benchmark = True
    
    self.log_to_screen = params.log_to_screen and self.world_rank==0
    self.log_to_wandb = params.log_to_wandb and self.world_rank==0
  
    self.device = torch.cuda.current_device()
    self.params = params
    self.params.device = self.device

    self.params['name'] = args.config + '_' + str(args.run_num)
    self.params['group'] = "era5_" + args.config

    self.config = args.config 
    self.run_num = args.run_num
    self.ckpt_fn = torch.utils.checkpoint.checkpoint if params.activation_ckpt else ckpt_identity

  def build_and_launch(self):
    self.params['in_channels'] = np.array(self.params['in_channels'])
    self.params['out_channels'] = np.array(self.params['out_channels'])
    self.params['N_in_channels'] = len(self.params['in_channels'])
    self.params['N_out_channels'] = len(self.params['out_channels'])

    if params.add_zenith:
        params.N_in_channels += 1

    # init wandb
    exp_dir = os.path.join(*[self.root_dir, self.config, self.run_num])


    self.params['experiment_dir'] = os.path.abspath(exp_dir)
    self.params['checkpoint_path'] = os.path.join(exp_dir, 'training_checkpoints/ckpt.tar')
    self.params['best_checkpoint_path'] = os.path.join(exp_dir, 'training_checkpoints/best_ckpt.tar')
    self.params['resuming'] = True if os.path.isfile(self.params.checkpoint_path) else False
    if self.log_to_screen:
        logging.info(self.params.log())

    self.params['global_batch_size'] = self.params.batch_size
    self.params['batch_size'] = 1 #int(self.params.batch_size//self.world_size)
    self.params.two_step_training = True


    #self.train_data_loader, self.train_dataset, self.train_sampler = get_data_loader(self.params, self.params.train_data_path, dist.is_initialized(), train=True)
    self.valid_data_loader, self.valid_dataset = get_data_loader(self.params, self.params.valid_data_path, dist.is_initialized(), train=False)

    if self.params.rmse_loss:
      self.loss_obj = weighted_rmse_loss
    else:
      self.loss_obj = LpLoss(relative=params.relative_loss)

    logging.info('rank %d, data loader initialized'%self.world_rank)

    self.params.crop_size_x = self.valid_dataset.crop_size_x
    self.params.crop_size_y = self.valid_dataset.crop_size_y
    self.params.img_shape_x = self.valid_dataset.img_shape_x
    self.params.img_shape_y = self.valid_dataset.img_shape_y

    if self.params.nettype == 'afno':
      self.model = AFNONet(self.params).to(self.device)
    elif self.params.nettype == 'swin':
      self.model = swinv2net(self.params).to(self.device)
    else:
      raise Exception("not implemented")

    if self.params.optimizer_type == 'FusedAdam':
      self.optimizer = optimizers.FusedAdam(self.model.parameters(), lr = self.params.lr)
    elif self.params.optimizer_type == 'FusedLAMB':
      self.optimizer = optimizers.FusedLAMB(self.model.parameters(), lr = self.params.lr, weight_decay=self.params.weight_decay, max_grad_norm=5.)
    elif self.params.optimizer_type == 'AdamW':
      self.optimizer = torch.optim.AdamW(self.model.parameters(), lr =self.params.lr, weight_decay=self.params.weight_decay)
    else:
      self.optimizer = torch.optim.Adam(self.model.parameters(), lr =self.params.lr)

    if self.params.enable_nhwc:
      # NHWC: Convert model to channels_last memory format
      self.model = self.model.to(memory_format=torch.channels_last)

    if self.params.enable_amp == True:
      self.gscaler = amp.GradScaler()

    if dist.is_initialized():
      self.model = DistributedDataParallel(self.model,
                                           device_ids=[self.local_rank],
                                           output_device=[self.local_rank],find_unused_parameters=False)

    self.iters = 0
    if self.params.resuming:
      logging.info("Loading checkpoint %s"%self.params.checkpoint_path)
      self.restore_checkpoint(self.params.checkpoint_path)

            
    if self.log_to_screen:
      #logging.info(self.model)
      logging.info("Number of trainable model parameters: {}".format(self.count_parameters()))

    # launch training
    self.compute_grads()


  def compute_grad_norm(self, p_list):
    norm_type = 2.0
    grads = [p.grad for p in p_list if p.grad is not None]
    norm_per = torch.stack([torch.norm(g.detach(), norm_type).to(self.params.device) for g in grads])
    total_norm = torch.norm(norm_per, norm_type)
    return total_norm.detach().cpu(), norm_per.detach().cpu().numpy()

  def switch_off_grad(self, model):
    for param in model.parameters():
      param.requires_grad = False


  def compute_grads(self):
    tr_time = 0
    data_time = 0
    self.model.train()
    grad1 = []
    grad2 = []
    pgrads = []

    self.iters = 0
    for i in range(0, 2900,100):
      self.iters += 1
      data = self.valid_dataset[i]
      logging.info('Step %d, sample %d'%(self.iters, i)) 
      data_start = time.time()
      inp, tar = map(lambda x: x.to(self.device, dtype = torch.float).unsqueeze(0), data)      
      if self.params.orography and self.params.two_step_training:
          orog = inp[:,-2:-1] 


      if self.params.enable_nhwc:
        inp = inp.to(memory_format=torch.channels_last)
        tar = tar.to(memory_format=torch.channels_last)


      if 'residual_field' in self.params.target:
        tar -= inp[:, 0:tar.size()[1]]
      data_time += time.time() - data_start

      tr_start = time.time()

      self.model.zero_grad()
      with amp.autocast(self.params.enable_amp):
        inp = inp.clone().detach().requires_grad_(True)
        gen_step_one = self.ckpt_fn(self.model,
                                    inp, 
                                    use_reentrant=False,
                                    preserve_rng_state=False).to(self.device, dtype = torch.float)
        gen_step_one_next = gen_step_one.detach()
        loss_step_one = self.loss_obj(gen_step_one, tar[:,0:self.params.N_out_channels])
        grad1.append(self.compute_grads_between(loss_step_one, inp))
        self.model.zero_grad()
        del gen_step_one, inp, loss_step_one
        

        if self.params.orography:
            gen_step_one_input = torch.cat( (gen_step_one_next, orog), axis = 1)
        elif params.add_zenith:
            zenith = tar[:,2*self.params.N_out_channels:2*self.params.N_out_channels+2]
            gen_step_one_input = torch.cat([gen_step_one_next, zenith], dim= 1)
        gen_step_one_input =  gen_step_one_input.clone().detach().requires_grad_(True)
        gen_step_two = self.ckpt_fn(self.model,
                                    gen_step_one_input, 
                                    use_reentrant=False,
                                    preserve_rng_state=False).to(self.device, dtype = torch.float)
        loss_step_two = self.loss_obj(gen_step_two, tar[:,self.params.N_out_channels:2*self.params.N_out_channels])
        grad2.append(self.compute_grads_between(loss_step_two, gen_step_one_input))
        g_norm, grads = self.compute_grad_norm(self.model.parameters())
        pgrads.append(grads)
        del gen_step_two, gen_step_one_input, loss_step_two, gen_step_one_next

      tr_time += time.time() - tr_start
    
    self.save_outputs(grad1, grad2, pgrads)
    logging.info('DONE')
    return


  def compute_grads_between(self, l, x, ret_graph=False):
      if self.params.enable_amp:
          self.gscaler.scale(l).backward(retain_graph=ret_graph)
          self.gscaler.step(self.optimizer)
          grads = x.grad.detach().cpu().numpy()/self.gscaler.get_scale()
          self.gscaler.update()
      else:
          l.backward(retain_graph=ret_graph)
          grads = x.grad.detach().cpu().numpy()
      return grads


  def save_outputs(self, grad1, grad2, pgrads):
      grad1 = np.concatenate(grad1)
      grad2 = np.concatenate(grad2)
      pgrads = np.concatenate(pgrads)

      for g, n in zip([grad1, grad2, pgrads], ['step1', 'step2', 'modelparams']):
          np.save(os.path.join(self.params.experiment_dir, 'grads_%s.npy'%n), g)

    

  def restore_checkpoint(self, checkpoint_path):
    """ We intentionally require a checkpoint_dir to be passed
        in order to allow Ray Tune to use this function """
    checkpoint = torch.load(checkpoint_path, map_location='cuda:{}'.format(self.local_rank))
    try:
        self.model.load_state_dict(checkpoint['model_state'])
    except:
        new_state_dict = OrderedDict()
        for key, val in checkpoint['model_state'].items():
            name = key[7:]
            new_state_dict[name] = val 
        self.model.load_state_dict(new_state_dict)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--run_num", default='00', type=str)
  parser.add_argument("--yaml_config", default='./config/AFNO.yaml', type=str)
  parser.add_argument("--config", default='default', type=str)
  parser.add_argument("--enable_amp", action='store_true')
  parser.add_argument("--epsilon_factor", default = 0, type = float)
  parser.add_argument("--sweep_id", default=None, type=str, help='sweep config from ./configs/sweeps.yaml')

  args = parser.parse_args()

  params = YParams(os.path.abspath(args.yaml_config), args.config)
  trainer = Trainer(params, args)

  trainer.build_and_launch()

  if dist.is_initialized():
      dist.barrier()
  logging.info('DONE')
