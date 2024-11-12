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
from utils.data_loader_hrrr_era5 import get_data_loader, get_absolute_path
from networks.swinv2_hrrr import swinv2net
from utils.img_utils import vis
from utils.wandb import get_wandb_names
import wandb
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

    # wandb setup
    if self.log_to_wandb:
      entity, project = get_wandb_names()
      self.params.entity = entity
      self.params.project = project
      self.params['name'] = args.config + '_' + str(args.run_num)
      self.params['group'] = args.config

    self.config = args.config 
    self.run_num = args.run_num
    self.ckpt_fn = torch.utils.checkpoint.checkpoint if params.activation_ckpt else ckpt_identity

  def build_and_launch(self):

    if self.params.boundary_padding_pixels > 0:
      self.params.era5_img_size = (self.params.hrrr_img_size[0] + 2*self.params.boundary_padding_pixels, self.params.hrrr_img_size[1] + 2*self.params.boundary_padding_pixels)
    else:
      self.params.era5_img_size = self.params.hrrr_img_size

    if self.params.add_zenith:
        params.N_in_channels += 1

    # init wandb
    if self.sweep_id:
      jid = os.environ['SLURM_JOBID'] # so different sweeps dont resume
      exp_dir = os.path.join(*[self.root_dir, 'sweeps', self.sweep_id, self.config, jid])
    else:
      exp_dir = os.path.join(*[self.root_dir, self.config, self.run_num])

    if self.world_rank==0:
      if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)
        os.makedirs(os.path.join(exp_dir, 'training_checkpoints/'))
        os.makedirs(os.path.join(exp_dir, 'wandb/'))

    self.params['experiment_dir'] = os.path.abspath(exp_dir)
    self.params['checkpoint_path'] = os.path.join(exp_dir, 'training_checkpoints/ckpt.tar')
    self.params['best_checkpoint_path'] = os.path.join(exp_dir, 'training_checkpoints/best_ckpt.tar')
    self.params['resuming'] = True if os.path.isfile(self.params.checkpoint_path) else False
    if self.log_to_wandb:
      if self.sweep_id:
        wandb.init(dir=os.path.join(exp_dir, "wandb"))
        hpo_config = wandb.config
        self.params.update_params(hpo_config)
        logging.info('HPO sweep %s, trial params:'%self.sweep_id)
        logging.info(self.params.log())
      else:
        wandb.init(dir=os.path.join(exp_dir, "wandb"),
                    config=self.params.params, name=self.params.name, group=self.params.group, project=self.params.project, 
                    entity=self.params.entity, resume=self.params.resuming)
        logging.info(self.params.log())

    if self.sweep_id and dist.is_initialized():
      from mpi4py import MPI
      comm = MPI.COMM_WORLD
      rank = comm.Get_rank()
      assert self.world_rank == rank
      if rank != 0: 
        self.params = None
      self.params = comm.bcast(self.params, root=0)
      self.params.device = self.device # dont broadcast 0s device

    # set_seed(self.params, self.world_size)

    if self.world_rank==0:
      logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(exp_dir, 'out.log'))
      logging_utils.log_versions()

    self.params['global_batch_size'] = self.params.batch_size
    self.params['local_batch_size'] = int(self.params.batch_size//self.world_size)
    logging.info('rank %d, global batch size %d, local batch size %d'%(self.world_rank, self.params.global_batch_size, self.params.local_batch_size))

    # dump the yaml used
    if self.world_rank == 0:
      hparams = ruamelDict()
      yaml = YAML()
      for key, value in self.params.params.items():
        hparams[str(key)] = value.tolist() if isinstance(value, np.ndarray) else value
        with open(os.path.join(self.params['experiment_dir'], 'hyperparams.yaml'), 'w') as hpfile:
            yaml.dump(hparams, hpfile)

    logging.info('rank %d, begin data loader init'%self.world_rank)
    self.train_data_loader, self.train_dataset, self.train_sampler = get_data_loader(self.params, dist.is_initialized(), train=True)
    self.valid_data_loader, self.valid_dataset = get_data_loader(self.params, dist.is_initialized(), train=False)

    self.base_hrrr_channels, self.kept_hrrr_channels = self.train_dataset._get_hrrr_channel_names()

    logging.info("training on hrrr channels: %s"%self.kept_hrrr_channels)

    if self.params.rmse_loss:
      #warn
      logging.warning('Using weighted rmse loss, latitude weighting is not implemented for HRRR')
      self.loss_obj = weighted_rmse_loss
    else:
      self.loss_obj = LpLoss(relative=params.relative_loss)

    logging.info('rank %d, data loader initialized'%self.world_rank)

    if self.params.nettype == 'afno':
      self.model = AFNONet(self.params).to(self.device) 
    elif self.params.nettype == 'swin_hrrr':
      self.model = swinv2net(self.params).to(self.device)
    else:
      raise Exception("not implemented")
     
    if self.params.enable_nhwc:
      # NHWC: Convert model to channels_last memory format
      self.model = self.model.to(memory_format=torch.channels_last)

    if self.log_to_wandb:
      wandb.watch(self.model)

    if self.params.optimizer_type == 'FusedAdam':
      self.optimizer = optimizers.FusedAdam(self.model.parameters(), lr = self.params.lr, betas=(0.9, 0.95))
    elif self.params.optimizer_type == 'FusedLAMB':
      self.optimizer = optimizers.FusedLAMB(self.model.parameters(), lr = self.params.lr, weight_decay=self.params.weight_decay, max_grad_norm=5.)
    elif self.params.optimizer_type == 'AdamW':
      self.optimizer = torch.optim.AdamW(self.model.parameters(), lr =self.params.lr, weight_decay=self.params.weight_decay)
    else:
      self.optimizer = torch.optim.Adam(self.model.parameters(), lr =self.params.lr, weight_decay=self.params.weight_decay, betas=(0.9, 0.95))

    if self.params.enable_amp == True:
      self.gscaler = amp.GradScaler()

    if dist.is_initialized():
      self.model = DistributedDataParallel(self.model,
                                           device_ids=[self.local_rank],
                                           output_device=[self.local_rank],find_unused_parameters=False)

    self.iters = 0
    self.startEpoch = 0
    if self.params.resuming:
      logging.info("Loading checkpoint %s"%self.params.checkpoint_path)
      self.restore_checkpoint(self.params.checkpoint_path)

    if self.params.two_step_training:
      if self.params.resuming == False and self.params.pretrained == True:
        logging.info("Starting from pretrained one-step afno model at %s"%self.params.pretrained_ckpt_path)
        self.restore_checkpoint(self.params.pretrained_ckpt_path)
        self.iters = 0
        self.startEpoch = 0

            
    self.epoch = self.startEpoch

    if self.params.scheduler == 'ReduceLROnPlateau':
      self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.2, patience=5, mode='min')
    elif self.params.scheduler == 'CosineAnnealingLR':
      self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.params.max_epochs, last_epoch=self.startEpoch-1)
    else:
      self.scheduler = None

    if self.log_to_screen:
      #logging.info(self.model)
      logging.info("Number of trainable model parameters: {}".format(self.count_parameters()))

    # launch training
    self.train()


  def compute_grad_norm(self, p_list):
    norm_type = 2.0
    grads = [p.grad for p in p_list if p.grad is not None]
    total_norm = torch.norm(torch.stack([torch.norm(g.detach(), norm_type).to(self.params.device) for g in grads]), norm_type)
    return total_norm

  def switch_off_grad(self, model):
    for param in model.parameters():
      param.requires_grad = False

  def train(self):
    if self.log_to_screen:
      logging.info("Starting Training Loop...")

    best_valid_loss = 1.e6
    for epoch in range(self.startEpoch, self.params.max_epochs):
      if dist.is_initialized():
        self.train_sampler.set_epoch(epoch)
#        self.valid_sampler.set_epoch(epoch)

      start = time.time()
      tr_time, data_time, train_logs = self.train_one_epoch()
      valid_time, valid_logs = self.validate_one_epoch()

      if self.params.scheduler == 'ReduceLROnPlateau':
        self.scheduler.step(valid_logs['valid_loss'])
      elif self.params.scheduler == 'CosineAnnealingLR':
        self.scheduler.step()
        if self.epoch >= self.params.max_epochs:
          logging.info("Terminating training after reaching params.max_epochs while LR scheduler is set to CosineAnnealingLR")
          exit()

      if self.log_to_wandb:
        for pg in self.optimizer.param_groups:
          lr = pg['lr']
        wandb.log({'lr': lr})

      if self.world_rank == 0:
        if self.params.save_checkpoint:
          #checkpoint at the end of every epoch
          self.save_checkpoint(self.params.checkpoint_path)
          if valid_logs['valid_loss'] <= best_valid_loss:
            #logging.info('Val loss improved from {} to {}'.format(best_valid_loss, valid_logs['valid_loss']))
            self.save_checkpoint(self.params.best_checkpoint_path)
            best_valid_loss = valid_logs['valid_loss']

      if self.log_to_screen:
        logging.info('Time taken for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
        logging.info('Train loss: {}. Valid loss: {}'.format(train_logs['loss'], valid_logs['valid_loss']))


  def train_one_epoch(self):
    self.epoch += 1
    tr_time = 0
    data_time = 0
    self.model.train()
    
    for i, data in enumerate(self.train_data_loader, 0):

      self.iters += 1

      if i % self.params.print_every == 0 and self.world_rank == 0:
        logging.info("iteration: {}, target: {}".format(i, self.params.target))
      
      # adjust_LR(optimizer, params, iters)
      data_start = time.time()
      #inp, tar = map(lambda x: x.to(self.device, dtype = torch.float), data)      
      #data: dict: {'hrrr': (inp, tar), 'era5': (inp, tar)}
      inp, tar = data['hrrr']
      inp, tar = inp.to(self.device, dtype = torch.float), tar.to(self.device, dtype = torch.float)
      boundary = data['era5'][0].to(self.device, dtype = torch.float)

      if self.params.orography and self.params.two_step_training:
          orog = inp[:,-2:-1] 


      if self.params.enable_nhwc:
        inp = inp.to(memory_format=torch.channels_last)
        tar = tar.to(memory_format=torch.channels_last)

      data_time += time.time() - data_start

      tr_start = time.time()

      if self.params.mask_ratio > 0:
          # Randomly select some channels to mask out, compute loss only on those
          mask = torch.zeros(inp.shape, device=self.device)
          b, c, _, _ = inp.shape
          ch_mask = np.random.choice(np.arange(self.params.n_hrrr_channels), (b,int(params.mask_ratio*c)), replace=False)
          mask[:, ch_mask] = 1.
          loss_mask = mask # or torch.ones(inp.shape, dtype=torch.bool) to compute over all channels
      else:
          # Nothing is masked, loss computed everywhere
          mask = torch.ones(inp.shape, device=self.device)
          loss_mask = mask


      self.model.zero_grad()
      if self.params.two_step_training:
          with amp.autocast(self.params.enable_amp):
            gen_step_one = self.ckpt_fn(self.model,
                                        inp,
                                        use_reentrant=False,
                                        ).to(self.device, dtype = torch.float)
            loss_step_one = self.loss_obj(gen_step_one, tar[:,0:self.params.n_hrrr_channels])
            if self.params.orography:
                gen_step_two = self.ckpt_fn(self.model,
                                            torch.cat( (gen_step_one, orog), axis = 1),
                                            use_reentrant=False,
                                            ).to(self.device, dtype = torch.float)
            elif self.params.add_zenith:
                zenith = tar[:,2*self.params.n_hrrr_channels:2*self.params.n_hrrr_channels+2]
                gen_step_one_zenith = torch.cat([gen_step_one, zenith], dim= 1)
                gen_step_two = self.ckpt_fn(self.model,
                                            gen_step_one_zenith, 
                                            use_reentrant=False,
                                            ).to(self.device, dtype = torch.float)
            else:
                gen_step_two = self.ckpt_fn(self.model,
                                            gen_step_one,
                                            use_reentrant=False,
                                            ).to(self.device, dtype = torch.float)
            loss_step_two = self.loss_obj(gen_step_two, tar[:,self.params.n_hrrr_channels:2*self.params.n_hrrr_channels])
            loss = loss_step_one + loss_step_two
      else:
          with amp.autocast(self.params.enable_amp):
            gen = self.model(inp, boundary, mask) #.to(self.device, dtype = torch.float)
            loss = self.loss_obj(gen*loss_mask, tar*loss_mask)

      for param in self.model.parameters():
        if param.grad is not None:
          torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)

      if self.params.enable_amp:
        self.gscaler.scale(loss).backward()
        self.gscaler.step(self.optimizer)
      else:
        loss.backward()
        self.optimizer.step()

      if self.params.enable_amp:
        self.gscaler.update()

      g_norm = self.compute_grad_norm(self.model.parameters())

      tr_time += time.time() - tr_start
    
    try:
        logs = {'loss': loss, 'loss_step_one': loss_step_one, 'loss_step_two': loss_step_two, 'grad_norm': g_norm}
    except:
        logs = {'loss': loss, 'grad_norm': g_norm}

    if dist.is_initialized():
      for key in sorted(logs.keys()):
        dist.all_reduce(logs[key].detach())
        logs[key] = float(logs[key]/dist.get_world_size())

    if self.log_to_wandb:
      wandb.log(logs, step=self.epoch)

    return tr_time, data_time, logs

  def validate_one_epoch(self):
    self.model.eval()
    n_valid_batches = 20 #do validation on first 20 images, just for LR scheduler
    stats_location = get_absolute_path(os.path.join(self.params.location, self.params.conus_dataset_name, self.params.hrrr_stats))
    global_stds = np.load(os.path.join(stats_location, 'stds.npy'))
    global_means = np.load(os.path.join(stats_location, 'means.npy'))

    if len(self.params.exclude_channels) > 0:

      excluded_indices = [self.base_hrrr_channels.index(ch) for ch in self.params.exclude_channels]
      global_stds = np.delete(global_stds, excluded_indices)
      global_means = np.delete(global_means, excluded_indices)

    if self.params.target == 'residual_field':
      tendency_stds = np.load(get_absolute_path(os.path.join(self.params.location, self.params.conus_dataset_name, 'stats_tendency/stds.npy')))
      if len(self.params.exclude_channels) > 0:
        tendency_stds = np.delete(tendency_stds, excluded_indices)
      tendency_stds = torch.as_tensor(tendency_stds).to(self.device)
      mult = tendency_stds
      mult_stds = torch.as_tensor(global_stds).to(self.device)
      ratio_factor = tendency_stds.unsqueeze(0).unsqueeze(2).unsqueeze(3)/mult_stds.unsqueeze(0).unsqueeze(2).unsqueeze(3)
      mult_rmse = mult_stds if self.params.log_scale else mult
    else:
      mult_rmse = torch.as_tensor(global_stds).to(self.device)



    valid_buff = torch.zeros((3), dtype=torch.float32, device=self.device)
    valid_loss = valid_buff[0].view(-1)
    valid_l1 = valid_buff[1].view(-1)
    valid_steps = valid_buff[2].view(-1)
    valid_rmse = torch.zeros((self.params.n_hrrr_channels), dtype=torch.float32, device=self.device)
    valid_acc = torch.zeros((self.params.n_hrrr_channels), dtype=torch.float32, device=self.device)

    valid_start = time.time()

    sample_idx = np.random.randint((n_valid_batches))
    with torch.no_grad():
      for i, data in enumerate(self.valid_data_loader, 0):
        if i>=n_valid_batches:
          break    

        inp, tar = data['hrrr']
        inp, tar = inp.to(self.device, dtype = torch.float), tar.to(self.device, dtype = torch.float)
        boundary = data['era5'][0].to(self.device, dtype = torch.float)

        if self.params.orography and self.params.two_step_training:
            orog = inp[:,-2:-1]


        if self.params.mask_ratio > 0:
            # Randomly select some channels to mask out, compute loss only on those
            mask = torch.zeros(inp.shape, device=self.device)
            b, c, _, _ = inp.shape
            ch_mask = np.random.choice(np.arange(self.params.n_hrrr_channels), (b,int(params.mask_ratio*c)), replace=False)
            mask[:, ch_mask] = 1.
            loss_mask = mask # or torch.ones(inp.shape, dtype=torch.bool) to compute over all channels
        else:
            # Nothing is masked, loss computed everywhere
            mask = torch.ones(inp.shape, device=self.device) 
            loss_mask = mask

        if self.params.two_step_training:
            gen_step_one = self.model(inp).to(self.device, dtype = torch.float)
            loss_step_one = self.loss_obj(gen_step_one, tar[:,0:self.params.n_hrrr_channels])

            if self.params.orography:
                gen_step_two = self.model(torch.cat( (gen_step_one, orog), axis = 1)  ).to(self.device, dtype = torch.float)
            elif self.params.add_zenith:
                zenith = tar[:,2*self.params.n_hrrr_channels:2*self.params.n_hrrr_channels+1]
                gen_step_one_zenith = torch.cat([gen_step_one, zenith], dim= 1)
                gen_step_two = self.model(gen_step_one_zenith).to(self.device, dtype = torch.float)
            else:
                gen_step_two = self.model(gen_step_one)

            loss_step_two = self.loss_obj(gen_step_two, tar[:,self.params.n_hrrr_channels:2*self.params.n_hrrr_channels])
            valid_loss += loss_step_one + loss_step_two
            valid_l1 += nn.functional.l1_loss(gen_step_one, tar[:,0:self.params.n_hrrr_channels])
        else:
            gen = self.model(inp, boundary, mask).to(self.device, dtype = torch.float)
            valid_loss += self.loss_obj(gen*loss_mask, tar*loss_mask) 
            # reconstruct full fields if log-tendency normed
            if self.params.log_scale:
                assert self.params.tendency_normalization, "log-scaling intended for use with tendency configuration"
                inp = torch.sign(inp)*torch.expm1(torch.abs(inp))
                gen = ratio_factor*torch.sign(gen)*torch.expm1(torch.abs(gen)) + inp
                tar = ratio_factor*torch.sign(tar)*torch.expm1(torch.abs(tar)) + inp
            valid_l1 += nn.functional.l1_loss(gen, tar)

        valid_steps += 1.
        # save fields for vis 
        if (i == sample_idx) and self.log_to_wandb:
          if self.params.two_step_training:
            fields = [gen_step_one[0,0].detach().cpu().numpy(), tar[0,0].detach().cpu().numpy()]
          else:
            fields = [gen[0,0].detach().cpu().numpy(), tar[0,0].detach().cpu().numpy()]

        if self.params.two_step_training:
            if self.params.target == 'residual_field':
                raise NotImplementedError
                #valid_rmse += nn.functional.mse_loss((gen_step_one + inp), (tar[:,0:self.params.n_hrrr_channels] + inp))
            else:
                valid_rmse += mult_rmse * torch.sqrt(nn.functional.mse_loss(gen_step_one, tar[:,0:self.params.n_hrrr_channels])) #rescale loss to original units
        else:
          # valid_rmse += mult_rmse * torch.sqrt(nn.functional.mse_loss(gen, tar)) #This line computes the global RMSE instead of per channel RMSE. 
          loss_per_element = torch.sqrt(nn.functional.mse_loss(gen, tar, reduction='none'))
          loss_per_channel = loss_per_element.mean(dim=(0,2,3))
          valid_rmse += mult_rmse * loss_per_channel

           
    if dist.is_initialized():
      dist.all_reduce(valid_buff)
      dist.all_reduce(valid_rmse)

    # divide by number of steps
    valid_buff[0:2] = valid_buff[0:2] / valid_buff[2]
    valid_rmse = valid_rmse / valid_buff[2]

    # download buffers
    valid_buff_cpu = valid_buff.detach().cpu().numpy()
    valid_rmse_cpu = valid_rmse.detach().cpu().numpy()

    valid_time = time.time() - valid_start

    try:
      logs = {'valid_l1': valid_buff_cpu[1], 'valid_loss': valid_buff_cpu[0], 'valid_rmse_u10': valid_rmse_cpu[0], 'valid_rmse_v10': valid_rmse_cpu[1]}
    except:
      logs = {'valid_l1': valid_buff_cpu[1], 'valid_loss': valid_buff_cpu[0], 'valid_rmse_u10': valid_rmse_cpu[0]}    
    if self.log_to_wandb:
      fig = vis(fields)
      logs['vis'] = wandb.Image(fig)
      plt.close(fig)
      wandb.log(logs, step=self.epoch)

    return valid_time, logs

  def load_model_wind(self, model_path):
    if self.log_to_screen:
      logging.info('Loading the wind model weights from {}'.format(model_path))
    checkpoint = torch.load(model_path, map_location='cuda:{}'.format(self.local_rank))
    if dist.is_initialized():
      self.model_wind.load_state_dict(checkpoint['model_state'])
    else:
      new_model_state = OrderedDict()
      model_key = 'model_state' if 'model_state' in checkpoint else 'state_dict'
      for key in checkpoint[model_key].keys():
          if 'module.' in key: # model was stored using ddp which prepends module
              name = str(key[7:])
              new_model_state[name] = checkpoint[model_key][key]
          else:
              new_model_state[key] = checkpoint[model_key][key]
      self.model_wind.load_state_dict(new_model_state)
      self.model_wind.eval()

  def save_checkpoint(self, checkpoint_path, model=None):
    """ We intentionally require a checkpoint_dir to be passed
        in order to allow Ray Tune to use this function """

    if not model:
      model = self.model

    torch.save({'iters': self.iters, 'epoch': self.epoch, 'model_state': model.state_dict(),
                  'optimizer_state_dict': self.optimizer.state_dict()}, checkpoint_path)

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
    self.iters = checkpoint['iters']
    self.startEpoch = checkpoint['epoch']
    if self.params.resuming:  #restore checkpoint is used for finetuning as well as resuming. If finetuning (i.e., not resuming), restore checkpoint does not load optimizer state, instead uses config specified lr.
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--run_num", default='00', type=str)
  parser.add_argument("--yaml_config", default='./config/hrrr_swin.yaml', type=str)
  parser.add_argument("--config", default='base', type=str)
  parser.add_argument("--enable_amp", action='store_true')
  parser.add_argument("--epsilon_factor", default = 0, type = float)
  parser.add_argument("--sweep_id", default=None, type=str, help='sweep config from ./configs/sweeps.yaml')

  args = parser.parse_args()

  params = YParams(os.path.abspath(args.yaml_config), args.config)
  trainer = Trainer(params, args)

  if args.sweep_id and trainer.world_rank==0:
    wandb.agent(args.sweep_id, function=trainer.build_and_launch, count=1, entity=trainer.params.entity, project=trainer.params.project)
  else:
    trainer.build_and_launch()

  if dist.is_initialized():
      dist.barrier()
  logging.info('DONE')
