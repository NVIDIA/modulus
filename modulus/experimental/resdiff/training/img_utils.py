# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import glob
from types import new_class
import torch
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch import Tensor
import h5py
import math
import torchvision.transforms.functional as TF


def reshape_fields(img, inp_or_tar, crop_size_x, crop_size_y,rnd_x, rnd_y, params, y_roll, train, normalize=True, grid=False):
    #Takes in np array of size (n_history+1, c, h, w) and returns torch tensor of size ((n_channels*(n_history+1), crop_size_x, crop_size_y)

    if len(np.shape(img)) == 3:
      img = np.expand_dims(img, 0)
      
    if img.shape[3] > 720:
        img = img[:, :, 0:720]         #remove last pixel for era5 data
        
    #print('img', img.shape)
              
    #n_history = np.shape(img)[0] - 1   #for era5
    n_history = params.n_history
    
    img_shape_x = np.shape(img)[-2]
    img_shape_y = np.shape(img)[-1]
    n_channels = np.shape(img)[1] #this will either be N_in_channels or N_out_channels
    channels = params.in_channels if inp_or_tar =='inp' else params.out_channels
    #print('channels', channels)
    
    # dist.print0('normalize', normalize)
    # dist.print0('train', train)

    if normalize and train:
        mins = np.load(params.min_path)[:, channels]
        maxs = np.load(params.max_path)[:, channels]
        means = np.load(params.global_means_path)[:, channels]
        stds = np.load(params.global_stds_path)[:, channels]
        
    if crop_size_x == None:
        crop_size_x = img_shape_x
    if crop_size_y == None:
        crop_size_y = img_shape_y

    
    if normalize and train:
        if params.normalization == 'minmax':
          img  -= mins
          img /= (maxs - mins)
        elif params.normalization == 'zscore':
          #print('params.normalization == zscore')
          img -=means
          img /=stds

    if grid:
        if inp_or_tar == 'inp':
            if params.gridtype == 'linear':
                assert params.N_grid_channels == 2, "N_grid_channels must be set to 2 for gridtype linear"
                x = np.meshgrid(np.linspace(-1, 1, img_shape_x))
                y = np.meshgrid(np.linspace(-1, 1, img_shape_y))
                grid_x, grid_y = np.meshgrid(y, x)
                grid = np.stack((grid_x, grid_y), axis = 0)
            elif params.gridtype == 'sinusoidal':
                #print('sinusuidal grid added ......')
                assert params.N_grid_channels == 4, "N_grid_channels must be set to 4 for gridtype sinusoidal"
                n_channels = n_channels + params.N_grid_channels
                x1 = np.meshgrid(np.sin(np.linspace(0, 2*np.pi, img_shape_x)))
                x2 = np.meshgrid(np.cos(np.linspace(0, 2*np.pi, img_shape_x)))
                y1 = np.meshgrid(np.sin(np.linspace(0, 2*np.pi, img_shape_y)))
                y2 = np.meshgrid(np.cos(np.linspace(0, 2*np.pi, img_shape_y)))
                grid_x1, grid_y1 = np.meshgrid(y1, x1)
                grid_x2, grid_y2 = np.meshgrid(y2, x2)
                grid = np.expand_dims(np.stack((grid_x1, grid_y1, grid_x2, grid_y2), axis = 0), axis = 0)
            img = np.concatenate((img, grid), axis = 1 )

    if params.roll:
        img = np.roll(img, y_roll, axis = -1)

    # if train and (crop_size_x or crop_size_y):
    #     img = img[:,:,rnd_x:rnd_x+crop_size_x, rnd_y:rnd_y+crop_size_y]

    if (crop_size_x or crop_size_y):
        img = img[:,:,rnd_x:rnd_x+crop_size_x, rnd_y:rnd_y+crop_size_y]

    if inp_or_tar == 'inp':
        img = np.reshape(img, (n_channels*(n_history+1), crop_size_x, crop_size_y))
    elif inp_or_tar == 'tar':
        img = np.reshape(img, (n_channels, crop_size_x, crop_size_y))
    

    return torch.as_tensor(img)
          


    

