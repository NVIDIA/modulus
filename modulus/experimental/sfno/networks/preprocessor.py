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

import copy
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from modulus.experimental.sfno.utils import comm
from modulus.experimental.sfno.utils.grids import GridConverter
from modulus.experimental.sfno.mpu.mappings import reduce_from_parallel_region, copy_to_parallel_region

class Preprocessor2D(nn.Module):
    def __init__(self, params):
        super(Preprocessor2D, self).__init__()

        self.n_history = params.n_history
        self.transform_to_nhwc = params.enable_nhwc
        self.history_normalization_mode = params.history_normalization_mode
        if self.history_normalization_mode == "exponential":
            self.history_normalization_decay = params.history_normalization_decay
            # inverse ordering, since first element is oldest
            history_normalization_weights = torch.exp( (-self.history_normalization_decay) * torch.arange(start=self.n_history, end=-1, step=-1, dtype=torch.float32))
            history_normalization_weights = history_normalization_weights / torch.sum(history_normalization_weights)
            history_normalization_weights = torch.reshape(history_normalization_weights, (1, -1, 1, 1, 1))
        elif self.history_normalization_mode == "mean":
            history_normalization_weights = torch.Tensor(1./float(self.n_history+1), dtype=torch.float32)
            history_normalization_weights = torch.reshape(history_normalization_weights, (1, -1, 1, 1, 1))
        else:
            history_normalization_weights = torch.ones(self.n_history+1, dtype=torch.float32)
        self.register_buffer("history_normalization_weights", history_normalization_weights, persistent=False)
        self.history_mean = None
        self.history_std = None
        self.history_diff_mean = None
        self.history_diff_var = None
        self.history_eps = 1e-6
        self.img_shape = [params.img_shape_x, params.img_shape_y]

        # unpredicted input channels:
        self.unpredicted_inp_train = None
        self.unpredicted_tar_train = None
        self.unpredicted_inp_eval = None
        self.unpredicted_tar_eval = None

        # process static features
        static_features = None
        # needed for sharding
        start_x = params.img_local_offset_x
        end_x = min(start_x + params.img_local_shape_x, params.img_shape_x)
        pad_x = params.img_local_shape_x - (end_x - start_x)
        start_y = params.img_local_offset_y
        end_y = min(start_y + params.img_local_shape_y, params.img_shape_y)
        pad_y =	params.img_local_shape_y - (end_y - start_y)

        # set up grid
        if params.add_grid:
            with torch.no_grad():
                if hasattr(params, "lat") and hasattr(params, "lon"):
                    lat = torch.tensor(params.lat).to(torch.float32)
                    lon = torch.tensor(params.lon).to(torch.float32)

                    # convert grid if required
                    gconv = GridConverter(params.data_grid_type, params.model_grid_type,
                                          torch.deg2rad(lat), torch.deg2rad(lon))
                    tx, ty = gconv.get_dst_coords()
                    tx = tx.to(torch.float32)
                    ty = ty.to(torch.float32)
                else:
                    tx = torch.linspace(0, 1, params.img_shape_x + 1, dtype=torch.float32)[0:-1]
                    ty = torch.linspace(0, 1, params.img_shape_y + 1, dtype=torch.float32)[0:-1]

                x_grid, y_grid = torch.meshgrid(tx, ty, indexing='ij')
                x_grid, y_grid = x_grid.unsqueeze(0).unsqueeze(0), y_grid.unsqueeze(0).unsqueeze(0)
                grid = torch.cat([x_grid, y_grid], dim=1)

                # shard spatially:
                grid = grid[:, :, start_x:end_x, start_y:end_y]

                # pad if needed
                grid = F.pad(grid, [0, pad_y, 0, pad_x])

                # transform if requested
                if params.gridtype == "sinusoidal":
                    num_freq = 1
                    if hasattr(params, "grid_num_frequencies"):
                        num_freq = int(params.grid_num_frequencies)
                    
                    singrid  = None
                    for freq in range(1, num_freq+1):
                        if singrid is None:
                            singrid = torch.sin(grid)
                        else:
                            singrid = torch.cat([singrid, torch.sin(freq * grid)], dim=1)
                        
                    static_features = singrid
                else:
                    static_features = grid

        if params.add_orography:
            from utils.conditioning_inputs import get_orography
            oro = torch.tensor(get_orography(params.orography_path), dtype=torch.float32)
            oro = torch.reshape(oro, (1, 1, oro.shape[0], oro.shape[1]))

            # shard
            oro = oro[:, :, start_x:end_x, start_y:end_y]

            # pad if needed
            oro = F.pad(oro, [0, pad_y, 0, pad_x])
            
            if static_features is None:
                static_features = oro
            else:
                static_features = torch.cat([static_features, oro], dim=1)

        if params.add_landmask:
            from utils.conditioning_inputs import get_land_mask
            lsm = torch.tensor(get_land_mask(params.landmask_path), dtype=torch.long)
            # one hot encode and move channels to front:
            lsm = torch.permute(torch.nn.functional.one_hot(lsm), (2, 0, 1)).to(torch.float32)
            lsm = torch.reshape(lsm, (1, lsm.shape[0], lsm.shape[1], lsm.shape[2]))
            
            # shard
            lsm = lsm[:, :, start_x:end_x, start_y:end_y]

            # pad if needed
            lsm = F.pad(lsm, [0, pad_y, 0, pad_x])

            if static_features is None:
                static_features = lsm
            else:
                static_features = torch.cat([static_features, lsm], dim=1)

        self.do_add_static_features = False
        if static_features is not None:
            self.do_add_static_features = True
            self.register_buffer("static_features", static_features, persistent=False)
        

    def flatten_history(self, x):
        
        # flatten input
        if x.dim() == 5:
            b_, t_, c_, h_, w_ = x.shape
            x = torch.reshape(x, (b_, t_*c_, h_, w_))

        return x

    def expand_history(self, x, nhist):
        if x.dim() == 4:
            b_, ct_, h_, w_ = x.shape
            x = torch.reshape(x, (b_, nhist, ct_ // nhist, h_, w_))
        return x

    def add_static_features(self, x):
        if self.do_add_static_features:
            # we need to replicate the grid for each batch:
            static = torch.tile(self.static_features, dims=(x.shape[0], 1, 1, 1))
            x = torch.cat([x, static], dim=1)

        return x

    
    def remove_static_features(self, x):
        # only remove if something was added in the first place
        if self.do_add_static_features:
            nfeat = self.static_features.shape[1]
            x = x[:, :x.shape[1]-nfeat, :, :]
        return x
    
    
    def append_history(self, x1, x2, step):

        # take care of unpredicted features first
        # this is necessary in order to copy the targets unpredicted features
        # (such as zenith angle) into the inputs unpredicted features,
        # such that they can be forward in the next autoregressive step
        # extract utar

        # update the unpredicted input
        if self.training:
            if (self.unpredicted_tar_train is not None) and (step < self.unpredicted_tar_train.shape[1]):
                utar = self.unpredicted_tar_train[:, step:(step+1), :, :, :]
                if (self.n_history == 0):
                    self.unpredicted_inp_train.copy_(utar)
                else:
                    self.unpredicted_inp_train.copy_(torch.cat([self.unpredicted_inp_train[:, 1:, :, :, :], utar], dim=1))
        else:
            if (self.unpredicted_tar_eval is not None) and (step < self.unpredicted_tar_eval.shape[1]):
                utar = self.unpredicted_tar_eval[:, step:(step+1), :, :, :]
                if (self.n_history == 0):
                    self.unpredicted_inp_eval.copy_(utar)
                else:
                    self.unpredicted_inp_eval.copy_(torch.cat([self.unpredicted_inp_eval[:, 1:, :, :, :], utar], dim=1))

        if self.n_history > 0:
            # this is more complicated
            x1 = self.expand_history(x1, nhist=self.n_history+1)
            x2 = self.expand_history(x2, nhist=1)

            # append
            res = torch.cat([x1[:,1:,:,:,:], x2], dim=1)

            # flatten again
            res = self.flatten_history(res)
        else:
            res = x2

        return res

    
    def append_channels(self, x, xc):
        xdim = x.dim()
        x = self.expand_history(x, self.n_history+1)

        xc = self.expand_history(xc, self.n_history+1)

        # concatenate
        xo = torch.cat([x, xc], dim=2)

        # flatten if requested
        if xdim == 4: 
            xo = self.flatten_history(xo)

        return xo

    def history_compute_stats(self, x):
        if self.history_normalization_mode == "none":
            self.history_mean = torch.zeros((1, 1, 1, 1), dtype=torch.float32, device=x.device)
            self.history_std = torch.ones((1, 1, 1, 1), dtype=torch.float32, device=x.device)
        elif self.history_normalization_mode == "timediff":
            # reshaping
            xdim = x.dim()
            if xdim == 4:
                b_, c_, h_, w_ = x.shape
                xr = torch.reshape(x, (b_, (self.n_history+1), c_ // (self.n_history+1), h_, w_))
            else:
                xshape = x.shape
                xr = x
            
            # time difference mean:
            self.history_diff_mean = torch.mean(torch.sum(xr[:, 1:, ...] - xr[:, 0:-1, ...], dim=(4,5)), dim=(1,2))
            # reduce across gpus
            if comm.get_size("spatial") > 1:
                self.history_diff_mean = reduce_from_parallel_region(self.history_diff_mean, "spatial")
            self.history_diff_mean = self.history_diff_mean / float(self.img_shape[0] * self.img_shape[1])

            # time difference std
            self.history_diff_var = torch.mean(torch.sum( torch.square( (xr[:, 1:, ...] - xr[:, 0:-1, ...]) - self.history_diff_mean), dim=(4,5)), dim=(1,2))
            # reduce across gpus
            if comm.get_size("spatial") > 1:
                self.history_diff_var = reduce_from_parallel_region(self.history_diff_var, "spatial")
            self.history_diff_var = self.history_diff_var / float(self.img_shape[0] * self.img_shape[1])

            # time difference stds
            self.history_diff_mean = copy_to_parallel_region(self.history_diff_mean, "spatial")
            self.history_diff_var = copy_to_parallel_region(self.history_diff_var, "spatial")
        else:
            xdim = x.dim()
            if xdim == 4:
                b_, c_, h_, w_ = x.shape
                xr = torch.reshape(x, (b_, (self.n_history+1), c_ // (self.n_history+1), h_, w_))
            else:
                xshape = x.shape
                xr = x

            # mean
            # compute weighted mean over dim 1, but sum over dim=3,4
            self.history_mean = torch.sum(xr * self.history_normalization_weights, dim=(1, 3, 4), keepdim=True)
            # reduce across gpus
            if comm.get_size("spatial") > 1:
                self.history_mean = reduce_from_parallel_region(self.history_mean, "spatial")
            self.history_mean = self.history_mean / float(self.img_shape[0] * self.img_shape[1])

            # compute std
            self.history_std = torch.sum( torch.square(xr - self.history_mean) * self.history_normalization_weights, dim=(1, 3, 4), keepdim=True)
            # reduce across gpus
            if comm.get_size("spatial") > 1:
                self.history_std = reduce_from_parallel_region(self.history_std, "spatial")
            self.history_std = torch.sqrt(self.history_std / float(self.img_shape[0] * self.img_shape[1]))
                
            # squeeze
            self.history_mean = torch.squeeze(self.history_mean, dim=1)
            self.history_std  = torch.squeeze(self.history_std, dim=1)

            # copy to parallel region
            self.history_mean = copy_to_parallel_region(self.history_mean, "spatial")
            self.history_std  =	copy_to_parallel_region(self.history_std,  "spatial")

        return    
    
    def history_normalize(self, x, target=False):
        if self.history_normalization_mode in ["none", "timediff"]:
            return x
        
        xdim = x.dim()
        if xdim == 4:
            b_, c_, h_, w_ = x.shape
            xr = torch.reshape(x, (b_, (self.n_history+1), c_ // (self.n_history+1), h_, w_))
        else:
            xshape = x.shape
            xr = x
            x = self.flatten_history(x)
            
        # normalize
        if target:
            # strip off the unpredicted channels
            xn = (x - self.history_mean[:, :x.shape[1], :, :]) / self.history_std[:, :x.shape[1], :, :]
        else:
            # tile to include history
            hm = torch.tile(self.history_mean, (1, self.n_history+1, 1, 1))
            hs = torch.tile(self.history_std, (1, self.n_history+1, 1, 1))
            xn = (x - hm) / hs

        if xdim == 5:
            xn = torch.reshape(xn, xshape)
            
        return xn

    def history_denormalize(self, xn, target=False):
        if self.history_normalization_mode in ["none", "timediff"]:
            return xn
        
        assert(self.history_mean is not None)
        assert(self.history_std is not None)

        xndim = xn.dim()
        if xndim == 5:
            xnshape = xn.shape
            xn = self.flatten_history(xn)

        # de-normalize
        if target:
            # strip off the unpredicted channels
            x = xn * self.history_std[:, :xn.shape[1], :, :] + self.history_mean[:, :xn.shape[1], :, :]
        else:
            # tile to include history
            hm = torch.tile(self.history_mean, (1, self.n_history+1, 1, 1))
            hs = torch.tile(self.history_std, (1, self.n_history+1, 1, 1))
            x = xn * hs + hm

        if xndim == 5:
            x = torch.reshape(x, xnshape)

        return x
    
    def cache_unpredicted_features(self, x, y, xz=None, yz=None):
        if self.training:
            if (self.unpredicted_inp_train is not None) and (xz is not None):
                self.unpredicted_inp_train.copy_(xz)
            else:
                self.unpredicted_inp_train = xz

            if (self.unpredicted_tar_train is not None) and (yz is not None):
                self.unpredicted_tar_train.copy_(yz)
            else:
                self.unpredicted_tar_train = yz
        else:
            if (self.unpredicted_inp_eval is not None) and (xz is not None):
                self.unpredicted_inp_eval.copy_(xz)
            else:
                self.unpredicted_inp_eval = xz

            if (self.unpredicted_tar_eval is not None) and (yz is not None):
                self.unpredicted_tar_eval.copy_(yz)
            else:
                self.unpredicted_tar_eval = yz

        return x, y

    def append_unpredicted_features(self, inp):
        if self.training:
            if self.unpredicted_inp_train is not None:
                inp = self.append_channels(inp, self.unpredicted_inp_train)
        else:
            if self.unpredicted_inp_eval is not None:
                inp = self.append_channels(inp, self.unpredicted_inp_eval)
        return inp

    def remove_unpredicted_features(self, inp):
        if self.training:
            if self.unpredicted_inp_train is not None:
                inpf = self.expand_history(inp, nhist=self.n_history+1)
                inpc = inpf[:, :, :inpf.shape[2]-self.unpredicted_inp_train.shape[2], :, :]
                inp = self.flatten_history(inpc)
        else:
            if self.unpredicted_inp_eval is not None:
                inpf = self.expand_history(inp, nhist=self.n_history+1)
                inpc = inpf[:, :, :inpf.shape[2]-self.unpredicted_inp_eval.shape[2], :, :]
                inp = self.flatten_history(inpc)
                
        return inp
        
       
def get_preprocessor(params):
    return Preprocessor2D(params)

