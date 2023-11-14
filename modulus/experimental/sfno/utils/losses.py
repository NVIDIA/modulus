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

from typing import Optional, Tuple, List
import math

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from modulus.experimental.sfno.utils import comm
from modulus.experimental.sfno.utils.grids import GridQuadrature
from modulus.experimental.sfno.mpu.mappings import reduce_from_parallel_region, gather_from_parallel_region

import torch_harmonics as harmonics
from torch_harmonics.quadrature import clenshaw_curtiss_weights, legendre_gauss_weights


class LossHandler(nn.Module):
    """
    Wrapper class that will handle computing losses.
    """

    def __init__(self, params):

        super(LossHandler, self).__init__()

        self.rank = comm.get_rank("matmul")
        self.n_future = params.n_future

        # get global image shape
        self.img_shape = (params.img_shape_x, params.img_shape_y)
        self.crop_shape = (params.img_crop_shape_x, params.img_crop_shape_y)
        self.crop_offset = (params.img_crop_offset_x, params.img_crop_offset_y)

        loss_type = self.loss_type = params.loss

        loss_type = set(loss_type.split())

        if 'pole-masked' in loss_type:
            pole_mask = 1
        else:
            pole_mask = 0

        if 'weighted' in loss_type:
            if params.channel_weights == 'auto':
                channel_weights = torch.ones(params.N_out_channels, dtype=torch.float32)
                for c, chn in enumerate(params.channel_names):
                    if chn in ['u10m', 'v10m', 'u100m', 'v100m', 'tp', 'sp', 'msl', 'tcwv']:
                        channel_weights[c] = 0.1
                    elif chn in ['t2m', '2d']:
                        channel_weights[c] = 1.0
                    elif chn[0] in ['z', 'u', 'v', 't', 'r', 'q']:
                        pressure_level = float(chn[1:])
                        channel_weights[c] = 0.001 * pressure_level
                    else:
                        channel_weights[c] = 0.01
            else:
                channel_weights = torch.Tensor(params.channel_weights).float()
        else:
            channel_weights = torch.ones(params.N_out_channels, dtype=torch.float32)
        
        # renormalize the weights to one
        channel_weights = channel_weights.reshape(1, -1, 1, 1)
        channel_weights = channel_weights / torch.sum(channel_weights)


        if 'absolute' in loss_type:
            absolute = True
        else:
            absolute = False           

        if 'squared' in loss_type:
            squared = True
        else:
            squared = False

        if 'temp-std' in loss_type:
            eps = 1e-6
            global_stds = torch.from_numpy(np.load(params.global_stds_path)).reshape(1, -1, 1, 1)[:, params.out_channels]
            time_diff_stds = np.sqrt(params.dt) * torch.from_numpy(np.load(params.time_diff_stds_path)).reshape(1, -1, 1, 1)[:, params.out_channels]
            time_var_weights = global_stds / (time_diff_stds+eps)
            # time_var_weights = 1 / (time_diff_stds+eps)
            if squared:
                time_var_weights = time_var_weights**2
            channel_weights = channel_weights * time_var_weights

        self.register_buffer('channel_weights', channel_weights)

        # which weights to use
        quadrature_rule_type = "naive"
        if params.model_grid_type == "legendre_gauss":
            quadrature_rule_type = "legendre-gauss"
        
        # decide which loss to use
        if 'l2' in loss_type:
            if 'geometric' in loss_type:
                self.loss_obj = GeometricLpLoss(self.img_shape, self.crop_shape, self.crop_offset,
                                                p=2, absolute=absolute, squared=squared, pole_mask=pole_mask, quadrature_rule=quadrature_rule_type)
            else:
                self.loss_obj = GeometricLpLoss(self.img_shape, self.crop_shape, self.crop_offset,
                                                p=2, absolute=absolute, pole_mask=pole_mask, jacobian='flat')
        elif 'l1' in loss_type:
            if 'geometric' in loss_type:
                self.loss_obj = GeometricLpLoss(self.img_shape, self.crop_shape, self.crop_offset,
                                                p=1, absolute=absolute, pole_mask=pole_mask, quadrature_rule=quadrature_rule_type)
            else:
                self.loss_obj = GeometricLpLoss(self.img_shape, self.crop_shape, self.crop_offset,
                                                p=1, absolute=absolute, pole_mask=pole_mask, jacobian='flat')
        elif 'geometric h1' in loss_type:
            self.loss_obj = GeometricH1Loss(self.img_shape, absolute=absolute, squared=squared)
        else:
            raise ValueError(f"Unknown loss function: {self.loss_type}")

        # weighting factor for the case of multistep training
        # this is a canonical uniform weight we found to work best
        # depending on the problem, a different weight strategy might work better
        multistep_weight = torch.ones(self.n_future+1, dtype=torch.float32) / float(self.n_future+1)
        multistep_weight = multistep_weight.reshape(-1, 1, 1, 1)

        self.register_buffer('multistep_weight', multistep_weight)

        # # decide whether to gather the input
        self.do_gather_input = False
        if comm.get_size("spatial") > 1:
            self.do_gather_input = True

    @torch.jit.ignore
    def _gather_input(self, x: torch.Tensor) -> torch.Tensor:
        # combine data
        # h
        xh = gather_from_parallel_region(x, -2, "h")
        xw = gather_from_parallel_region(xh, -1, "w")
    
        # crop
        x = xw[...,
               self.crop_offset[0]:self.crop_offset[0]+self.crop_shape[0],
               self.crop_offset[1]:self.crop_offset[1]+self.crop_shape[1]].contiguous()

        return x

    def is_distributed(self):
        return False

    def forward(self, prd: torch.Tensor, tar: torch.Tensor, inp: torch.Tensor):

        if self.do_gather_input:
            prd = self._gather_input(prd)
            tar = self._gather_input(tar)

        if hasattr(self, "minmax"):
            chw = torch.ones_like(self.channel_weights)
            chw = chw / torch.sum(chw)
            chw += self.channel_weights.abs() / torch.sum(self.channel_weights.abs())
        else:
            chw = self.channel_weights
        
        
        # print(chw.reshape(-1))
        if self.training:
            chw = (chw * self.multistep_weight).reshape(1, -1)
        else:
            chw = chw.reshape(1, -1)

        return self.loss_obj(prd, tar, chw)

    
# double check if polar optimization has an effect - we use 5 here by default
class GeometricLpLoss(nn.Module):
    """
    Computes the Lp loss on the sphere.
    """

    def __init__(self,
                 img_shape: Tuple[int, int],
                 crop_shape: Tuple[int, int],
                 crop_offset: Tuple[int, int],
                 p: Optional[float]=2.,
                 size_average: Optional[bool]=False,
                 reduction: Optional[bool]=True,
                 absolute: Optional[bool]=False,
                 squared: Optional[bool]=False,
                 pole_mask: Optional[int]=0,
                 jacobian: Optional[str]='s2',
                 quadrature_rule: Optional[str]='naive'):
        super(GeometricLpLoss, self).__init__()

        self.p = p
        self.img_shape = img_shape
        self.crop_shape = crop_shape
        self.crop_offset = crop_offset
        self.reduction = reduction
        self.size_average = size_average
        self.absolute = absolute
        self.squared = squared
        self.pole_mask = pole_mask

        # get the quadrature
        self.quadrature = GridQuadrature(quadrature_rule,
                                         img_shape=self.img_shape, crop_shape=self.crop_shape, crop_offset=self.crop_offset,
                                         normalize=True, pole_mask=self.pole_mask)

    def abs(self, prd: torch.Tensor, tar: torch.Tensor, chw: torch.Tensor):
        num_examples = prd.size()[0]

        all_norms = self.quadrature(torch.abs(prd-tar)**self.p)
        all_norms = all_norms.reshape(num_examples, -1)
        
        if not self.squared:
            all_norms = all_norms**(1./self.p)
        
        # apply channel weighting
        all_norms = chw * all_norms

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, prd: torch.Tensor, tar: torch.Tensor, chw: torch.Tensor):
        num_examples = prd.size()[0]

        diff_norms = self.quadrature(torch.abs(prd-tar)**self.p)
        diff_norms = diff_norms.reshape(num_examples, -1)

        tar_norms = self.quadrature(torch.abs(tar)**self.p)
        tar_norms = tar_norms.reshape(num_examples, -1)

        # divide the ratios
        frac_norms = (diff_norms / tar_norms)
        
        if not self.squared:
            frac_norms = frac_norms**(1./self.p)

        # setup return value
        retval = chw * frac_norms

        if self.reduction:
            if self.size_average:
                retval = torch.mean(retval)
            else:
                retval = torch.sum(retval)

        return retval

    def forward(self, prd: torch.Tensor, tar: torch.Tensor, chw: torch.Tensor):
        if self.absolute:
            loss = self.abs(prd, tar, chw)
        else:
            loss = self.rel(prd, tar, chw)

        return loss


# double check if polar optimization has an effect - we use 5 here by default
class GeometricH1Loss(nn.Module):
    """
    Computes the weighted H1 loss on the sphere.
    Alpha is a parameter which balances the respective seminorms.
    """

    def __init__(self,
                 img_shape: Tuple[int, int],
                 p: Optional[float]=2.,
                 size_average: Optional[bool]=False,
                 reduction: Optional[bool]=True,
                 absolute: Optional[bool]=False,
                 squared: Optional[bool]=False,
                 alpha: Optional[float]=0.5,):
        super(GeometricH1Loss, self).__init__()

        self.reduction = reduction
        self.size_average = size_average
        self.absolute = absolute
        self.squared = squared
        self.alpha = alpha

        self.sht = harmonics.RealSHT(*img_shape, grid='equiangular').float()
        h1_weights = torch.arange(self.sht.lmax).float()
        h1_weights = h1_weights * (h1_weights + 1)
        self.register_buffer("h1_weights", h1_weights)


    def abs(self, prd: torch.Tensor, tar: torch.Tensor):
        num_examples = prd.size()[0]

        coeffs = torch.view_as_real(self.sht(prd - tar))
        coeffs = coeffs[..., 0]**2 + coeffs[..., 1]**2
        norm2 = coeffs[..., :, 0] + 2 * torch.sum(coeffs[..., :, 1:], dim=-1)
        l2_norm2 = norm2.reshape(num_examples, -1).sum(dim=-1)
        h1_norm2 = (norm2 * self.h1_weights).reshape(num_examples, -1).sum(dim=-1)

        if not self.squared:
            all_norms = self.alpha*torch.sqrt(l2_norm2) + (1 - self.alpha)*torch.sqrt(h1_norm2)
        else: 
            all_norms = self.alpha*l2_norm2 + (1 - self.alpha)*h1_norm2

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, prd: torch.Tensor, tar: torch.Tensor, mask: Optional[torch.Tensor] = None):
        num_examples = prd.size()[0]

        coeffs = torch.view_as_real(self.sht(prd - tar))
        coeffs = coeffs[..., 0]**2 + coeffs[..., 1]**2
        norm2 = coeffs[..., :, 0] + 2 * torch.sum(coeffs[..., :, 1:], dim=-1)
        l2_norm2 = norm2.reshape(num_examples, -1).sum(dim=-1)
        h1_norm2 = (norm2 * self.h1_weights).reshape(num_examples, -1).sum(dim=-1)

        tar_coeffs = torch.view_as_real(self.sht(tar))
        tar_coeffs = tar_coeffs[..., 0]**2 + tar_coeffs[..., 1]**2
        tar_norm2 = tar_coeffs[..., :, 0] + 2 * torch.sum(tar_coeffs[..., :, 1:], dim=-1)
        tar_l2_norm2 = tar_norm2.reshape(num_examples, -1).sum(dim=-1)
        tar_h1_norm2 = (tar_norm2 * self.h1_weights).reshape(num_examples, -1).sum(dim=-1)

        if not self.squared:
            diff_norms = self.alpha*torch.sqrt(l2_norm2) + (1 - self.alpha)*torch.sqrt(h1_norm2)
            tar_norms  = self.alpha*torch.sqrt(tar_l2_norm2) + (1 - self.alpha)*torch.sqrt(tar_h1_norm2)
        else: 
            diff_norms = self.alpha*l2_norm2 + (1 - self.alpha)*h1_norm2
            tar_norms  = self.alpha*tar_l2_norm2 + (1 - self.alpha)*tar_h1_norm2

        # setup return value
        retval = diff_norms / tar_norms
        if mask is not None:
            retval = retval * mask
            
        if self.reduction:
            if self.size_average:
                if mask is None:
                    retval = torch.mean(retval)
                else:
                    retval = torch.sum(retval) / torch.sum(mask)
            else:
                retval = torch.sum(retval)

        return retval

    def forward(self, prd: torch.Tensor, tar: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if self.absolute:
            loss = self.abs(prd, tar)
        else:
            loss = self.rel(prd, tar, mask)

        return loss
