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

from functools import partial
from collections import OrderedDict
from copy import Error, deepcopy
from re import S
from numpy.lib.arraypad import pad
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from apex.normalization import FusedLayerNorm
import torch.fft
from torch.nn.modules.container import Sequential
from torch.utils.checkpoint import checkpoint_sequential
from typing import Optional
import math

# helpers
from modulus.experimental.sfno.networks.layers import DropPath, MLP
from modulus.experimental.sfno.networks.activations import ComplexReLU

class PatchEmbed(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=(16, 16), inp_chans=3, embed_dim=768):
        super(PatchEmbed, self).__init__()
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(inp_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # gather input
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # new: B, C, H*W
        x = self.proj(x).flatten(2)
        return x

@torch.jit.script
def compl_mul_add_fwd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    tmp = torch.einsum("bkixys,kior->srbkoxy", a, b)
    res = torch.stack([tmp[0,0,...] - tmp[1,1,...], tmp[1,0,...] + tmp[0,1,...]], dim=-1)
    return res


@torch.jit.script
def compl_mul_add_fwd_c(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    resc = torch.einsum("bkixy,kio->bkoxy", ac, bc)
    res = torch.view_as_real(resc)
    return res


class AFNO2D(nn.Module):
    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.0, hard_thresholding_fraction=1, hidden_size_factor=1,
                 use_complex_kernels=False):
        super(AFNO2D, self).__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"
        
        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02
        self.mult_handle = compl_mul_add_fwd_c if use_complex_kernels else compl_mul_add_fwd
        
        # new
        self.w1 = nn.Parameter(self.scale * torch.randn(self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor, 2))
        self.b1 = nn.Parameter(self.scale * torch.randn(1, self.num_blocks*self.block_size, 1, 1))
        self.w2 = nn.Parameter(self.scale * torch.randn(self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size, 2))
        # self.b2 = nn.Parameter(self.scale * torch.randn(self.num_blocks, self.block_size, 1, 1, 2))
        
        #self.act = nn.ReLU()
        self.act = ComplexReLU(negative_slope=0.0, mode="cartesian")
        
    def forward(self, x):
        
        bias = x
    
        dtype = x.dtype
        x = x.float()
        B, C, H, W = x.shape
        total_modes_H = H // 2 + 1
        total_modes_W = W // 2 + 1
        kept_modes_H = int(total_modes_H * self.hard_thresholding_fraction)
        kept_modes_W = int(total_modes_W * self.hard_thresholding_fraction)
        
        x = torch.fft.rfft2(x, dim=(-2, -1), norm="ortho")
        x = x.view(B, self.num_blocks, self.block_size, H, W // 2 + 1)
        
        # do spectral conv
        x = torch.view_as_real(x)
        x_fft = torch.zeros(x.shape, device=x.device)
        
        if kept_modes_H == total_modes_H:
            oac = torch.view_as_complex(self.mult_handle(x[:, :, :, :,  :kept_modes_W, :], self.w1))
            oa = torch.view_as_real(self.act(oac))
            x_fft[:, :, :, :, :kept_modes_W, :] = self.mult_handle(oa, self.w2)
        else:
            olc = torch.view_as_complex(self.mult_handle(x[:, :, :, :kept_modes_H,  :kept_modes_W, :], self.w1))
            ohc = torch.view_as_complex(self.mult_handle(x[:, :, :, -kept_modes_H:, :kept_modes_W, :], self.w1))

            ol = torch.view_as_real(self.act(olc))
            oh = torch.view_as_real(self.act(ohc))
            
            x_fft[:, :, :, :kept_modes_H,  :kept_modes_W, :] = self.mult_handle(ol, self.w2)
            x_fft[:, :, :, -kept_modes_H:, :kept_modes_W, :] = self.mult_handle(oh, self.w2)
            
        # finalize
        x = F.softshrink(x_fft, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, C, H, W // 2 + 1)
        x = torch.fft.irfft2(x, s=(H, W), dim=(-2,-1), norm="ortho")
        x = x.type(dtype)
        
        return x + self.b1 + bias


class Block(nn.Module):
    def __init__(
            self,
            h,
            w,
            dim,
            mlp_ratio=4.,
            drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            num_blocks=8,
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1.0,
            use_complex_kernels=True,
            skip_fno='linear',
            nested_skip_fno=True,
            checkpointing=False,
            verbose=True,
        ):
        super(Block, self).__init__()
        
        # norm layer
        self.norm1 = norm_layer() #((h,w))

        if skip_fno is None:
            if verbose:
                print('Using no skip connection around FNO.')

        elif skip_fno == 'linear':
            # self.skip_layer = nn.Linear(dim, dim)
            self.skip_layer = nn.Conv2d(dim, dim, 1, 1)
            if verbose:
                print('Using Linear skip connection around FNO.')

        elif skip_fno == 'identity':
            self.skip_layer = nn.Identity()
            if verbose:
                print('Using Identity skip connection around FNO.')
        
        else:
            if verbose:
                print(f'Got skip_fno={skip_fno}, not using any skip around FNO -- use linear or identity to change this.')
        self.skip_fno = skip_fno

        self.nested_skip_fno = nested_skip_fno

        # filter
        self.filter = AFNO2D(dim, num_blocks, sparsity_threshold, hard_thresholding_fraction,
                             use_complex_kernels=use_complex_kernels) 
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # norm layer
        self.norm2 = norm_layer() #((h,w))
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop_rate=drop,
                       checkpointing=checkpointing)


    def forward(self, x):
        residual = x
        
        x = self.norm1(x)
        x = self.filter(x)

        if self.skip_fno is not None:
            x = x + self.skip_layer(residual)
            if not self.nested_skip_fno:
                residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + residual
        return x


class AdaptiveFourierNeuralOperatorNet(nn.Module):
    def __init__(
            self,
            inp_shape = (720, 1440),
            patch_size = (16, 16),
            inp_chans = 2,
            out_chans = 2,
            embed_dim = 768,
            num_layers = 12,
            mlp_ratio = 4.,
            drop_rate = 0.,
            drop_path_rate = 0.,
            num_blocks = 16,
            sparsity_threshold = 0.01,
            normalization_layer = 'instance_norm',
            skip_fno = 'linear',
            nested_skip_fno = True,
            hard_thresholding_fraction = 1.0,
            checkpointing = False,
            use_complex_kernels = True,
            verbose = False,
            **kwargs
        ):
        super(AdaptiveFourierNeuralOperatorNet, self).__init__()
        self.img_size = inp_shape
        self.patch_size = patch_size
        self.inp_chans = inp_chans
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        # some sanity checks
        assert (len(patch_size) == 2), f"Expected patch_size to have two entries but got {patch_size} instead"
        assert ( (self.img_size[0] % self.patch_size[0] == 0) and (self.img_size[1] % self.patch_size[1] == 0) ), f"Error, the patch size {self.patch_size} does not divide the image dimensions {self.img_size} evenly."
        
        self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=self.patch_size, inp_chans=self.inp_chans, embed_dim=self.embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, num_patches))
        self.pos_drop = nn.Dropout(p=drop_rate) if drop_rate > 0. else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]

        # compute the downscaled image size
        self.h = self.img_size[0] // self.patch_size[0]
        self.w = self.img_size[1] // self.patch_size[1]

        # pick norm layer
        if normalization_layer == "layer_norm":
            norm_layer = partial(nn.LayerNorm, normalized_shape=(self.h, self.w), eps=1e-6)
        elif normalization_layer == "instance_norm":
            norm_layer = partial(nn.InstanceNorm2d, num_features=embed_dim, eps=1e-6, affine=True, track_running_stats=False)
        else:
            raise NotImplementedError(f"Error, normalization {normalization_layer} not implemented.") 

        self.blocks = nn.ModuleList([
            Block(h = self.h,
                  w = self.w,
                  dim = self.embed_dim,
                  mlp_ratio = mlp_ratio,
                  drop = drop_rate,
                  drop_path = dpr[i],
                  norm_layer = norm_layer,
                  num_blocks = num_blocks,
                  sparsity_threshold = sparsity_threshold,
                  hard_thresholding_fraction = hard_thresholding_fraction,
                  use_complex_kernels = use_complex_kernels,
                  skip_fno = skip_fno,
                  nested_skip_fno = nested_skip_fno, 
                  checkpointing = checkpointing,
                  verbose = verbose) 
        for i in range(num_layers)])

        # head
        self.head = nn.Conv2d(embed_dim, self.out_chans*self.patch_size[0]*self.patch_size[1], 1, bias=False)

        with torch.no_grad():
            nn.init.trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=.02)
            #nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, FusedLayerNorm) or isinstance(m, nn.InstanceNorm3d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # reshape
        x = x.reshape(B, self.embed_dim, self.h, self.w)

        for blk in self.blocks:
            x = blk(x)
            
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        # new: B, C, H, W
        b = x.shape[0]
        xv = x.view(b, self.patch_size[0], self.patch_size[1], -1, self.h, self.w)
        xvt = torch.permute(xv, (0, 3, 4, 1, 5, 2)).contiguous()
        x = xvt.view(b, -1, (self.h * self.patch_size[0]), (self.w * self.patch_size[1]))
        
        return x
