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

import math
import torch
from torch import nn
import torch.nn.functional as F

from modulus.experimental.sfno.utils import comm
from modulus.experimental.sfno.mpu.layers import distributed_transpose_h, distributed_transpose_w

# 3D routines
# forward
class RealFFT3(nn.Module):
    """
    Helper routine to wrap FFT similarly to the SHT
    """
    def __init__(self,
                 nd,
                 nh,
                 nw,
                 ldmax = None,
                 lhmax = None,
                 lwmax = None):
        super(RealFFT3, self).__init__()
        
        # dimensions
        self.nd = nd
        self.nh = nh
        self.nw = nw
        self.ldmax = min(ldmax or self.nd, self.nd)
        self.lhmax = min(lhmax or self.nh, self.nh)
        self.lwmax = min(lwmax or self.nw // 2 + 1, self.nw // 2 + 1)
        
        # half-modes
        self.ldmax_high = math.ceil(self.ldmax / 2)
        self.ldmax_low = math.floor(self.ldmax / 2)
        self.lhmax_high = math.ceil(self.lhmax / 2)
        self.lhmax_low = math.floor(self.lhmax / 2)

    def forward(self, x):
        y = torch.fft.rfftn(x, s=(self.nd, self.nh, self.nw), dim=(-3, -2, -1), norm="ortho")
        
        # truncate in w
        yt = y[..., :self.lwmax]
        
        # truncate in h
        yt = torch.cat([yt[..., :self.lhmax_high, :], y[..., -self.lhmax_low:, :]], dim=-2)
        
        # truncate in d
        y = torch.cat([yt[..., :self.ldmax_high, :, :], y[..., -self.ldmax_low:, :, :]], dim=-3)
        
        return y

class DistributedRealFFT3(nn.Module):
    """
    Helper routine to wrap FFT similarly to the SHT
    """
    def __init__(self,
                 nd,
                 nh,
                 nw,
                 ldmax = None,
                 lhmax = None,
                 lwmax = None):
        super(DistributedRealFFT3, self).__init__()

        # get the comms grid:
        self.comm_size_h = comm.get_size("h")
        self.comm_size_w = comm.get_size("w")
        self.comm_rank_w = comm.get_rank("w")

        # dimensions
        self.nd = nd
        self.nh = nh
        self.nw = nw
        self.ldmax = min(ldmax or self.nd, self.nd)
        self.lhmax = min(lhmax or self.nh, self.nh)
        self.lwmax = min(lwmax or self.nw // 2 + 1, self.nw // 2 + 1)
        
        # half-modes
        self.ldmax_high = math.ceil(self.ldmax / 2)
        self.ldmax_low = math.floor(self.ldmax / 2)
        self.lhmax_high = math.ceil(self.lhmax / 2)
        self.lhmax_low = math.floor(self.lhmax / 2)

        # frequency paddings: 
        # we assume the d-dim is always 
        # local, so we do not need padding here.
        lhdist = (self.lhmax + self.comm_size_h - 1) // self.comm_size_h
        self.lhpad = lhdist * self.comm_size_h - self.lhmax
        lwdist = (self.lwmax + self.comm_size_w - 1) // self.comm_size_w
        self.lwpad = lwdist * self.comm_size_w - self.lwmax


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # we need to ensure that we can split the channels evenly
        assert(x.dim() == 5)
        assert(x.shape[1] % self.comm_size_h == 0)
        assert(x.shape[1] % self.comm_size_w == 0)
        
        # h and w is split. First we make w local by transposing into channel dim
        if self.comm_size_w > 1:
            xt = distributed_transpose_w.apply(x, (1, 4))
        else:
            xt = x
        
        # do first 2D FFT
        xtf = torch.fft.rfft2(xt, s=(self.nd, self.nw), dim=(2, 4), norm="ortho")
        
        # truncate width-modes
        xtft = xtf[..., :self.lwmax]
        
        # truncate depth-modes
        xtftt = torch.cat([xtft[:, :, :self.ldmax_high, ...], 
                           xtft[:, :, -self.ldmax_low:, ...]], dim=2)
        
        # pad the dim to allow for splitting
        xtfp = F.pad(xtftt, (0, self.lwpad), mode="constant")

        # transpose: after this, m is split and c is local
        if self.comm_size_w > 1:
            y = distributed_transpose_w.apply(xtfp, (4, 1))
        else:
            y = xtfp
            
        # transpose: after this, c is split and h is local
        if self.comm_size_h > 1:
            yt = distributed_transpose_h.apply(y, (1, 3))
        else:
            yt = y

        # the input data might be padded, make sure to truncate to nlat:
        ytt = yt[..., :self.nh, :].contiguous()

        # do second FFT:
        yo = torch.fft.fft(ytt, n=self.nh, dim=3, norm="ortho")
        
        # truncate the modes
        yot = torch.cat([yo[..., :self.lhmax_high, :], 
                         yo[..., -self.lhmax_low:, :]], dim=3)

        # pad if required
        yop = F.pad(yot, (0, 0, 0, self.lhpad), mode="constant")
        
        # transpose: after this, l is split and c is local
        if self.comm_size_h > 1:
            y = distributed_transpose_h.apply(yop, (3, 1))
        else:
            y = yop

        return y
        
    
class InverseRealFFT3(nn.Module):
    """
    Helper routine to wrap FFT similarly to the SHT
    """
    def __init__(self,
                 nd,
                 nh,
                 nw,
                 ldmax = None,
                 lhmax = None,
                 lwmax = None):
        super(InverseRealFFT3, self).__init__()

        # dimensions
        self.nd = nd
        self.nh = nh
        self.nw = nw
        self.ldmax = min(ldmax or self.nd, self.nd)
        self.lhmax = min(lhmax or self.nh, self.nh)
        self.lwmax = min(lwmax or self.nw // 2 + 1, self.nw // 2 + 1)
        
        # half-modes
        self.ldmax_high = math.ceil(self.ldmax / 2)
        self.ldmax_low = math.floor(self.ldmax / 2)
        self.lhmax_high = math.ceil(self.lhmax / 2)
        self.lhmax_low = math.floor(self.lhmax / 2)

    def forward(self, x):

        # truncation is implicit but better do it manually
        xt = x[..., :self.lwmax]
           
        # pad in d 
        if (self.ldmax < self.nd):
            # pad
            xth = xt[..., :self.ldmax_high, :, :]
            xtl = xt[..., -self.ldmax_low:, :, :]
            xthp = F.pad(xth, (0,0,0,0,0,self.nd-self.ldmax))
            xt = torch.cat([xthp, xtl], dim=-3)
            
        # pad in h
        if (self.lhmax < self.nh):
            # pad
            xth = xt[..., :self.lhmax_high, :]
            xtl = xt[..., -self.lhmax_low:, :]
            xthp = F.pad(xth, (0,0,0,self.nh-self.lhmax))
            xt = torch.cat([xthp, xtl], dim=-3)
        
        out = torch.fft.irfftn(xt, s=(self.nd, self.nh, self.nw), dim=(-3, -2, -1), norm="ortho")
        
        return out
        

class DistributedInverseRealFFT3(nn.Module):
    """
    Helper routine to wrap FFT similarly to the SHT
    """
    def __init__(self,
                 nd,
                 nh,
                 nw,
                 ldmax = None,
                 lhmax = None,
                 lwmax = None):
        super(DistributedInverseRealFFT3, self).__init__()

        # get the comms grid:
        self.comm_size_h = comm.get_size("h")
        self.comm_size_w = comm.get_size("w")
        self.comm_rank_w = comm.get_rank("w")

        # dimensions
        self.nd = nd
        self.nh = nh
        self.nw = nw
        self.ldmax = min(ldmax or self.nd, self.nd)
        self.lhmax = min(lhmax or self.nh, self.nh)
        self.lwmax = min(lwmax or self.nw // 2 + 1, self.nw // 2 + 1)
        
        # half-modes
        self.ldmax_high = math.ceil(self.ldmax / 2)
        self.ldmax_low = math.floor(self.ldmax / 2)
        self.lhmax_high = math.ceil(self.lhmax / 2)
        self.lhmax_low = math.floor(self.lhmax / 2)

        # spatial paddings
        hdist = (self.nh + self.comm_size_h - 1) // self.comm_size_h
        self.hpad = hdist * self.comm_size_h - self.nh
        wdist = (self.nw + self.comm_size_w - 1) // self.comm_size_w
        self.wpad = wdist * self.comm_size_w - self.nw
        
        # frequency paddings
        lhdist = (self.lhmax + self.comm_size_h - 1) // self.comm_size_h
        self.lhpad = lhdist * self.comm_size_h - self.lhmax
        lwdist = (self.lwmax + self.comm_size_w - 1) // self.comm_size_w
        self.lwpad = lwdist * self.comm_size_w - self.lwmax


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # we need to ensure that we can split the channels evenly
        assert(x.dim() == 5)
        assert(x.shape[1] % self.comm_size_h == 0)
        assert(x.shape[1] % self.comm_size_w == 0)

        # transpose: after that, channels are split, l is local:
        if self.comm_size_h > 1:
            xt = distributed_transpose_h.apply(x, (1, 3))
        else:
            xt = x
            
        # truncate: assumes that data was padded at the END!
        # this is compatibel with the forward transform and easier to handle
        # in a distributed setting
        xtt = xt[..., :self.lhmax, :]

        # we should pad the middle here manually, so that the inverse FFT is correct
        if self.lhmax < self.nh:
            xtth = xtt[..., :self.lhmax_high, :]
            xttl = xtt[..., -self.lhmax_low, :]
            xtthp = F.pad(xtth, (0, 0, 0, self.nlat-self.lhmax), mode="constant")
            xtt = torch.cat([xtthp, xttl], dim=-2)
            
        if self.ldmax < self.nd:
            xtth = xtt[:, :, :self.ldmax_high, ...]
            xttl = xtt[:, :, -self.ldmax_low, ...]
            xtthp = F.pad(xtth, (0, 0, 0, 0, 0, self.nd-self.ldmax), mode="constant")
            xtt = torch.cat([xtthp, xttl], dim=-2)
        
        # do first fft
        xf = torch.fft.ifft2(xtt, s=(self.nd, self.nh), dim=(2, 3), norm="ortho")
        
        # transpose: after this, l is split and channels are local
        xfp = F.pad(xf, (0, 0, 0, self.hpad), mode="constant")

        if self.comm_size_h > 1:
            y = distributed_transpose_h.apply(xfp, (3, 1))
        else:
            y = xfp

        # transpose: after this, channels are split and m is local
        if self.comm_size_w > 1:
            yt = distributed_transpose_w.apply(y, (1, 4))
        else:
            yt = y

        # truncate
        ytt = yt[..., :self.lwmax]

        # apply the inverse (real) FFT
        x = torch.fft.irfft(ytt, n=self.nw, dim=-1, norm="ortho")
        
        # pad before we transpose back
        xp = F.pad(x, (0, self.wpad), mode="constant")

        # transpose: after this, m is split and channels are local
        if self.comm_size_w > 1:
            out = distributed_transpose_w.apply(xp, (4, 1))
        else:
            out = xp

        return out
