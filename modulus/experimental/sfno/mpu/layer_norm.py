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

import torch
import torch.nn as nn

from torch.cuda import amp

from typing import Tuple

# for spatial model-parallelism
from modulus.experimental.sfno.utils import comm
from modulus.experimental.sfno.mpu.mappings import gather_from_parallel_region, copy_to_parallel_region

class DistributedInstanceNorm2d(nn.Module):
    """
    Computes a distributed instance norm using Welford's online algorithm
    """
    def __init__(self,
                 num_features,
                 eps=1e-05,
                 affine=False,
                 device=None,
                 dtype=None):
        super(DistributedInstanceNorm2d, self).__init__()

        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias   = nn.Parameter(torch.zeros(num_features))
            self.weight.is_shared_mp = ["spatial"] #["h", "w"]
            self.bias.is_shared_mp   = ["spatial"] #["h", "w"]

        self.gather_mode = 'welford'

    @torch.jit.ignore
    def _gather_hw(self, x: torch.Tensor) -> torch.Tensor:
	    # gather the data over the spatial communicator
        xh = gather_from_parallel_region(x, -2, "h")
        xw = gather_from_parallel_region(xh, -1, "w")
        return xw

    @torch.jit.ignore
    def _gather_spatial(self, x: torch.Tensor) -> torch.Tensor:
	    # gather the data over the spatial communicator
        xs = gather_from_parallel_region(x, -1, "spatial")
        return xs

    def _stats_naive(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the statistics in the naive way by first gathering the tensors and then computing them"""
    
        x = self._gather_hw(x)
        var, mean = torch.var_mean(x, dim=(-2,-1), unbiased=False, keepdim=True)

        return var, mean

    def _stats_welford(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes the statistics locally, then uses the Welford online algorithm to reduce them"""
        
        var, mean = torch.var_mean(x, dim=(-2,-1), unbiased=False, keepdim=False)
        # workaround to not use shapes, as otherwise cuda graphs won't work
        count = torch.ones_like(x[0,0], requires_grad=False)
        count = torch.sum(count, dim=(-2,-1), keepdim=False)

        vars = self._gather_spatial(var.unsqueeze(-1))
        means = self._gather_spatial(mean.unsqueeze(-1))
        counts = self._gather_spatial(count.unsqueeze(-1))

        m2s = vars * counts

        mean = means[..., 0]
        m2 = m2s[..., 0]
        count = counts[..., 0]

        # use Welford's algorithm to accumulate them into a single mean and variance
        for i in range(1, comm.get_size("spatial")):
            delta = means[..., i] - mean
            m2 = m2 + m2s[..., i] + delta**2 * count * counts[..., i] / (count + counts[..., i])
            if i == 1:
                mean = (mean*count + means[..., i]*counts[..., i]) / (count + counts[..., i])
            else:
                mean = mean + delta * counts[..., i] / (count + counts[..., i])

            # update the current count
            count = count + counts[..., i]

        var = m2 / count

        var = var.reshape(1, -1, 1, 1)
        mean = mean.reshape(1, -1, 1, 1)

        return var, mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        with amp.autocast(enabled=False):
            dtype = x.dtype
            x = x.float()

            # start by computing std and mean
            if self.gather_mode == 'naive':
                var, mean = self._stats_naive(x)
            elif self.gather_mode == 'welford':
                var, mean = self._stats_welford(x)
            else:
                raise ValueError(f"Unknown gather mode {self.gather_mode}")

            # this is absolutely necessary to get the correct graph in the backward pass
            mean = copy_to_parallel_region(mean, "spatial")
            var = copy_to_parallel_region(var, "spatial")

        x = x.to(dtype)
        mean = mean.to(dtype)
        var = var.to(dtype)

        # apply the normalization
        x = (x - mean) / torch.sqrt(var + self.eps)

        # affine transform if we use it
        if self.affine:
            x = self.weight.reshape(-1, 1, 1) * x + self.bias.reshape(-1, 1, 1)

        return x
        
        
class DistributedLayerNorm(nn.Module):
    """
    This is a lightweight wrapper which only computed norm across channels.
    This norm breaks equivariance since the norm across channels is different per grid
    point.
    """
    def __init__(self,
                 normalized_shape,
                 eps=1e-05,
                 elementwise_affine=True,
                 bias=True,
                 device=None,
                 dtype=None):
        super(DistributedLayerNorm, self).__init__()
        
        assert(comm.get_size("matmul") == 1)
        
        self.norm = nn.LayerNorm(normalized_shape,
                                 eps=eps,
                                 elementwise_affine=elementwise_affine,
                                 bias=bias, device=device, dtype=dtype)

        if elementwise_affine:
            # set up weight sharing and sharding
            self.norm.weight.is_shared_mp = ["model"]
            self.norm.weight.sharded_dims_mp = [None]
            if bias:
                self.norm.bias.is_shared_mp = ["model"]
                self.norm.bias.sharded_dims_mp = [None]


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # assume input is NCHW, so we transpose
        xt = torch.transpose(x, 1, 3)
        xn = self.norm(xt)
        x = torch.transpose(xn, 1, 3).contiguous()

        return x
