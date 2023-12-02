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
from modulus.experimental.sfno.utils.grids import GridQuadrature

class GeometricL1(torch.nn.Module):
    def __init__(self, grid_type,
                 img_shape, crop_shape=None, crop_offset=(0,0),
                 normalize=False,
                 channel_reduction='mean',
                 batch_reduction='mean'):
        super(GeometricL1, self).__init__()

        self.quadrature = GridQuadrature(grid_type,
                                         img_shape=img_shape,
                                         crop_shape=crop_shape,
                                         crop_offset=crop_offset,
                                         normalize=normalize)

        # determine how to reduce
        self.channel_reduction = channel_reduction
        self.batch_reduction = batch_reduction

    def forward(self, x, y):
        diff = self.quadrature(torch.abs(x-y))

        # reduce:
        if self.channel_reduction == 'mean':
            diff = torch.mean(diff, dim=1)
        elif self.channel_reduction == 'sum':
            diff = torch.sum(diff, dim=1)

        if self.batch_reduction == 'mean':
            diff = torch.mean(diff, dim=0)
        elif self.batch_reduction == 'sum':
            diff = torch.sum(diff, dim=0)
        
        return diff


class GeometricRMSE(torch.nn.Module):
    def __init__(self, grid_type,
                 img_shape, crop_shape=None, crop_offset=(0,0),
                 normalize=False,
                 channel_reduction='mean',
                 batch_reduction='mean'):
        super(GeometricRMSE, self).__init__()

        self.quadrature	= GridQuadrature(grid_type,
                                         img_shape=img_shape,
                                         crop_shape=crop_shape,
                                         crop_offset=crop_offset,
                                         normalize=normalize)
        
        # determine how to reduce
        self.channel_reduction = channel_reduction
        self.batch_reduction =	batch_reduction

    def forward(self, x, y):
        diff = self.quadrature(torch.square(x-y))

        # reduce:
        if self.channel_reduction == 'mean':
            diff = torch.mean(diff, dim=1)
        elif self.channel_reduction == 'sum':
            diff = torch.sum(diff, dim=1)

        if self.batch_reduction == 'mean':
            diff = torch.mean(diff, dim=0)
        elif self.batch_reduction == 'sum':
            diff = torch.sum(diff, dim=0)
        
        # compute square root:
        result = torch.sqrt(diff)

        return result


class GeometricACC(torch.nn.Module):
    def __init__(self, grid_type,
                 img_shape, crop_shape=None, crop_offset=(0,0),
                 normalize=False,
                 channel_reduction='mean',
                 batch_reduction='mean',
                 eps=1e-8):
        super(GeometricACC, self).__init__()
         
        self.eps = eps
        self.quadrature = GridQuadrature(grid_type,
                                         img_shape=img_shape,
                                         crop_shape=crop_shape,
                                         crop_offset=crop_offset,
                                         normalize=normalize)
         
        # determine how to reduce
        self.channel_reduction = channel_reduction
        self.batch_reduction =  batch_reduction
                 
    def forward(self, x, y):
        cov_xy = self.quadrature(x * y)
        var_x = self.quadrature(torch.square(x))
        var_y = self.quadrature(torch.square(y))

        # compute ratio
        acc = cov_xy / (torch.sqrt( var_x * var_y ) + self.eps)

        # reduce
        if self.channel_reduction == 'mean':
            acc = torch.mean(acc, dim=1)
        elif self.channel_reduction == 'sum':
            acc = torch.sum(acc, dim=1)

        if self.batch_reduction == 'mean':
            acc = torch.mean(acc, dim=0)
        elif self.batch_reduction == 'sum':
            acc = torch.sum(acc, dim=0)

        return acc

    
class SimpsonQuadrature(torch.nn.Module):
    def __init__(self, num_intervals, interval_width, device):
        super(SimpsonQuadrature, self).__init__()

        # set up integration weights
        weights = [0. for _ in range(num_intervals+1)]
        if (num_intervals % 2 == 0):
            # Simpsons 1/3
            for j in range(1, (num_intervals // 2 + 1)):
                weights[2*j-2] += 1.
                weights[2*j-1] += 4.
                weights[2*j] += 1.
            self.weights = torch.tensor(weights, dtype=torch.float32, device=device)
            self.weights *= interval_width/3.
        else:
            raise NotImplementedError("Error, please specify an even number of intervals")

    def forward(self, x, dim=1):
        # reshape weights to handle channels
        shape = [1 for _ in range(x.dim())]
        shape[dim] = -1
        weights = torch.reshape(self.weights, shape)

        return torch.sum(x * weights, dim=dim)


class TrapezoidQuadrature(torch.nn.Module):
    def __init__(self, num_intervals, interval_width, device):
        super(TrapezoidQuadrature, self).__init__()

        # set up integration weights
        weights = [interval_width for _ in range(num_intervals+1)]
        weights[0] *= 0.5
        weights[-1] *= 0.5
        self.weights = torch.tensor(weights, dtype=torch.float32, device=device)

    def forward(self, x, dim=1):
        shape = [1 for _ in range(x.dim())]
        shape[dim] = -1
        weights = torch.reshape(self.weights, shape)

        return torch.sum(x * weights, dim=dim)


class Quadrature(torch.nn.Module):
    def __init__(self, num_intervals, interval_width, device):
        super(Quadrature, self).__init__()
        if (num_intervals % 2 == 0):
            self.quad = SimpsonQuadrature(num_intervals, interval_width, device)
        else:
            self.quad = TrapezoidQuadrature(num_intervals, interval_width, device)

    def forward(self, x, dim=1):
        return self.quad(x, dim)
