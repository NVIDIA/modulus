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

import numpy as np
import torch

from torch_harmonics.quadrature import legendre_gauss_weights, clenshaw_curtiss_weights

class GridConverter(torch.nn.Module):
    def __init__(self, src_grid, dst_grid, lat_rad, lon_rad):
        super(GridConverter, self).__init__()
        self.src = src_grid
        self.dst = dst_grid
        self.src_lat = lat_rad
        self.src_lon = lon_rad

        if (self.src != self.dst):
            if (self.dst == "legendre-gauss"):
                cost_lg, _ = legendre_gauss_weights(lat_rad.shape[0], -1, 1)
                tq = torch.arccos(torch.from_numpy(cost_lg)) - torch.pi/2.
                self.dst_lat = tq.to(lat_rad.device)
                self.dst_lon = lon_rad

                # compute indices
                permutation = torch.arange(lat_rad.shape[0]-1, -1, -1).to(torch.long).to(lat_rad.device)
                jj = torch.searchsorted(lat_rad, self.dst_lat, sorter=permutation) - 1
                self.indices = jj[permutation]
            
                # compute weights
                self.interp_weights = ( (self.dst_lat - lat_rad[self.indices]) / torch.diff(lat_rad)[self.indices] ).reshape(-1, 1)
            else:
                raise NotImplementedError(f"Error, destination grid type {self.dst} not implemented.")
        else:
            self.dst_lat = self.src_lat
            self.dst_lon = self.src_lon

    def get_src_coords(self):
        return self.src_lat, self.src_lon
            
    def get_dst_coords(self):
        return self.dst_lat, self.dst_lon

    def forward(self, data):
        if self.src == self.dst:
            return data
        else:
            return torch.lerp(data[..., self.indices, :], data[..., self.indices+1, :], self.interp_weights.to(dtype=data.dtype))


class GridQuadrature(torch.nn.Module):
    def __init__(self, quadrature_rule, img_shape,
                 crop_shape=None, crop_offset=(0, 0),
                 normalize=False, pole_mask=None):
        super(GridQuadrature, self).__init__()

        if quadrature_rule == 'naive':
            jacobian = torch.clamp(torch.sin(torch.linspace(0, torch.pi, img_shape[0])), min=0.)
            dtheta = torch.pi / img_shape[0]
            dlambda =  2 * torch.pi / img_shape[1]
            dA = dlambda * dtheta
            quad_weight = dA * jacobian.unsqueeze(1)
            quad_weight = quad_weight.tile(1, img_shape[1])
            # numerical precision can be an issue here, make sure it sums to 4pi:
            quad_weight = quad_weight * (4. * torch.pi) / torch.sum(quad_weight)
        elif quadrature_rule == 'clenshaw-curtiss':
            cost, w = clenshaw_curtiss_weights(img_shape[0], -1, 1)
            weights = torch.from_numpy(w)
            dlambda = 2 * torch.pi / img_shape[1]
            quad_weight = dlambda * torch.from_numpy(w).unsqueeze(1)
            quad_weight = quad_weight.tile(1, img_shape[1])
        elif quadrature_rule == 'legendre-gauss':
            cost, w = legendre_gauss_weights(img_shape[0], -1, 1)
            weights = torch.from_numpy(w)
            dlambda = 2 * torch.pi / img_shape[1]
            quad_weight = dlambda * torch.from_numpy(w).unsqueeze(1)
            quad_weight = quad_weight.tile(1, img_shape[1])
        else:
            raise ValueError(f"Unknown quadrature rule {quadrature_rule}")

        # apply normalization
        if normalize:
            quad_weight = quad_weight / (4.*torch.pi)

        # apply pole mask
        if (pole_mask is not None) and (pole_mask > 0):
            quad_weight[:pole_mask, :] = 0.
            quad_weight[sizes[0]-pole_mask:, :] = 0.

        # crop globally if requested
        if crop_shape is not None:
            quad_weight = quad_weight[crop_offset[0]:crop_offset[0]+crop_shape[0],
                                      crop_offset[1]:crop_offset[1]+crop_shape[1]]

        # make it contiguous
        quad_weight = quad_weight.contiguous()
            
        # reshape
        H, W = quad_weight.shape
        quad_weight = quad_weight.reshape(1, 1, H, W)

        self.register_buffer('quad_weight', quad_weight)

    def forward(self, x):
        # integrate over last two axes only:
        return torch.sum(x * self.quad_weight, dim=(-2,-1))
