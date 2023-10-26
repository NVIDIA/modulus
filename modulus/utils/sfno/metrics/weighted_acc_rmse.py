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
import torch.nn as nn

from modulus.utils.sfno import logging_utils

# distributed stuff
from modulus.utils.sfno.distributed import comm
from modulus.utils.sfno.distributed.mappings import reduce_from_parallel_region

logging_utils.config_logger()

# torch version for rmse comp
@torch.jit.script
def l1_torch_local(
    pred: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """Calculates the L1 loss between prediction and target in a local setting."""
    return nn.functional.l1_loss(pred, target)


def l1_torch_distributed(
    pred: torch.Tensor, target: torch.Tensor, group_name: str
) -> torch.Tensor:  # pragma: no cover
    """Calculates the L1 loss between prediction and target in a distributed setting."""
    res = nn.functional.l1_loss(pred, target)
    res = reduce_from_parallel_region(res, group_name) / float(
        comm.get_size(group_name)
    )
    return res


@torch.jit.script
def lat_torch(j: torch.Tensor, num_lat: int) -> torch.Tensor:  # pragma: no cover
    """Calculates the latitude in degrees."""
    return 90.0 - j * 180.0 / float(num_lat - 1)


@torch.jit.script
def latitude_weighting_factor_torch(
    lat: torch.Tensor,
) -> torch.Tensor:  # pragma: no cover
    """Calculates the latitude weighting factor."""
    cos_lat = torch.cos(torch.deg2rad(lat))
    cos_lat_norm = torch.sum(cos_lat)
    lwf = torch.clamp(cos_lat / cos_lat_norm, min=0.0) * float(cos_lat.shape[0])
    return lwf


@torch.jit.script
def weighted_rmse_torch_kernel(
    pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """Calculates the weighted rmse between prediction and target."""
    result = torch.mean(weight * torch.square(pred - target), dim=(-1, -2))
    return result


@torch.jit.script
def weighted_rmse_torch_local(
    pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """Calculates the weighted rmse between prediction and target in a local setting."""

    # compute the rmse
    res = weighted_rmse_torch_kernel(pred, target, weight)

    # average over batches
    res = torch.mean(torch.sqrt(res), dim=0)

    return res


def weighted_rmse_torch_distributed(
    pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, group_name: str
) -> torch.Tensor:  # pragma: no cover
    """Calculates the weighted rmse between prediction and target in a distributed
    setting."""

    # compute the local rmse
    res = weighted_rmse_torch_kernel(pred, target, weight)

    # perform model parallel mean:
    res = reduce_from_parallel_region(res, group_name) / float(
        comm.get_size(group_name)
    )

    # average over batches
    res = torch.mean(torch.sqrt(res), dim=0)

    return res


# FIXME: needs to be adopted like above
@torch.jit.script
def weighted_acc_torch_kernel(
    pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor
):  # pragma: no cover
    """takes in arrays of size [n, c, h, w]  and returns latitude-weighted acc"""
    cov = torch.sum(weight * pred * target, dim=(-1, -2))
    var1 = torch.sum(weight * torch.square(pred), dim=(-1, -2))
    var2 = torch.sum(weight * torch.square(target), dim=(-1, -2))

    return cov, var1, var2


@torch.jit.script
def weighted_acc_torch_local(
    pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """Calculates the weighted acc between prediction and target in a local
    setting."""
    eps = 1e-6
    cov, var1, var2 = weighted_acc_torch_kernel(pred, target, weight)
    res = cov / (torch.sqrt(var1 * var2) + eps)

    # average over batches
    res = torch.mean(res, dim=0)

    return res


def weighted_acc_torch_local_no_reduction(
    pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:  # pragma: no cover
    """Calculates the weighted acc between prediction and target in a local
    setting without averaging."""
    eps = 1e-6
    cov, var1, var2 = weighted_acc_torch_kernel(pred, target, weight)
    res = cov / (torch.sqrt(var1 * var2) + eps)

    return res


def weighted_acc_torch_distributed(
    pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, group_name: str
) -> torch.Tensor:  # pragma: no cover
    """Calculates the weighted acc between prediction and target in a distributed
    setting."""
    eps = 1e-6
    cov, var1, var2 = weighted_acc_torch_kernel(pred, target, weight)

    # reductions:
    cov = reduce_from_parallel_region(cov, group_name)
    var1 = reduce_from_parallel_region(var1, group_name)
    var2 = reduce_from_parallel_region(var2, group_name)

    # compute ratio
    res = cov / (torch.sqrt(var1 * var2) + eps)

    # average over batches
    res = torch.mean(res, dim=0)

    return res


class SimpsonQuadrature(nn.Module):
    """Implements the Simpson's rule for numerical integration."""

    def __init__(self, num_intervals, interval_width, device):  # pragma: no cover
        super(SimpsonQuadrature, self).__init__()

        # set up integration weights
        weights = [0.0 for _ in range(num_intervals + 1)]
        if num_intervals % 2 == 0:
            # Simpsons 1/3
            for j in range(1, (num_intervals // 2 + 1)):
                weights[2 * j - 2] += 1.0
                weights[2 * j - 1] += 4.0
                weights[2 * j] += 1.0
            self.weights = torch.tensor(weights, dtype=torch.float32, device=device)
            self.weights *= interval_width / 3.0
        else:
            raise NotImplementedError(
                "Error, please specify an even number of intervals"
            )

    def forward(self, x, dim=1):  # pragma: no cover
        # reshape weights to handle channels
        shape = [1 for _ in range(x.dim())]
        shape[dim] = -1
        weights = torch.reshape(self.weights, shape)

        return torch.sum(x * weights, dim=dim)


class TrapezoidQuadrature(nn.Module):
    """Implements the trapezoidal rule for numerical integration."""

    def __init__(self, num_intervals, interval_width, device):  # pragma: no cover
        super(TrapezoidQuadrature, self).__init__()

        # set up integration weights
        weights = [interval_width for _ in range(num_intervals + 1)]
        weights[0] *= 0.5
        weights[-1] *= 0.5
        self.weights = torch.tensor(weights, dtype=torch.float32, device=device)

    def forward(self, x, dim=1):  # pragma: no cover
        # reshape weights to handle channels
        shape = [1 for _ in range(x.dim())]
        shape[dim] = -1
        weights = torch.reshape(self.weights, shape)

        return torch.sum(x * weights, dim=dim)


class Quadrature(nn.Module):
    """Implements the numerical integration using either Simpson's or Trapezoid rule."""

    def __init__(self, num_intervals, interval_width, device):  # pragma: no cover
        super(Quadrature, self).__init__()
        if num_intervals % 2 == 0:
            self.quad = SimpsonQuadrature(num_intervals, interval_width, device)
        else:
            self.quad = TrapezoidQuadrature(num_intervals, interval_width, device)

    def forward(self, x, dim=1):  # pragma: no cover
        return self.quad(x, dim)
