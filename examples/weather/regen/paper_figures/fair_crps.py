# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

from typing import Union

import numpy as np
import torch

Tensor = torch.Tensor


@torch.jit.script
def _kernel_crps_implementation(pred: Tensor, obs: Tensor, biased: bool) -> Tensor:
    """An O(m log m) implementation of the kernel CRPS formulas"""
    skill = torch.abs(pred - obs[..., None]).mean(-1)
    pred, _ = torch.sort(pred)

    # derivation of fast implementation of spread-portion of CRPS formula when x is sorted
    # sum_(i,j=1)^m |x_i - x_j| = sum_(i<j) |x_i -x_j| + sum_(i > j) |x_i - x_j|
    #                           = 2 sum_(i <= j) |x_i -x_j|
    #                           = 2 sum_(i <= j) (x_j - x_i)
    #                           = 2 sum_(i <= j) x_j - 2 sum_(i <= j) x_i
    #                           = 2 sum_(j=1)^m j x_j - 2 sum (m - i + 1) x_i
    #                           = 2 sum_(i=1)^m (2i - m - 1) x_i
    m = pred.size(-1)
    i = torch.arange(1, m + 1, device=pred.device, dtype=pred.dtype)
    denom = m * m if biased else m * (m - 1)
    factor = (2 * i - m - 1) / denom
    spread = torch.sum(factor * pred, dim=-1)
    return skill - spread


def kcrps(pred: Tensor, obs: Tensor, dim: int = 0, biased: bool = True):
    """Estimate the CRPS from a finite ensemble

    Computes the local Continuous Ranked Probability Score (CRPS) by using
    the kernel version of CRPS. The cost is O(m log m).

    Creates a map of CRPS and does not accumulate over lat/lon regions.
    Approximates:
    .. math::
        CRPS(X, y) = E[X - y] - 0.5 E[X-X']

    with
    .. math::
        sum_i=1^m |X_i - y| / m - 1/(2m^2) sum_i,j=1^m |x_i - x_j|

    Parameters
    ----------
    pred : Tensor
        Tensor containing the ensemble predictions. The ensemble dimension
        is assumed to be the leading dimension unless 'dim' is specified.
    obs : Union[Tensor, np.ndarray]
        Tensor or array containing an observation over which the CRPS is computed
        with respect to.
    dim : int, optional
        The dimension over which to compute the CRPS, assumed to be 0.
    biased :
        When False, uses the unbiased estimators described in (Zamo and Naveau, 2018)::

            E|X-y|/m - 1/(2m(m-1)) sum_(i,j=1)|x_i - x_j|

        Unlike ``crps`` this is fair for finite ensembles. Non-fair ``crps`` favors less
        dispersive ensembles since it is biased high by E|X- X'|/ m where m is the
        ensemble size.

    Returns
    -------
    Tensor
        Map of CRPS
    """
    pred = torch.movedim(pred, dim, -1)
    return _kernel_crps_implementation(pred, obs, biased=biased)


def _crps_gaussian(mean: Tensor, std: Tensor, obs: Union[Tensor, np.ndarray]) -> Tensor:
    """
    Computes the local Continuous Ranked Probability Score (CRPS)
    using assuming that the forecast distribution is normal.

    Creates a map of CRPS and does not accumulate over lat/lon regions.

    Computes:

    .. math::

        CRPS(mean, std, y) = std * [ \\frac{1}{\\sqrt{\\pi}}} - 2 \\phi ( \\frac{x-mean}{std} ) -
                ( \\frac{x-mean}{std} ) * (2 \\Phi(\\frac{x-mean}{std}) - 1) ]

    where \\phi and \\Phi are the normal gaussian pdf/cdf respectively.

    Parameters
    ----------
    mean : Tensor
        Tensor of mean of forecast distribution.
    std : Tensor
        Tensor of standard deviation of forecast distribution.
    obs : Union[Tensor, np.ndarray]
        Tensor or array containing an observation over which the CRPS is computed
        with respect to. Broadcasting dimensions must be compatible with the non-zeroth
        dimensions of bins and cdf.

    Returns
    -------
    Tensor
        Map of CRPS
    """
    if isinstance(obs, np.ndarray):
        obs = torch.from_numpy(obs).to(mean.device)
    # Check shape compatibility
    if mean.shape != std.shape:
        raise ValueError(
            "Mean and standard deviation must have"
            + "compatible shapes but found"
            + str(mean.shape)
            + " and "
            + str(std.shape)
            + "."
        )
    if mean.shape != obs.shape:
        raise ValueError(
            "Mean and obs must have"
            + "compatible shapes but found"
            + str(mean.shape)
            + " and "
            + str(obs.shape)
            + "."
        )

    d = (obs - mean) / std
    phi = torch.exp(-0.5 * d**2) / torch.sqrt(torch.as_tensor(2 * torch.pi))

    # Note, simplified expression below is not exactly Gaussian CDF
    Phi = torch.erf(d / torch.sqrt(torch.as_tensor(2.0)))

    return std * (2 * phi + d * Phi - 1.0 / torch.sqrt(torch.as_tensor(torch.pi)))
