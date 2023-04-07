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
import numpy as np
from typing import Union
from .histogram import cdf as cdf_function

Tensor = torch.Tensor


def _crps_gaussian(mean: Tensor, std: Tensor, obs: Union[Tensor, np.ndarray]) -> Tensor:
    """Computes the local Continuous Ranked Probability Score (CRPS)
    using assuming that the forecast distribution is normal.

    Creates a map of CRPS and does not accumulate over lat/lon regions.
    Computes:

    .. math:
        CRPS(mean, std, y) = std * [ \\frac{1}{\\pi} - 2 \\phi ( \\frac{x-mean}{std} ) -
                ( \\frac{x-mean}{std} ) * (2 \\Phi(\\frac{x-mean}{std}) - 1) ]

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
    assert mean.shape == std.shape, (
        "Mean and standard deviation must have"
        + "compatible shapes but found"
        + str(mean.shape)
        + " and "
        + str(std.shape)
        + "."
    )
    assert mean.shape == obs.shape, (
        "Mean and obs must have"
        + "compatible shapes but found"
        + str(mean.shape)
        + " and "
        + str(obs.shape)
        + "."
    )

    d = (obs - mean) / std
    phi = torch.exp(-0.5 * d**2) / torch.sqrt(torch.as_tensor(2 * torch.pi))
    Phi = torch.erf(d / torch.sqrt(torch.as_tensor(2.0)))

    return 2 * phi + (obs - mean) * Phi - std / torch.sqrt(torch.as_tensor(torch.pi))


def _crps_from_cdf(
    bin_edges: Tensor, cdf: Tensor, obs: Union[Tensor, np.ndarray]
) -> Tensor:
    """Computes the local Continuous Ranked Probability Score (CRPS)
    using a cumulative distribution function.

    Creates a map of CRPS and does not accumulate over lat/lon regions.
    Computes:
        CRPS(X, y) = int[ (F(x) - 1[x - y])^2 ] dx
        where F is the empirical cdf of X.

    Parameters
    ----------
    bins_edges : Tensor
        Tensor [N+1, ...] containing bin edges. The leading dimension must represent the
        N+1 bin edges.
    cdf : Tensor
        Tensor [N, ...] containing a cdf, defined over bins. The non-zeroth dimensions
        of bins and cdf must be compatible.
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
        obs = torch.from_numpy(obs).to(cdf.device)
    assert bin_edges.shape[1:] == cdf.shape[1:], (
        "Expected bins and cdf to have compatible non-zeroth dimensions but have shapes"
        + str(bin_edges.shape[1:])
        + " and "
        + str(cdf.shape[1:])
        + "."
    )
    assert bin_edges.shape[1:] == obs.shape, (
        "Expected bins and observations to have compatible broadcasting dimensions but have shapes"
        + str(bin_edges.shape[1:])
        + " and "
        + str(obs.shape)
        + "."
    )
    assert bin_edges.shape[0] == cdf.shape[0] + 1, (
        "Expected zeroth dimension of cdf to be equal to the zeroth dimension of bins + 1 but have shapes"
        + str(bin_edges.shape[0])
        + " and "
        + str(cdf.shape[0])
        + "+1."
    )
    dbins = bin_edges[1, ...] - bin_edges[0, ...]
    bin_mids = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    obs = torch.ge(bin_mids, obs).int()
    return torch.sum(torch.abs(cdf - obs) ** 2 * dbins, dim=0)


def _crps_from_counts(
    bin_edges: Tensor, counts: Tensor, obs: Union[Tensor, np.ndarray]
) -> Tensor:
    """Computes the local Continuous Ranked Probability Score (CRPS)
    using a histogram of counts.

    Creates a map of CRPS and does not accumulate over lat/lon regions.
    Computes:
        CRPS(X, y) = int[ (F(x) - 1[x - y])^2 ] dx
        where F is the empirical cdf of X.

    Parameters
    ----------
    bins_edges : Tensor
        Tensor [N+1, ...] containing bin edges. The leading dimension must represent the
        N+1 bin edges.
    counts : Tensor
        Tensor [N, ...] containing counts, defined over bins. The non-zeroth dimensions
        of bins and counts must be compatible.
    obs : Union[Tensor, np.ndarray]
        Tensor or array containing an observation over which the CRPS is computed
        with respect to. Broadcasting dimensions must be compatible with the non-zeroth
        dimensions of bins and counts.

    Returns
    -------
    Tensor
        Map of CRPS
    """
    if isinstance(obs, np.ndarray):
        obs = torch.from_numpy(obs).to(counts.device)
    assert bin_edges.shape[1:] == counts.shape[1:], (
        "Expected bins and cdf to have compatible non-zeroth dimensions but have shapes"
        + str(bin_edges.shape[1:])
        + " and "
        + str(counts.shape[1:])
        + "."
    )
    assert bin_edges.shape[1:] == obs.shape, (
        "Expected bins and observations to have compatible broadcasting dimensions but have shapes"
        + str(bin_edges.shape[1:])
        + " and "
        + str(obs.shape)
        + "."
    )
    assert bin_edges.shape[0] == counts.shape[0] + 1, (
        "Expected zeroth dimension of cdf to be equal to the zeroth dimension of bins + 1 but have shapes"
        + str(bin_edges.shape[0])
        + " and "
        + str(counts.shape[0])
        + "+1."
    )
    cdf_hat = torch.cumsum(counts / torch.sum(counts, dim=0), dim=0)
    return _crps_from_cdf(bin_edges, cdf_hat, obs)


def crps(
    pred: Tensor, obs: Union[Tensor, np.ndarray], bins: Union[int, Tensor] = 10
) -> Tensor:
    """
    Computes the local Continuous Ranked Probability Score (CRPS) by computing
    a histogram and CDF of the predictions.

    Creates a map of CRPS and does not accumulate over lat/lon regions.
    Computes:
        CRPS(X, y) = int[ (F(x) - 1[x - y])^2 ] dx
        where F is the empirical cdf of X.

    Parameters
    ----------
    pred : Tensor
        Tensor [B, ...] containing the ensemble predictions. The leading dimension must represent the
        ensemble dimension.
    obs : Union[Tensor, np.ndarray]
        Tensor or array containing an observation over which the CRPS is computed
        with respect to. Broadcasting dimensions must be compatible with the non-zeroth
        dimensions of bins and cdf.
    bins : Union[int, Tensor], optional
        Either the number of bins, or a tensor of bin edges with dimension [N+1, ...]
        where N is the number of bins, by default 10.

    Returns
    -------
    Tensor
        Map of CRPS
    """
    bin_edges, cdf = cdf_function(pred, bins=bins)
    return _crps_from_cdf(bin_edges, cdf, obs)
