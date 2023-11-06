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

Tensor = torch.Tensor


def wasserstein(bin_edges: Tensor, cdf_x: Tensor, cdf_y: Tensor) -> Tensor:
    """1-Wasserstein distance between two discrete CDF functions

    This norm is typically used to compare two different forecast ensembles (for X and
    Y). Creates a map of distance and does not accumulate over lat/lon regions.
    Computes

    .. math::

        W(F_X, F_Y) = int[ |F_X(x) - F_Y(x)| ] dx

    where F_X is the empirical cdf of X and F_Y is the empirical cdf of Y.

    Parameters
    ----------
    bin_edges : Tensor
        Tensor containing bin edges. The leading dimension must represent the N+1 bin
        edges.
    cdf_x : Tensor
        Tensor containing a CDF one, defined over bins. The non-zeroth dimensions of
        bins and cdf must be compatible.
    cdf_y : Tensor
        Tensor containing a CDF two, defined over bins. Must be compatible with cdf_x in
        terms of bins and shape.

    Returns
    -------
    Tensor
        The 1-Wasserstein distance between cdf_x and cdf_y
    """
    return torch.sum(
        torch.abs(cdf_x - cdf_y) * (bin_edges[1, ...] - bin_edges[0, ...]), dim=0
    )
