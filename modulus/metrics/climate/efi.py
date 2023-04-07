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

# TODO(Dallas) Introduce Distributed Class for computation.

import torch
from modulus.metrics.general.histogram import normal_pdf, normal_cdf, histogram
from modulus.metrics.general.entropy import _entropy_from_counts

Tensor = torch.Tensor


def efi(
    pred_cdf: Tensor,
    bin_edges: Tensor,
    climatology_mean: Tensor,
    climatology_std: Tensor,
) -> Tensor:
    """Calculates the Extreme Forecast Index (EFI) for an ensemble forecast against
    a climatological distribution.

    Parameters
    ----------
    pred_cdf : Tensor
        Cumulative distribution function of predictions of shape [N, ...]
        where N is the number of bins. This cdf must be defined over the
        passed bin_edges.
    bin_edges : Tensor
        Tensor of bin edges with shape [N+1, ...]
        where N is the number of bins.
    climatology_mean : Tensor
        Tensor of climatological mean with shape [...]
    climatology_std : Tensor
        Tensor of climatological std with shape [...]

    Returns
    -------
    Tensor
        EFI values of each of the batched dimensions.

    Note
    ----
    Reference: https://www.atmos.albany.edu/daes/atmclasses/atm401/spring_2016/ppts_pdfs/ECMWF_EFI.pdf
    """

    clim_cdf = normal_cdf(climatology_mean, climatology_std, bin_edges, grid="right")

    return torch.trapz(
        (clim_cdf - pred_cdf) / torch.sqrt(1e-8 + clim_cdf * (1.0 - clim_cdf)),
        clim_cdf,
        dim=0,
    )


def normalized_entropy(
    pred_pdf: Tensor,
    bin_edges: Tensor,
    climatology_mean: Tensor,
    climatology_std: Tensor,
) -> Tensor:
    """Calculates the relative entropy, or surprise, of using the prediction distribution as opposed to the
    climatology distribution.

    Parameters
    ----------
    pred_cdf : Tensor
        Cumulative distribution function of predictions of shape [N, ...]
        where N is the number of bins. This cdf must be defined over the
        passed bin_edges.
    bin_edges : Tensor
        Tensor of bin edges with shape [N+1, ...]
        where N is the number of bins.
    climatology_mean : Tensor
        Tensor of climatological mean with shape [...]
    climatology_std : Tensor
        Tensor of climatological std with shape [...]

    Returns
    -------
    Tensor
        Relative Entropy values of each of the batched dimensions.

    Note
    ----
    Reference: https://www.atmos.albany.edu/daes/atmclasses/atm401/spring_2016/ppts_pdfs/ECMWF_EFI.pdf
    """

    clim_pdf = normal_pdf(climatology_mean, climatology_std, bin_edges, grid="right")

    return 1.0 - _entropy_from_counts(pred_pdf, bin_edges) / _entropy_from_counts(
        clim_pdf, bin_edges
    )
