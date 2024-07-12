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

from typing import Optional
import torch


def compute_drag_coefficient(normals, area, coeff, p, s):
    """
    Compute drag coefficient for a given mesh.

    Parameters:
    -----------
    normals: Tensor
        The surface normals mapped onto nodes
    area: Tensor
        The surface areas of each cell mapped onto nodes
    coeff: Tensor
        Dynamic pressure times the frontal area
    p: Tensor
        Pressure distribution on the mesh
    s: Tensor
        Wall shear stress distribution on the mesh

    Returns:
    --------
    c_drag: float:
        Computed drag coefficient
    """

    # Compute coefficients
    c_p = coeff * torch.dot(normals[:, 0], area * p)
    c_f = -coeff * torch.dot(s[:, 0], area)

    # Compute total drag coefficients
    c_drag = c_p + c_f

    return c_drag


def relative_lp_error(pred, y, p=2):
    """
    Calculate relative L2 error norm
    Parameters:
    -----------
    pred: torch.Tensor
        Prediction
    y: torch.Tensor
        Ground truth
    Returns:
    --------
    error: float
        Calculated relative L2 error norm (percentage) on cpu
    """

    error = (
        torch.mean(torch.linalg.norm(pred - y, ord=p) / torch.linalg.norm(y, ord=p))
        .cpu()
        .numpy()
    )
    return error * 100



def degree(index: torch.Tensor, num_nodes: Optional[int] = None,
           dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    r"""Computes the (unweighted) degree of a given one-dimensional index
    tensor.

    Args:
        index (LongTensor): Index tensor.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
        dtype (:obj:`torch.dtype`, optional): The desired data type of the
            returned tensor.

    :rtype: :class:`Tensor`

    Example:
        >>> row = torch.tensor([0, 1, 0, 2, 0])
        >>> degree(row, dtype=torch.long)
        tensor([3, 1, 1])
    """
    N = torch.max(index) + 1
    N = int(N)
    out = torch.zeros((N, ), dtype=dtype, device=index.device)
    one = torch.ones((index.size(0), ), dtype=out.dtype, device=out.device)
    return out.scatter_add_(0, index, one)


def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src

def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)
