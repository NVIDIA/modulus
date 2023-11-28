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
from torch import Tensor


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
