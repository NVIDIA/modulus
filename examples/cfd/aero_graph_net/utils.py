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

from typing import Any, Mapping, Optional

import torch
from torch import Tensor

from dgl import DGLGraph


class RRMSELoss(torch.nn.Module):
    """Relative RMSE loss."""

    def forward(self, pred: Tensor, target: Tensor):
        return (
            torch.linalg.vector_norm(pred - target) / torch.linalg.vector_norm(target)
        ).mean()


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


def batch_as_dict(
    batch, device: Optional[torch.device | str] = None
) -> Mapping[str, Any]:
    """Wraps provided batch in a dictionary, if needed.

    If `device` is not None, moves all Tensor items to the device.
    """

    batch = batch if isinstance(batch, Mapping) else {"graph": batch}
    if device is None:
        return batch
    return {
        k: v.to(device) if isinstance(v, (Tensor, DGLGraph)) else v
        for k, v in batch.items()
    }
