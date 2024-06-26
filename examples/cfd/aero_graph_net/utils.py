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
import numpy as np
import pyvista as pv
import shapely

from dgl import DGLGraph


class RRMSELoss(torch.nn.Module):
    """Relative RMSE loss."""

    def forward(self, pred: Tensor, target: Tensor):
        return (
            torch.linalg.vector_norm(pred - target) / torch.linalg.vector_norm(target)
        ).mean()


def compute_frontal_area(mesh: pv.PolyData, direction: str = "x"):
    """
    Compute frontal area of a given mesh
    Ref: https://github.com/pyvista/pyvista/discussions/5211#discussioncomment-7794449

    Parameters:
    -----------
    mesh: pv.PolyData
        Input mesh
    direction: str, optional
        Direction to project the area. Defaults to "x".

    Raises:
    -------
    ValueError: Only supports x, y and z projection for computing frontal area.

    Returns:
    --------
    frontal area: float
        Frontal area of the mesh in the given direction
    """
    direction_map = {
        "x": ((1, 0, 0), [1, 2]),
        "y": ((0, 1, 0), [0, 2]),
        "z": ((0, 0, 1), [0, 1]),
    }

    if direction not in direction_map:
        raise ValueError("Direction must be x, y or z only")

    normal, indices = direction_map[direction]
    areas_proj = mesh.project_points_to_plane(origin=(0, 0, 0), normal=normal)
    merged = shapely.union_all(
        [
            shapely.Polygon(
                np.stack(
                    [
                        areas_proj.points[tri, indices[0]],
                        areas_proj.points[tri, indices[1]],
                    ],
                    axis=1,
                )
            )
            for tri in areas_proj.triangulate().regular_faces
        ]
    )

    return merged.area


def compute_force_coefficients(
    normals: np.ndarray,
    area: np.ndarray,
    coeff: float,
    p: np.ndarray,
    wss: np.ndarray,
    force_direction: np.ndarray = np.array([1, 0, 0]),
):
    """
    Computes force coefficients for a given mesh. Output includes the pressure and skin
    friction components. Can be used to compute lift and drag.
    For drag, use the `force_direction` as the direction of the motion,
    e.g. [1, 0, 0] for flow in x direction.
    For lift, use the `force_direction` as the direction perpendicular to the motion,
    e.g. [0, 1, 0] for flow in x direction and weight in y direction.

    Parameters:
    -----------
    normals: np.ndarray
        The surface normals on cells of the mesh
    area: np.ndarray
        The surface areas of each cell
    coeff: float
        Reciprocal of dynamic pressure times the frontal area, i.e. 2/(A * rho * U^2)
    p: np.ndarray
        Pressure distribution on the mesh (on each cell)
    wss: np.ndarray
        Wall shear stress distribution on the mesh (on each cell)
    force_direction: np.ndarray
        Direction to compute the force, default is np.array([1, 0, 0])

    Returns:
    --------
    c_total: float
        Computed total force coefficient
    c_p: float
        Computed pressure force coefficient
    c_f: float
        Computed skin friction coefficient
    """

    # Compute coefficients
    c_p = coeff * np.sum(np.dot(normals, force_direction) * area * p)
    c_f = -coeff * np.sum(np.dot(wss, force_direction) * area)

    # Compute total force coefficients
    c_total = c_p + c_f

    return c_total, c_p, c_f


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
