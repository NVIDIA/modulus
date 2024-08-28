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

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pyvista as pv


def _midpoint_data_interp(
    pt1: np.ndarray, pt2: np.ndarray, points: np.ndarray, field: np.ndarray
) -> np.ndarray:
    """
    Interpolate data on the midpoint of two points

    Parameters:
    -----------
    pt1 : np.ndarray
        Numpy array defining first point. Expected shape [1, 3]
    pt2 : np.ndarray
        Numpy array defining second point. Expected shape [1, 3]
    points : np.ndarray
        Numpy array containing all the points in the mesh. Expected shape [N, 3]
    field : np.ndarray
        Numpy array containing field values at all the points in the mesh.
        Expected shape [N, m]

    Returns:
    --------
    np.ndarray
        Value at the midpoint
    """
    idx1 = np.where(np.all(points == pt1, axis=1))[0]
    idx2 = np.where(np.all(points == pt2, axis=1))[0]

    return 0.5 * (field[idx1][0] + field[idx2][0])


def line_integral(
    edges: Tuple[np.ndarray, np.ndarray],
    points: np.ndarray,
    field: np.ndarray,
    criteria: Dict[str, float] = None,
) -> float:
    """
    Compute integral along a line / edge. Currently assumes the curve in xy plane.

    Parameters:
    -----------
    edges : Tuple[np.ndarray, np.ndarray]
        Tuple of two points. Each point of [1, 3] shape
    points : np.ndarray
        Points for the edge
    field : np.ndarray
        Field values at each point
    criteria : Dict[str, float], optional
        Criteria to sub-sample the integration points. Defaults to None.

    Returns:
    --------
    float
        Integral along the given curve / line.
    """
    midpts = []
    midpt_vals = []
    normals = []
    lengths = []

    # TODO: generalize to different planes
    for e in edges:
        vec = e[0] - e[1]
        normal = [vec[1], -vec[0], vec[2]]
        normal = normal / np.linalg.norm(normal)
        midpt = (e[0] + e[1]) / 2
        midpt_val = _midpoint_data_interp(e[0], e[1], points, field)

        midpts.append(midpt)
        midpt_vals.append(midpt_val)
        normals.append(normal)
        lengths.append(np.linalg.norm(e[0] - e[1]))

    midpts = np.array(midpts)
    normals = np.array(normals)
    midpt_vals = np.array(midpt_vals)
    lengths = np.array(lengths).reshape(-1, 1)

    if criteria is not None:
        idx = (
            (midpts[:, 0] >= criteria["x_min"])
            & (midpts[:, 0] <= criteria["x_max"])
            & (midpts[:, 1] >= criteria["y_min"])
            & (midpts[:, 1] <= criteria["y_max"])
        )
        midpts = midpts[idx]
        normals = normals[idx]
        midpt_vals = midpt_vals[idx]
        lengths = lengths[idx]

    integral = np.sum(normals * midpt_vals * lengths, axis=0) / np.sum(lengths, axis=0)
    return integral


def surface_integral(
    mesh: pv.PolyData,
    data_type: str = "point_data",
    array_name: Optional[Union[str, List[str]]] = None,
) -> Dict[np.ndarray]:
    """
    Computes the surface integral of a given mesh

    Parameters:
    -----------
    mesh : pv.PolyData
        Mesh data with field information
    data_type : str, optional
        Whether to use "cell_data" or "point_data" to integrate. by default "point_data"
    array_name : Optional[Union[str, List[str]]], optional
        Array names to integrate. by default None which integrates all arrays.

    Returns:
    --------
    Dict[np.ndarray]
        Dictionary containing surface integrals for requested arrays.
    """
    # compute normals
    mesh = mesh.compute_normals()

    data = getattr(mesh, data_type)

    if not data.keys():
        raise ValueError(
            f"No data arrays found using type: {data_type}, try switching between 'point_data' and 'cell_data'"
        )

    available_vars = data.keys()

    if array_name is None:
        data_arr = data.keys()
    else:
        if not isinstance(array_name, (list, tuple)):
            data_arr = [array_name]
        else:
            data_arr = array_name

        if not set(data_arr) <= set(available_vars):
            raise ValueError(
                f"Requested vars not found in the provided mesh file. Choose from {available_vars}"
            )

    for arr in data_arr:
        if len(data[arr].shape) == 1:
            # Scalar quantity
            data[f"integral_{arr}"] = data[arr]
        elif len(data[arr].shape) == 2:
            # Vector quantity
            data[f"integral_{arr}"] = np.sum(data[arr] * data["Normals"], axis=1)

    # integrate the results
    integrated = mesh.integrate_data()

    results = {}
    for arr in data_arr:
        results[f"integral_{arr}"] = np.array(integrated[f"integral_{arr}"])

    return results
