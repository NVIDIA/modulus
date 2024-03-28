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

import numpy as np
import pyvista as pv
from scipy.interpolate import griddata
from numpy.fft import fft, fftfreq
from typing import List, Dict, Union, Tuple


def generate_mesh(
    nodes: np.ndarray, faces: np.ndarray, fields: np.ndarray
) -> pv.PolyData:
    """
    Generate mesh from given nodes, faces and fields arrays

    Args:
        nodes (np.ndarray): Nodes of the mesh
        faces (np.ndarray): Faces of the mesh
        fields (np.ndarray): Field values at each node

    Returns:
        pv.PolyData: Output mesh
    """
    points_3d = np.hstack([nodes, np.zeros((nodes.shape[0], 1))])
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces]).flatten()
    mesh = pv.PolyData(points_3d, faces_pv)
    for k, v in fields.items():
        mesh.point_data[k] = v

    return mesh


def compute_gradients(mesh: pv.PolyData, scalars: List[str]) -> pv.PolyData:
    """
    Compute the gradients of requested scalars for the given mesh

    Args:
        mesh (pv.PolyData): Input mesh
        scalars (List[str]): List of scalars to compute gradients for

    Returns:
        pv.PolyData: Output mesh with gradient information
    """

    for s in scalars:
        mesh = mesh.compute_derivative(scalars=s, gradient=f"grad_{s}")

    return mesh


def modulus_geometry_interpolator(
    mesh: pv.PolyData, modulus_geometry, num_samples: int
) -> Dict[str, np.ndarray]:
    """
    Interpolate mesh results on the boundary of a modulus geometry object

    Args:
        mesh (pv.PolyData): Input mesh
        modulus_geometry : Modulus Geometry
        num_samples (int): Number of samples

    Returns:
        Dict[str, np.ndarray]: Samples with interpolated data
    """

    samples = modulus_geometry.sample_boundary(num_samples)

    coords = np.concatenate((samples["x"], samples["y"]), axis=1)
    for k in mesh.point_data.keys():
        if k == "pyvistaOriginalPointIds":
            pass
        else:
            interp_vals = griddata(
                mesh.points.view(np.ndarray)[:, 0:2],
                mesh.point_data[k].view(np.ndarray),
                coords,
                method="linear",
            )

            samples[k] = interp_vals.reshape(-1, 1)

    return samples


def modulus_geometry_interior_interpolator(
    mesh: pv.PolyData, modulus_geometry, num_samples: int
) -> Dict[str, np.ndarray]:
    """
    Interpolate mesh results in the interior of a modulus geometry object

    Args:
        mesh (pv.PolyData): Input mesh
        modulus_geometry: Modulus Geometry
        num_samples (int): Number of samples

    Returns:
        Dict[str, np.ndarray]: Samples with interpolated data
    """
    samples = modulus_geometry.sample_interior(num_samples)

    coords = np.concatenate((samples["x"], samples["y"]), axis=1)
    for k in mesh.point_data.keys():
        if k == "pyvistaOriginalPointIds":
            pass
        else:
            interp_vals = griddata(
                mesh.points.view(np.ndarray)[:, 0:2],
                mesh.point_data[k].view(np.ndarray),
                coords,
                method="linear",
            )

            samples[k] = interp_vals.reshape(-1, 1)

    return samples


def dominant_freq_calc(signal: List[Union[int, float]]) -> float:
    """
    Compute the dominant frequency in the signal

    Args:
        signal (List[Union[int, float]]): Signal

    Returns:
        float: Dominant frequency
    """
    N = len(signal)
    yf = fft(signal)

    mag = np.abs(yf)
    mag[0] = 0
    dom_idx = np.argmax(mag)
    dom_freq = fftfreq(N, 1)[dom_idx]

    return np.abs(dom_freq)


def midpoint_data_interp(
    pt1: np.ndarray, pt2: np.ndarray, points: np.ndarray, field: np.ndarray
) -> np.ndarray:
    """
    Interpolate data on the midpoint of two points

    Args:
        pt1 (np.ndarray): Numpy array defining first point. Expected shape [1, 3]
        pt2 (np.ndarray): Numpy array defining second point. Expected shape [1, 3]
        points (np.ndarray): Numpy array containing all the points in the mesh.
        Expected shape [N, 3]
        field (np.ndarray): Numpy array containing field values at all the points in
        the mesh. Expected shape [N, m]

    Returns:
        np.ndarray: Value at the midpoint
    """
    idx1 = np.where(np.all(points == pt1, axis=1))[0]
    idx2 = np.where(np.all(points == pt2, axis=1))[0]

    return 0.5 * (field[idx1][0] + field[idx2][0])


def compute_line_integral(
    edges: Tuple[np.ndarray, np.ndarray],
    points: np.ndarray,
    field: np.ndarray,
    criteria: Dict[str, float] = None,
) -> float:
    """
    Compute integral along a line / edge. Currently assumes the curve in xy plane.

    Args:
        edges (Tuple[np.ndarray, np.ndarray]): Tuple of two points.
        Each point of [1, 3] shape
        points (np.ndarray): Points for the edge
        field (np.ndarray): Field values at each point
        criteria (Dict[str, float], optional): Criteria to sub-sample the integration
        points. Defaults to None.

    Returns:
        float: Integral along the given curve / line.
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
        midpt_val = midpoint_data_interp(e[0], e[1], points, field)

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
