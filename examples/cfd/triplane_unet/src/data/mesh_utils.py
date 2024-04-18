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
from typing import Tuple, Optional, List

import torch
import numpy as np
from sklearn.utils import check_array, check_consistent_length

try:
    import ensightreader
except ImportError:
    print(
        "Could not import ensightreader. Please install it from `pip install ensight-reader`"
    )


from jaxtyping import Float


def compute_drag_coefficient(
    poly_normals: np.ndarray,
    poly_area: np.ndarray,
    coeff: float,
    poly_pressure: np.ndarray,
    poly_wss: np.ndarray,
    dir_movement: np.ndarray = np.array([-1, 0, 0]),
):
    """Compute drag coefficient of the mesh assuming the movement direction is negative x-axis.
    Reference: https://www.idealsimulations.com/simworks-tutorials/drag-coefficient-of-a-sphere-cfd-simulation/

    Parameters:
    -----------
    poly_normals: The surface normals on cells (e.g. polygons, triangles) on the mesh
    poly_area: The surface areas of each cell
    coeff: 2/(A * rho * U^2) where rho is the density, U the velocity, and A the cross-sectional area along the movement direction
    poly_pressure: The pressure on each cell
    poly_wss: The wall shear stress on each cell
    dir_movement: The direction of movement, default is -x axis

    Returns:
    --------
    c_drag: float:
        Computed drag coefficient
    """

    # Compute coefficients
    c_p = coeff * np.dot(np.dot(poly_normals, dir_movement), poly_area * poly_pressure)
    c_f = coeff * np.abs(np.dot(np.dot(poly_wss, dir_movement), poly_area))

    # Compute total drag coefficients
    c_drag = c_p + c_f
    return c_drag


def mean_rel_L2_error(y_true, y_pred):
    """Mean relative L2 regression loss.

    Parameters
    ----------
    y_true : array-like of shape = (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape = (n_samples, n_outputs)
        Estimated target values.

    Returns
    -------
    loss : float
         A non-negative floating point value (best is 0.0),
    """
    return np.mean(rel_errors(y_true, y_pred, norm=2))


def rel_errors(y_true, y_pred, norm):
    """Relative L_norm regression losses per sample, where norm is specified as argument.

    Parameters
    ----------
    y_true : array-like of shape = (n_samples, n_outputs)
        Ground truth (correct) target values.
    y_pred : array-like of shape = (n_samples, n_outputs)
        Estimated target values.
    norm: either 1, 2 , np.inf or -np.inf
        Vector norm to compute.

    Returns
    -------
    losses : list
         A list of non-negative floating point value (best is 0.0) for each sample.
    """
    y_true = check_array(y_true, ensure_2d=True)
    y_pred = check_array(y_pred, ensure_2d=True)
    check_consistent_length(y_true, y_pred)

    if y_true.shape[1] != y_pred.shape[1]:
        raise ValueError(
            "y_true and y_pred have different number of output "
            "({}!={})".format(y_true.shape[1], y_pred.shape[1])
        )

    return np.linalg.norm(y_true - y_pred, axis=1, ord=norm) / np.linalg.norm(
        y_true, axis=1, ord=norm
    )


class Normalizer:
    """Normalizer."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def encode(self, x):
        return (x - self.mean) / self.std

    def decode(self, x):
        return x * self.std + self.mean


# error_metrics_example.py
# # Drag Coefficient #
# ####################
#
# # y_true and y_pred have shape (3, 1); 3 samples, each with one drag coeff value
# y_true = np.array([0.300, 0.310, 0.320]).reshape(3, 1)
# y_pred = np.array([0.302, 0.313, 0.315]).reshape(3, 1)
# print("Drag coefficient")
#
# # 1. R^2-score
# r2 = r2_score(y_true, y_pred)
# print(f"$R^2$-score: {r2:.3f}")
#
# # 2. Mean Absolute Error (MAE)
# mae = mean_absolute_error(y_true, y_pred)
# print(f"MAE: {mae:.4f}")
#
#
# # 3. Max. Absolute Error (MAXAE)
# maxae = np.max(np.abs(y_true - y_pred))
# print(f"MAXAE: {maxae:.4f}")
# print()
#
#
# # Fields (velocity, pressure, wall shear stress) #
# ##################################################
# print("Fields")
#
# # y_true and y_pred have shape (3, n_cells * n_components), where n_components
# # corresponds to the number of cells in the mesh; n_components is 1 for pressure
# # and 3 for velocity/wall shear stress
# # here an example is given where n_cells * n_components =  4
# y_true = np.array([[1., 1., 1., 1.], [2., 2., 2., 2.], [3., 3., 3., 3.]])
# y_pred = np.array([[1., 0.5, 1., 1.], [2., 1., 2., 2.], [3., 3., 2., 3.]])
#
# # 1. Mean relative L^2-error
# mrl2e = mean_rel_L2_error(y_true, y_pred)
# print(f"MRL2E: {mrl2e * 100.:.2f} %")


def convert_to_pyvista(ensight_block, fp):
    """
    Convert an ensight block to a pyvista face format that has pyvista face
    format:
        # The first number in each sub-array is the number of points in the face
        faces = np.hstack([
            [N, index_0, index_1, ..., index_N-1],  # N-sided face
            # Other faces...
        ])
    """

    def _node_connectivity_to_face(node_counts, connectivity):
        # Then for 1-dim connectivity, insert the node count before each face
        conn_index = 0
        face_index = 0
        faces = np.zeros(len(node_counts) + np.sum(node_counts), dtype=int)
        for num_sides in node_counts:
            faces[face_index] = num_sides
            faces[face_index + 1 : face_index + 1 + num_sides] = connectivity[
                conn_index : conn_index + num_sides
            ]
            conn_index += num_sides
            face_index += num_sides + 1
        return faces

    if ensight_block.element_type == ensightreader.ElementType.NFACED:
        (
            polyhedra_face_counts,
            face_node_counts,
            face_connectivity,
        ) = ensight_block.read_connectivity_nfaced(fp)
        faces = _node_connectivity_to_face(face_node_counts, face_connectivity - 1)
    elif ensight_block.element_type == ensightreader.ElementType.NSIDED:
        (
            polygon_node_counts,
            polygon_connectivity,
        ) = ensight_block.read_connectivity_nsided(fp)
        faces = _node_connectivity_to_face(
            polygon_node_counts, polygon_connectivity - 1
        )
    else:
        connectivity = ensight_block.read_connectivity(fp)  # 1-based
        num_sides = np.repeat(connectivity.shape[1], len(connectivity)).reshape(-1, 1)
        faces = np.hstack([num_sides, connectivity - 1]).flatten()
    return faces


def point_cloud_to_sdf(
    points: Float[torch.Tensor, "N 3"], sdf_points: Float[torch.Tensor, "M 3"]
) -> Float[torch.Tensor, "M"]:
    """Compute the signed distance function (SDF) of a point cloud with respect to a set of points.

    Parameters:
    -----------
    points: torch.Tensor[N, 3]
        The point cloud
    sdf_points: torch.Tensor[M, 3]
        The points to compute the SDF from

    Returns:
    --------
    sdf: torch.Tensor[M]
        The signed distance function of the point cloud with respect to the sdf_points
    """

    # Compute the pairwise distances between the points and the sdf_points
    dist = torch.cdist(points, sdf_points)

    # Compute the minimum distance to each sdf_point
    sdf = torch.min(dist, dim=1).values
    return sdf


def bbox_to_centers(
    bbox_min: Float[torch.Tensor, "3"],
    bbox_max: Float[torch.Tensor, "3"],
    resolution: Tuple[int, int, int] = [64, 64, 64],
):
    """Compute the centers of the cells in a 3D grid defined by a bounding box.

    Parameters:
    -----------
    bbox_min: torch.Tensor[3]
        The minimum coordinates of the bounding box
    bbox_max: torch.Tensor[3]
        The maximum coordinates of the bounding box
    resolution: Tuple[int, int, int]
        The resolution of the grid

    Returns:
    --------
    centers: torch.Tensor[resolution[0] * resolution[1] * resolution[2], 3]
        The centers of the cells in the grid
    """

    # Compute the cell size
    cell_size = (bbox_max - bbox_min) / torch.tensor(resolution)

    # Compute the grid
    x = torch.linspace(bbox_min[0], bbox_max[0], resolution[0])
    y = torch.linspace(bbox_min[1], bbox_max[1], resolution[1])
    z = torch.linspace(bbox_min[2], bbox_max[2], resolution[2])

    # Compute the centers of the cells
    centers = (
        torch.stack(torch.meshgrid(x, y, z), dim=-1).reshape(-1, 3) + cell_size / 2
    )
    return centers
