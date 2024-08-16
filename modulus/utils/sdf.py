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

# ruff: noqa: F401

import numpy as np
import warp as wp
from numpy.typing import NDArray


@wp.kernel
def _bvh_query_distance(
    mesh: wp.uint64,
    points: wp.array(dtype=wp.vec3f),
    max_dist: wp.float32,
    sdf: wp.array(dtype=wp.float32),
    sdf_hit_point: wp.array(dtype=wp.vec3f),
    sdf_hit_point_id: wp.array(dtype=wp.int32),
):

    """
    Computes the signed distance from each point in the given array `points`
    to the mesh represented by `mesh`,within the maximum distance `max_dist`,
    and stores the result in the array `sdf`.

    Parameters:
        mesh (wp.uint64): The identifier of the mesh.
        points (wp.array): An array of 3D points for which to compute the
            signed distance.
        max_dist (wp.float32): The maximum distance within which to search
            for the closest point on the mesh.
        sdf (wp.array): An array to store the computed signed distances.
        sdf_hit_point (wp.array): An array to store the computed hit points.
        sdf_hit_point_id (wp.array): An array to store the computed hit point ids.

    Returns:
        None
    """
    tid = wp.tid()

    res = wp.mesh_query_point(mesh, points[tid], max_dist)

    mesh_ = wp.mesh_get(mesh)

    p0 = mesh_.points[mesh_.indices[3 * res.face + 0]]
    p1 = mesh_.points[mesh_.indices[3 * res.face + 1]]
    p2 = mesh_.points[mesh_.indices[3 * res.face + 2]]

    p_closest = res.u * p0 + res.v * p1 + (1.0 - res.u - res.v) * p2

    sdf[tid] = res.sign * wp.abs(wp.length(points[tid] - p_closest))
    sdf_hit_point[tid] = p_closest
    sdf_hit_point_id[tid] = res.face


def signed_distance_field(
    mesh_vertices: list[tuple[float, float, float]],
    mesh_indices: NDArray[float],
    input_points: list[tuple[float, float, float]],
    max_dist: float = 1e8,
    include_hit_points: bool = False,
    include_hit_points_id: bool = False,
) -> wp.array:
    """
    Computes the signed distance field (SDF) for a given mesh and input points.

    Parameters:
    ----------
        mesh_vertices (list[tuple[float, float, float]]): List of vertices defining the mesh.
        mesh_indices (list[tuple[int, int, int]]): List of indices defining the triangles of the mesh.
        input_points (list[tuple[float, float, float]]): List of input points for which to compute the SDF.
        max_dist (float, optional): Maximum distance within which to search for
            the closest point on the mesh. Default is 1e8.
        include_hit_points (bool, optional): Whether to include hit points in
            the output. Default is False.
        include_hit_points_id (bool, optional): Whether to include hit point
            IDs in the output. Default is False.

    Returns:
    -------
        wp.array: An array containing the computed signed distance field.

    Example:
    -------
    >>> mesh_vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    >>> mesh_indices = np.array((0, 1, 2))
    >>> input_points = [(0.5, 0.5, 0.5)]
    >>> signed_distance_field(mesh_vertices, mesh_indices, input_points).numpy()
    Module modulus.utils.sdf load on device 'cuda:0' took ...
    array([0.5], dtype=float32)
    """

    wp.init()
    mesh = wp.Mesh(
        wp.array(mesh_vertices, dtype=wp.vec3), wp.array(mesh_indices, dtype=wp.int32)
    )

    sdf_points = wp.array(input_points, dtype=wp.vec3)
    sdf = wp.zeros(shape=sdf_points.shape, dtype=wp.float32)
    sdf_hit_point = wp.zeros(shape=sdf_points.shape, dtype=wp.vec3f)
    sdf_hit_point_id = wp.zeros(shape=sdf_points.shape, dtype=wp.int32)

    wp.launch(
        kernel=_bvh_query_distance,
        dim=len(sdf_points),
        inputs=[mesh.id, sdf_points, max_dist, sdf, sdf_hit_point, sdf_hit_point_id],
    )

    if include_hit_points and include_hit_points_id:
        return (sdf, sdf_hit_point, sdf_hit_point_id)
    elif include_hit_points:
        return (sdf, sdf_hit_point)
    elif include_hit_points_id:
        return (sdf, sdf_hit_point_id)
    else:
        return sdf
