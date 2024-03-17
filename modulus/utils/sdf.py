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

import warp as wp


@wp.kernel
def bvh_query_distance(
    mesh: wp.uint64,
    points: wp.array(dtype=wp.vec3f),
    max_dist: wp.float32,
    sdf: wp.array(dtype=wp.float32),
):
    tid = wp.tid()

    res = wp.mesh_query_point(mesh, points[tid], max_dist)

    mesh_ = wp.mesh_get(mesh)

    p0 = mesh_.points[mesh_.indices[3 * res.face + 0]]
    p1 = mesh_.points[mesh_.indices[3 * res.face + 1]]
    p2 = mesh_.points[mesh_.indices[3 * res.face + 2]]

    p_closest = res.u * p0 + res.v * p1 + (1.0 - res.u - res.v) * p2

    sdf[tid] = res.sign * wp.abs(wp.length(points[tid] - p_closest))


def signed_distance_field(
    mesh_vertices,
    mesh_indices,
    input_points,
    max_dist=1e8,
    include_hit_points=False,
    include_hit_points_and_id=False,
):
    wp.init()
    mesh = wp.Mesh(
        wp.array(mesh_vertices, dtype=wp.vec3), wp.array(mesh_indices, dtype=wp.int32)
    )
    sdf_points = wp.array(input_points, dtype=wp.vec3)
    sdf = wp.zeros(shape=sdf_points.shape, dtype=wp.float32)
    wp.launch(
        kernel=bvh_query_distance,
        dim=len(sdf_points),
        inputs=[mesh.id, sdf_points, max_dist, sdf],
    )

    return sdf
