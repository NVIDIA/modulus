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
# ruff: noqa: E402

import urllib.request

import numpy as np
import pytest
from pytest_utils import import_or_fail
from stl import mesh

from modulus.utils.sdf import sdf_to_stl, signed_distance_field


@pytest.fixture
def download_stl(tmp_path):
    url = "https://upload.wikimedia.org/wikipedia/commons/4/43/Stanford_Bunny.stl"

    parsed_url = urllib.parse.urlparse(url)
    if parsed_url.scheme not in ("http", "https"):
        raise ValueError(f"URL scheme '{parsed_url.scheme}' is not permitted.")

    file_path = tmp_path / "Stanford_Bunny.stl"

    # Download the STL file
    urllib.request.urlretrieve(url, file_path)  # noqa: S310

    # Return the path to the downloaded file
    return file_path


def tet_verts(flip_x=1):
    tet = np.array(
        [
            flip_x * 0,
            0,
            0,  # bottom
            flip_x * 0,
            1,
            0,
            flip_x * 1,
            0,
            0,
            flip_x * 0,
            0,
            0,  # front
            flip_x * 1,
            0,
            0,
            flip_x * 0,
            0,
            1,
            flip_x * 0,
            0,
            0,  # left
            flip_x * 0,
            0,
            1,
            flip_x * 0,
            1,
            0,
            flip_x * 1,
            0,
            0,  # "top"
            flip_x * 0,
            1,
            0,
            flip_x * 0,
            0,
            1,
        ],
        dtype=np.float64,
    )

    return tet


@import_or_fail("warp")
def test_sdf(pytestconfig):

    tet = tet_verts()

    sdf_tet = signed_distance_field(
        tet,
        np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),
        np.array([1, 1, 1, 0.1, 0.1, 0.1], dtype=np.float64),
    )
    np.testing.assert_allclose(sdf_tet.numpy(), [1.15470052, -0.1], atol=1e-7)

    sdf_tet, sdf_hit_point, sdf_hit_point_id = signed_distance_field(
        tet,
        np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=np.int32),
        np.array([1, 1, 1, 0.12, 0.11, 0.1], dtype=np.float64),
        include_hit_points=True,
        include_hit_points_id=True,
    )
    np.testing.assert_allclose(
        sdf_hit_point.numpy(),
        [[0.33333322, 0.33333334, 0.3333334], [0.12000002, 0.11, 0.0]],
        atol=1e-7,
    )
    np.testing.assert_allclose(sdf_hit_point_id.numpy(), [3, 0], atol=1e-7)


@import_or_fail("warp")
def test_stl_gen(pytestconfig, download_stl, tmp_path):

    bunny_mesh = mesh.Mesh.from_file(str(download_stl))

    vertices = np.array(bunny_mesh.vectors, dtype=np.float64)
    vertices_3d = vertices.reshape(-1, 3)
    vert_indices = np.arange((vertices_3d.shape[0]))

    bounds = {
        "x": (np.min(vertices_3d[:, 0]), np.max(vertices_3d[:, 0])),
        "y": (np.min(vertices_3d[:, 1]), np.max(vertices_3d[:, 1])),
        "z": (np.min(vertices_3d[:, 2]), np.max(vertices_3d[:, 2])),
    }

    res = {k: v[1] - v[0] for k, v in bounds.items()}
    min_res = min(res.values()) / 100
    n = [int((bounds[k][1] - bounds[k][0] + 0.1) // min_res) for k in bounds.keys()]
    x = np.linspace(bounds["x"][0] - 0.5, bounds["x"][1] + 0.5, n[0], dtype=np.float64)
    y = np.linspace(bounds["y"][0] - 0.5, bounds["y"][1] + 0.5, n[1], dtype=np.float64)
    z = np.linspace(bounds["z"][0] - 0.5, bounds["z"][1] + 0.5, n[2], dtype=np.float64)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

    coords = np.concatenate(
        [xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], axis=1
    )

    sdf_test = signed_distance_field(
        vertices_3d, vert_indices, coords.flatten()
    ).numpy()
    output_filename = tmp_path / "output_stl.stl"
    sdf_to_stl(sdf_test.reshape(n[0], n[1], n[2]), filename=output_filename)

    # read the saved stl
    saved_stl = mesh.Mesh.from_file(str(output_filename))

    assert saved_stl.vectors is not None
