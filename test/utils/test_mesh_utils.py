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

import hashlib
import os
import random
import urllib

import numpy as np
import pytest
from pytest_utils import import_or_fail

stl = pytest.importorskip("stl")


def compute_checksum(file_path):
    """Compute the SHA256 checksum of a given file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def write_multi_body_stl(filename, center=(0, 0, 0), side_length=1.0):
    """Creates a multi-body ASCII STL cube."""

    half = side_length / 2
    cx, cy, cz = center

    # Define vertices
    vertices = np.array(
        [
            [cx - half, cy - half, cz - half],  # 0
            [cx + half, cy - half, cz - half],  # 1
            [cx + half, cy + half, cz - half],  # 2
            [cx - half, cy + half, cz - half],  # 3
            [cx - half, cy - half, cz + half],  # 4
            [cx + half, cy - half, cz + half],  # 5
            [cx + half, cy + half, cz + half],  # 6
            [cx - half, cy + half, cz + half],  # 7
        ]
    )

    # Define faces
    faces = [
        ([0, 1, 2], [0, 2, 3]),  # Bottom
        ([4, 5, 6], [4, 6, 7]),  # Top
        ([0, 1, 5], [0, 5, 4]),  # Front
        ([2, 3, 7], [2, 7, 6]),  # Back
        ([0, 3, 7], [0, 7, 4]),  # Left
        ([1, 2, 6], [1, 6, 5]),  # Right
    ]

    with open(filename, "w") as f:
        for i, (tri1, tri2) in enumerate(faces):
            f.write(f"solid body_{i}\n")
            for tri in [tri1, tri2]:
                v1, v2, v3 = vertices[tri]
                normal = np.cross(v2 - v1, v3 - v1)
                normal /= np.linalg.norm(normal)  # Normalize the normal vector

                f.write(f"  facet normal {normal[0]} {normal[1]} {normal[2]}\n")
                f.write("    outer loop\n")
                f.write(f"      vertex {v1[0]} {v1[1]} {v1[2]}\n")
                f.write(f"      vertex {v2[0]} {v2[1]} {v2[2]}\n")
                f.write(f"      vertex {v3[0]} {v3[1]} {v3[2]}\n")
                f.write("    endloop\n")
                f.write("  endfacet\n")
            f.write(f"endsolid body_{i}\n")


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


@import_or_fail(["vtk", "warp"])
def test_mesh_utils(tmp_path, pytestconfig):
    """Tests the utility for combining VTP files and converting tesselated files."""

    import vtk

    from modulus.utils.mesh import (
        combine_vtp_files,
        convert_tesselated_files_in_directory,
    )

    def _create_random_vtp_mesh(num_points: int, num_triangles: int, dir: str) -> tuple:
        """
        Create a random VTP (VTK PolyData) mesh with triangles.

        Parameters:
            num_points (int): Number of random points.
            num_triangles (int): Number of triangles.
            dir (str): Directory to save the VTP and VTU files.

        Returns:
            tuple: A tuple containing the random VTP mesh (vtk.vtkPolyData).
        """

        # make directory if it does not exist
        os.makedirs(dir, exist_ok=True)

        # Create random points
        points = vtk.vtkPoints()
        for _ in range(num_points):
            x, y, z = (
                random.uniform(-10, 10),
                random.uniform(-10, 10),
                random.uniform(-10, 10),
            )
            points.InsertNextPoint(x, y, z)

        # Create triangles
        triangles = vtk.vtkCellArray()
        for _ in range(num_triangles):
            p1, p2, p3 = (
                random.randint(0, num_points - 1),
                random.randint(0, num_points - 1),
                random.randint(0, num_points - 1),
            )
            triangle = vtk.vtkTriangle()
            triangle.GetPointIds().SetId(0, p1)
            triangle.GetPointIds().SetId(1, p2)
            triangle.GetPointIds().SetId(2, p3)
            triangles.InsertNextCell(triangle)

        # Create a PolyData object (VTP mesh)
        vtp_mesh = vtk.vtkPolyData()
        vtp_mesh.SetPoints(points)
        vtp_mesh.SetPolys(triangles)

        # Assign random scalar values (features) to points in VTP mesh
        scalar_values = vtk.vtkDoubleArray()
        scalar_values.SetName("RandomFeatures")
        for _ in range(num_points):
            scalar_values.InsertNextValue(random.uniform(0, 1))
        vtp_mesh.GetPointData().SetScalars(scalar_values)

        # Write VTP mesh to file
        vtp_writer = vtk.vtkXMLPolyDataWriter()
        vtp_writer.SetFileName(os.path.join(dir, "random.vtp"))
        vtp_writer.SetInputData(vtp_mesh)
        vtp_writer.Write()

    def _create_random_obj_mesh(num_vertices: int, num_faces: int, dir: str) -> None:
        """
        Create a random OBJ file with the specified number of vertices and faces.

        Parameters:
            num_vertices (int): Number of vertices in the mesh.
            num_faces (int): Number of faces in the mesh.
            dir (str): Directory to save the OBJ file.
        """

        # make directory if it does not exist
        os.makedirs(dir, exist_ok=True)

        # Generate random vertices
        vertices = []
        for _ in range(num_vertices):
            x, y, z = (
                random.uniform(-10, 10),
                random.uniform(-10, 10),
                random.uniform(-10, 10),
            )
            vertices.append((x, y, z))

        # Generate random faces
        faces = []
        for _ in range(num_faces):
            # Randomly select 3 vertices for a face
            v1, v2, v3 = random.sample(range(num_vertices), 3)
            # OBJ format uses 1-based indexing
            faces.append((v1 + 1, v2 + 1, v3 + 1))

        # Write vertices and faces to OBJ file
        with open(os.path.join(dir, "random.obj"), "w") as obj_file:
            # Write vertices
            for v in vertices:
                obj_file.write("v {} {} {}\n".format(v[0], v[1], v[2]))
            # Write faces
            for f in faces:
                obj_file.write("f {} {} {}\n".format(f[0], f[1], f[2]))

    tmp_dir_1 = tmp_path / "temp_data_1"
    tmp_dir_2 = tmp_path / "temp_data_2"
    _create_random_vtp_mesh(num_points=10, num_triangles=20, dir=tmp_dir_1)
    _create_random_vtp_mesh(num_points=8, num_triangles=15, dir=tmp_dir_2)
    combine_vtp_files(
        [tmp_dir_1 / "random.vtp", tmp_dir_2 / "random.vtp"],
        output_file=tmp_path / "combined.vtp",
    )
    assert os.path.exists(tmp_path / "combined.vtp")

    tmp_dir = tmp_path / "temp_data"

    _create_random_vtp_mesh(num_points=10, num_triangles=20, dir=tmp_dir)
    convert_tesselated_files_in_directory(
        "vtp2stl", tmp_dir, output_dir=tmp_path / "converted"
    )
    assert os.path.exists(tmp_path / "converted/random.stl")

    _create_random_obj_mesh(num_vertices=30, num_faces=12, dir=tmp_dir)
    convert_tesselated_files_in_directory(
        "obj2vtp", tmp_dir, output_dir=tmp_path / "converted"
    )
    assert os.path.exists(tmp_path / "converted/random.vtp")


@import_or_fail(["warp", "skimage", "stl"])
@pytest.mark.parametrize("backend", ["warp", "skimage"])
def test_stl_gen(pytestconfig, backend, download_stl, tmp_path):

    from stl import mesh

    from modulus.utils.mesh import (
        sdf_to_stl,
    )
    from modulus.utils.sdf import signed_distance_field

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
    n = [int((bounds[k][1] - bounds[k][0] + 2) // min_res) for k in bounds.keys()]
    x = np.linspace(bounds["x"][0] - 1, bounds["x"][1] + 1, n[0], dtype=np.float64)
    y = np.linspace(bounds["y"][0] - 1, bounds["y"][1] + 1, n[1], dtype=np.float64)
    z = np.linspace(bounds["z"][0] - 1, bounds["z"][1] + 1, n[2], dtype=np.float64)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

    coords = np.concatenate(
        [xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], axis=1
    )

    sdf_test = signed_distance_field(
        vertices_3d, vert_indices, coords.flatten()
    ).numpy()
    output_filename = tmp_path / "output_stl.stl"
    sdf_to_stl(
        sdf_test.reshape(n[0], n[1], n[2]),
        threshold=0.0,
        backend=backend,
        filename=output_filename,
    )

    # read the saved stl
    saved_stl = mesh.Mesh.from_file(str(output_filename))

    assert saved_stl.vectors is not None


@pytest.fixture
def generate_test_stls(tmp_path):
    """Fixture to generate STL files in a temporary directory before running tests."""

    from modulus.utils.mesh.combine_stl_files import combine_stls

    cube_1_path = tmp_path / "cube_1.stl"
    cube_2_path = tmp_path / "cube_2.stl"
    cube_1_combined_path = tmp_path / "cube_1_combined.stl"
    all_cubes_combined_path = tmp_path / "all_cubes_combined.stl"

    # Generate STL files in tmp directory
    write_multi_body_stl(cube_1_path, center=(0, 0, 0), side_length=1.0)
    write_multi_body_stl(cube_2_path, center=(2, 2, 2), side_length=1.0)

    # Combine STL files
    combine_stls(
        input_files=str(cube_1_path),
        output_file=str(cube_1_combined_path),
    )

    combine_stls(
        input_files=[str(cube_1_path), str(cube_2_path)],
        output_file=str(all_cubes_combined_path),
    )

    return {
        "cube_1_combined": cube_1_combined_path,
        "all_cubes_combined": all_cubes_combined_path,
    }


@import_or_fail(["pyvista"])
def test_combined_stl(generate_test_stls, pytestconfig):
    """Test to check combining stls."""

    EXPECTED_CHECKSUMS = {
        "cube_1_combined": "b5be925cbdfe6867a782c94321a4702cef397b4d17139dab2453d9ee8cbe0998",
        "all_cubes_combined": "c2830f65700dfa66b3e65d16a982d76cd9841c4b7fbd37e2597c5b409acd1fee",
    }

    for key, expected_checksum in EXPECTED_CHECKSUMS.items():
        computed_checksum = compute_checksum(generate_test_stls[key])
        assert computed_checksum == expected_checksum, f"Checksum mismatch for {key}"
