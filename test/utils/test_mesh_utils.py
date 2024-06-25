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

import os
import random

from pytest_utils import import_or_fail

from modulus.utils.mesh import combine_vtp_files, convert_tesselated_files_in_directory


@import_or_fail(["vtk"])
def test_mesh_utils(tmp_path, pytestconfig):
    """Tests the utility for combining VTP files and converting tesselated files."""

    import vtk

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
