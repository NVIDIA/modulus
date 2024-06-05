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
from typing import Tuple

import numpy as np
import pymesh


def load_collapse_edges_save(
    file_path: str,
    output_path: str,
    rel_threshold: float = 10.0,
    preserve_feature: bool = True,
) -> None:
    """
    Load a mesh from the specified file, collapse short edges,
    and save the modified mesh as an STL file.
    Supports parsing the following formats: .obj, .ply, .off,
    .stl, .mesh, .node, .poly and .msh.

    Args:
    - file_path (str): Path to the input mesh file (e.g., "input.stl").
    - output_path (str): Path to save the modified mesh as an STL file.
    - rel_threshold (float): Relative threshold for edge collapse.
    - preserve_feature (bool): Whether to preserve features.

    Raises:
    - ValueError: If the input file format is not supported.
    """
    supported_formats = [
        ".obj",
        ".ply",
        ".off",
        ".stl",
        ".mesh",
        ".node",
        ".poly",
        ".msh",
    ]
    _, input_extension = os.path.splitext(file_path)
    if input_extension not in supported_formats:
        raise ValueError(
            "Unsupported file format. Supported formats are: "
            + ", ".join(supported_formats)
        )

    mesh = pymesh.load_mesh(file_path)
    mesh, _ = pymesh.collapse_short_edges(mesh, rel_threshold, preserve_feature)
    pymesh.save_mesh(output_path, mesh)


def collapse_edges(
    vertices: np.ndarray,
    faces: np.ndarray,
    rel_threshold: float = 10.0,
    preserve_feature: bool = True,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Collapse short edges in a mesh defined by its vertices and faces.

    Args:
    - vertices (numpy.ndarray): Array of shape (N, 3) representing vertex coordinates.
    - faces (numpy.ndarray): Array of shape (M, 3) representing vertex indices forming faces.
    - rel_threshold (float): Relative threshold for edge collapse.
    - preserve_feature (bool): Whether to preserve features.

    Returns:
    - numpy.ndarray: The modified vertices.
    - numpy.ndarray: The modified faces.
    - dict: Information about the operation. 'source_face_index' contains the source face index for each face.
    """
    mesh = pymesh.form_mesh(vertices, faces)
    vertices, faces, information = pymesh.collapse_short_edges_raw(
        mesh.vertices,
        mesh.faces,
        rel_threshold=rel_threshold,
        preserve_feature=preserve_feature,
    )
    return vertices, faces, information
