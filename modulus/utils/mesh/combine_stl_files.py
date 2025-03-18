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

from typing import List, Union

import pyvista as pv


def combine_stls(
    input_files: Union[str, List[str]], output_file: str, binary: bool = True
) -> None:
    """Combine multiple STL files into a single-body STL file using PyVista.
    Also converts a single multi-body STL to a single-body STL.

    Parameters
    ----------
    input_files : Union[str, List[str]]
        Path or list of paths to the input STL file(s) to be combined.
    output_file : str
        Path to save the combined STL file.
    binary : bool, optional
        Writes the file as binary when True and ASCII when False, by default True.
    """

    # Ensure input_files is a list
    if isinstance(input_files, str):
        input_files = [input_files]

    # Load all STL files as PyVista meshes
    combined_mesh = pv.PolyData()
    for file in input_files:
        mesh = pv.read(file)
        combined_mesh = combined_mesh.merge(mesh)  # Merge all meshes into one

    # Save the combined mesh as an STL file
    combined_mesh.save(output_file, binary=binary)
