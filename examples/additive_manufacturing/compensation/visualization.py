# ignore_header_test
# ruff: noqa: E402

# © Copyright 2023 HP Development Company, L.P.
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


import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pytorch3d.loss import chamfer_distance


def stl_to_vertices_and_faces(file_path):
    """
    Function to load STL and convert to vertices and triangles
    """
    mesh = o3d.io.read_triangle_mesh(file_path)
    if mesh.is_empty():
        raise ValueError(f"Mesh at {file_path} is empty or invalid.")
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    return verts, faces


def plot_mesh(mesh, ax, face_color, edge_color, label):
    """
    Function to plot meshes
    """
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    # Add faces with transparency
    ax.add_collection3d(
        Poly3DCollection(
            vertices[triangles], facecolor=face_color, edgecolor=edge_color, alpha=0.3
        )
    )
    ax.scatter(
        vertices[:, 0], vertices[:, 1], vertices[:, 2], color=edge_color, s=0.1
    )  # Points
    ax.set_title(label, fontsize=12)
    ax.set_box_aspect([1, 1, 1])  # Equal scaling
    ax.grid(True)


# RMS Calculation
def calculate_rms(source, target):
    """Calculate Root Mean Square (RMS) error."""
    diff = source - target
    return torch.sqrt(torch.mean(diff**2))


# Sample usage: Load STL files
target_mesh_o3d = o3d.io.read_triangle_mesh("/content/cad_4.stl")
uncompensated_mesh_o3d = o3d.io.read_triangle_mesh("/content/cad_4.stl")
compensated_mesh_o3d = o3d.io.read_triangle_mesh("/content/cad_4.stl")

# Load STL files for Chamfer Distance and RMS calculation
target_verts, _ = stl_to_vertices_and_faces("/content/cad_4.stl")
uncompensated_verts, _ = stl_to_vertices_and_faces("/content/cad_4.stl")
compensated_verts, _ = stl_to_vertices_and_faces("/content/cad_4.stl")

# Convert vertices to PyTorch tensors
target_verts_tensor = torch.tensor(target_verts, dtype=torch.float32).unsqueeze(0)
uncompensated_verts_tensor = torch.tensor(
    uncompensated_verts, dtype=torch.float32
).unsqueeze(0)
compensated_verts_tensor = torch.tensor(
    compensated_verts, dtype=torch.float32
).unsqueeze(0)

# Chamfer Distance
cd_uncomp, _ = chamfer_distance(uncompensated_verts_tensor, target_verts_tensor)
cd_comp, _ = chamfer_distance(compensated_verts_tensor, target_verts_tensor)

# RMS Error
min_len = min(len(uncompensated_verts), len(target_verts))
uncomp_rms = calculate_rms(
    uncompensated_verts_tensor[:, :min_len, :], target_verts_tensor[:, :min_len, :]
)
comp_rms = calculate_rms(
    compensated_verts_tensor[:, :min_len, :], target_verts_tensor[:, :min_len, :]
)

# Fitness (normalized metric based on distances)
fitness_uncomp = 1 - cd_uncomp.item()
fitness_comp = 1 - cd_comp.item()

# Print evaluation metrics
print(f"Chamfer Distance (Uncompensated): {cd_uncomp.item()}")
print(f"Chamfer Distance (Compensated): {cd_comp.item()}")
print(f"RMS Error (Uncompensated): {uncomp_rms.item()}")
print(f"RMS Error (Compensated): {comp_rms.item()}")
print(f"Fitness (Uncompensated): {fitness_uncomp}")
print(f"Fitness (Compensated): {fitness_comp}")

# Visualization with enhanced plot
fig = plt.figure(figsize=(18, 6))
ax1 = fig.add_subplot(131, projection="3d")
ax2 = fig.add_subplot(132, projection="3d")
ax3 = fig.add_subplot(133, projection="3d")

plot_mesh(
    target_mesh_o3d,
    ax1,
    face_color="lightcoral",
    edge_color="red",
    label="Target\n(Desired Shape)",
)
plot_mesh(
    uncompensated_mesh_o3d,
    ax2,
    face_color="lightgreen",
    edge_color="green",
    label=f"Uncompensated\nCD: {cd_uncomp.item():.4f}\nRMS: {uncomp_rms.item():.4f}",
)
plot_mesh(
    compensated_mesh_o3d,
    ax3,
    face_color="lightblue",
    edge_color="blue",
    label=f"Compensated\nCD: {cd_comp.item():.4f}\nRMS: {comp_rms.item():.4f}",
)

plt.tight_layout()
plt.show()
