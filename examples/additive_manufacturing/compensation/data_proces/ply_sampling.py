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


import os

import numpy as np
import open3d as o3d
import pandas as pd
import torch
import torch_geometric
import trimesh


def generate_mesh_train(
    ply_path,
    scan_pcd_path,
    save_csv=True,
    save_mesh_path=None,
    part_name="bar",
    part_id="3",
    export_format="pth",
    filter_dist=False,
):
    """
    A PLY file is a computer file format for storing 3D data as a collection of polygons.
    PLY stands for Polygon File Format, and it's also known as the Stanford Triangle Format.
    PLY files are used to store 3D data from 3D scanners.

    This function load a CAD file in PLY format, or STL format with trimesh:
         i.e. <trimesh.Trimesh(vertices.shape=(point_cnt, 3), faces.shape=(cnt, 3), name=`ply_path`)>

    Load the raw scan file sampled points in PCD format, then save the updated scan mesh in OBJ format.

    Parameters:
        - ply_path = os.path.join(root_data_path, "data_pipeline_bar/remesh98.ply")
        - scan_pcd_path = os.path.join(root_data_path, "data_pipeline_bar/bar_98/scan/98_SAMPLED_POINTS_aligned.pcd")
        - save_mesh_path = "test_data_pipeline"

    Return:
        Saved scan mesh path
    """
    os.makedirs(save_mesh_path, exist_ok=True)

    # Load cad mesh from PLY file
    cad_mesh = trimesh.load(ply_path)

    # Centralize the coordinates
    cad_pts = torch.FloatTensor(np.asarray(cad_mesh.vertices)) - torch.FloatTensor(
        cad_mesh.bounds.mean(0)
    )

    # Load raw scan file in PCD, o3d function to read PointCloud from file
    scan_pts = o3d.io.read_point_cloud(scan_pcd_path)

    # Centralize the coordinates
    scan_pts = torch.FloatTensor(np.asarray(scan_pts.points)) - torch.FloatTensor(
        cad_mesh.bounds.mean(0)
    )

    # Fined one-to-one matching
    idx1, idx2 = torch_geometric.nn.knn(scan_pts, cad_pts, 1)
    new_vert = scan_pts[idx2]

    if filter_dist:
        dist = torch.sqrt(torch.sum(torch.pow(cad_pts - new_vert, 2), 1))
        filt = dist > 1.2
        new_vert[filt] = cad_pts[filt]

    # Updates the scan coordinates to the original CAD mesh
    scan_mesh = cad_mesh
    vertices = new_vert + torch.FloatTensor(cad_mesh.bounds.mean(0))
    scan_mesh.vertices = vertices

    if export_format == "obj":
        scan_mesh.export(os.path.join(save_mesh_path, "data_out.obj"))
    elif export_format == "pth":
        torch.save(vertices, os.path.join(save_mesh_path, f"{part_id}/{part_name}.pth"))
    else:
        print("Export format should be OBJ or PTH")
        exit()

    if save_csv:
        # save the original CAD points with centralize coordinates
        np.savetxt(
            os.path.join(save_mesh_path, f"{part_id}/{part_name}_cad.csv"), cad_pts
        )
        # save the mapped scan_pts points with centralize coordinates
        np.savetxt(
            os.path.join(save_mesh_path, f"{part_id}/{part_name}_scan.csv"), new_vert
        )

    return os.path.join(save_mesh_path, "data_out.obj")


def generate_mesh_eval(cad_path, comp_out_path, export_path, view=False):
    """
    Function to load a 3D object pair (Original design file v.s. Scanned printed / Compensated part),
        - CAD design in format of OBJ or STL
        - Scanned printed, or compensated part points, in CSV or TXT
    Export the Scanned in mesh, OBJ format

    Parameters:
    - object_name = "bar"
    - part_id = 5
    - cad_path = "%s_%d/cad/%s_%d_uptess.obj" % (object_name, part_id, object_name, part_id)
    - comp_out_path = comp/out__%02d.csv" % (part_id)

    Return:
        Saved scan mesh path
    """
    os.makedirs(export_path, exist_ok=True)

    # Sample design CAD name
    cad_mesh = trimesh.load(cad_path)

    # Sample scanned printed file, or generated compensated file, in CSV or TXT
    # change the reading format, if data was saved with other separators, " ", ","
    scan_pts = pd.read_csv(comp_out_path, sep=",").values

    # Define the new vertices as the scanned printed points coordinates
    new_vert = torch.FloatTensor(scan_pts)

    # Define the mesh from the Design CAD
    scan_mesh = cad_mesh

    # Export new mesh
    scan_mesh.vertices = new_vert
    scan_mesh.export(os.path.join(export_path, "export_out.obj"))
    if view:
        scan_mesh.show()
