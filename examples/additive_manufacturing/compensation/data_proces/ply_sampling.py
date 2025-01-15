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


import os, glob
import numpy as np
import trimesh
import open3d as o3d
import torch
import torch_geometric


def generate_mesh_train():
    cad_mesh = trimesh.load(
        "/home/juheonlee/juheon_work/new_data/CustomerLarge-2_003-iso_offset.ply"
    )

    x = torch.FloatTensor(np.asarray(cad_mesh.vertices)) - torch.FloatTensor(
        cad_mesh.bounds.mean(0)
    )

    # raw scan
    scan_pts = o3d.io.read_point_cloud(
        "/home/juheonlee/juheon_work/new_data/CustomerLarge-2_003-iso_offset_aligned.pcd"
    )
    y = torch.FloatTensor(np.asarray(scan_pts.points)) - torch.FloatTensor(
        cad_mesh.bounds.mean(0)
    )

    # fine one-to-one matching
    idx1, idx2 = torch_geometric.nn.knn(y, x, 1)
    new_vert = y[idx2]

    scan_mesh = cad_mesh
    scan_mesh.vertices = new_vert + torch.FloatTensor(cad_mesh.bounds.mean(0))
    scan_mesh.export("data_out.obj")


def generate_mesh_eval(view=False):
    object_name = "bar"
    part_id = 5
    cad_mesh = trimesh.load(
        "%s_%d/cad/%s_%d_uptess.obj" % (object_name, part_id, object_name, part_id)
    )
    scan_pts = np.loadtxt(
        "C:/Users/leejuhe/Documents/comp/out__%02d.csv" % (part_id), delimiter=","
    )
    x = torch.FloatTensor(np.asarray(cad_mesh.vertices))
    y = torch.FloatTensor(scan_pts)
    new_vert = y
    scan_mesh = cad_mesh
    scan_mesh.vertices = new_vert
    scan_mesh.export("%s_%d_out.obj" % (object_name, part_id))
    if view:
        scan_mesh.show()


def generate_scan_file(
    root_dir="/home/chenle/codes/DL_prediction_compensation-master/data",
    save_csv=True,
    part_id=0,
    part_name=None,
):
    # cad_mesh = trimesh.load('/home/juheonlee/juheon_work/new_data/CustomerLar  ge-2_003-iso_offset.ply')
    # scan_pts = o3d.io.read_point_cloud('/home/juheonlee/juheon_work/new_data/CustomerLarge-2_003-iso_offset_scan_aligned.pcd')
    # cad_mesh = trimesh.load('/home/chenle/codes/Model_Aligner-master/output/0/scan/Part_1_S+U_CD.stl')
    # scan_pts = o3d.io.read_point_cloud(
    #     '/home/chenle/codes/Model_Aligner-master/output/0/scan/Part_1_S+U_CD.pcd')
    #
    print("cad file: ", glob.glob(f"{root_dir}/{part_id}/cad/*.{'stl'}"))
    cad_mesh = trimesh.load(glob.glob(f"{root_dir}/{part_id}/cad/*.{'stl'}")[0])
    print("loading... ", glob.glob(f"{root_dir}/{part_id}/scan/*.{'pcd'}"))
    scan_pts = o3d.io.read_point_cloud(
        glob.glob(f"{root_dir}/{part_id}/scan/*.{'pcd'}")[0]
    )

    x = torch.FloatTensor(np.asarray(cad_mesh.vertices)) - torch.FloatTensor(
        cad_mesh.bounds.mean(0)
    )
    y = torch.FloatTensor(np.asarray(scan_pts.points)) - torch.FloatTensor(
        cad_mesh.bounds.mean(0)
    )

    # find one-to-one match
    idx1, idx2 = torch_geometric.nn.knn(y, x, 1)
    new_vert = y[idx2]

    vertices = new_vert + torch.FloatTensor(cad_mesh.bounds.mean(0))
    torch.save(vertices, os.path.join(root_dir, f"{part_id}/{part_name}.pth"))
    if save_csv:
        np.savetxt(os.path.join(root_dir, f"{part_id}/{part_name}_cad.csv"), x)
        print("saved to cad: ", x.shape)
        np.savetxt(os.path.join(root_dir, f"{part_id}/{part_name}_scan.csv"), new_vert)
        print("saved to _scan: ", new_vert.shape)
        print("saved text, ", os.path.join(root_dir, part_name + "_cad.csv"))


def generate_scan():
    path1 = "C:/Users/leejuhe/Documents/flytte2"
    path2 = "C:/Users/leejuhe/Documents/dataset/bar_outputs"

    if not path2:
        raise Exception("no data path defined")

    subfolders = [f.path for f in os.scandir(path2) if f.is_dir()]
    print(subfolders)
    part_id = subfolders[0].split("\\")[1].split("_")
    # print(part_id)
    #'''
    for i in range(len(subfolders)):
        print("processing: %s" % subfolders[i])

        object_name = "bar"
        part_id = subfolders[i].split("\\")[1].split("_")[0]
        if part_id == "23":
            continue
        cad_mesh = trimesh.load(
            "%s/%s_%s/cad/%s_%s_uptess.obj"
            % (path1, object_name, part_id, object_name, part_id)
        )
        print(cad_mesh)
        x = torch.FloatTensor(np.asarray(cad_mesh.vertices)) - torch.FloatTensor(
            cad_mesh.bounds.mean(0)
        )

        # raw scan
        scan_pts = o3d.io.read_point_cloud(
            "%s/scan/%s_scan_u_aligned.pcd" % (subfolders[i], part_id)
        )
        y = torch.FloatTensor(np.asarray(scan_pts.points)) - torch.FloatTensor(
            cad_mesh.bounds.mean(0)
        )

        idx1, idx2 = torch_geometric.nn.knn(y, x, 1)
        new_vert = y[idx2]
        dist = torch.sqrt(torch.sum(torch.pow(x - new_vert, 2), 1))
        filt = dist > 1.2

        new_vert[filt] = x[filt]
        vertices = new_vert + torch.FloatTensor(cad_mesh.bounds.mean(0))
        print(vertices)
        torch.save(
            vertices, "%s/scan/%s_%s_test.pth" % (subfolders[i], object_name, part_id)
        )
    #'''


if __name__ == "__main__":
    generate_scan_file(part_id="11", part_name="4")
