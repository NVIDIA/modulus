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
import pandas as pd
import torch
import torch_geometric

# import open3d as o3d
from utils import log_string

torch.manual_seed(0)


class Bar(torch.utils.data.Dataset):
    """
    To import the dataset, you can use files in either .txt or .csv format. Below is the folder structure for sample input data:
        - input_data.txt: Contains logs, with each row corresponding to a build part geometry
            - /part_folder_i (aligned with logs in input_data.txt):
                - cad_<part_id>.txt: Contains 3 columns, each representing a point's coordinates.
                - scan_red<part_id>.csv: Includes 3 columns representing point locations.
    """

    def __init__(
        self,
        data_path="insert data path, default in cfg.data_options.data_path",
        num_points=50000,
        partition="train",
        random_sample=False,
        transform=None,
        LOG_FOUT=None,
    ):
        self.num_points = num_points
        self.data_path = data_path
        log_string(LOG_FOUT, f"Process from data_path: {data_path}")

        self.random_sample = random_sample
        self.partition = partition
        if self.partition == "train":
            lists = [
                line.rstrip() for line in open(self.data_path + "/24hrs.txt")
            ]  # [28:]
        elif self.partition == "val":
            lists = [line.rstrip() for line in open(self.data_path + "/24hrs_val.txt")]
        self.items = []
        len_ds = len(lists)

        print("total data_size = %02d" % len_ds)
        for i in range(len_ds):  # load all CAD & scan pairs
            tag = lists[i].split("/")[-2][2:]
            cad = torch.FloatTensor(
                np.loadtxt(lists[i] + "cad%s.txt" % (tag), delimiter="\t")
            )[:, :3]
            scan = torch.FloatTensor(
                np.loadtxt(lists[i] + "scan_res%s.csv" % (tag), delimiter=",")
            )[:, :3]
            self.items.append((i + 1, cad, scan))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        part_id, mesh, scan = self.items[idx]
        m = torch.mean(mesh)

        # random sampling for 50k points
        if self.random_sample and (
            self.partition == "train" or self.partition == "val"
        ):
            sample = torch.randint(
                low=0, high=min(len(mesh), len(scan)) - 1, size=(self.num_points,)
            )

            # find correspondence between CAD - scan points
            pts1 = mesh[sample]
            pts2 = scan[sample]
        else:
            pts1 = mesh
            pts2 = scan
        with torch.no_grad():
            edge_index = torch_geometric.nn.knn_graph(torch.FloatTensor(pts1), 20)

        s = pts1.std(0)
        # output in torch_geometric format
        out = torch_geometric.data.Data(
            x=pts1,
            y=pts2,
            edge_index=edge_index,
            m=m,
            s=s,
            part_id=part_id,
        )
        return out


class Ocardo(torch.utils.data.Dataset):
    """

    :param data_path:
          # contains the list of paths for dataset
            # in the input_data.txt file, each row contains the input part data folder
            # i.e.
            # /home/DL_engine/input_data/1/
            # /home/DL_engine/input_data/2/
              Under each data folder:
                i.e.
                - /home/DL_engine/input_data/1/cad{tag_id}.csv
                - /home/DL_engine/input_data/1/scan_res{tag_id}.csv
            # for complete description of sample data format, refer to README.md
            # each part shape scan/ cad: i.e. torch.Size([12775, 3])
    :param num_points:
    :param partition:
    :param random_sample:
    :param transform:
    """

    def __init__(
        self,
        data_path="./input_data/",
        num_points=50000,
        partition="train",
        random_sample=True,
        transform=None,
        LOG_FOUT=None,
    ):
        self.num_points = num_points
        self.data_path = data_path
        log_string(LOG_FOUT, f"Process from data_path: {data_path}")

        self.random_sample = random_sample
        # Read each row as the input part data ID
        lists = [
            line.rstrip()
            for line in open(os.path.join(self.data_path, "input_data.txt"))
        ]
        log_string(LOG_FOUT, f"read data folder name lists: {lists}")

        self.items = []
        # Initialize the tranform function to: Converts mesh faces [3, num_faces] to edge indices [2, num_edges] (functional name: face_to_edge).
        # https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html
        self.transform = torch_geometric.transforms.FaceToEdge()
        for i in range(len(lists)):
            # process for each data build in the input_data file
            # read the build id
            tag = lists[i].split("/")[-2]
            # Read each row from the cad_<part_id>.txt, store as torch.FloatTensor the point coordinates
            # cad = torch.FloatTensor(np.loadtxt(f"{self.data_path}/{lists[i]}cad{tag}.txt", delimiter='\t'))[:,:3] #input_data_bar_sample

            log_string(LOG_FOUT, f"{self.data_path}/{lists[i]}scan_res{tag}.csv")

            cad = torch.FloatTensor(
                pd.read_csv(f"{self.data_path}/{lists[i]}cad{tag}.csv", sep=" ").values
            )  # molded_fiber

            # Read each row from the scan_res<part_id>.csv, store as torch.FloatTensor the point coordinates
            # scan = torch.FloatTensor(np.loadtxt(f"{self.data_path}/{lists[i]}scan_res{tag}.csv", delimiter=','))[:,:3]    #input_data_bar_sample
            scan = torch.FloatTensor(
                pd.read_csv(
                    f"{self.data_path}/{lists[i]}scan_res{tag}.csv", sep=" "
                ).values
            )  # molded_fiber

            self.items.append((i + 1, cad, scan))
            log_string(LOG_FOUT, f"loaded scan {scan.shape}")

            if not cad.shape == scan.shape:
                raise Exception("Part CAD and Scan files not match ")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        """
        For each item that contains the dataset
            - part_id (i.e. 1)
            - mesh(the original cad of size: pt_cnt, 3),
            - scan(the scan of the printed part of size: pt_cnt, 3)
            i.e. torch.Size([650, 3]) torch.Size([650, 3])


        """
        part_id, mesh, scan = self.items[idx]

        # todo: reason to compute mean/ what means for mean < 0?
        # torch.mean(mesh):  tensor(-0.8895)
        # torch.mean(mesh):  tensor(4.4421)
        m = torch.mean(mesh)

        # find correspondence between CAD - scan points
        if self.random_sample:
            # Get random sampling index from 0 to self.num_points
            # i.e. sample id:  tensor([ 1653, 27927,  3942,  ..., 24202,  1684, 23686])
            # resulting pts1, pts2 size: [self.num_points, 3], i.e. torch.Size([190000, 3])
            # todo: if meaningful with the sample >> the pcloud original scanning/ sampling density -> los of duplicated samples ?
            sample = torch.randint(
                low=0, high=min(len(mesh), len(scan)) - 1, size=(self.num_points,)
            )

            # find correspondence between CAD - scan points
            pts1 = mesh[sample]
            pts2 = scan[sample]
        else:
            pts1 = mesh
            pts2 = scan

        with torch.no_grad():
            # taking 10 nodes for nearest neighbors, this lead to the edge numbers to be ~ 10 x sample number,
            # i.e. sample#=190k, neighbor#=10, edge_index.shape=[2, ~1900k]
            # i.e.torch.Size([2, 2082861]) / torch.Size([2, 1911460])
            # knn compueted edge index:  tensor([[ 45007, 130923,  79760,  ..., 147219, 146399, 132629],
            #                                 [     0,      0,      0,  ..., 189999, 189999, 189999]])
            edge_index = torch_geometric.nn.knn_graph(torch.FloatTensor(pts1), 10)

        out = torch_geometric.data.Data(
            x=pts1,
            y=pts2,
            edge_index=edge_index,
            m=m,
            part_id=part_id,
        )
        return out
