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
import sys

import numpy as np
import pandas as pd
import trimesh
import torch
from torch.utils.data import Dataset
import torch_geometric

import open3d as o3d
import trimesh


torch.manual_seed(0)


class Bar(torch.utils.data.Dataset):
    def __init__(self,
                 data_path='/home/leejuhe/juheon_work/bucket_data/',
                 num_points=50000,
                 partition='train',
                 random_sample=False,
                 transform = None):
        self.num_points = num_points
        self.data_path = data_path # default
        print("data_path: ", data_path)

        self.random_sample = random_sample 
        self.partition = partition
        if self.partition == 'train':
            lists = [line.rstrip() for line in open(self.data_path + '/24hrs.txt')]#[28:]
        elif self.partition =='val':
            lists = [line.rstrip() for line in open(self.data_path + '/24hrs_val.txt')]    
        self.items = [] 
        l = len(lists)
       
        print('total data_size = %02d'%l)
        for i in range(l): # load all CAD & scan pairs 
            tag = lists[i].split('/')[-2][2:]
            cad  = torch.FloatTensor(np.loadtxt(lists[i]+'cad%s.txt'%(tag), delimiter='\t'))[:,:3]
            scan = torch.FloatTensor(np.loadtxt(lists[i]+'scan_res%s.csv'%(tag), delimiter=','))[:,:3]
            self.items.append((i+1,cad,scan))

    def __len__(self):
        return len(self.items)

    def __getitem__(self,idx):
        part_name = 'bar'
        
        part_id, mesh, scan = self.items[idx]
        m = torch.mean(mesh)
        
        # random sampling for 50k points 
        if self.random_sample and (self.partition == 'train' or self.partition == 'val'):
            sample = torch.randint(low=0, high=min(len(mesh),len(scan))-1,size=(self.num_points,))
            
            # find correspondence between CAD - scan points 
            pts1 = mesh[sample]
            pts2 = scan[sample]
        else:
            pts1 = mesh 
            pts2 = scan
        with torch.no_grad():
            edge_index = torch_geometric.nn.knn_graph(torch.FloatTensor(pts1),20)
            
        s = pts1.std(0)
        # output in torch_geometric format
        out = torch_geometric.data.Data(x = pts1, y = pts2, edge_index=edge_index,m=m, s=s,part_id=part_id,part_name=part_name)
        return out


class Ocardo(torch.utils.data.Dataset):
    def __init__(self,
                 data_path='./input_data/',
                 num_points=50000,
                 partition='train',
                 random_sample=True,
                 transform = None,
                 ):
        """

        :param data_path:
              # contains the list of paths for dataset
                # in the input_data.txt file, each row contains the input part data folder
                # i.e.
                # /home/DL_engine/input_data/1/
                # /home/DL_engine/input_data/2/
        :param num_points:
        :param partition:
        :param random_sample:
        :param transform:
        """
        self.num_points = num_points
        self.data_path = data_path
        print("data_path: ", data_path)

        self.random_sample= random_sample
        # Read each row as the input part data ID
        lists = [line.rstrip() for line in open(os.path.join(self.data_path, 'input_data.txt'))]
        print("read lists: ", lists)
        
        self.items = []
        # Initialize the tranform function to: Converts mesh faces [3, num_faces] to edge indices [2, num_edges] (functional name: face_to_edge).
        # https://pytorch-geometric.readthedocs.io/en/latest/modules/transforms.html
        self.transform = torch_geometric.transforms.FaceToEdge()
        for i in range(len(lists)):
            # process for each data build in the input_data file
            # read the build id
            tag = lists[i].split('/')[-2]
            print("tag: ", tag)
            print("start loading with np: ", f"{self.data_path}/{lists[i]}cad{tag}.txt")

            # Read each row from the cad_<part_id>.txt, store as torch.FloatTensor the point coordinates
            # cad = torch.FloatTensor(np.loadtxt(lists[i]+'cad%s.txt'%(tag), delimiter='\t'))[:,:3]
            # cad = torch.FloatTensor(np.loadtxt(f"{self.data_path}/{lists[i]}cad{tag}.csv", delimiter='\t'))[:,:3]
            cad = torch.FloatTensor(pd.read_csv(f"{self.data_path}/{lists[i]}cad{tag}.csv", sep=' ').values) #[:,:3]
            print("loaded cad", cad.shape)

            # Read each row from the scan_res<part_id>.csv, store as torch.FloatTensor the point coordinates
            # scan = torch.FloatTensor(np.loadtxt(lists[i]+'scan_res%s.csv'%(tag), delimiter=','))[:,:3]
            scan = torch.FloatTensor(pd.read_csv(f"{self.data_path}/{lists[i]}scan_res{tag}.csv", sep=' ').values)
            self.items.append((i+1, cad, scan))
            print("loaded scan", scan.shape)

            assert cad.shape == scan.shape, print("part CAD and Scan files not match ")

        # print("finished preprocessing data: ", self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        # todo: remove the hard-coded part name
        part_name = 'ocardo'
        
        part_id, mesh, scan = self.items[idx]
        print("__getitem__ part_id, mesh, scan: ", part_id, mesh.shape, scan.shape)
        m = torch.mean(mesh)
        
        # find correspondence between CAD - scan points 
        if self.random_sample:
            sample = torch.randint(low=0, high=min(len(mesh),len(scan))-1,size=(self.num_points,))
            # find correspondence between CAD - scan points 
            pts1 = mesh[sample]
            pts2 = scan[sample]
            # print("check train data format, pst1/pst2: ", pts1.shape, pts2.shape)
            # exit()
        #     check train data format, pst1/pst2:  torch.Size([190000, 3]) torch.Size([190000, 3])
        else:
            pts1 = mesh
            pts2 = scan
        
        with torch.no_grad():
            edge_index = torch_geometric.nn.knn_graph(torch.FloatTensor(pts1),10)
        s = pts1.std(0)

        print("get_item printed sample size, pst1/2: ", pts1.shape, pts2.shape)

        # print("check points/ edge_index: ", pts1.shape, edge_index.shape)
        # print("sample edge index: ", edge_index[:,:10])
        # check points/ edge_index:  torch.Size([50000, 3]) torch.Size([2, 500000])

        out = torch_geometric.data.Data(x = pts1, y = pts2,
                                        edge_index=edge_index,
                                        m=m,
                                        part_id=part_id,
                                        part_name=part_name)
        return out


if __name__=='__main__':
    dataset = Ocardo()
    print(dataset[0])
    
