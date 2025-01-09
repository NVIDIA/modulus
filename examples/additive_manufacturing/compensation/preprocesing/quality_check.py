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
import torch 
import pytorch3d
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)

import torch_geometric
import open3d as o3d

#data_top  = np.loadtxt('./48hrs/MF1/15913_1_Top_SAMPLED_POINTS.pcd_aligned_SAMPLED_POINTS_C2C_DIST.csv', delimiter=',', skiprows=1)
#data_bot  = np.loadtxt('./48hrs/MF1/15913_1_Bottom_SAMPLED_POINTS.pcd_aligned_SAMPLED_POINTS_C2C_DIST.csv', delimiter=',', skiprows=1)

def sanity():
    ''''
    data_top  = np.loadtxt('./24hrs/MF1/15914_1_Top.pcd_aligned_SAMPLED_POINTS_C2C_DIST.csv', delimiter=',', skiprows=1)
    data_bot  = np.loadtxt('./24hrs/MF1/15914_1_Bottom.pcd_aligned_SAMPLED_POINTS_C2C_DIST.csv', delimiter=',', skiprows=1)
    #data_top  = np.loadtxt('./24hrs/MF1/15914_1_Top.pcd_aligned_SAMPLED_POINTS_C2C_DIST.csv', delimiter=',', skiprows=1)
    

    # scan top 
    cad_top_x = data_top[:,0] - data_top[:,5] # x
    cad_top_y = data_top[:,1] - data_top[:,6] # y
    cad_top_z = data_top[:,2] - data_top[:,3] # z

    cad_top = np.stack([cad_top_x,cad_top_y,cad_top_z],1)
    print(cad_top.shape)

    # scan bottom
    cad_bot_x = data_bot[:,0] - data_bot[:,5] # x 
    cad_bot_y = data_bot[:,1] - data_bot[:,6] # y
    cad_bot_z = data_bot[:,2] - data_bot[:,3] # z

    cad_bot = np.stack([cad_bot_x,cad_bot_y,cad_bot_z],1)

    cad = np.concatenate([cad_top,cad_bot],0)

    #cad = np.concatenate([data_top[:,:3],data_bot[:,:3]],0)
    # case of a processed
    '''
    data_bot  = np.loadtxt('./bucket_1_16402/cad2scan5.csv', delimiter=',', skiprows=1)
    print(data_bot.shape)
    cad_bot_x = data_bot[:,0] - data_bot[:,5] # x 
    cad_bot_y = data_bot[:,1] - data_bot[:,6] # y
    cad_bot_z = data_bot[:,2] - data_bot[:,3] # z

    cad = np.stack([cad_bot_x,cad_bot_y,cad_bot_z],1)

    #'''
    np.savetxt('./bucket_1_16402/MF5/scan_res5.csv',cad,delimiter=',')

def sanity2():
    data_bot  = np.loadtxt('./cad2scan5.txt', delimiter=',', skiprows=1)
    print(data_bot.shape)
    cad_bot_x = data_bot[:,0] - data_bot[:,8] # x 
    cad_bot_y = data_bot[:,1] - data_bot[:,9] # y
    cad_bot_z = data_bot[:,2] - data_bot[:,10] # z

    cad = np.stack([cad_bot_x,cad_bot_y,cad_bot_z],1)

    np.savetxt('scan5_res.csv',cad,delimiter=',')

np.random.seed(0)
def conversion():
    # In case mistakenly sample 400k instead 200k    
    for idx in range(1,6):
        print('processing %0d'%idx)
        data1  = np.loadtxt('./bucket_1_16402/MF%d/16402_%d_cad.csv'%(idx,idx), delimiter=',', skiprows=1)
        data2  = np.loadtxt('./bucket_1_16402/MF%d/16402_%d_scan.csv'%(idx,idx), delimiter=',', skiprows=1)
        
        idx1 = np.random.choice(np.arange(len(data1)), 200000+idx, replace=False)
        idx2 = np.random.choice(np.arange(len(data2)), 200000+idx, replace=False)

        #data1 = data1[:,:3]
        #data2 = data2[:,:3]
        
        data1 = data1[idx1,:3]
        data2 = data2[idx2,:3]

        np.savetxt('./bucket_1_16402/MF%d/cad%d.txt'%(idx,idx),data1,delimiter='\t')
        np.savetxt('./bucket_1_16402/MF%d/scan%d.txt'%(idx,idx),data2,delimiter='\t')

import trimesh
def quality():
    cad = trimesh.load_mesh('/home/juheonlee/juheon_work/bucket_data/part1111.stl')
    pts = np.asarray(cad.vertices)
    np.savetxt('quality4.txt',pts)

def matching():
    scan  = np.loadtxt('./24hrs/MF1/15914_1_Top.pcd_aligned_SAMPLED_POINTS_C2C_DIST.csv', delimiter=',', skiprows=1)
    cad   = np.loadtxt('./merged_cad2.csv', delimiter=',', skiprows=1)
    scan = torch.from_numpy(scan)[:,:3].cuda()
    cad = torch.from_numpy(cad)[:,:3].cuda()
    print(scan.shape)
    print(cad.shape)
    # compute knn distance
    idx1,idx2 = torch_geometric.nn.knn(scan, cad, 1, None,cosine=True)
    print(idx1)
    print(idx2)
    pts = scan[idx2].cpu()
    np.savetxt('match_co.csv', pts.numpy(), delimiter=',')

if __name__=='__main__':
    sanity()
    #sanity2()
    #quality()
    #conversion()
    #matching()
    #