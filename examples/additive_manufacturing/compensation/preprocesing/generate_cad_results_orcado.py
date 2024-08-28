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


import trimesh
import numpy as np


l = 2
for i in range(0,l): 
    print('generating Part# %d'%(i+1))
    cad = trimesh.load_mesh('/home/leejuhe/juheon_work/ocardo_iso/%d/MHC_Fuse_1_%d_OPT_860Kpoints.stl'%(i+1,i+1))
    #cad = trimesh.load_mesh('/home/leejuhe/juheon_work/MHC_Fuse_1_%d_OPT.stl'%(i+1))
    #cad = trimesh.load_mesh('/home/leejuhe/juheon_work/ocardo_split/MHC_Fuse_1_%d_OPT_remeshed_bottom.stl'%(i+1))
    #cad = trimesh.load_mesh('/home/leejuhe/juheon_work/ocardo_split/MHC_Fuse_1_%d_OPT_remeshed_top.stl'%(i+1))
    #cad = trimesh.load_mesh('/home/leejuhe/juheon_work/isometric/15914/MF1-HUH-10EGG-02-SB3-A1_part_%d_remeshed.ply'%(i+1))
    #cad = trimesh.load_mesh('/home/leejuhe/juheon_work/isometric/15914/MF1-HUH-10EGG-02-SB3-A1_part_1_NTU_%d.stl'%(i+1))
    #cad = trimesh.load_mesh('/home/leejuhe/juheon_work/isometric/15914/MF1-HUH-10EGG-02-SB3-A1_part_1_NTU20MB_%d.stl'%(i+1))
    #cad = trimesh.load_mesh('/home/leejuhe/juheon_work/isometric/17134/MF%d.stl'%(i+1))
    #cad = trimesh.load_mesh('/home/leejuhe/juheon_work/isometric/18182/remeshed_pack_Part-%d.obj'%(i+1))
    
    pts = np.loadtxt('/home/leejuhe/juheon_work/DL_engine/results/ocardo/MHC_Fuse_1_%d_OPT_860Kpoints_comp.csv'%(i+1),delimiter=',') 
    #pts = np.loadtxt('/home/leejuhe/juheon_work/DL_engine/results/ocardo/v3/MHC_Fuse_1_%d_OPT_remeshed_bottom_comp.csv'%(i+1),delimiter=',')    
    #pts = np.loadtxt('/home/leejuhe/juheon_work/DL_engine/results/ocardo/v3/MHC_Fuse_1_%d_OPT_remeshed_top_comp.csv'%(i+1),delimiter=',') 
    #pts = np.loadtxt('/home/leejuhe/juheon_work/DL_engine/results/r1/MF1-HUH-10EGG-02-SB3-A1_part_%d_remeshed_comp.csv'%(i+1),delimiter=',')
    #pts = np.loadtxt('/home/leejuhe/juheon_work/DL_engine/results/r1/MF1-HUH-10EGG-02-SB3-A1_part_1_NTU_%d_comp.csv'%(i+1),delimiter=',')
    #pts = np.loadtxt('/home/leejuhe/juheon_work/DL_engine/results/r1/MF1-HUH-10EGG-02-SB3-A1_part_1_NTU20MB_%d_comp.csv'%(i+1),delimiter=',')    
    #pts = np.loadtxt('/home/leejuhe/juheon_work/DL_engine/results/r1/remeshed_pack_Part-%d_comp.csv'%(i+1),delimiter=',')    
    #pts = np.loadtxt('/home/leejuhe/juheon_work/DL_engine/results/r3/MF%d_comp.csv'%(i+1),delimiter=',')    
    
    cad.vertices = pts 
    cad.export('/home/leejuhe/juheon_work/DL_engine/results/ocardo/MHC_Fuse_1_%d_OPT_860Kpoints_comp.stl'%(i+1))
    #cad.export('/home/leejuhe/juheon_work/DL_engine/results/ocardo/MHC_Fuse_1_%d_OPT_remeshed_bottom_comp.stl'%(i+1))
    #cad.export('/home/leejuhe/juheon_work/DL_engine/results/ocardo/MHC_Fuse_1_%d_OPT_remeshed_top_comp.stl'%(i+1))
    #cad.export('/home/leejuhe/juheon_work/DL_engine/results/r1/MF1-HUH-10EGG-02-SB3-A1_part_%d_remeshed_comp.stl'%(i+1))
    #cad.export('/home/leejuhe/juheon_work/DL_engine/results/r1/MF1-HUH-10EGG-02-SB3-A1_part_1_NTU_%d_comp.stl'%(i+1))
    #cad.export('/home/leejuhe/juheon_work/DL_engine/results/r1/MF1-HUH-10EGG-02-SB3-A1_part_1_NTU20MB_%d_comp.stl'%(i+1))
    #cad.export('/home/leejuhe/juheon_work/DL_engine/results/r3/MF%d_comp.stl'%(i+1))
    #cad.export('/home/leejuhe/juheon_work/DL_engine/results/r1/remeshed_pack_Part-%d_comp.stl'%(i+1))
