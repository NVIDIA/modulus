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
for i in range(1,l): 
    print('generating Part# %d'%(i+1))
    #cad = trimesh.load_mesh('/home/leejuhe/juheon_work/bucket_data/val2/MF1-HUH-10EGG-02-SB3-A1_part_1_V5-NTU_%d.stl'%(i+1))
    cad = trimesh.load_mesh('/home/leejuhe/juheon_work/MHC_Fuse_1_2_OPT.stl')
    #cad = trimesh.load_mesh('/home/juheonlee/juheon_work/bucket_data/masks/Mask_%d.stl'%(i+1))
    pts = np.loadtxt('outp_%d.csv'%(i+1),delimiter=',')
    cad.vertices = pts 
    cad.export('outp%d.stl'%(i+1))
