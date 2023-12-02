# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import numpy as np
import torch
from modulus.experimental.sfno.third_party.climt.zenith_angle import cos_zenith_angle
import datetime
from netCDF4 import Dataset as DS
import h5py

def get_orography(orography_path):
    """ returns the surface geopotential for each grid point after normalizing it to be in the range [0, 1]"""

    with DS(orography_path, "r") as f:
            
        orography = f.variables["Z"][0,:,:]

        orography = (orography - orography.min()) / (orography.max() - orography.min())

    return orography

def get_land_mask(land_mask_path):
    """ returns the land mask for each grid point. land sea mask is between 0 and 1 """

    with h5py.File(land_mask_path, "r") as f:

        lsm = f["LSM"][0,:,:]
    
    return lsm


if __name__ == "__main__":

    lsm = get_land_mask("/code/utils/e5.oper.invariant.128_172_lsm.ll025sc.1979010100_1979010100.nc")
    figure = plt.figure(figsize=(10, 10))
    ax = figure.add_subplot(111)
    im = ax.imshow(lsm)
    ax.set_title("Land Mask")
    #add colorbar
    cbar = figure.colorbar(im, ax=ax)
    plt.savefig("land_mask.png")

    orography = get_orography("/code/utils/e5.oper.invariant.128_129_z.ll025sc.1979010100_1979010100.nc")
    figure = plt.figure(figsize=(10, 10))
    ax = figure.add_subplot(111)
    im = ax.imshow(orography)
    ax.set_title("Orography")
    #add colorbar
    cbar = figure.colorbar(im, ax=ax)
    plt.savefig("orography.png")
