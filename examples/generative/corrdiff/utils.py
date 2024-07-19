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

   
def set_patch_shape(img_shape, patch_shape):
    img_shape_y, img_shape_x = img_shape
    patch_shape_y, patch_shape_x = patch_shape
    if (patch_shape_x is None) or (patch_shape_x > img_shape_x):
        patch_shape_x = img_shape_x
    if (patch_shape_y is None) or (patch_shape_y > img_shape_y):
        patch_shape_y = img_shape_y
    if patch_shape_x != img_shape_x or patch_shape_y != img_shape_y:
        if patch_shape_x != patch_shape_y:
            raise NotImplementedError("Rectangular patch not supported yet")
        if patch_shape_x % 32 != 0 or patch_shape_y % 32 != 0:
            raise ValueError("Patch shape needs to be a multiple of 32")
    return (img_shape_y, img_shape_x) , (patch_shape_y, patch_shape_x)