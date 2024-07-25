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

import torch
import numpy as np

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

def set_seed(rank):
    """
    Set seeds for NumPy and PyTorch to ensure reproducibility in distributed settings
    """
    np.random.seed(rank % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))

def configure_cuda_for_consistent_precision():
    """
    Configures CUDA and cuDNN settings to ensure consistent precision by
    disabling TensorFloat-32 (TF32) and reduced precision settings.
    """
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

def compute_num_accumulation_rounds(total_batch_size, batch_size_per_gpu, world_size):
    """
    Calculate the total batch size per GPU in a distributed setting, log the batch size per GPU, ensure it's within valid limits,
    determine the number of accumulation rounds, and validate that the global batch size matches the expected value.
    """
    batch_gpu_total = total_batch_size // world_size
    batch_size_per_gpu = batch_size_per_gpu
    if batch_size_per_gpu is None or batch_size_per_gpu > batch_gpu_total:
        batch_size_per_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_size_per_gpu
    if total_batch_size != batch_size_per_gpu * num_accumulation_rounds * world_size:
        raise ValueError(
            "total_batch_size must be equal to batch_size_per_gpu * num_accumulation_rounds * world_size"
        )
    return batch_gpu_total, num_accumulation_rounds

