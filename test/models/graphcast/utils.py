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

import dgl
import torch
import sys, os
import numpy as np


def fix_random_seeds():
    """Fix random seeds for reproducibility"""
    dgl.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)


def create_random_input(input_res, dim):
    """Create random input for testing"""
    return torch.randn(1, dim, *input_res)


def get_icosphere_path():
    """Get path to icosphere mesh"""
    script_path = os.path.abspath(__file__)
    icosphere_path = os.path.join(os.path.dirname(script_path), "icospheres.json")
    return icosphere_path
