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

import os
from pathlib import Path
from pydantic import BaseModel
from typing import Tuple, Optional


class Constants(BaseModel):
    """Ahmed Body model constants"""

    ckpt_path: str = "./checkpoints"
    ckpt_name: str = "./ahmed_body.pt"
    data_dir: str = "../dataset"
    results_dir: str = "./results"

    input_dim_nodes: int = 11
    input_dim_edges: int = 4
    output_dim: int = 4
    aggregation: int = "sum"
    hidden_dim_node_encoder = 256
    hidden_dim_edge_encoder = 256
    hidden_dim_node_decoder = 256

    batch_size: int = 1
    epochs: int = 500
    num_training_samples: int = 683
    num_validation_samples: int = 100
    num_test_samples: int = 100

    lr: float = 1e-4
    lr_decay_rate: float = 0.99985

    amp: bool = False
    jit: bool = False

    wandb_mode = "disabled"
