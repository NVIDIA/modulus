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
    """Stokes flow model constants"""

    ckpt_path: str = "./checkpoints"
    ckpt_name: str = "./stokes.pt"
    data_dir: str = "./dataset"
    results_dir: str = "./results"

    input_dim_nodes: int = 7
    input_dim_edges: int = 3
    output_dim: int = 3
    hidden_dim_node_encoder: int = 256
    hidden_dim_edge_encoder: int = 256
    hidden_dim_node_decoder: int = 256
    aggregation: int = "sum"

    batch_size: int = 1
    epochs: int = 500
    num_training_samples: int = 500

    num_validation_samples: int = 10
    num_test_samples: int = 10

    lr: float = 1e-4
    lr_decay_rate: float = 0.99985

    amp: bool = False
    jit: bool = False

    wandb_mode: str = "disabled"

    # Physics-informed constants
    graph_path: str = "graph_7.vtp"

    mlp_hidden_dim: int = 256
    mlp_num_layers: int = 6
    mlp_input_dim: int = 2
    mlp_output_dim: int = 3

    pi_iters: int = 10000
    pi_lr: float = 0.001
