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
    """vortex shedding constants"""

    # data configs
    data_dir: str = "./raw_dataset/cylinder_flow/cylinder_flow"

    # training configs
    batch_size: int = 1
    epochs: int = 25
    num_training_samples: int = 400
    num_training_time_steps: int = 300
    lr: float = 0.0001
    lr_decay_rate: float = 0.9999991
    num_input_features: int = 6
    num_output_features: int = 3
    num_edge_features: int = 3
    ckpt_path: str = "checkpoints"
    ckpt_name: str = "model.pt"

    # performance configs
    amp: bool = False
    jit: bool = False

    # test & visualization configs
    num_test_samples: int = 10
    num_test_time_steps: int = 300
    viz_vars: Tuple[str, ...] = ("u", "v", "p")
    frame_skip: int = 10
    frame_interval: int = 1

    # wb configs
    wandb_mode: str = "disabled"
    watch_model: bool = False
