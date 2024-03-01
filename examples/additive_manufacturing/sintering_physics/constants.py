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
from pathlib import Path
from pydantic import BaseModel
from typing import Tuple, Optional


class Constants(BaseModel):
    """Virtual Foundry (Digital Sintering) Graphnet constants"""

    # Train model, one step evaluation or rollout evaluation, options: ['train', 'eval', 'eval_rollout']
    mode: str = "train"
    # eval_split is Split to use when running evaluation, options: ['train', 'valid', 'test']
    eval_split: str = "test"

    # data configs
    data_path: str = "./data/test_validation"

    # training configs
    batch_size: int = 1
    num_steps: int = int(2e7)
    eval_steps: int = 1
    # default 6.7e-4, for eval, set noise_std=0
    noise_std: float = 1e-9
    # loss type options: ['standard', 'anchor', 'me', 'correlation', 'anchor_me']
    loss: str = "me"
    # loss_decay_factor for loss type = me, for example, range (0, 1]
    loss_decay_factor: float = 0.6
    l_plane: float = 30
    l_me: float = 3
    # The path for saving checkpoints of the model.
    model_path_vfgn: str = "models/test"
    # The path for saving outputs (e.g. rollouts).
    output_path: str = "rollouts/test24"
    prefetch_buffer_size: int = 100

    # devices settings
    device: str = "cuda:0"
    # flags.DEFINE_string('message_passing_devices', 'cuda:0', or "['cuda:0', 'cuda:1]",help="The devices for message passing")
    message_passing_devices: str = "['cuda:0']"

    # performance configs
    fp16: bool = False

    # test & visualization configs
    rollout_refine: bool = False
    meta1: str = "step100_s1"
    meta2: str = "step100_s2"
    version_modulus: bool = False

    # Rollout settings
    rollout_path: str = "rollouts/rollout_test_0.json"
    metadata_path: str = "./data/test_validation"
    step_stride: int = 3
    block_on_show: bool = True
    # test data type: ['standard', 'train', 'test']
    ds_type: str = "standard"
    # Test build name
    test_build: str = "test0"
    plot_tolerance_range: bool = True
    plot_3d: bool = False

    # Data preprocessing settings
    ADD_ANCHOR: bool = True
    raw_data_dir: str = None
    process_step: int = 1
