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

import sys
import torch
import numpy as np

from modulus.models.graphcast.graph_cast_net import GraphCastNet

# Fix random seeds
torch.manual_seed(0)
np.random.seed(0)

# Random input
x = torch.randn(1, 2, 721, 1440)

# Instantiate the model
model = GraphCastNet(
    meshgraph_path="./icospheres.pickle",
    static_dataset_path=None,
    input_dim_grid_nodes=2,
    input_dim_mesh_nodes=3,
    input_dim_edges=4,
    output_dim_grid_nodes=2,
    processor_layers=3,
    hidden_dim=4,
    do_concat_trick=True,
)

# Fix random seeds again
torch.manual_seed(0)
np.random.seed(0)

# Instantiate the model with concat trick enabled
model_ct = GraphCastNet(
    meshgraph_path="./icospheres.pickle",
    static_dataset_path=None,
    input_dim_grid_nodes=2,
    input_dim_mesh_nodes=3,
    input_dim_edges=4,
    output_dim_grid_nodes=2,
    processor_layers=3,
    hidden_dim=4,
    do_concat_trick=False,
)

# Forward pass without checkpointing
y_pred = model(x)
y_pred_ct = model_ct(x)

# Check that the results are the same
assert torch.allclose(y_pred_ct, y_pred), "Concat trick failed!"
