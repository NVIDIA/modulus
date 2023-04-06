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

sys.path.append("../")
from modulus.models.graphcast.graph_cast_net import GraphCastNet

# Fix random seeds
torch.manual_seed(0)
np.random.seed(0)

# Random input
x = torch.randn(1, 3, 721, 1440)

# Instantiate the model
model = GraphCastNet(
    meshgraph_path="./icospheres.pickle",
    static_dataset_path=None,
    input_dim_grid_nodes=3,
    input_dim_mesh_nodes=3,
    input_dim_edges=4,
    output_dim_grid_nodes=3,
    processor_layers=4,
    hidden_dim=16,
    do_concat_trick=True,
)

# Set gradient checkpointing
model.set_checkpoint_model(False)
model.set_checkpoint_encoder(True)
model.set_checkpoint_processor(2)
model.set_checkpoint_decoder(True)

# Forward pass with checkpointing
y_pred_checkpointed = x
for i in range(2):
    y_pred_checkpointed = model(y_pred_checkpointed)

# Set gradient checkpointing
model.set_checkpoint_model(False)
model.set_checkpoint_encoder(False)
model.set_checkpoint_processor(1)
model.set_checkpoint_decoder(False)

# Forward pass without checkpointing
y_pred = x
for i in range(2):
    y_pred = model(y_pred)

# Check that the results are the same
assert torch.allclose(y_pred_checkpointed, y_pred), "Checkpointing failed!"
