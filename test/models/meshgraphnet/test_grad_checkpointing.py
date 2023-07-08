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

import torch
import pytest
import dgl
import numpy as np

from modulus.models.meshgraphnet import MeshGraphNet


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_grad_checkpointing(device):
    """Test gradient checkpointing"""

    # constants
    model_kwds = {
        "input_dim_nodes": 4,
        "input_dim_edges": 3,
        "output_dim": 2,
        "do_concat_trick": True,
        "num_processor_checkpoint_segments": 0,
    }

    # Fix random seeds
    dgl.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # Instantiate the model without checkpointing
    model = MeshGraphNet(**model_kwds).to(device)

    # Fix random seeds
    dgl.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # Instantiate the model with checkpointing
    model_kwds["num_processor_checkpoint_segments"] = 2
    model_checkpointed = MeshGraphNet(**model_kwds).to(device)

    # Random input
    num_nodes, num_edges = 18, 12
    src = torch.tensor([np.random.randint(num_nodes) for _ in range(num_edges)])
    dst = torch.tensor([np.random.randint(num_nodes) for _ in range(num_edges)])
    graph = dgl.graph((src, dst)).to(device)
    node_features = torch.randn(num_nodes, model_kwds["input_dim_nodes"]).to(device)
    edge_features = torch.randn(num_edges, model_kwds["input_dim_edges"]).to(device)

    # Forward pass with checkpointing
    y_pred = model(node_features, edge_features, graph)
    y_pred_checkpointed = model_checkpointed(node_features, edge_features, graph)

    # dummy loss
    loss = y_pred.sum()
    loss_checkpointed = y_pred_checkpointed.sum()

    # compute gradients without checkpointing
    loss.backward()
    computed_grads = {}
    for name, param in model.named_parameters():
        computed_grads[name] = param.grad.clone()

    # compute gradients with checkpointing
    loss_checkpointed.backward()
    computed_grads_checkpointed = {}
    for name, param in model_checkpointed.named_parameters():
        computed_grads_checkpointed[name] = param.grad.clone()

    # Compare the gradients
    for name in computed_grads:
        torch.allclose(
            computed_grads_checkpointed[name], computed_grads[name]
        ), "Gradient do not match. Checkpointing failed!"

    # Check that the results are the same
    assert torch.allclose(
        y_pred_checkpointed, y_pred
    ), "Outputs do not match. Checkpointing failed!"
