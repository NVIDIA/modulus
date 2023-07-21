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
import numpy as np

from utils import get_icosphere_path, fix_random_seeds
from modulus.models.graphcast.graph_cast_net import GraphCastNet

icosphere_path = get_icosphere_path()


def test_concat_trick(num_channels=2, res_h=11, res_w=20):
    """Test concat trick"""

    # Fix random seeds
    fix_random_seeds()

    # Random input
    device = "cuda"
    x = torch.rand(1, num_channels, res_h, res_w, device=device)
    x_ct = x.clone().detach()

    for recomp_act in [False, True]:
        # Fix random seeds
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        np.random.seed(0)

        # Instantiate the model
        model = GraphCastNet(
            meshgraph_path=icosphere_path,
            static_dataset_path=None,
            input_res=(res_h, res_w),
            input_dim_grid_nodes=num_channels,
            input_dim_mesh_nodes=3,
            input_dim_edges=4,
            output_dim_grid_nodes=num_channels,
            processor_layers=3,
            hidden_dim=4,
            do_concat_trick=False,
            recompute_activation=False,
        ).to("cuda")

        # Fix random seeds again
        fix_random_seeds()

        # Instantiate the model with concat trick enabled
        model_ct = GraphCastNet(
            meshgraph_path=icosphere_path,
            static_dataset_path=None,
            input_res=(res_h, res_w),
            input_dim_grid_nodes=num_channels,
            input_dim_mesh_nodes=3,
            input_dim_edges=4,
            output_dim_grid_nodes=num_channels,
            processor_layers=3,
            hidden_dim=4,
            do_concat_trick=True,
            recompute_activation=recomp_act,
        ).to(device)

        # Forward pass without checkpointing
        x.requires_grad_()
        y_pred = model(x)
        loss = y_pred.sum()
        loss.backward()
        x_grad = x.grad

        x_ct.requires_grad_()
        y_pred_ct = model_ct(x_ct)
        loss_ct = y_pred_ct.sum()
        loss_ct.backward()
        x_grad_ct = x_ct.grad

        # Check that the results are the same
        # tolerances quite large on GPU
        assert torch.allclose(
            y_pred_ct,
            y_pred,
            atol=5.0e-3,
        ), "Concat trick failed, outputs do not match!"
        assert torch.allclose(
            x_grad_ct,
            x_grad,
            atol=1.0e-2,
        ), "Concat trick failed, gradients do not match!"


if __name__ == "__main__":
    test_concat_trick()
