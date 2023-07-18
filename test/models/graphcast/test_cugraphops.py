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


def test_cugraphops(num_channels=2, res_h=21, res_w=10):
    """Test cugraphops"""
    icosphere_path = get_icosphere_path()

    # Fix random seeds
    fix_random_seeds()

    # Random input
    x = torch.randn(1, num_channels, res_h, res_w, device="cuda")
    x_dgl = x.clone().detach()

    for concat_trick in [False, True]:
        for recomp_act in [False, True]:
            # Fix random seeds
            torch.manual_seed(0)
            torch.cuda.manual_seed(0)
            np.random.seed(0)

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
                do_concat_trick=concat_trick,
                use_cugraphops_decoder=True,
                use_cugraphops_encoder=True,
                use_cugraphops_processor=True,
                recompute_activation=recomp_act,
            ).to("cuda")

            # Fix random seeds again
            fix_random_seeds()

            model_dgl = GraphCastNet(
                meshgraph_path=icosphere_path,
                static_dataset_path=None,
                input_res=(res_h, res_w),
                input_dim_grid_nodes=num_channels,
                input_dim_mesh_nodes=3,
                input_dim_edges=4,
                output_dim_grid_nodes=num_channels,
                processor_layers=3,
                hidden_dim=4,
                do_concat_trick=concat_trick,
                use_cugraphops_decoder=False,
                use_cugraphops_encoder=False,
                use_cugraphops_processor=False,
                recompute_activation=False,
            ).to("cuda")

            # Forward pass without checkpointing
            x.requires_grad_()
            y_pred = model(x)
            loss = y_pred.sum()
            loss.backward()
            x_grad = x.grad

            x_dgl.requires_grad_()
            y_pred_dgl = model_dgl(x_dgl)
            loss_dgl = y_pred_dgl.sum()
            loss_dgl.backward()
            x_grad_dgl = x_dgl.grad

            # Check that the results are the same
            assert torch.allclose(
                y_pred_dgl, y_pred, atol=1.0e-6
            ), "testing DGL against cugraph-ops: outputs do not match!"
            assert torch.allclose(
                x_grad_dgl, x_grad, atol=1.0e-4, rtol=1.0e-3
            ), "testing DGL against cugraph-ops: gradients do not match!"


if __name__ == "__main__":
    test_cugraphops()
