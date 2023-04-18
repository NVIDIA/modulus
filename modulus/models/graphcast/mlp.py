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

from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd.function import once_differentiable

from .utils import sum_efeat_dgl
from modulus.models.layers.fused_silu import silu_backward_for

try:
    from pylibcugraphops.pytorch.operators import update_efeat_bipartite_e2e
    from pylibcugraphops.pytorch import BipartiteCSC
except:
    update_efeat_bipartite_e2e = None
    BipartiteCSC = None

try:
    from apex.normalization import FusedLayerNorm

    apex_imported = True
except:
    apex_imported = False


class CustomSiLuLinearAutogradFunction(torch.autograd.Function):
    """Custom SiLU + Linear autograd function"""

    @staticmethod
    def forward(
        ctx,
        features: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
    ) -> torch.Tensor:
        # by combining SiLU and a Linear transformation
        # we can avoid storing the activation
        # at the cost of recomputing it during the backward
        out = F.silu(features)
        out = F.linear(out, weight, bias)
        ctx.save_for_backward(features, weight)
        return out

    @staticmethod
    @once_differentiable
    def backward(
        ctx, grad_output: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor],]:
        """backward pass of the SiLU + Linear function"""
        (
            need_dgrad,
            need_wgrad,
            need_bgrad,
        ) = ctx.needs_input_grad
        features, weight = ctx.saved_tensors

        grad_features = None
        grad_weight = None
        grad_bias = None

        if need_bgrad:
            grad_bias = grad_output.sum(dim=0)

        if need_wgrad:
            out = F.silu(features)
            grad_weight = grad_output.T @ out

        if need_dgrad:
            grad_features = grad_output @ weight
            silu_backward = silu_backward_for(features.dtype, features.dim())
            grad_silu = silu_backward.execute([features])[0]
            grad_features = grad_features * grad_silu

        return grad_features, grad_weight, grad_bias


class MLP(nn.Module):
    """MLP with normalization in the last layer

    Parameters
    ----------
    input_dim : int
        dimensionality of the input features
    output_dim : int, optional
        dimensionality of the output features, by default 512
    hidden_dim : int, optional
        number of neurons in each hidden layer, by default 512
    hidden_layers : int, optional
        number of hidden layers, by default 1
    activation_fn : nn.Module, optional
        , by default nn.SiLU()
    norm_type : str, optional
        normalization type, by default "LayerNorm"
    recompute_activation : bool, optional
        Flag for recomputing recompute_activation in backward to save memory, by default False.
        Currently, only SiLU is supported.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
        recompute_activation: bool = False,
    ):
        super().__init__()

        layers = [nn.Linear(input_dim, hidden_dim), activation_fn]
        self.hidden_layers = hidden_layers
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), activation_fn]
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.norm_type = norm_type
        if norm_type is not None:
            assert norm_type in [
                "LayerNorm",
                "GraphNorm",
                "InstanceNorm",
                "BatchNorm",
                "MessageNorm",
            ]
            if norm_type == "LayerNorm" and apex_imported:
                norm_layer = FusedLayerNorm
            else:
                norm_layer = getattr(nn, norm_type)
            layers.append(norm_layer(output_dim))

        self.model = nn.Sequential(*layers)

        if recompute_activation:
            assert isinstance(activation_fn, nn.SiLU)
            self.forward_fn = self.custom_silu_linear_forward
        else:
            self.forward_fn = self.default_forward

    def default_forward(self, x: Tensor) -> Tensor:
        """default forward pass of the MLP"""
        return self.model(x)

    def custom_silu_linear_forward(self, x: Tensor) -> Tensor:
        """forward pass of the MLP where SiLU is recomputed in backward"""
        lin = self.model[0]
        hidden = lin(x)
        for i in range(1, self.hidden_layers + 1):
            lin = self.model[2 * i]
            hidden = CustomSiLuLinearAutogradFunction.apply(
                hidden, lin.weight, lin.bias
            )

        if self.norm_type is not None:
            norm = self.model[2 * self.hidden_layers + 1]
            hidden = norm(hidden)
        return hidden

    def forward(self, x: Tensor) -> Tensor:
        return self.forward_fn(x)


class TruncatedMLP(nn.Module):
    """Truncated MLP where concat+MLP is replaced by MLP+sum

    Parameters
    ----------
    efeat_dim: int
        dimension of the input edge features
    src_dim: int
        dimension of the input src-node features
    dst_dim: int
        dimension of the input dst-node features
    output_dim : int, optional
        dimensionality of the output features, by default 512
    hidden_dim : int, optional
        number of neurons in each hidden layer, by default 512
    hidden_layers : int, optional
        number of hidden layers, by default 1
    activation_fn : nn.Module, optional
        type of activation function, by default nn.SiLU()
    norm_type : str, optional
        normalization type, by default "LayerNorm"
    bias : bool, optional
        whether to use bias in the MLP, by default True
    recompute_activation : bool, optional
        Flag for recomputing activation in backward to save memory, by default False.
        Currently, only SiLU is supported.
    """

    def __init__(
        self,
        efeat_dim: int,
        src_dim: int,
        dst_dim: int,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
        bias: bool = True,
        recompute_activation: bool = False,
    ):
        super().__init__()

        self.efeat_dim = efeat_dim
        self.src_dim = src_dim
        self.dst_dim = dst_dim

        # this should ensure the same sequence of initializations
        # as the original MLP-Layer in combination with a concat operation
        tmp_lin = nn.Linear(efeat_dim + src_dim + dst_dim, hidden_dim, bias=bias)
        # orig_weight has shape (hidden_dim, efeat_dim + src_dim + dst_dim)
        orig_weight = tmp_lin.weight
        w_efeat, w_src, w_dst = torch.split(
            orig_weight, [efeat_dim, src_dim, dst_dim], dim=1
        )
        self.lin_efeat = nn.Parameter(w_efeat)
        self.lin_src = nn.Parameter(w_src)
        self.lin_dst = nn.Parameter(w_dst)

        if bias:
            self.bias = tmp_lin.bias
        else:
            self.bias = None

        layers = [activation_fn]
        self.hidden_layers = hidden_layers
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), activation_fn]
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.norm_type = norm_type
        if norm_type is not None:
            assert norm_type in [
                "LayerNorm",
                "GraphNorm",
                "InstanceNorm",
                "BatchNorm",
                "MessageNorm",
            ]
            if norm_type == "LayerNorm" and apex_imported:
                norm_layer = FusedLayerNorm
            else:
                norm_layer = getattr(nn, norm_type)
            layers.append(norm_layer(output_dim))

        self.model = nn.Sequential(*layers)

        if recompute_activation:
            assert isinstance(activation_fn, nn.SiLU)
            self.forward_fn = self.custom_silu_linear_forward
        else:
            self.forward_fn = self.default_forward

    def forward_truncated_sum(
        self,
        efeat: Tensor,
        src_feat: Tensor,
        dst_feat: Tensor,
        src_idx: Tensor,
        dst_idx: Tensor,
    ) -> Tensor:
        """forward pass of the truncated MLP. This uses separate linear layers without
        bias. Bias is added to one MLP, as we sum afterwards. This adds the bias to the
         total sum, too. Having it in one F.linear should allow a fusion of the bias
         addition while avoiding adding the bias to the "edge-level" result.
        """
        mlp_efeat = F.linear(efeat, self.lin_efeat, None)
        mlp_src = F.linear(src_feat, self.lin_src, None)
        mlp_dst = F.linear(dst_feat, self.lin_dst, self.bias)
        mlp_sum = sum_efeat_dgl(mlp_efeat, mlp_src, mlp_dst, src_idx, dst_idx)
        return mlp_sum

    def default_forward(
        self,
        efeat: Tensor,
        src_feat: Tensor,
        dst_feat: Tensor,
        src_idx: Tensor,
        dst_idx: Tensor,
    ) -> Tensor:
        """Default forward pass of the truncated MLP."""
        mlp_sum = self.forward_truncated_sum(
            efeat, src_feat, dst_feat, src_idx, dst_idx
        )
        return self.model(mlp_sum)

    def custom_silu_linear_forward(
        self,
        efeat: Tensor,
        src_feat: Tensor,
        dst_feat: Tensor,
        src_idx: Tensor,
        dst_idx: Tensor,
    ) -> Tensor:
        """Forward pass of the truncated MLP with custom SiLU function."""
        mlp_sum = self.forward_truncated_sum(
            efeat, src_feat, dst_feat, src_idx, dst_idx
        )
        lin = self.model[1]
        hidden = CustomSiLuLinearAutogradFunction.apply(mlp_sum, lin.weight, lin.bias)
        for i in range(2, self.hidden_layers + 1):
            lin = self.model[2 * i - 1]
            hidden = CustomSiLuLinearAutogradFunction.apply(
                hidden, lin.weight, lin.bias
            )

        if self.norm_type is not None:
            norm = self.model[2 * self.hidden_layers]
            hidden = norm(hidden)
        return hidden

    def forward(
        self,
        efeat: Tensor,
        src_feat: Tensor,
        dst_feat: Tensor,
        src_idx: Tensor,
        dst_idx: Tensor,
    ) -> Tensor:
        return self.forward_fn(efeat, src_feat, dst_feat, src_idx, dst_idx)


class TruncatedMLPCuGraph(TruncatedMLP):
    """Truncated MLP where concat+MLP is replaced by MLP+sum
       which uses CuGraph as the GNN backend.

    Parameters
    ----------
    efeat_dim: int
        dimension of the input edge features
    src_dim: int
        dimension of the input src-node features
    dst_dim: int
        dimension of the input dst-node features
    output_dim : int, optional
        dimensionality of the output features, by default 512
    hidden_dim : int, optional
        number of neurons in each hidden layer, by default 512
    hidden_layers : int, optional
        number of hidden layers, by default 1
    activation_fn : nn.Module, optional
        type of activation function, by default nn.SiLU()
    norm_type : str, optional
        normalization type, by default "LayerNorm"
    bias : bool, optional
        whether to use bias in the MLP, by default True
    recompute_activation : bool, optional
        Flag for recomputing activation in backward to save memory, by default False.
        Currently, only SiLU is supported.
    """

    def __init__(
        self,
        efeat_dim: int,
        src_dim: int,
        dst_dim: int,
        output_dim: int = 512,
        hidden_dim: int = 512,
        hidden_layers: int = 1,
        activation_fn: nn.Module = nn.SiLU(),
        norm_type: str = "LayerNorm",
        bias: bool = True,
        recompute_activation: bool = False,
    ):
        super().__init__(
            efeat_dim,
            src_dim,
            dst_dim,
            output_dim,
            hidden_dim,
            hidden_layers,
            activation_fn,
            norm_type,
            bias,
            recompute_activation,
        )

    def forward_truncated_sum(
        self,
        efeat: Tensor,
        src_feat: Tensor,
        dst_feat: Tensor,
        graph: BipartiteCSC,
    ) -> Tensor:
        """Forward pass of the truncated MLP. This uses separate linear layers without
        bias. Bias is added to one MLP, as we sum afterwards. This adds the bias to the
        total sum, too."""
        mlp_efeat = F.linear(efeat, self.lin_efeat, None)
        mlp_src = F.linear(src_feat, self.lin_src, None)
        mlp_dst = F.linear(dst_feat, self.lin_dst, self.bias)
        mlp_sum = update_efeat_bipartite_e2e(
            mlp_efeat, mlp_src, mlp_dst, graph, mode="sum"
        )
        return mlp_sum

    def default_forward(
        self,
        efeat: Tensor,
        src_feat: Tensor,
        dst_feat: Tensor,
        graph: BipartiteCSC,
    ) -> Tensor:
        """Default forward pass of the truncated MLP."""
        mlp_sum = self.forward_truncated_sum(efeat, src_feat, dst_feat, graph)
        return self.model(mlp_sum)

    def custom_silu_linear_forward(
        self,
        efeat: Tensor,
        src_feat: Tensor,
        dst_feat: Tensor,
        graph: BipartiteCSC,
    ) -> Tensor:
        """Forward pass of the truncated MLP with custom SiLU function."""
        mlp_sum = self.forward_truncated_sum(efeat, src_feat, dst_feat, graph)
        lin = self.model[1]
        hidden = CustomSiLuLinearAutogradFunction.apply(mlp_sum, lin.weight, lin.bias)
        for i in range(2, self.hidden_layers + 1):
            lin = self.model[2 * i - 1]
            hidden = CustomSiLuLinearAutogradFunction.apply(
                hidden, lin.weight, lin.bias
            )

        if self.norm_type is not None:
            norm = self.model[2 * self.hidden_layers]
            hidden = norm(hidden)
        return hidden

    def forward(
        self, efeat: Tensor, src_feat: Tensor, dst_feat: Tensor, graph
    ) -> Tensor:
        return self.forward_fn(efeat, src_feat, dst_feat, graph)
