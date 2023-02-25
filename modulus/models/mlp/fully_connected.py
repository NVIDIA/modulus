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
import torch.nn as nn
import modulus

from torch import Tensor
from dataclasses import dataclass
from typing import Optional, Union, List
from modulus.models.layers import FCLayer
from ..meta import ModelMetaData
from ..module import Module


@dataclass
class MetaData(ModelMetaData):
    name: str = "FullyConnected"
    # Optimization
    jit: bool = True
    cuda_graphs: bool = True
    amp: bool = True
    torch_fx: bool = True
    # Inference
    onnx: bool = True
    onnx_runtime: bool = True
    # Physics informed
    func_torch: bool = True
    auto_grad: bool = True


class FullyConnected(Module):
    """A densely-connected MLP architecture

    Parameters
    ----------
    in_features : int, optional
        Size of input features, by default 512
    layer_size : int, optional
        Size of every hidden layer, by default 512
    out_features : int, optional
        Size of output features, by default 512
    num_layers : int, optional
        Number of hidden layers, by default 6
    activation_fn : Union[nn.Module, List[nn.Module]], optional
        Activation function to use, by default nn.SILU
    skip_connections : bool, optional
        Add skip connections every 2 hidden layers, by default False
    adaptive_activations : bool, optional
        Use an adaptive activation function, by default False
    weight_norm : bool, optional
        Use weight norm on fully connected layers, by default False

    Example
    -------
    >>> model = modulus.models.mlp.FullyConnected(in_features=32, out_features=64)
    >>> input = torch.randn(128, 32)
    >>> output = model(input)
    >>> output.size()
    torch.Size([128, 64])
    """

    def __init__(
        self,
        in_features: int = 512,
        layer_size: int = 512,
        out_features: int = 512,
        num_layers: int = 6,
        activation_fn: Union[nn.Module, List[nn.Module]] = nn.SiLU(),
        skip_connections: bool = False,
        adaptive_activations: bool = False,
        weight_norm: bool = False,
    ) -> None:
        super().__init__(meta=MetaData())

        self.skip_connections = skip_connections

        if adaptive_activations:
            activation_par = nn.Parameter(torch.ones(1))
        else:
            activation_par = None

        if not isinstance(activation_fn, list):
            activation_fn = [activation_fn] * num_layers
        if len(activation_fn) < num_layers:
            activation_fn = activation_fn + [activation_fn[-1]] * (
                num_layers - len(activation_fn)
            )

        self.layers = nn.ModuleList()

        layer_in_features = in_features
        for i in range(num_layers):
            self.layers.append(
                FCLayer(
                    layer_in_features,
                    layer_size,
                    activation_fn[i],
                    weight_norm,
                    activation_par,
                )
            )
            layer_in_features = layer_size

        self.final_layer = FCLayer(
            in_features=layer_size,
            out_features=out_features,
            activation_fn=None,
            weight_norm=False,
            activation_par=None,
        )

    def forward(self, x: Tensor) -> Tensor:
        x_skip: Optional[Tensor] = None
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.skip_connections and i % 2 == 0:
                if x_skip is not None:
                    x, x_skip = x + x_skip, x
                else:
                    x_skip = x

        x = self.final_layer(x)
        return x
