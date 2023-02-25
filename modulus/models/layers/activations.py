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

Tensor = torch.Tensor


class Identity(nn.Module):
    """Identity activation function

    Dummy function for removing activations from a model

    Example
    -------
    >>> idnt_func = modulus.models.layers.Identity()
    >>> input = torch.randn(2, 2)
    >>> output = idnt_func(input)
    >>> torch.allclose(input, output)
    True
    """

    def forward(self, x: Tensor) -> Tensor:
        return x


class Stan(nn.Module):
    """Self-scalable Tanh (Stan) for 1D Tensors

    Parameters
    ----------
    out_features : int, optional
        Number of features, by default 1

    Note
    ----
    References: Gnanasambandam, Raghav and Shen, Bo and Chung, Jihoon and Yue, Xubo and others.
    Self-scalable Tanh (Stan): Faster Convergence and Better Generalization
    in Physics-informed Neural Networks. arXiv preprint arXiv:2204.12589, 2022.

    Example
    -------
    >>> stan_func = modulus.models.layers.Stan(out_features=1)
    >>> input = torch.Tensor([[0],[1],[2]])
    >>> stan_func(input)
    tensor([[0.0000],
            [1.5232],
            [2.8921]], grad_fn=<MulBackward0>)
    """

    def __init__(self, out_features: int = 1):
        super().__init__()
        self.beta = nn.Parameter(torch.ones(out_features))

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.beta.shape[-1]:
            raise ValueError(
                f"The last dimension of the input must be equal to the dimension of Stan parameters. Got inputs: {x.shape}, params: {self.beta.shape}"
            )
        return torch.tanh(x) * (1.0 + self.beta * x)


class SquarePlus(nn.Module):
    """Squareplus activation

    Note
    ----
    Reference: arXiv preprint arXiv:2112.11687

    Example
    -------
    >>> sqr_func = modulus.models.layers.SquarePlus()
    >>> input = torch.Tensor([[1,2],[3,4]])
    >>> sqr_func(input)
    tensor([[1.6180, 2.4142],
            [3.3028, 4.2361]])
    """

    def __init__(self):
        super().__init__()
        self.b = 4

    def forward(self, x: Tensor) -> Tensor:
        return 0.5 * (x + torch.sqrt(x * x + self.b))
