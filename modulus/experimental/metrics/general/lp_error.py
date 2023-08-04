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
from typing import Any, Tuple

Tensor = torch.Tensor


def lp_error(
    pred: Tensor,
    target: Tensor,
    dim: Any = -1,
    p: int = 2,
    relative: bool = False,
    reduce: bool = False,
) -> Tensor:
    """Calculates the (relative) Lp error norm between two tensors

    Parameters
    ----------
    pred : Tensor
        Input prediction tensor
    target : Tensor
        Target tensor
    dim : Any[int, Tuple], optional
        The dimensions to calculate the Lp norm over, by default -1
    p : int, optional
        The norm, by default 2
    relative : bool, optional
        Whether to calculate the relative Lp norm, by default False
    reduce : bool, optional
        Reduce the output by averaging across remaining dimensions, by default False

    Returns
    -------
    Tensor
        Root mean squared error value(s)
    """
    diff_norms = torch.linalg.norm(pred - target, ord=p, dim=dim)
    if relative:
        target_norms = torch.linalg.norm(target, ord=p, dim=dim)
        if reduce:
            return torch.mean(diff_norms / target_norms)
        return diff_norms / target_norms
    if reduce:
        return torch.mean(diff_norms)
    return diff_norms
