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
import torch.nn.functional as F

from torch import Tensor
from modulus.utils.graphcast.loss import DefaultLoss, CustomLoss

dtypes = (torch.float, torch.bfloat16)
n = (100, 1000)
d = (100, 200, 400)

for dt in dtypes:
    for nn in n:
        area = torch.rand(nn, dtype=dt, device="cuda") + 0.01
        default_loss = DefaultLoss(area)
        custom_loss = CustomLoss(area)

        for dd in d:
            invar1 = torch.rand(nn, dd, dtype=dt, device="cuda")
            outvar1 = torch.rand(nn, dd, dtype=dt, device="cuda")

            invar2 = invar1.clone().detach()
            outvar2 = outvar1.clone().detach()

            invar1.requires_grad_()
            invar2.requires_grad_()

            loss1 = default_loss(invar1, outvar1)
            loss1.backward()
            grad1 = invar1.grad

            loss2 = custom_loss(invar2, outvar2)
            loss2.backward()
            grad2 = invar2.grad

            atol = 1.0e-8 if dt == torch.float else 1.0e-6
            loss_diff = torch.abs(loss1 - loss2)
            loss_diff_msg = (
                f"{dt}-{nn}-{dd}: loss diff - min/max/mean: {loss_diff.min()} / "
                f"{loss_diff.max()} / {loss_diff.mean()}"
            )
            grad_diff = torch.abs(grad1 - grad2)
            grad_diff_msg = (
                f"{dt}-{nn}-{dd}: grad diff - min/max/mean: {grad_diff.min()} / "
                f"{grad_diff.max()} / {grad_diff.mean()}"
            )

            assert torch.allclose(loss1, loss2, atol=atol), loss_diff_msg
            assert torch.allclose(grad1, grad2, atol=atol), grad_diff_msg
