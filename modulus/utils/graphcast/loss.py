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


class DefaultLoss(nn.Module):
    """Default loss function with cell area weighting.

    Parameters
    ----------
    area : torch.Tensor
        Cell area with shape (N,).
    """

    def __init__(self, area):
        super().__init__()
        self.area = area

    def forward(self, invar, outvar):
        loss = (invar - outvar) ** 2
        loss = loss.mean(dim=(0, 1))

        loss = torch.mul(loss, self.area)
        loss = loss.mean()
        return loss


class CustomLossFunction(torch.autograd.Function):
    """Custom loss function with cell area weighting.

    Parameters
    ----------
    invar : torch.Tensor
        Invar.
    outvar : torch.Tensor
        Outvar.
    area : torch.Tensor
        Cell area with shape (N,).
    """

    @staticmethod
    def forward(ctx, invar, outvar, area):
        with torch.no_grad():
            diff = invar - outvar
            loss = diff**2
            loss = loss.mean(dim=1)
            loss = torch.mul(loss, area)
            loss = torch.mean(loss)

            loss_grad = 2 * (diff)
            loss_grad *= 1.0 / (invar.size(0) * invar.size(1))
            loss_grad *= area.unsqueeze(-1)
        ctx.save_for_backward(loss_grad)
        return loss

    @staticmethod
    def backward(ctx, _):
        """Backward method"""
        # "grad_output" should be 1, here
        # hence simply ignore
        (grad_invar,) = ctx.saved_tensors
        return grad_invar, None, None


class CustomLoss(nn.Module):
    """Custom loss with cell area weighting.

    Parameters
    ----------
    area : torch.Tensor
        Cell area with shape (N,).
    """

    def __init__(self, area):
        super().__init__()
        self.area = area

    def forward(self, invar, outvar):
        loss = CustomLossFunction.apply(invar, outvar, self.area)
        return loss
