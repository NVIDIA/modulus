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

from typing import Optional, List

import torch
from torch import nn

import torch_harmonics as harmonics
from torch_harmonics.quadrature import clenshaw_curtiss_weights


# TODO (era5_wind) double check if polar optimization has an effect - we use 5 here by default
class GeometricLpLoss(nn.Module):
    """Geometric Lp loss"""

    def __init__(
        self,
        inp_size: List[int],
        p: float = 2.0,
        size_average: bool = False,
        reduction: bool = True,
        absolute: bool = False,
        squared: bool = False,
        pole_mask: int = 0,
        jacobian: str = "s2",
        quadrature_rule: str = "naive",
    ):  # pragma: no cover
        super(GeometricLpLoss, self).__init__()

        self.p = p
        self.inp_size = inp_size
        self.reduction = reduction
        self.size_average = size_average
        self.absolute = absolute
        self.squared = squared
        self.pole_mask = pole_mask

        if jacobian == "s2":
            jacobian = torch.sin(
                torch.linspace(0, torch.pi, self.inp_size[0])
            ).unsqueeze(1)
        else:
            jacobian = torch.ones(self.inp_size[0], 1)

        if quadrature_rule == "naive":
            dtheta = torch.pi / self.inp_size[0]
            dlambda = 2 * torch.pi / self.inp_size[1]
            dA = dlambda * dtheta
            quad_weight = dA * jacobian
        elif quadrature_rule == "clenshaw-curtiss":
            _, w = clenshaw_curtiss_weights(self.inp_size[0], -1, 1)
            dlambda = 2 * torch.pi / self.inp_size[1]
            quad_weight = dlambda * torch.from_numpy(w).unsqueeze(-1)
        else:
            raise ValueError(f"Unknown quadrature rule {quadrature_rule}")

        self.register_buffer(
            "quad_weight", quad_weight
        )  # TODO (mnabian): check if this works with model packaging

    def abs(
        self, prd: torch.Tensor, tar: torch.Tensor, channel_weights: torch.Tensor
    ):  # pragma: no cover
        """Computes the absolute loss"""
        num_examples = prd.size()[0]
        if self.pole_mask:
            all_norms = torch.sum(
                torch.abs(
                    prd[..., self.pole_mask : -self.pole_mask, :]
                    - tar[..., self.pole_mask : -self.pole_mask, :]
                )
                ** self.p
                * self.quad_weight[..., self.pole_mask : -self.pole_mask, :],
                dim=(-2, -1),
            )
        else:
            all_norms = torch.sum(
                torch.abs(prd - tar) ** self.p * self.quad_weight,
                dim=(-2, -1),
            )

        all_norms = all_norms.reshape(num_examples, -1).sum()

        if not self.squared:
            all_norms = all_norms ** (1 / self.p)

        # apply channel weighting
        all_norms = channel_weights.reshape(1, -1) * all_norms

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(
        self,
        prd: torch.Tensor,
        tar: torch.Tensor,
        channel_weights: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):  # pragma: no cover
        """Computes the relative loss"""
        num_examples = prd.size()[0]

        if self.pole_mask:
            diff_norms = torch.sum(
                torch.abs(
                    prd[..., self.pole_mask : -self.pole_mask, :]
                    - tar[..., self.pole_mask : -self.pole_mask, :]
                )
                ** self.p
                * self.quad_weight[..., self.pole_mask : -self.pole_mask, :],
                dim=(-2, -1),
            )
        else:
            diff_norms = torch.sum(
                torch.abs(prd - tar) ** self.p * self.quad_weight, dim=(-2, -1)
            )

        diff_norms = diff_norms.reshape(num_examples, -1)

        tar_norms = torch.sum(torch.abs(tar) ** self.p * self.quad_weight, dim=(-2, -1))
        tar_norms = tar_norms.reshape(num_examples, -1)

        if not self.squared:
            diff_norms = diff_norms ** (1 / self.p)
            tar_norms = tar_norms ** (1 / self.p)

        # setup return value
        retval = channel_weights.reshape(1, -1) * (diff_norms / tar_norms)
        if mask is not None:
            retval = retval * mask

        if self.reduction:
            if self.size_average:
                if mask is None:
                    retval = torch.mean(retval)
                else:
                    retval = torch.sum(retval) / torch.sum(mask)
            else:
                retval = torch.sum(retval)

        return retval

    def forward(
        self,
        prd: torch.Tensor,
        tar: torch.Tensor,
        channel_weights: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):  # pragma: no cover
        if self.absolute:
            loss = self.abs(prd, tar, channel_weights)
        else:
            loss = self.rel(prd, tar, channel_weights, mask)

        return loss


# double check if polar optimization has an effect - we use 5 here by default
class GeometricH1Loss(nn.Module):
    """Geometric H1 loss"""

    def __init__(
        self,
        inp_size: List[int],
        size_average: bool = False,
        reduction: bool = True,
        absolute: bool = False,
        squared: bool = False,
        alpha: float = 0.5,
    ):  # pragma: no cover
        super(GeometricH1Loss, self).__init__()

        self.reduction = reduction
        self.size_average = size_average
        self.absolute = absolute
        self.squared = squared
        self.alpha = alpha

        self.sht = harmonics.RealSHT(*inp_size, grid="equiangular").float()
        h1_weights = torch.arange(self.sht.lmax).float()
        h1_weights = h1_weights * (h1_weights + 1)
        self.register_buffer("h1_weights", h1_weights)

    def abs(self, prd: torch.Tensor, tar: torch.Tensor):  # pragma: no cover
        """Computes the absolute loss"""
        num_examples = prd.size()[0]

        coeffs = torch.view_as_real(self.sht(prd - tar))
        coeffs = coeffs[..., 0] ** 2 + coeffs[..., 1] ** 2
        norm2 = coeffs[..., :, 0] + 2 * torch.sum(coeffs[..., :, 1:], dim=-1)
        l2_norm2 = norm2.reshape(num_examples, -1).sum(dim=-1)
        h1_norm2 = (norm2 * self.h1_weights).reshape(num_examples, -1).sum(dim=-1)

        if not self.squared:
            all_norms = self.alpha * torch.sqrt(l2_norm2) + (
                1 - self.alpha
            ) * torch.sqrt(h1_norm2)
        else:
            all_norms = self.alpha * l2_norm2 + (1 - self.alpha) * h1_norm2

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(
        self, prd: torch.Tensor, tar: torch.Tensor, mask: Optional[torch.Tensor] = None
    ):  # pragma: no cover
        """Computes the relative loss"""
        num_examples = prd.size()[0]

        coeffs = torch.view_as_real(self.sht(prd - tar))
        coeffs = coeffs[..., 0] ** 2 + coeffs[..., 1] ** 2
        norm2 = coeffs[..., :, 0] + 2 * torch.sum(coeffs[..., :, 1:], dim=-1)
        l2_norm2 = norm2.reshape(num_examples, -1).sum(dim=-1)
        h1_norm2 = (norm2 * self.h1_weights).reshape(num_examples, -1).sum(dim=-1)

        tar_coeffs = torch.view_as_real(self.sht(tar))
        tar_coeffs = tar_coeffs[..., 0] ** 2 + tar_coeffs[..., 1] ** 2
        tar_norm2 = tar_coeffs[..., :, 0] + 2 * torch.sum(
            tar_coeffs[..., :, 1:], dim=-1
        )
        tar_l2_norm2 = tar_norm2.reshape(num_examples, -1).sum(dim=-1)
        tar_h1_norm2 = (
            (tar_norm2 * self.h1_weights).reshape(num_examples, -1).sum(dim=-1)
        )

        if not self.squared:
            diff_norms = self.alpha * torch.sqrt(l2_norm2) + (
                1 - self.alpha
            ) * torch.sqrt(h1_norm2)
            tar_norms = self.alpha * torch.sqrt(tar_l2_norm2) + (
                1 - self.alpha
            ) * torch.sqrt(tar_h1_norm2)
        else:
            diff_norms = self.alpha * l2_norm2 + (1 - self.alpha) * h1_norm2
            tar_norms = self.alpha * tar_l2_norm2 + (1 - self.alpha) * tar_h1_norm2

        # setup return value
        retval = diff_norms / tar_norms
        if mask is not None:
            retval = retval * mask

        if self.reduction:
            if self.size_average:
                if mask is None:
                    retval = torch.mean(retval)
                else:
                    retval = torch.sum(retval) / torch.sum(mask)
            else:
                retval = torch.sum(retval)

        return retval

    def forward(
        self, prd: torch.Tensor, tar: torch.Tensor, mask: Optional[torch.Tensor] = None
    ):  # pragma: no cover
        if self.absolute:
            loss = self.abs(prd, tar)
        else:
            loss = self.rel(prd, tar, mask)

        return loss
