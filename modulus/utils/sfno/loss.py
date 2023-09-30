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

from typing import Optional, Tuple

import numpy as np
import torch
import torch_harmonics as harmonics
from torch import nn
from torch_harmonics.quadrature import clenshaw_curtiss_weights

from modulus.utils.sfno.distributed import comm
from modulus.utils.sfno.distributed.mappings import gather_from_parallel_region


class LossHandler(nn.Module):
    """
    Wrapper class that will handle computing losses.
    """

    def __init__(self, params, img_size=(720, 1440), d=2):  # pragma: no cover

        super(LossHandler, self).__init__()

        self.rank = comm.get_rank("matmul")
        self.n_future = params.n_future

        # TODO: allow for crop offset, otherwise the weighting will not be correct
        self.img_shape = (params.img_crop_shape_x, params.img_crop_shape_y)

        loss_type = self.loss_type = params.loss

        if loss_type[:11] == "pole-masked":
            pole_mask = 1
            loss_type = loss_type[12:]
        else:
            pole_mask = 0

        if loss_type[:8] == "weighted":
            if params.channel_weights == "auto":
                channel_weights = torch.ones(params.N_out_channels, dtype=torch.float32)
                for c, chn in enumerate(params.channel_names):
                    if chn in ["u10m", "v10m", "u100m", "v100m", "sp", "msl", "tcwv"]:
                        channel_weights[c] = 0.1
                    elif chn in ["t2m", "2d"]:
                        channel_weights[c] = 1.0
                    elif chn[0] in ["z", "u", "v", "t", "r", "q"]:
                        pressure_level = float(chn[1:])
                        channel_weights[c] = 0.001 * pressure_level
                    else:
                        channel_weights[c] = 0.01
            else:
                channel_weights = torch.Tensor(params.channel_weights).float()

            loss_type = loss_type[9:]
        else:
            channel_weights = torch.ones(params.N_out_channels, dtype=torch.float32)

        # renormalize the weights to one
        channel_weights = channel_weights.reshape(1, -1, 1, 1)
        channel_weights = channel_weights / torch.sum(channel_weights)

        if loss_type[:8] == "absolute":
            absolute = True
            loss_type = loss_type[9:]
        else:
            absolute = False

        if loss_type[:7] == "squared":
            squared = True
            loss_type = loss_type[8:]
        else:
            squared = False

        if loss_type[:8] == "temp-std":
            eps = 1e-6
            global_stds = torch.from_numpy(np.load(params.global_stds_path)).reshape(
                1, -1, 1, 1
            )[:, params.in_channels]
            time_diff_stds = torch.from_numpy(
                np.load(params.time_diff_stds_path)
            ).reshape(1, -1, 1, 1)[:, params.in_channels]
            time_var_weights = global_stds / (time_diff_stds + eps)
            # time_var_weights = 1 / (time_diff_stds+eps)
            if squared:
                time_var_weights = time_var_weights**2
            channel_weights = channel_weights * time_var_weights
            loss_type = loss_type[9:]

        self.register_buffer("channel_weights", channel_weights)

        # TODO: clean this up and replace it with string parsing to set the parameters
        if loss_type == "l2":
            self.loss_obj = GeometricLpLoss(
                self.img_shape,
                p=2,
                absolute=absolute,
                pole_mask=pole_mask,
                jacobian="flat",
            )
        elif loss_type == "l1":
            self.loss_obj = GeometricLpLoss(
                self.img_shape,
                p=1,
                absolute=absolute,
                pole_mask=pole_mask,
                jacobian="flat",
            )
        elif loss_type == "geometric l2":
            self.loss_obj = GeometricLpLoss(
                self.img_shape,
                p=2,
                absolute=absolute,
                squared=squared,
                pole_mask=pole_mask,
            )
        elif loss_type == "geometric l1":
            self.loss_obj = GeometricLpLoss(
                self.img_shape, p=1, absolute=absolute, pole_mask=pole_mask
            )
        elif loss_type == "geometric h1":
            self.loss_obj = GeometricH1Loss(
                self.img_shape, absolute=absolute, squared=squared
            )
        else:
            raise ValueError(f"Unknown loss function: {loss_type}")

        # weighting factor for the case of multistep training
        # TODO change hardcoded weighting
        multistep_weight = torch.arange(1, self.n_future + 2, dtype=torch.float32)
        multistep_weight = multistep_weight / torch.sum(multistep_weight)
        multistep_weight = multistep_weight.reshape(-1, 1, 1, 1)

        self.register_buffer("multistep_weight", multistep_weight)

        # # decide whether to gather the input
        self.do_gather_input = False
        if comm.get_size("h") * comm.get_size("w") > 1:
            self.do_gather_input = True

    @torch.jit.ignore
    def _gather_input(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        # combine data
        # h
        xh = gather_from_parallel_region(x, -2, "h")
        xw = gather_from_parallel_region(xh, -1, "w")

        # crop
        x = xw[:, :, : self.img_shape[0], : self.img_shape[1]]

        return x

    def is_distributed(self):  # pragma: no cover
        """Returns whether the loss is distributed or not (always False)"""
        return False

    def forward(
        self, prd: torch.Tensor, tar: torch.Tensor, inp: torch.Tensor
    ):  # pragma: no cover

        if self.do_gather_input:
            prd = self._gather_input(prd)
            tar = self._gather_input(tar)

        if hasattr(self, "minmax"):
            chw = torch.ones_like(self.channel_weights)
            chw = chw / torch.sum(chw)
            chw += self.channel_weights.abs() / torch.sum(self.channel_weights.abs())
        else:
            chw = self.channel_weights

        if self.training:
            chw = (chw * self.multistep_weight).reshape(1, -1, 1, 1)
        else:
            chw = chw

        return self.loss_obj(prd, tar, chw)


# double check if polar optimization has an effect - we use 5 here by default
class GeometricLpLoss(nn.Module):
    """Geometric Lp loss"""

    def __init__(
        self,
        img_size: Tuple[int, int],
        p: Optional[float] = 2.0,
        size_average: Optional[bool] = False,
        reduction: Optional[bool] = True,
        absolute: Optional[bool] = False,
        squared: Optional[bool] = False,
        pole_mask: Optional[int] = 0,
        jacobian: Optional[str] = "s2",
        quadrature_rule: Optional[str] = "naive",
    ):  # pragma: no cover
        super(GeometricLpLoss, self).__init__()

        self.p = p
        self.img_size = img_size
        self.reduction = reduction
        self.size_average = size_average
        self.absolute = absolute
        self.squared = squared
        self.pole_mask = pole_mask

        if jacobian == "s2":
            jacobian = torch.sin(
                torch.linspace(0, torch.pi, self.img_size[0])
            ).unsqueeze(1)
        else:
            jacobian = torch.ones(self.img_size[0], 1)

        if quadrature_rule == "naive":
            dtheta = torch.pi / self.img_size[0]
            dlambda = 2 * torch.pi / self.img_size[1]
            dA = dlambda * dtheta
            quad_weight = dA * jacobian
        elif quadrature_rule == "clenshaw-curtiss":
            cost, w = clenshaw_curtiss_weights(self.img_size[0], -1, 1)
            dlambda = 2 * torch.pi / self.img_size[1]
            quad_weight = dlambda * torch.from_numpy(w).unsqueeze(-1)
        else:
            raise ValueError(f"Unknown quadrature rule {quadrature_rule}")

        self.register_buffer("quad_weight", quad_weight)

    def abs(
        self, prd: torch.Tensor, tar: torch.Tensor, chw: torch.Tensor
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
        all_norms = chw.reshape(1, -1) * all_norms

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
        chw: torch.Tensor,
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
        retval = chw.reshape(1, -1) * (diff_norms / tar_norms)
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
        chw: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):  # pragma: no cover
        if self.absolute:
            loss = self.abs(prd, tar, chw)
        else:
            loss = self.rel(prd, tar, chw, mask)

        return loss


# double check if polar optimization has an effect - we use 5 here by default
class GeometricH1Loss(nn.Module):
    """Geometric H1 loss"""

    def __init__(
        self,
        img_size: Tuple[int, int],
        p: Optional[float] = 2.0,
        size_average: Optional[bool] = False,
        reduction: Optional[bool] = True,
        absolute: Optional[bool] = False,
        squared: Optional[bool] = False,
        alpha: Optional[float] = 0.5,
    ):  # pragma: no cover
        super(GeometricH1Loss, self).__init__()

        self.reduction = reduction
        self.size_average = size_average
        self.absolute = absolute
        self.squared = squared
        self.alpha = alpha

        self.sht = harmonics.RealSHT(*img_size, grid="equiangular").float()
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
