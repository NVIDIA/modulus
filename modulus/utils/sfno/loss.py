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

import torch
from torch import nn
import torch.nn.functional as F

from modulus.utils.sfno.distributed import comm
from modulus.utils.sfno.distributed.mappings import gather_from_parallel_region

import torch_harmonics as harmonics


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

        channel_weights = torch.ones(params.N_out_channels).float()

        # if loss_type[:20] == 'dynamically weighted':
        #     self.dynamic_weight_obj = GeometricLpLoss(self.img_shape, p=2)
        #     loss_type = loss_type[21:]

        if loss_type[:11] == "pole-masked":
            pole_mask = 1
            loss_type = loss_type[12:]
        else:
            pole_mask = 0

        if loss_type[:8] == "weighted":
            channel_weights = torch.Tensor(params.channel_weights).float()
            channel_weights = (
                params.N_out_channels * channel_weights / torch.sum(channel_weights)
            )
            loss_type = loss_type[9:]

        if loss_type[:8] == "absolute":
            absolute = True
            loss_type = loss_type[9:]
        else:
            absolute = False

        if self.loss_type[:7] == "squared":
            squared = True
            loss_type = loss_type[8:]
        else:
            squared = False

        # TODO: clean this up and replace it with string parsing to set the parameters
        if loss_type == "l2":
            self.loss_obj = (
                DistributedLpLoss(d=d, p=2)
                if params.split_data_channels
                else LpLoss(d=d, p=2)
            )
        elif loss_type == "l1":
            self.loss_obj = (
                DistributedLpLoss(d=d, p=1)
                if params.split_data_channels
                else LpLoss(d=d, p=1)
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
        elif loss_type == "spectral":
            self.loss_obj = SpectralLoss(
                self.img_shape, absolute=absolute, squared=squared
            )
        else:
            raise ValueError(f"Unknown loss function: {loss_type}")

        # weighting factor for the case of multistep training
        # TODO change hardcoded weighting
        multistep_weight = torch.arange(1, self.n_future + 2, dtype=torch.float32)
        multistep_weight = multistep_weight / torch.sum(multistep_weight)
        multistep_weight = multistep_weight.reshape(1, -1, 1, 1, 1)

        self.register_buffer("multistep_weight", multistep_weight)

        # renormalize the weights to one
        channel_weights = channel_weights.reshape(1, -1, 1, 1)
        self.register_buffer("channel_weights", channel_weights)

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

        chw = self.channel_weights

        if self.training:
            B, C, H, W = prd.shape

            # adding weighting factors
            prd = (
                prd.reshape(B, -1, C // (self.n_future + 1), H, W)
                * self.multistep_weight
            )
            tar = (
                tar.reshape(B, -1, C // (self.n_future + 1), H, W)
                * self.multistep_weight
            )

            # compute the dynamic weighting factor which depends on the variability of the channels
            if hasattr(self, "dynamic_weight_obj"):
                dynamic_weight = self.dynamic_weight_obj(tar[:, -1], inp[:])
                dynamic_weight = dynamic_weight / torch.sum(
                    dynamic_weight, dim=-1, keepdim=True
                )
                prd = prd * dynamic_weight
                tar = tar * dynamic_weight

            prd = prd.reshape(B, C, H, W)
            tar = tar.reshape(B, C, H, W)

            chw = chw.repeat(1, self.n_future + 1, 1, 1)

        return self.loss_obj(prd, tar, chw)


# double check if polar optimization has an effect - we use 5 here by default
class GeometricLpLoss(nn.Module):
    """Geometric Lp loss"""

    def __init__(
        self,
        img_size: Tuple[int, int],
        p: Optional[float] = 2.0,
        size_average: Optional[bool] = True,
        reduction: Optional[bool] = True,
        absolute: Optional[bool] = False,
        squared: Optional[bool] = False,
        pole_mask: Optional[int] = 0,
    ):  # pragma: no cover
        super(GeometricLpLoss, self).__init__()

        self.d = 2
        self.p = p

        self.img_size = img_size
        self.reduction = reduction
        self.size_average = size_average
        self.absolute = absolute
        self.squared = squared
        self.pole_mask = pole_mask

        jacobian = torch.sin(torch.linspace(0, torch.pi, self.img_size[0])).unsqueeze(1)
        dtheta = torch.pi / self.img_size[0]
        dlambda = 2 * torch.pi / self.img_size[1]

        self.register_buffer("jacobian", jacobian)
        self.dA = dlambda * dtheta

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
                * self.jacobian[..., self.pole_mask : -self.pole_mask, :]
                * self.dA,
                dim=(-2, -1),
            )
        else:
            all_norms = torch.sum(
                torch.abs(prd - tar) ** self.p * self.jacobian * self.dA, dim=(-2, -1)
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
                * self.jacobian[..., self.pole_mask : -self.pole_mask, :]
                * self.dA,
                dim=(-2, -1),
            )
        else:
            diff_norms = torch.sum(
                torch.abs(prd - tar) ** self.p * self.jacobian * self.dA, dim=(-2, -1)
            )

        diff_norms = diff_norms.reshape(num_examples, -1)

        tar_norms = torch.sum(
            torch.abs(tar) ** self.p * self.jacobian * self.dA, dim=(-2, -1)
        )
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


class LpLoss(nn.Module):
    """Lp loss"""

    def __init__(
        self,
        d: Optional[float] = 2.0,
        p: Optional[float] = 2.0,
        size_average: Optional[bool] = True,
        reduction: Optional[bool] = True,
    ):  # pragma: no cover
        super(LpLoss, self).__init__()

        # Dimension and Lp-norm type are postive
        assert d > 0.0 and p > 0.0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(
        self, prd: torch.Tensor, tar: torch.Tensor, chw: torch.Tensor
    ):  # pragma: no cover
        """Computes the absolute loss"""
        num_examples = prd.size()[0]

        # Assume uniform mesh
        h = 1.0 / (prd.size()[1] - 1.0)

        prdv = prd.reshape(num_examples, -1)
        tarv = y.reshape(num_examples, -1)

        # all_norms = (h**(self.d/self.p)) * torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)
        all_norms = (h ** (self.d / self.p)) * torch.linalg.norm(
            prdv - tarv, ord=self.p, dim=1
        )

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

        prdv = prd.reshape(num_examples, -1)
        tarv = tar.reshape(num_examples, -1)

        # diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        # y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        diff_norms = torch.linalg.norm(prdv - tarv, ord=self.p, dim=1)
        tar_norms = torch.linalg.norm(tarv, ord=self.p, dim=1)

        # setup return value
        retval = chw.reshape(1, -1) * diff_norms / tar_norms
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
        return self.rel(prd, tar, chw, mask)


# class DistributedLpLoss(nn.Module):
#     def __init__(self, d: Optional[float]=2., p: Optional[float]=2., size_average: Optional[bool]=True, reduction: Optional[bool]=True):
#         super(DistributedLpLoss, self).__init__()

#         # get matmul parallel size
#         self.matmul_comm_size = comm.get_size("matmul")

#         #Dimension and Lp-norm type are postive
#         assert d > 0. and p == 2.

#         self.d = d
#         self.p = p
#         self.reduction = reduction
#         self.size_average = size_average

#     def rel(self, prd: torch.Tensor, tar: torch.Tensor, mask: Optional[torch.Tensor] = None):
#         num_examples = prd.size()[0]

#         prdv = prd.reshape(num_examples, -1)
#         tarv = tar.reshape(num_examples, -1)

#         diff_norms = torch.sum(torch.square(prdv-tarv), dim=1)
#         tar_norms = torch.sum(torch.square(tarv), dim=1)
#         combined = torch.stack([diff_norms, tar_norms], dim=1)

#         # reduce:
#         combined = reduce_from_matmul_parallel_region(combined)
#         norm = torch.sqrt(combined[:, 0] / combined[:, 1])

#         if mask is not None:
#             norm *= mask

#         if self.reduction:
#             if self.size_average:
#                 if mask is None:
#                     norm = torch.mean(norm)
#                 else:
#                     norm = torch.sum(norm) / torch.sum(mask)
#             else:
#                 return torch.sum(norm)

#         return norm

#     def forward(self, prd: torch.Tensor, tar: torch.Tensor, mask: Optional[torch.Tensor] = None):
#         return self.rel(prd, tar, mask)

# double check if polar optimization has an effect - we use 5 here by default
class GeometricH1Loss(nn.Module):
    """Geometric H1 loss"""

    def __init__(
        self,
        img_size: Tuple[int, int],
        p: Optional[float] = 2.0,
        size_average: Optional[bool] = True,
        reduction: Optional[bool] = True,
        absolute: Optional[bool] = False,
        squared: Optional[bool] = False,
        alpha: Optional[float] = 0.9,
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


# double check if polar optimization has an effect - we use 5 here by default
class SpectralLoss(nn.Module):
    """Spectral loss"""

    def __init__(
        self,
        img_size: Tuple[int, int],
        p: Optional[float] = 2.0,
        size_average: Optional[bool] = True,
        reduction: Optional[bool] = True,
        absolute: Optional[bool] = False,
        squared: Optional[bool] = False,
    ):  # pragma: no cover
        super(SpectralLoss, self).__init__()

        self.reduction = reduction
        self.size_average = size_average
        self.absolute = absolute
        self.squared = squared

        self.sht = harmonics.RealSHT(*img_size, grid="equiangular").float()
        spectral_weights = torch.arange(self.sht.lmax).float()
        spectral_weights = spectral_weights + 1
        self.register_buffer("spectral_weights", spectral_weights)

    def abs(self, prd: torch.Tensor, tar: torch.Tensor):  # pragma: no cover
        """Computes the absolute loss"""
        num_examples = prd.size()[0]

        # compute coefficients
        coeffs = torch.view_as_real(self.sht(prd - tar))
        coeffs = coeffs[..., 0] ** 2 + coeffs[..., 1] ** 2
        norm2 = coeffs[..., :, 0] + 2 * torch.sum(coeffs[..., :, 1:], dim=-1)
        norm2 = (self.spectral_weights * norm2).reshape(num_examples, -1).sum(dim=-1)

        if not self.squared:
            norm2 = torch.sqrt(norm2)

        all_norms = norm2

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

        # compute coefficients
        coeffs = torch.view_as_real(self.sht(prd - tar))
        coeffs = coeffs[..., 0] ** 2 + coeffs[..., 1] ** 2
        norm2 = coeffs[..., :, 0] + 2 * torch.sum(coeffs[..., :, 1:], dim=-1)
        norm2 = (self.spectral_weights * norm2).reshape(num_examples, -1).sum(dim=-1)

        # compute coefficients
        tar_coeffs = torch.view_as_real(self.sht(tar))
        tar_coeffs = tar_coeffs[..., 0] ** 2 + tar_coeffs[..., 1] ** 2
        tar_norm2 = tar_coeffs[..., :, 0] + 2 * torch.sum(
            tar_coeffs[..., :, 1:], dim=-1
        )
        tar_norm2 = (
            (self.spectral_weights * tar_norm2).reshape(num_examples, -1).sum(dim=-1)
        )

        retval = tar_norm2 / norm2

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

        if not self.squared:
            retval

        return retval

    def forward(
        self, prd: torch.Tensor, tar: torch.Tensor, mask: Optional[torch.Tensor] = None
    ):  # pragma: no cover
        if self.absolute:
            loss = self.abs(prd, tar)
        else:
            loss = self.rel(prd, tar, mask)

        return loss
