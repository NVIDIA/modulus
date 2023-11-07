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

# import FactorizedTensor from tensorly for tensorized operations
import tensorly as tl
import torch
import torch.nn as nn
import torch_harmonics.distributed as thd
from tltorch.factorized_tensors.core import FactorizedTensor
from torch.cuda import amp

# import convenience functions for factorized tensors
from modulus.models.sfno.activations import ComplexReLU

# for the experimental module
from modulus.models.sfno.contractions import (
    _contract_rank,
    compl_exp_mul2d_fwd,
    compl_exp_muladd2d_fwd,
    compl_mul2d_fwd,
    compl_muladd2d_fwd,
)
from modulus.models.sfno.factorizations import get_contract_fun

tl.set_backend("pytorch")


class SpectralConv(nn.Module):  # pragma: no cover
    """
    Spectral Convolution according to Driscoll & Healy. Designed for convolutions on
    the two-sphere S2 using the Spherical Harmonic Transforms in torch-harmonics, but
    supports convolutions on the periodic domain via the RealFFT2 and InverseRealFFT2
    wrappers.
    """

    def __init__(
        self,
        forward_transform,
        inverse_transform,
        in_channels,
        out_channels,
        operator_type="diagonal",
        separable=False,
        bias=False,
    ):  # pragma: no cover
        super(SpectralConv, self).__init__()

        self.forward_transform = forward_transform
        self.inverse_transform = inverse_transform

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.modes_lat = self.inverse_transform.lmax
        self.modes_lon = self.inverse_transform.mmax

        self.scale_residual = (
            self.forward_transform.nlat != self.inverse_transform.nlat
        ) or (self.forward_transform.nlon != self.inverse_transform.nlon)
        if hasattr(self.forward_transform, "grid"):
            self.scale_residual = self.scale_residual or (
                self.forward_transform.grid != self.inverse_transform.grid
            )

        # remember factorization details
        self.operator_type = operator_type
        self.separable = separable

        if self.inverse_transform.lmax != self.modes_lat:
            raise AssertionError
        if self.inverse_transform.mmax != self.modes_lon:
            raise AssertionError

        weight_shape = [in_channels]

        if not self.separable:
            weight_shape += [out_channels]

        if isinstance(self.inverse_transform, thd.DistributedInverseRealSHT):
            self.modes_lat_local = self.inverse_transform.lmax_local
            self.modes_lon_local = self.inverse_transform.mmax_local
            self.lpad_local = self.inverse_transform.lpad_local
            self.mpad_local = self.inverse_transform.mpad_local
            self.nlat_local = self.inverse_transform.nlat_local
            self.nlon_local = self.inverse_transform.nlon_local
        else:
            self.modes_lat_local = self.modes_lat
            self.modes_lon_local = self.modes_lon
            self.lpad = 0
            self.mpad = 0
            self.nlat_local = self.inverse_transform.nlat
            self.nlon_local = self.inverse_transform.nlon

        # unpadded weights
        if self.operator_type == "diagonal":
            weight_shape += [self.modes_lat_local, self.modes_lon_local]
        elif self.operator_type == "dhconv":
            weight_shape += [self.modes_lat_local]
        else:
            raise ValueError(f"Unsupported operator type f{self.operator_type}")

        scale = 2 / (in_channels + out_channels)
        init = scale * torch.randn(*weight_shape, 2)
        self.weight = nn.Parameter(init)

        if self.operator_type == "dhconv":
            self.weight.is_shared_mp = ["matmul", "w"]
            self.weight.sharded_dims_mp = [None for _ in weight_shape]
            self.weight.sharded_dims_mp[-1] = "h"
        else:
            self.weight.is_shared_mp = ["matmul"]
            self.weight.sharded_dims_mp = [None for _ in weight_shape]
            self.weight.sharded_dims_mp[-1] = "w"
            self.weight.sharded_dims_mp[-2] = "h"

        # get the contraction handle. This should return a pyTorch contraction
        self._contract = get_contract_fun(
            self.weight,
            implementation="factorized",
            separable=separable,
            complex=True,
            operator_type=operator_type,
        )

        if bias == "constant":
            self.bias = nn.Parameter(torch.zeros(1, self.out_channels, 1, 1))
        elif bias == "position":
            self.bias = nn.Parameter(
                torch.zeros(1, self.out_channels, self.nlat_local, self.nlon_local)
            )
            self.bias.is_shared_mp = ["matmul"]
            self.bias.sharded_dims_mp = [None, None, "h", "w"]

    def _init_weights(self):  # pragma: no cover
        scale = (
            2 / self.weight.shape[0]
        ) ** 0.5  # only for Relu activation after the convolution
        nn.init.normal_(self.weight, std=scale)
        if hasattr(self, "bias"):
            nn.init.zeros_(self.bias)

    def forward(self, x):  # pragma: no cover
        dtype = x.dtype
        residual = x
        x = x.float()
        B, C, H, W = x.shape

        with amp.autocast(enabled=False):
            x = self.forward_transform(x)
            if self.scale_residual:
                x = x.contiguous()
                residual = self.inverse_transform(x)
                residual = residual.to(dtype)

        # approach with unpadded weights
        xp = torch.zeros(
            x.shape[0],
            self.out_channels,
            self.modes_lat_local,
            self.modes_lon_local,
            dtype=x.dtype,
            device=x.device,
        )
        xp[..., : self.modes_lat_local, : self.modes_lon_local] = self._contract(
            x[..., : self.modes_lat_local, : self.modes_lon_local],
            self.weight,
            separable=self.separable,
            operator_type=self.operator_type,
        )
        x = xp.contiguous()

        with amp.autocast(enabled=False):
            x = self.inverse_transform(x)

        if hasattr(self, "bias"):
            x = x + self.bias

        x = x.type(dtype)

        return x, residual


class FactorizedSpectralConv(nn.Module):  # pragma: no cover
    """
    Factorized version of SpectralConv. Uses tensorly-torch to keep the weights factorized
    """

    def __init__(
        self,
        forward_transform,
        inverse_transform,
        in_channels,
        out_channels,
        operator_type="diagonal",
        rank=0.2,
        factorization=None,
        separable=False,
        decomposition_kwargs=dict(),
        bias=False,
    ):  # pragma: no cover
        super(FactorizedSpectralConv, self).__init__()

        self.forward_transform = forward_transform
        self.inverse_transform = inverse_transform

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.modes_lat = self.inverse_transform.lmax
        self.modes_lon = self.inverse_transform.mmax

        self.scale_residual = (
            self.forward_transform.nlat != self.inverse_transform.nlat
        ) or (self.forward_transform.nlon != self.inverse_transform.nlon)
        if hasattr(self.forward_transform, "grid"):
            self.scale_residual = self.scale_residual or (
                self.forward_transform.grid != self.inverse_transform.grid
            )

        # Make sure we are using a Complex Factorized Tensor
        if factorization is None:
            factorization = "ComplexDense"  # No factorization
        complex_weight = factorization[:7].lower() == "complex"

        # remember factorization details
        self.operator_type = operator_type
        self.rank = rank
        self.factorization = factorization
        self.separable = separable

        if self.inverse_transform.lmax != self.modes_lat:
            raise AssertionError
        if self.inverse_transform.mmax != self.modes_lon:
            raise AssertionError

        weight_shape = [in_channels]

        if not self.separable:
            weight_shape += [out_channels]

        if isinstance(self.inverse_transform, thd.DistributedInverseRealSHT):
            self.modes_lat_local = self.inverse_transform.lmax_local
            self.modes_lon_local = self.inverse_transform.mmax_local
            self.lpad_local = self.inverse_transform.lpad_local
            self.mpad_local = self.inverse_transform.mpad_local
        else:
            self.modes_lat_local = self.modes_lat
            self.modes_lon_local = self.modes_lon
            self.lpad = 0
            self.mpad = 0

        # unpadded weights
        if self.operator_type == "diagonal":
            weight_shape += [self.modes_lat_local, self.modes_lon_local]
        elif self.operator_type == "dhconv":
            weight_shape += [self.modes_lat_local]
        elif self.operator_type == "rank":
            weight_shape += [self.rank]
        else:
            raise ValueError(f"Unsupported operator type f{self.operator_type}")

        # form weight tensors
        self.weight = FactorizedTensor.new(
            weight_shape,
            rank=self.rank,
            factorization=factorization,
            fixed_rank_modes=False,
            **decomposition_kwargs,
        )
        # initialization of weights
        scale = (2 / weight_shape[0]) ** 0.5
        self.weight.normal_(0, scale)

        # get the contraction handle
        if operator_type == "rank":
            self._contract = _contract_rank
        else:
            self._contract = get_contract_fun(
                self.weight,
                implementation="reconstructed",
                separable=separable,
                complex=complex_weight,
                operator_type=operator_type,
            )

        if bias == "constant":
            self.bias = nn.Parameter(torch.zeros(1, self.out_channels, 1, 1))
        elif bias == "position":
            self.bias = nn.Parameter(
                torch.zeros(1, self.out_channels, self.nlat_local, self.nlon_local)
            )
            self.bias.is_shared_mp = ["matmul"]
            self.bias.sharded_dims_mp = [None, None, "h", "w"]

    def _init_weights(self):  # pragma: no cover
        scale = (2 / self.weight.shape[0]) ** 0.5  # only for Relu activation
        self.weight.normal_(0, scale)
        if hasattr(self, "bias"):
            nn.init.zeros_(self.bias)

    def forward(self, x):  # pragma: no cover
        dtype = x.dtype
        residual = x
        x = x.float()

        with amp.autocast(enabled=False):
            x = self.forward_transform(x)
            if self.scale_residual:
                x = x.contiguous()
                residual = self.inverse_transform(x)
                residual = residual.to(dtype)

        # approach with unpadded weights
        xp = torch.zeros(
            x.shape[0],
            self.out_channels,
            self.modes_lat_local,
            self.modes_lon_local,
            dtype=x.dtype,
            device=x.device,
        )

        if self.operator_type == "rank":
            xp[..., : self.modes_lat_local, : self.modes_lon_local] = self._contract(
                x[..., : self.modes_lat_local, : self.modes_lon_local],
                self.weight,
                self.lat_weight,
                self.lon_weight,
            )
        else:
            xp[..., : self.modes_lat_local, : self.modes_lon_local] = self._contract(
                x[..., : self.modes_lat_local, : self.modes_lon_local],
                self.weight,
                separable=self.separable,
                operator_type=self.operator_type,
            )
        x = xp.contiguous()

        with amp.autocast(enabled=False):
            x = self.inverse_transform(x)

        if hasattr(self, "bias"):
            x = x + self.bias

        x = x.type(dtype)

        return x, residual


class SpectralAttention(nn.Module):  # pragma: no cover
    """
    Spherical non-linear FNO layer
    """

    def __init__(
        self,
        forward_transform,
        inverse_transform,
        in_channels,
        out_channels,
        operator_type="diagonal",
        sparsity_threshold=0.0,
        hidden_size_factor=2,
        complex_activation="real",
        bias=False,
        spectral_layers=1,
        drop_rate=0.0,
    ):  # pragma: no cover
        super(SpectralAttention, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sparsity_threshold = sparsity_threshold
        self.operator_type = operator_type
        self.spectral_layers = spectral_layers

        self.scale = (2 / in_channels) ** 0.5

        self.modes_lat = forward_transform.lmax
        self.modes_lon = forward_transform.mmax

        # only storing the forward handle to be able to call it
        self.forward_transform = forward_transform
        self.inverse_transform = inverse_transform

        self.scale_residual = (
            (self.forward_transform.nlat != self.inverse_transform.nlat)
            or (self.forward_transform.nlon != self.inverse_transform.nlon)
            or (self.forward_transform.grid != self.inverse_transform.grid)
        )

        if inverse_transform.lmax != self.modes_lat:
            raise AssertionError
        if inverse_transform.mmax != self.modes_lon:
            raise AssertionError

        hidden_size = int(hidden_size_factor * self.in_channels)

        if operator_type == "diagonal":
            self.mul_add_handle = compl_muladd2d_fwd
            self.mul_handle = compl_mul2d_fwd

            # weights
            w = [self.scale * torch.randn(self.in_channels, hidden_size, 2)]

            [
                w.append(self.scale * torch.randn(hidden_size, hidden_size, 2))
                for _ in range(1, self.spectral_layers)
            ]

            self.w = nn.ParameterList(w)

            self.wout = nn.Parameter(
                self.scale * torch.randn(hidden_size, self.out_channels, 2)
            )

            if bias:
                self.b = nn.ParameterList(
                    [
                        self.scale * torch.randn(hidden_size, 1, 1, 2)
                        for _ in range(self.spectral_layers)
                    ]
                )

            self.activations = nn.ModuleList([])
            for _ in range(0, self.spectral_layers):
                self.activations.append(
                    ComplexReLU(
                        mode=complex_activation,
                        bias_shape=(hidden_size, 1, 1),
                        scale=self.scale,
                    )
                )

        elif operator_type == "l-dependant":
            self.mul_add_handle = compl_exp_muladd2d_fwd
            self.mul_handle = compl_exp_mul2d_fwd

            # weights
            w = [
                self.scale
                * torch.randn(self.modes_lat, self.in_channels, hidden_size, 2)
            ]
            for _ in range(1, self.spectral_layers):
                w.append(
                    self.scale
                    * torch.randn(self.modes_lat, hidden_size, hidden_size, 2)
                )
            self.w = nn.ParameterList(w)

            if bias:
                self.b = nn.ParameterList(
                    [
                        self.scale * torch.randn(hidden_size, 1, 1, 2)
                        for _ in range(self.spectral_layers)
                    ]
                )

            self.wout = nn.Parameter(
                self.scale
                * torch.randn(self.modes_lat, hidden_size, self.out_channels, 2)
            )

            self.activations = nn.ModuleList([])
            for _ in range(0, self.spectral_layers):
                self.activations.append(
                    ComplexReLU(
                        mode=complex_activation,
                        bias_shape=(hidden_size, 1, 1),
                        scale=self.scale,
                    )
                )

        else:
            raise ValueError("Unknown operator type")

        self.drop = nn.Dropout(drop_rate) if drop_rate > 0.0 else nn.Identity()

    def _init_weights(self):  # pragma: no cover
        raise NotImplementedError()

    def forward_mlp(self, x):  # pragma: no cover
        """forward pass of the MLP"""
        B, C, H, W = x.shape

        xr = torch.view_as_real(x)

        for layer in range(self.spectral_layers):
            if hasattr(self, "b"):
                xr = self.mul_add_handle(xr, self.w[layer], self.b[layer])
            else:
                xr = self.mul_handle(xr, self.w[layer])
            xr = torch.view_as_complex(xr)
            xr = self.activations[layer](xr)
            xr = self.drop(xr)
            xr = torch.view_as_real(xr)

        # final MLP
        x = self.mul_handle(xr, self.wout)

        x = torch.view_as_complex(x)

        return x

    def forward(self, x):  # pragma: no cover
        dtype = x.dtype
        residual = x
        x = x.to(torch.float32)

        # FWD transform
        with amp.autocast(enabled=False):
            x = self.forward_transform(x)
            if self.scale_residual:
                x = x.contiguous()
                residual = self.inverse_transform(x)
                residual = residual.to(dtype)

        # MLP
        x = self.forward_mlp(x)

        # BWD transform
        x = x.contiguous()
        with amp.autocast(enabled=False):
            x = self.inverse_transform(x)

        # cast back to initial precision
        x = x.to(dtype)

        return x, residual
