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
from modulus.experimental.models.sfno.activations import ComplexReLU

# for the experimental module
from modulus.experimental.models.sfno.contractions import (
    compl_exp_mul2d_fwd,
    compl_exp_muladd2d_fwd,
    compl_mul2d_fwd,
    compl_muladd2d_fwd,
)
from modulus.experimental.models.sfno.factorizations import get_contract_fun

tl.set_backend("pytorch")


class SpectralConvS2(nn.Module):
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
        scale="auto",
        operator_type="diagonal",
        rank=0.2,
        factorization=None,
        separable=False,
        decomposition_kwargs=dict(),
        bias=False,
        use_tensorly=True,
    ):  # pragma: no cover
        super(SpectralConvS2, self).__init__()

        if scale == "auto":
            # heuristic
            scale = 2 / (in_channels + out_channels)

        self.forward_transform = forward_transform
        self.inverse_transform = inverse_transform

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
        else:
            raise ValueError(f"Unsupported operator type f{self.operator_type}")

        if use_tensorly:
            # form weight tensors
            self.weight = FactorizedTensor.new(
                weight_shape,
                rank=self.rank,
                factorization=factorization,
                fixed_rank_modes=False,
                **decomposition_kwargs,
            )
            # initialization of weights
            self.weight.normal_(0, scale)
        else:
            if complex_weight:
                init = scale * torch.randn(*weight_shape, 2)
                self.weight = nn.Parameter(init)
            else:
                init = scale * torch.randn(*weight_shape)

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

        # get the contraction handle
        self._contract = get_contract_fun(
            self.weight,
            implementation="factorized",
            separable=separable,
            complex=complex_weight,
            operator_type=operator_type,
        )

        if bias:
            self.bias = nn.Parameter(scale * torch.zeros(1, out_channels, 1, 1))

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
        xp = torch.zeros_like(x)
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


class SpectralAttentionS2(nn.Module):
    """
    Spherical non-linear FNO layer
    """

    def __init__(
        self,
        forward_transform,
        inverse_transform,
        embed_dim,
        operator_type="diagonal",
        sparsity_threshold=0.0,
        hidden_size_factor=2,
        complex_activation="real",
        scale="auto",
        bias=False,
        spectral_layers=1,
        drop_rate=0.0,
    ):  # pragma: no cover
        super(SpectralAttentionS2, self).__init__()

        self.embed_dim = embed_dim
        self.sparsity_threshold = sparsity_threshold
        self.operator_type = operator_type
        self.spectral_layers = spectral_layers

        if scale == "auto":
            self.scale = 1 / (embed_dim * embed_dim)

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

        hidden_size = int(hidden_size_factor * self.embed_dim)

        if operator_type == "diagonal":
            self.mul_add_handle = compl_muladd2d_fwd
            self.mul_handle = compl_mul2d_fwd

            # weights
            w = [self.scale * torch.randn(self.embed_dim, hidden_size, 2)]
            for lay in range(1, self.spectral_layers):
                w.append(  # noqa: PERF401
                    self.scale * torch.randn(hidden_size, hidden_size, 2)
                )  # noqa: PERF401
            self.w = nn.ParameterList(w)

            self.wout = nn.Parameter(
                self.scale * torch.randn(hidden_size, self.embed_dim, 2)
            )

            if bias:
                self.b = nn.ParameterList(
                    [
                        self.scale * torch.randn(hidden_size, 1, 1, 2)
                        for _ in range(self.spectral_layers)
                    ]
                )

            self.activations = nn.ModuleList([])
            for lay in range(0, self.spectral_layers):
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
                self.scale * torch.randn(self.modes_lat, self.embed_dim, hidden_size, 2)
            ]
            for lay in range(1, self.spectral_layers):
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
                self.scale * torch.randn(self.modes_lat, hidden_size, self.embed_dim, 2)
            )

            self.activations = nn.ModuleList([])
            for lay in range(0, self.spectral_layers):
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

    def forward_mlp(self, x):  # pragma: no cover
        """forward pass of the MLP"""
        B, C, H, W = x.shape

        xr = torch.view_as_real(x)

        for lay in range(self.spectral_layers):
            if hasattr(self, "b"):
                xr = self.mul_add_handle(xr, self.w[lay], self.b[lay])
            else:
                xr = self.mul_handle(xr, self.w[lay])
            xr = torch.view_as_complex(xr)
            xr = self.activations[lay](xr)
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
