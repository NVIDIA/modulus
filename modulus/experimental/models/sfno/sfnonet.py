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

from dataclasses import dataclass
from functools import partial
from typing import Any, List

import torch
import torch.nn as nn

# get spectral transforms from torch_harmonics
import torch_harmonics as th
import torch_harmonics.distributed as thd

# layer normalization
from apex.normalization import FusedLayerNorm
from torch.cuda import amp
from torch.utils.checkpoint import checkpoint

# helpers
# import global convolution and non-linear spectral layers
# wrap fft, to unify interface to spectral transforms
from modulus.experimental.models.sfno.layers import (
    MLP,
    DropPath,
    EncoderDecoder,
    RealFFT2,
    SpectralAttention2d,
    SpectralAttentionS2,
)
from modulus.experimental.models.sfno.s2convolutions import SpectralConvS2
from modulus.experimental.utils.sfno.distributed.dist_layers import (
    DistributedEncoderDecoder,
    DistributedInverseRealFFT2,
    DistributedMLP,
    DistributedRealFFT2,
)
from modulus.models.layers import get_activation
from modulus.models.meta import ModelMetaData
from modulus.models.module import Module
from modulus.models.sfno.initialization import trunc_normal_

# more distributed stuff
from modulus.utils.sfno.distributed import comm
from modulus.utils.sfno.distributed.layer_norm import (
    DistributedInstanceNorm2d,
)
from modulus.utils.sfno.distributed.mappings import (
    gather_from_parallel_region,
    scatter_to_parallel_region,
)


class SpectralFilterLayer(nn.Module):
    """Spectral filter layer"""

    def __init__(
        self,
        forward_transform,
        inverse_transform,
        embed_dim,
        filter_type="linear",
        operator_type="diagonal",
        sparsity_threshold=0.0,
        use_complex_kernels=True,
        hidden_size_factor=1,
        rank=1.0,
        factorization=None,
        separable=False,
        complex_network=True,
        complex_activation="real",
        spectral_layers=1,
        drop_rate=0.0,
    ):  # pragma: no cover
        super(SpectralFilterLayer, self).__init__()

        if filter_type == "non-linear" and (
            isinstance(forward_transform, th.RealSHT)
            or isinstance(forward_transform, thd.DistributedRealSHT)
        ):
            self.filter = SpectralAttentionS2(
                forward_transform,
                inverse_transform,
                embed_dim,
                sparsity_threshold=sparsity_threshold,
                hidden_size_factor=hidden_size_factor,
                complex_activation=complex_activation,
                spectral_layers=spectral_layers,
                drop_rate=drop_rate,
                bias=False,
            )

        elif filter_type == "non-linear" and (
            isinstance(forward_transform, RealFFT2)
            or isinstance(forward_transform, DistributedRealFFT2)
        ):
            self.filter = SpectralAttention2d(
                forward_transform,
                inverse_transform,
                embed_dim,
                sparsity_threshold=sparsity_threshold,
                hidden_size_factor=hidden_size_factor,
                complex_activation=complex_activation,
                spectral_layers=spectral_layers,
                drop_rate=drop_rate,
                bias=False,
            )

        # spectral transform is passed to the module
        elif filter_type == "linear":
            self.filter = SpectralConvS2(
                forward_transform,
                inverse_transform,
                embed_dim,
                embed_dim,
                operator_type=operator_type,
                rank=rank,
                factorization=factorization,
                separable=separable,
                bias=False,
                use_tensorly=False if factorization is None else True,
            )

        else:
            raise (NotImplementedError)

    def forward(self, x):  # pragma: no cover
        return self.filter(x)


class FourierNeuralOperatorBlock(nn.Module):
    """Fourier Neural Operator Block"""

    def __init__(
        self,
        forward_transform,
        inverse_transform,
        embed_dim,
        filter_type="linear",
        operator_type="diagonal",
        mlp_ratio=2.0,
        drop_rate=0.0,
        drop_path=0.0,
        act_layer="gelu",
        norm_layer=(nn.LayerNorm, nn.LayerNorm),
        sparsity_threshold=0.0,
        use_complex_kernels=True,
        rank=1.0,
        factorization=None,
        separable=False,
        inner_skip="linear",
        outer_skip=None,  # None, nn.linear or nn.Identity
        use_mlp=False,
        comm_feature_inp_name=None,
        comm_feature_hidden_name=None,
        complex_network=True,
        complex_activation="real",
        spectral_layers=1,
        checkpointing=0,
    ):  # pragma: no cover
        super().__init__()

        if (comm.get_size("h") > 1) or (comm.get_size("w") > 1):
            self.input_shape_loc = (
                forward_transform.nlat_local,
                forward_transform.nlon_local,
            )
            self.output_shape_loc = (
                inverse_transform.nlat_local,
                inverse_transform.nlon_local,
            )
        else:
            self.input_shape_loc = (forward_transform.nlat, forward_transform.nlon)
            self.output_shape_loc = (inverse_transform.nlat, inverse_transform.nlon)

        # norm layer
        self.norm0 = norm_layer[0]()

        # convolution layer
        self.filter = SpectralFilterLayer(
            forward_transform,
            inverse_transform,
            embed_dim,
            filter_type,
            operator_type,
            sparsity_threshold,
            use_complex_kernels=use_complex_kernels,
            hidden_size_factor=mlp_ratio,
            rank=rank,
            factorization=factorization,
            separable=separable,
            complex_network=complex_network,
            complex_activation=complex_activation,
            spectral_layers=spectral_layers,
            drop_rate=drop_rate,
        )

        if inner_skip == "linear":
            self.inner_skip = nn.Conv2d(embed_dim, embed_dim, 1, 1)
        elif inner_skip == "identity":
            self.inner_skip = nn.Identity()

        if filter_type == "linear" or filter_type == "real linear":
            self.act_layer = get_activation(act_layer)

        # dropout
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # norm layer
        self.norm1 = norm_layer[1]()

        if use_mlp:
            MLPH = DistributedMLP if (comm.get_size("matmul") > 1) else MLP
            mlp_hidden_dim = int(embed_dim * mlp_ratio)
            self.mlp = MLPH(
                in_features=embed_dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop_rate=drop_rate,
                comm_inp_name=comm_feature_inp_name,
                comm_hidden_name=comm_feature_hidden_name,
                checkpointing=checkpointing,
            )

        if outer_skip == "linear":
            self.outer_skip = nn.Conv2d(embed_dim, embed_dim, 1, 1)
        elif outer_skip == "identity":
            self.outer_skip = nn.Identity()

    def forward(self, x):  # pragma: no cover
        x_norm = torch.zeros_like(x)
        x_norm[..., : self.input_shape_loc[0], : self.input_shape_loc[1]] = self.norm0(
            x[..., : self.input_shape_loc[0], : self.input_shape_loc[1]]
        )
        x, residual = self.filter(x_norm)

        if hasattr(self, "inner_skip"):
            x = x + self.inner_skip(residual)

        if hasattr(self, "act_layer"):
            x = self.act_layer(x)

        x_norm = torch.zeros_like(x)
        x_norm[
            ..., : self.output_shape_loc[0], : self.output_shape_loc[1]
        ] = self.norm1(x[..., : self.output_shape_loc[0], : self.output_shape_loc[1]])
        x = x_norm

        if hasattr(self, "mlp"):
            x = self.mlp(x)

        x = self.drop_path(x)

        if hasattr(self, "outer_skip"):
            x = x + self.outer_skip(residual)

        return x


@dataclass
class SFNOMetaData(ModelMetaData):
    """Data class for storing SFNO meta data"""

    name: str = "SFNO"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = True
    amp_cpu: bool = True
    amp_gpu: bool = True
    torch_fx: bool = False
    # Data type
    bf16: bool = False
    # Inference
    onnx: bool = False
    # Physics informed
    func_torch: bool = False
    auto_grad: bool = False


class SFNO(Module):
    """
    Spherical Fourier Neural Operator Network

    Parameters
    ----------
    spectral_transform : str, optional
        Type of spectral transformation to use, by default "sht"
    grid : str, optional
        Type of grid to use, by default "legendre-gauss"
    filter_type : str, optional
        Type of filter to use ('linear', 'non-linear'), by default "non-linear"
    operator_type : str, optional
        Type of operator to use ('diaginal', 'dhconv'), by default "diagonal"
    inp_shape : List[int], optional
        Shape of the input channels, by default [721, 1440]
    scale_factor : int, optional
        Scale factor to use, by default 16
    in_channels : int, optional
        Number of input channels, by default 2
    out_channels : int, optional
        Number of output channels, by default 2
    out_channels_id : List[int], optional
        List of output channel ids, by default [0, 1, 2]
    embed_dim : int, optional
        Dimension of the embeddings, by default 256
    num_layers : int, optional
        Number of layers in the network, by default 12
    repeat_layers : int, optional
        Number of times to repeat the layers, by default 1
    use_mlp : int, optional
        Whether to use MLP, by default True
    mlp_ratio : int, optional
        Ratio of MLP to use, by default 2.0
    activation_function : str, optional
        Activation function to use, by default "gelu"
    encoder_layers : int, optional
        Number of layers in the encoder, by default 1
    pos_embed : str, optional
        Type of positional embedding to use, by default "direct"
    drop_rate : float, optional
        Dropout rate, by default 0.0
    drop_path_rate : float, optional
        Dropout path rate, by default 0.0
    sparsity_threshold : float, optional
        Threshold for sparsity, by default 0.0
    normalization_layer : str, optional
        Type of normalization layer to use ("layer_norm", "instance_norm", "none"), by default "instance_norm"
    max_modes : Any, optional
        Maximum modes to use, by default None
    hard_thresholding_fraction : float, optional
        Fraction of hard thresholding to apply, by default 1.0
    use_complex_kernels : bool, optional
        Whether to use complex kernels, by default True
    big_skip : bool, optional
        Whether to use big skip connections, by default True
    rank : float, optional
        Rank of the approximation, by default 1.0
    factorization : Any, optional
        Type of factorization to use, by default None
    separable : bool, optional
        Whether to use separable convolutions, by default False
    complex_network : bool, optional
        Whether to use a complex network architecture, by default True
    complex_activation : str, optional
        Type of complex activation function to use, by default "real"
    spectral_layers : int, optional
        Number of spectral layers, by default 3
    output_transform : bool, optional
        Whether to use an output transform, by default False
    checkpointing : int, optional
        Number of checkpointing segments, by default 0

    Example:
    --------
    #>>> from modulus.models.sfno.sfnonet import SFNO
    #>>> model = SFNO(
    #...         inp_shape=(8, 16),
    #...         scale_factor=4,
    #...         in_channels=2,
    #...         out_channels=2,
    #...         embed_dim=16,
    #...         num_layers=2,
    #...         encoder_layers=1,
    #...         spectral_layers=2,
    #...         use_mlp=True,)
    #>>> model(torch.randn(1, 2, 8, 16)).shape
    #torch.Size([1, 2, 8, 16])
    """

    def __init__(
        self,
        spectral_transform: str = "sht",
        grid="legendre-gauss",
        filter_type: str = "non-linear",
        operator_type: str = "diagonal",
        inp_shape: List[int] = [721, 1440],
        out_shape: Any = None,
        scale_factor: int = 16,
        in_channels: int = 2,
        out_channels: int = 2,
        out_channels_id: List[int] = [0, 1, 2],
        channel_names: List[str] = ["u10m", "v10m", "u100m"],
        embed_dim: int = 256,
        num_layers: int = 12,
        repeat_layers=1,
        use_mlp: int = True,
        mlp_ratio: int = 2.0,
        activation_function: str = "gelu",
        encoder_layers: int = 1,
        pos_embed: str = "direct",
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        sparsity_threshold: float = 0.0,
        normalization_layer: str = "instance_norm",
        max_modes: Any = None,
        hard_thresholding_fraction: float = 1.0,
        use_complex_kernels: bool = True,
        big_skip: bool = True,
        rank: float = 1.0,
        factorization: Any = None,
        separable: bool = False,
        complex_network: bool = True,
        complex_activation: str = "real",
        spectral_layers: int = 3,
        output_transform: bool = False,
        checkpointing: int = 0,
    ):  # pragma: no cover
        super().__init__(meta=SFNOMetaData())

        self.spectral_transform = spectral_transform
        self.grid = grid
        self.filter_type = filter_type
        self.operator_type = operator_type
        self.inp_shape = inp_shape
        self.out_shape = inp_shape if out_shape is None else out_shape
        self.scale_factor = scale_factor
        self.in_chans = in_channels
        self.out_chans = out_channels
        self.out_channels_id = out_channels_id
        self.channel_names = channel_names
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.repeat_layers = repeat_layers
        self.max_modes = max_modes
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.normalization_layer = normalization_layer
        self.use_mlp = use_mlp
        self.mlp_ratio = mlp_ratio
        self.activation_function = activation_function
        self.encoder_layers = encoder_layers
        self.pos_embed = pos_embed
        self.big_skip = big_skip
        self.rank = rank
        self.factorization = factorization
        self.separable = separable
        self.complex_network = complex_network
        self.complex_activation = complex_activation
        self.spectral_layers = spectral_layers
        self.output_transform = output_transform
        self.checkpointing = checkpointing

        # compute the downscaled image size
        self.h = int(self.inp_shape[0] // self.scale_factor)
        self.w = int(self.inp_shape[1] // self.scale_factor)

        # Compute the maximum frequencies in h and in w
        if self.max_modes is not None:
            modes_lat, modes_lon = self.max_modes
        else:
            modes_lat = int(self.h * self.hard_thresholding_fraction)
            modes_lon = int((self.w // 2 + 1) * self.hard_thresholding_fraction)

        # prepare the spectral transforms
        if self.spectral_transform == "sht":
            sht_handle = th.RealSHT
            isht_handle = th.InverseRealSHT

            # parallelism
            if (comm.get_size("h") > 1) or (comm.get_size("w") > 1):
                polar_group = None if (comm.get_size("h") == 1) else comm.get_group("h")
                azimuth_group = (
                    None if (comm.get_size("w") == 1) else comm.get_group("w")
                )
                thd.init(polar_group, azimuth_group)
                sht_handle = thd.DistributedRealSHT
                isht_handle = thd.DistributedInverseRealSHT

            # set up
            self.trans_down = sht_handle(
                *self.inp_shape, lmax=modes_lat, mmax=modes_lon, grid="equiangular"
            ).float()
            self.itrans_up = isht_handle(
                *self.out_shape, lmax=modes_lat, mmax=modes_lon, grid="equiangular"
            ).float()
            self.trans = sht_handle(
                self.h, self.w, lmax=modes_lat, mmax=modes_lon, grid=self.grid
            ).float()
            self.itrans = isht_handle(
                self.h, self.w, lmax=modes_lat, mmax=modes_lon, grid=self.grid
            ).float()

        elif self.spectral_transform == "fft":
            fft_handle = th.RealFFT2
            ifft_handle = th.InverseRealFFT2

            # determine the global padding
            inp_dist_h = (
                (self.inp_shape[0] + comm.get_size("h")) - 1
            ) // comm.get_size("h")
            inp_dist_w = (
                (self.inp_shape[1] + comm.get_size("w")) - 1
            ) // comm.get_size("w")
            self.inp_padding = (
                inp_dist_h * comm.get_size("h") - self.inp_shape[0],
                inp_dist_w * comm.get_size("w") - self.inp_shape[1],
            )
            out_dist_h = (
                (self.out_shape[0] + comm.get_size("h")) - 1
            ) // comm.get_size("h")
            out_dist_w = (
                (self.out_shape[1] + comm.get_size("w")) - 1
            ) // comm.get_size("w")
            self.out_padding = (
                out_dist_h * comm.get_size("h") - self.out_shape[0],
                out_dist_w * comm.get_size("w") - self.out_shape[1],
            )
            # effective image size:
            self.inp_shape_eff = [
                self.inp_shape[0] + self.inp_padding[0],
                self.inp_shape[1] + self.inp_padding[1],
            ]
            self.inp_shape_loc = [
                self.inp_shape_eff[0] // comm.get_size("h"),
                self.inp_shape_eff[1] // comm.get_size("w"),
            ]
            self.out_shape_eff = [
                self.out_shape[0] + self.out_padding[0],
                self.out_shape[1] + self.out_padding[1],
            ]
            self.out_shape_loc = [
                self.out_shape_eff[0] // comm.get_size("h"),
                self.out_shape_eff[1] // comm.get_size("w"),
            ]

            if (comm.get_size("h") > 1) or (comm.get_size("w") > 1):
                fft_handle = DistributedRealFFT2
                ifft_handle = DistributedInverseRealFFT2

            self.trans_down = fft_handle(
                *self.inp_shape_eff, lmax=modes_lat, mmax=modes_lon
            ).float()
            self.itrans_up = ifft_handle(
                *self.out_shape_eff, lmax=modes_lat, mmax=modes_lon
            ).float()
            self.trans = fft_handle(
                self.h, self.w, lmax=modes_lat, mmax=modes_lon
            ).float()
            self.itrans = ifft_handle(
                self.h, self.w, lmax=modes_lat, mmax=modes_lon
            ).float()
        else:
            raise (ValueError("Unknown spectral transform"))

        # use the SHT/FFT to compute the local, downscaled grid dimensions
        if (comm.get_size("h") > 1) or (comm.get_size("w") > 1):
            self.inp_shape_loc = (
                self.trans_down.nlat_local,
                self.trans_down.nlon_local,
            )
            self.inp_shape_eff = [
                self.trans_down.nlat_local + self.trans_down.nlatpad_local,
                self.trans_down.nlon_local + self.trans_down.nlonpad_local,
            ]
            self.h_loc = self.itrans.nlat_local
            self.w_loc = self.itrans.nlon_local
        else:
            self.inp_shape_loc = (self.trans_down.nlat, self.trans_down.nlon)
            self.inp_shape_eff = [self.trans_down.nlat, self.trans_down.nlon]
            self.h_loc = self.itrans.nlat
            self.w_loc = self.itrans.nlon

        # encoder
        if comm.get_size("matmul") > 1:
            self.encoder = DistributedEncoderDecoder(
                num_layers=self.encoder_layers,
                input_dim=self.in_chans,
                output_dim=self.embed_dim,
                hidden_dim=int(1 * self.embed_dim),
                act=self.activation_function,
                comm_inp_name="fin",
                comm_out_name="fout",
            )
            fblock_mlp_inp_name = self.encoder.comm_out_name
            fblock_mlp_hidden_name = (
                "fout" if (self.encoder.comm_out_name == "fin") else "fin"
            )
        else:
            self.encoder = EncoderDecoder(
                num_layers=self.encoder_layers,
                input_dim=self.in_chans,
                output_dim=self.embed_dim,
                hidden_dim=int(1 * self.embed_dim),
                act=self.activation_function,
            )
            fblock_mlp_inp_name = "fin"
            fblock_mlp_hidden_name = "fout"

        # dropout
        self.pos_drop = nn.Dropout(p=drop_rate) if drop_rate > 0.0 else nn.Identity()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.num_layers)]

        # pick norm layer
        if self.normalization_layer == "layer_norm":
            norm_layer_inp = partial(
                nn.LayerNorm,
                normalized_shape=(self.inp_shape_loc[0], self.inp_shape_loc[1]),
                eps=1e-6,
            )
            norm_layer_mid = partial(
                nn.LayerNorm, normalized_shape=(self.h_loc, self.w_loc), eps=1e-6
            )
            norm_layer_out = partial(
                nn.LayerNorm,
                normalized_shape=(self.out_shape_loc[0], self.out_shape_loc[1]),
                eps=1e-6,
            )
        elif self.normalization_layer == "instance_norm":
            if comm.get_size("spatial") > 1:
                norm_layer_inp = partial(
                    DistributedInstanceNorm2d,
                    num_features=self.embed_dim,
                    eps=1e-6,
                    affine=True,
                )
            else:
                norm_layer_inp = partial(
                    nn.InstanceNorm2d,
                    num_features=self.embed_dim,
                    eps=1e-6,
                    affine=True,
                    track_running_stats=False,
                )
            norm_layer_out = norm_layer_mid = norm_layer_inp
        elif self.normalization_layer == "none":
            norm_layer_out = norm_layer_mid = norm_layer_inp = nn.Identity
        else:
            raise NotImplementedError(
                f"Error, normalization {self.normalization_layer} not implemented."
            )

        # FNO blocks
        self.blocks = nn.ModuleList([])
        for i in range(self.num_layers):

            first_layer = i == 0
            last_layer = i == self.num_layers - 1

            forward_transform = self.trans_down if first_layer else self.trans
            inverse_transform = self.itrans_up if last_layer else self.itrans

            inner_skip = "linear"
            outer_skip = "identity"

            if first_layer and last_layer:
                norm_layer = (norm_layer_inp, norm_layer_out)
            elif first_layer:
                norm_layer = (norm_layer_inp, norm_layer_mid)
            elif last_layer:
                norm_layer = (norm_layer_mid, norm_layer_out)
            else:
                norm_layer = (norm_layer_mid, norm_layer_mid)

            filter_type = self.filter_type

            operator_type = self.operator_type

            block = FourierNeuralOperatorBlock(
                forward_transform,
                inverse_transform,
                self.embed_dim,
                filter_type=filter_type,
                operator_type=operator_type,
                mlp_ratio=self.mlp_ratio,
                drop_rate=drop_rate,
                drop_path=dpr[i],
                act_layer=self.activation_function,
                norm_layer=norm_layer,
                sparsity_threshold=sparsity_threshold,
                use_complex_kernels=use_complex_kernels,
                inner_skip=inner_skip,
                outer_skip=outer_skip,
                use_mlp=self.use_mlp,
                comm_feature_inp_name=fblock_mlp_inp_name,
                comm_feature_hidden_name=fblock_mlp_hidden_name,
                rank=self.rank,
                factorization=self.factorization,
                separable=self.separable,
                complex_network=self.complex_network,
                complex_activation=self.complex_activation,
                spectral_layers=self.spectral_layers,
                checkpointing=self.checkpointing,
            )

            self.blocks.append(block)

        # decoder
        if comm.get_size("matmul") > 1:
            comm_inp_name = fblock_mlp_inp_name
            comm_out_name = fblock_mlp_hidden_name
            self.decoder = DistributedEncoderDecoder(
                num_layers=self.encoder_layers,
                input_dim=self.embed_dim,
                output_dim=self.out_chans,
                hidden_dim=int(1 * self.embed_dim),
                act=self.activation_function,
                comm_inp_name=comm_inp_name,
                comm_out_name=comm_out_name,
            )
        else:
            self.decoder = EncoderDecoder(
                num_layers=self.encoder_layers,
                input_dim=self.embed_dim + self.big_skip * self.out_chans,
                output_dim=self.out_chans,
                hidden_dim=int(1 * self.embed_dim),
                act=self.activation_function,
            )

        # output transform
        if self.big_skip:
            self.residual_transform = nn.Conv2d(
                self.out_chans, self.out_chans, 1, bias=False
            )

        # learned position embedding
        if self.pos_embed == "direct":
            # currently using deliberately a differently shape position embedding
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1, self.embed_dim, self.inp_shape_loc[0], self.inp_shape_loc[1]
                )
            )
            self.pos_embed.is_shared_mp = ["matmul"]
            self.pos_embed.sharded_dims_mp = [None, None, "h", "w"]
            self.pos_embed.type = "direct"
            trunc_normal_(self.pos_embed, std=0.02)
        elif self.pos_embed == "frequency":
            if (comm.get_size("h") > 1) or (comm.get_size("w") > 1):
                lmax_loc = self.itrans_up.lmax_local
                mmax_loc = self.itrans_up.mmax_local
            else:
                lmax_loc = self.itrans_up.lmax
                mmax_loc = self.itrans_up.mmax

            rcoeffs = nn.Parameter(
                torch.tril(
                    torch.randn(1, self.embed_dim, lmax_loc, mmax_loc), diagonal=0
                )
            )
            ccoeffs = nn.Parameter(
                torch.tril(
                    torch.randn(1, self.embed_dim, lmax_loc, mmax_loc - 1), diagonal=-1
                )
            )
            trunc_normal_(rcoeffs, std=0.02)
            trunc_normal_(ccoeffs, std=0.02)
            self.pos_embed = nn.ParameterList([rcoeffs, ccoeffs])
            self.pos_embed.type = "frequency"

        elif self.pos_embed == "none" or self.pos_embed == "None":
            delattr(self, "pos_embed")
        else:
            raise ValueError("Unknown position embedding type")

        if self.output_transform:
            minmax_channels = []
            for o, c in enumerate(self.out_channels_id):
                if self.channel_names[c][0] == "r":
                    minmax_channels.append(o)
            self.register_buffer(
                "minmax_channels",
                torch.Tensor(minmax_channels).to(torch.long),
                persistent=False,
            )

        self.apply(self._init_weights)

    def _init_weights(self, m):  # pragma: no cover
        """Helper routine for weight initialization"""
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, FusedLayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):  # pragma: no cover
        """Helper"""
        return {"pos_embed", "cls_token"}

    def _forward_features(self, x):  # pragma: no cover

        for r in range(self.repeat_layers):
            for blk in self.blocks:
                if self.checkpointing >= 3:
                    x = checkpoint(blk, x)
                else:
                    x = blk(x)

        return x

    def forward(self, x):  # pragma: no cover

        if comm.get_size("fin") > 1:
            x = scatter_to_parallel_region(x, "fin", 1)

        # save big skip
        if self.big_skip:
            # if output shape differs, use the spectral transforms to change resolution
            if self.out_shape != self.inp_shape:
                xtype = x.dtype
                # only take the predicted channels as residual
                residual = x[..., : self.out_chans, :, :].to(torch.float32)
                with amp.autocast(enabled=False):
                    residual = self.trans_down(residual)
                    residual = residual.contiguous()
                    # residual = self.inverse_transform(residual)
                    residual = self.itrans_up(residual)
                    residual = residual.to(dtype=xtype)
            else:
                # only take the predicted channels
                residual = x[..., : self.out_chans, :, :]

        if self.checkpointing >= 1:
            x = checkpoint(self.encoder, x)
        else:
            x = self.encoder(x)

        if hasattr(self, "pos_embed"):

            if self.pos_embed.type == "frequency":

                pos_embed = torch.stack(
                    [
                        self.pos_embed[0],
                        nn.functional.pad(self.pos_embed[1], (1, 0), "constant", 0),
                    ],
                    dim=-1,
                )
                with amp.autocast(enabled=False):
                    pos_embed = self.itrans_up(torch.view_as_complex(pos_embed))
            else:
                pos_embed = self.pos_embed

            # old way of treating unequally shaped weights
            if (
                self.pos_embed.type == "direct"
                and self.inp_shape_loc != self.inp_shape_eff
            ):
                xp = torch.zeros_like(x)
                xp[..., : self.inp_shape_loc[0], : self.inp_shape_loc[1]] = (
                    x[..., : self.inp_shape_loc[0], : self.inp_shape_loc[1]] + pos_embed
                )
                x = xp
            else:
                x = x + pos_embed

        # maybe clean the padding jsut in case
        x = self.pos_drop(x)

        # do the feature extraction
        x = self._forward_features(x)

        if self.big_skip:
            x = torch.cat((x, residual), dim=1)

        if self.checkpointing >= 1:
            x = checkpoint(self.decoder, x)
        else:
            x = self.decoder(x)

        if hasattr(self.decoder, "comm_out_name") and (
            comm.get_size(self.decoder.comm_out_name) > 1
        ):
            x = gather_from_parallel_region(x, self.decoder.comm_out_name, 1)

        if self.big_skip:
            x = x + self.residual_transform(residual)

        if self.output_transform:
            x[:, self.minmax_channels] = torch.sigmoid(x[:, self.minmax_channels])

        return x
