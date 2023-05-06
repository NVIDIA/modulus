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

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import torch.distributed as dist
from dataclasses import dataclass

from modulus.models.sfno.factorizations import get_contract_fun, _contract_dense

# helpers
from modulus.models.sfno.layers import trunc_normal_, DropPath, MLP

# import global convolution and non-linear spectral layers
from modulus.models.sfno.layers import SpectralAttention2d
from modulus.models.sfno.s2convolutions import SpectralConvS2, SpectralAttentionS2
from modulus.models.sfno.s2convolutions import RealSpectralAttentionS2
from modulus.models.sfno.s2convolutions import LocalConvS2

# get spectral transforms from torch_harmonics
import torch_harmonics as th
import torch_harmonics.distributed as thd

# wrap fft, to unify interface to spectral transforms
from modulus.models.sfno.layers import RealFFT2, InverseRealFFT2
from modulus.utils.sfno.distributed.layers import (
    DistributedRealFFT2,
    DistributedInverseRealFFT2,
    DistributedMLP,
)

# more distributed stuff
from modulus.utils.sfno.distributed import comm
from modulus.utils.sfno.distributed.mappings import (
    scatter_to_parallel_region,
    gather_from_parallel_region,
)

# layer normalization
from apex.normalization import FusedLayerNorm
from modulus.utils.sfno.distributed.layer_norm import DistributedInstanceNorm2d

from modulus.models.module import Module
from modulus.models.meta import ModelMetaData


@dataclass
class MetaData(ModelMetaData):
    name: str = "SFNO"
    # Optimization
    jit: bool = True
    cuda_graphs: bool = True
    amp_cpu: bool = True
    amp_gpu: bool = True
    torch_fx: bool = False
    # Inference
    onnx: bool = False
    # Physics informed
    func_torch: bool = False
    auto_grad: bool = False


class SpectralFilterLayer(nn.Module):
    def __init__(
        self,
        forward_transform,
        inverse_transform,
        embed_dim,
        filter_type="linear",
        operator_type="block-diagonal",
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
    ):
        super(SpectralFilterLayer, self).__init__()

        if filter_type == "non-linear" and (
            isinstance(forward_transform, th.RealSHT)
            or isinstance(forward_transform, thd.DistributedRealSHT)
        ):
            self.filter = SpectralAttentionS2(
                forward_transform,
                inverse_transform,
                embed_dim,
                operator_type=operator_type,
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
        elif filter_type == "linear" and (
            isinstance(forward_transform, th.RealSHT)
            or isinstance(forward_transform, thd.DistributedRealSHT)
        ):
            self.filter = SpectralConvS2(
                forward_transform,
                inverse_transform,
                embed_dim,
                embed_dim,
                operator_type=operator_type,
                rank=rank,
                factorization=factorization,
                separable=separable,
                bias=True,
                use_tensorly=False if factorization is None else True,
            )

        else:
            raise (NotImplementedError)

    def forward(self, x):
        return self.filter(x)


class FourierNeuralOperatorBlock(nn.Module):
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
        act_layer=nn.GELU,
        norm_layer=(nn.LayerNorm, nn.LayerNorm),
        sparsity_threshold=0.0,
        use_complex_kernels=True,
        rank=1.0,
        factorization=None,
        separable=False,
        inner_skip="linear",
        outer_skip=None,  # None, nn.linear or nn.Identity
        concat_skip=False,
        use_mlp=False,
        complex_network=True,
        complex_activation="real",
        spectral_layers=1,
        checkpointing=0,
    ):
        super(FourierNeuralOperatorBlock, self).__init__()

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

        self.concat_skip = concat_skip

        if concat_skip and inner_skip is not None:
            self.inner_skip_conv = nn.Conv2d(2 * embed_dim, embed_dim, 1, bias=False)

        if filter_type == "linear" or filter_type == "real linear":
            self.act_layer = act_layer()

        # dropout
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # norm layer
        self.norm1 = norm_layer[1]()

        if use_mlp == True:
            MLPH = DistributedMLP if (comm.get_size("matmul") > 1) else MLP
            mlp_hidden_dim = int(embed_dim * mlp_ratio)
            self.mlp = MLPH(
                in_features=embed_dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop_rate=drop_rate,
                checkpointing=checkpointing,
            )

        if outer_skip == "linear":
            self.outer_skip = nn.Conv2d(embed_dim, embed_dim, 1, 1)
        elif outer_skip == "identity":
            self.outer_skip = nn.Identity()

        if concat_skip and outer_skip is not None:
            self.outer_skip_conv = nn.Conv2d(2 * embed_dim, embed_dim, 1, bias=False)

    def forward(self, x):

        x_norm = torch.zeros_like(x)
        x_norm[..., : self.input_shape_loc[0], : self.input_shape_loc[1]] = self.norm0(
            x[..., : self.input_shape_loc[0], : self.input_shape_loc[1]]
        )
        x, residual = self.filter(x_norm)

        if hasattr(self, "inner_skip"):
            if self.concat_skip:
                x = torch.cat((x, self.inner_skip(residual)), dim=1)
                x = self.inner_skip_conv(x)
            else:
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
            if self.concat_skip:
                x = torch.cat((x, self.outer_skip(residual)), dim=1)
                x = self.outer_skip_conv(x)
            else:
                x = x + self.outer_skip(residual)

        return x


class SphericalFourierNeuralOperatorNet(Module):
    def __init__(
        self,
        params,
        spectral_transform="sht",
        filter_type="non-linear",
        operator_type="diagonal",
        img_shape=(721, 1440),
        scale_factor=16,
        in_chans=2,
        out_chans=2,
        embed_dim=256,
        num_layers=12,
        use_mlp=True,
        mlp_ratio=2.0,
        activation_function="gelu",
        encoder_layers=1,
        pos_embed=True,
        drop_rate=0.0,
        drop_path_rate=0.0,
        num_blocks=16,
        sparsity_threshold=0.0,
        normalization_layer="instance_norm",
        hard_thresholding_fraction=1.0,
        use_complex_kernels=True,
        big_skip=True,
        rank=1.0,
        factorization=None,
        separable=False,
        complex_network=True,
        complex_activation="real",
        spectral_layers=3,
        checkpointing=0,
    ):
        super(SphericalFourierNeuralOperatorNet, self).__init__(meta=MetaData())

        self.params = params
        self.spectral_transform = (
            params.spectral_transform
            if hasattr(params, "spectral_transform")
            else spectral_transform
        )
        self.filter_type = (
            params.filter_type if hasattr(params, "filter_type") else filter_type
        )
        self.operator_type = (
            params.operator_type if hasattr(params, "operator_type") else operator_type
        )
        self.img_shape = (
            (params.img_shape_x, params.img_shape_y)
            if hasattr(params, "img_shape_x") and hasattr(params, "img_shape_y")
            else img_shape
        )
        self.scale_factor = (
            params.scale_factor if hasattr(params, "scale_factor") else scale_factor
        )
        self.in_chans = (
            params.N_in_channels if hasattr(params, "N_in_channels") else in_chans
        )
        self.out_chans = (
            params.N_out_channels if hasattr(params, "N_out_channels") else out_chans
        )
        self.embed_dim = self.num_features = (
            params.embed_dim if hasattr(params, "embed_dim") else embed_dim
        )
        self.num_layers = (
            params.num_layers if hasattr(params, "num_layers") else num_layers
        )
        self.num_blocks = (
            params.num_blocks if hasattr(params, "num_blocks") else num_blocks
        )
        self.hard_thresholding_fraction = (
            params.hard_thresholding_fraction
            if hasattr(params, "hard_thresholding_fraction")
            else hard_thresholding_fraction
        )
        self.normalization_layer = (
            params.normalization_layer
            if hasattr(params, "normalization_layer")
            else normalization_layer
        )
        self.use_mlp = params.use_mlp if hasattr(params, "use_mlp") else use_mlp
        self.activation_function = (
            params.activation_function
            if hasattr(params, "activation_function")
            else activation_function
        )
        self.encoder_layers = (
            params.encoder_layers
            if hasattr(params, "encoder_layers")
            else encoder_layers
        )
        self.pos_embed = params.pos_embed if hasattr(params, "pos_embed") else pos_embed
        self.big_skip = params.big_skip if hasattr(params, "big_skip") else big_skip
        self.rank = params.rank if hasattr(params, "rank") else rank
        self.factorization = (
            params.factorization if hasattr(params, "factorization") else factorization
        )
        self.separable = params.separable if hasattr(params, "separable") else separable
        self.complex_network = (
            params.complex_network
            if hasattr(params, "complex_network")
            else complex_network
        )
        self.complex_activation = (
            params.complex_activation
            if hasattr(params, "complex_activation")
            else complex_activation
        )
        self.spectral_layers = (
            params.spectral_layers
            if hasattr(params, "spectral_layers")
            else spectral_layers
        )
        self.checkpointing = (
            params.checkpointing if hasattr(params, "checkpointing") else checkpointing
        )
        # self.pretrain_encoding = params.pretrain_encoding if hasattr(params, "pretrain_encoding") else False

        # compute the downscaled image size
        self.h = int(self.img_shape[0] // self.scale_factor)
        self.w = int(self.img_shape[1] // self.scale_factor)

        # Compute the maximum frequencies in h and in w
        modes_lat = int(self.h * self.hard_thresholding_fraction)
        modes_lon = int((self.w // 2 + 1) * self.hard_thresholding_fraction)

        # determine the global padding
        img_dist_h = (self.img_shape[0] + comm.get_size("h") - 1) // comm.get_size("h")
        img_dist_w = (self.img_shape[1] + comm.get_size("w") - 1) // comm.get_size("w")
        self.padding = (
            img_dist_h * comm.get_size("h") - self.img_shape[0],
            img_dist_w * comm.get_size("w") - self.img_shape[1],
        )

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
                *self.img_shape, lmax=modes_lat, mmax=modes_lon, grid="equiangular"
            ).float()
            self.itrans_up = isht_handle(
                *self.img_shape, lmax=modes_lat, mmax=modes_lon, grid="equiangular"
            ).float()
            self.trans = sht_handle(
                self.h, self.w, lmax=modes_lat, mmax=modes_lon, grid="legendre-gauss"
            ).float()
            self.itrans = isht_handle(
                self.h, self.w, lmax=modes_lat, mmax=modes_lon, grid="legendre-gauss"
            ).float()

        elif self.spectral_transform == "fft":
            fft_handle = th.RealFFT2
            ifft_handle = th.InverseRealFFT2

            # effective image size:
            self.img_shape_eff = [
                self.img_shape[0] + self.padding[0],
                self.img_shape[1] + self.padding[1],
            ]
            self.img_shape_loc = [
                self.img_shape_eff[0] // comm.get_size("h"),
                self.img_shape_eff[1] // comm.get_size("w"),
            ]

            if (comm.get_size("h") > 1) or (comm.get_size("w") > 1):
                fft_handle = DistributedRealFFT2
                ifft_handle = DistributedInverseRealFFT2

            self.trans_down = fft_handle(
                *self.img_shape_eff, lmax=modes_lat, mmax=modes_lon
            ).float()
            self.itrans_up = ifft_handle(
                *self.img_shape_eff, lmax=modes_lat, mmax=modes_lon
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
            self.img_shape_loc = (
                self.trans_down.nlat_local,
                self.trans_down.nlon_local,
            )
            self.img_shape_eff = [
                self.trans_down.nlat_local + self.trans_down.nlatpad_local,
                self.trans_down.nlon_local + self.trans_down.nlonpad_local,
            ]
            self.h_loc = self.itrans.nlat_local
            self.w_loc = self.itrans.nlon_local
        else:
            self.img_shape_loc = (self.trans_down.nlat, self.trans_down.nlon)
            self.img_shape_eff = [self.trans_down.nlat, self.trans_down.nlon]
            self.h_loc = self.itrans.nlat
            self.w_loc = self.itrans.nlon

        # determine activation function
        if self.activation_function == "relu":
            self.activation_function = nn.ReLU
        elif self.activation_function == "gelu":
            self.activation_function = nn.GELU
        elif self.activation_function == "silu":
            self.activation_function = nn.SiLU
        else:
            raise ValueError(f"Unknown activation function {self.activation_function}")

        # encoder
        encoder_hidden_dim = self.embed_dim
        current_dim = self.in_chans
        encoder_modules = []
        for i in range(self.encoder_layers):
            encoder_modules.append(
                nn.Conv2d(current_dim, encoder_hidden_dim, 1, bias=True)
            )
            encoder_modules.append(self.activation_function())
            current_dim = encoder_hidden_dim
        encoder_modules.append(nn.Conv2d(current_dim, self.embed_dim, 1, bias=False))
        self.encoder = nn.Sequential(*encoder_modules)

        # dropout
        self.pos_drop = nn.Dropout(p=drop_rate) if drop_rate > 0.0 else nn.Identity()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.num_layers)]

        # pick norm layer
        if self.normalization_layer == "layer_norm":
            norm_layer0 = partial(
                nn.LayerNorm,
                normalized_shape=(self.img_shape_loc[0], self.img_shape_loc[1]),
                eps=1e-6,
            )
            norm_layer1 = partial(
                nn.LayerNorm, normalized_shape=(self.h_loc, self.w_loc), eps=1e-6
            )
        elif self.normalization_layer == "instance_norm":
            if comm.get_size("spatial") > 1:
                norm_layer0 = partial(
                    DistributedInstanceNorm2d,
                    num_features=self.embed_dim,
                    eps=1e-6,
                    affine=True,
                )
            else:
                norm_layer0 = partial(
                    nn.InstanceNorm2d,
                    num_features=self.embed_dim,
                    eps=1e-6,
                    affine=True,
                    track_running_stats=False,
                )
            norm_layer1 = norm_layer0
        elif self.normalization_layer == "none":
            norm_layer0 = nn.Identity
            norm_layer1 = norm_layer0
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

            if first_layer:
                norm_layer = (norm_layer0, norm_layer1)
            elif last_layer:
                norm_layer = (norm_layer1, norm_layer0)
            else:
                norm_layer = (norm_layer1, norm_layer1)

            filter_type = self.filter_type

            operator_type = self.operator_type

            block = FourierNeuralOperatorBlock(
                forward_transform,
                inverse_transform,
                self.embed_dim,
                filter_type=filter_type,
                operator_type=operator_type,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                drop_path=dpr[i],
                act_layer=self.activation_function,
                norm_layer=norm_layer,
                sparsity_threshold=sparsity_threshold,
                use_complex_kernels=use_complex_kernels,
                inner_skip=inner_skip,
                outer_skip=outer_skip,
                use_mlp=self.use_mlp,
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
        decoder_hidden_dim = self.embed_dim
        current_dim = self.embed_dim + self.big_skip * self.in_chans
        decoder_modules = []
        for i in range(self.encoder_layers):
            decoder_modules.append(
                nn.Conv2d(current_dim, decoder_hidden_dim, 1, bias=True)
            )
            decoder_modules.append(self.activation_function())
            current_dim = decoder_hidden_dim
        decoder_modules.append(nn.Conv2d(current_dim, self.out_chans, 1, bias=False))
        self.decoder = nn.Sequential(*decoder_modules)

        # learned position embedding
        if self.pos_embed:
            # currently using deliberately a differently shape position embedding
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1, self.embed_dim, self.img_shape_loc[0], self.img_shape_loc[1]
                )
            )
            # self.pos_embed = nn.Parameter( torch.zeros(1, self.embed_dim, self.img_shape_eff[0], self.img_shape_eff[1]) )
            self.pos_embed.is_shared_mp = ["matmul"]
            trunc_normal_(self.pos_embed, std=0.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Helper routine for weight initialization"""
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, FusedLayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def _forward_features(self, x):

        for blk in self.blocks:
            if self.checkpointing >= 3:
                x = checkpoint(blk, x)
            else:
                x = blk(x)

        return x

    def forward(self, x):

        # save big skip
        if self.big_skip:
            residual = x

        if self.checkpointing >= 1:
            x = checkpoint(self.encoder, x)
        else:
            x = self.encoder(x)

        if hasattr(self, "pos_embed"):

            # old way of treating unequally shaped weights
            if self.img_shape_loc != self.img_shape_eff:
                xp = torch.zeros_like(x)
                xp[..., : self.img_shape_loc[0], : self.img_shape_loc[1]] = (
                    x[..., : self.img_shape_loc[0], : self.img_shape_loc[1]]
                    + self.pos_embed
                )
                x = xp
            else:
                x = x + self.pos_embed

        # maybe clean the padding jsut in case

        x = self.pos_drop(x)

        x = self._forward_features(x)

        if self.big_skip:
            x = torch.cat((x, residual), dim=1)

        if self.checkpointing >= 1:
            x = checkpoint(self.decoder, x)
        else:
            x = self.decoder(x)

        return x
