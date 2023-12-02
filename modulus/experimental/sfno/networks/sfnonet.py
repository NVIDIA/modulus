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
from torch.utils.checkpoint import checkpoint
from torch.cuda import amp

# helpers
from modulus.experimental.sfno.networks.layers import DropPath, MLP, EncoderDecoder

# import global convolution and non-linear spectral layers
from modulus.experimental.sfno.networks.spectral_convolution import SpectralConv, FactorizedSpectralConv, SpectralAttention

# get spectral transforms from torch_harmonics
import torch_harmonics as th
import torch_harmonics.distributed as thd
# wrap fft, to unify interface to spectral transforms
from modulus.experimental.sfno.networks.layers import RealFFT2, InverseRealFFT2
from modulus.experimental.sfno.mpu.layers import DistributedRealFFT2, DistributedInverseRealFFT2, DistributedMLP, DistributedEncoderDecoder

# more distributed stuff
from modulus.experimental.sfno.utils import comm

# layer normalization
from modulus.experimental.sfno.mpu.mappings import scatter_to_parallel_region, gather_from_parallel_region
from modulus.experimental.sfno.mpu.layer_norm import DistributedInstanceNorm2d, DistributedLayerNorm


class SpectralFilterLayer(nn.Module):
    def __init__(
        self,
        forward_transform,
        inverse_transform,
        embed_dim,
        filter_type = "linear",
        operator_type = "diagonal",
        sparsity_threshold = 0.0,
        hidden_size_factor = 1,
        factorization = None,
        rank = 1.0,
        separable = False,
        complex_activation = "real",
        spectral_layers = 1,
        bias = False,
        drop_rate = 0.):
        super(SpectralFilterLayer, self).__init__()

        if filter_type == "non-linear":
            self.filter = SpectralAttention(forward_transform,
                                            inverse_transform,
                                            embed_dim,
                                            embed_dim,
                                            operator_type = operator_type,
                                            sparsity_threshold = sparsity_threshold,
                                            hidden_size_factor = hidden_size_factor,
                                            complex_activation = complex_activation,
                                            spectral_layers = spectral_layers,
                                            drop_rate = drop_rate,
                                            bias = bias)

        elif filter_type == "linear" and factorization is None:
            self.filter = SpectralConv(forward_transform,
                                       inverse_transform,
                                       embed_dim,
                                       embed_dim,
                                       operator_type = operator_type,
                                       separable = separable,
                                       bias = bias)

        elif filter_type == "linear" and factorization is not None:
            self.filter = FactorizedSpectralConv(forward_transform,
                                                 inverse_transform,
                                                 embed_dim,
                                                 embed_dim,
                                                 operator_type = operator_type,
                                                 rank = rank,
                                                 factorization = factorization,
                                                 separable = separable,
                                                 bias = bias)

        
        else:
            raise(NotImplementedError)

    def forward(self, x):
        return self.filter(x)


class FourierNeuralOperatorBlock(nn.Module):
    def __init__(
            self,
            forward_transform,
            inverse_transform,
            embed_dim,
            filter_type = "linear",
            operator_type = "diagonal",
            mlp_ratio = 2.,
            mlp_drop_rate = 0.,
            path_drop_rate = 0.,
            act_layer = nn.GELU,
            norm_layer = (nn.Identity, nn.Identity),
            sparsity_threshold = 0.0,
            rank = 1.0,
            factorization = None,
            separable = False,
            inner_skip = "linear",
            outer_skip = None,
            use_mlp = False,
            comm_feature_inp_name = None,
            comm_feature_hidden_name = None,
            complex_activation = "real",
            spectral_layers = 1,
            final_activation = False,
            bias = False,
            checkpointing = 0):
        super(FourierNeuralOperatorBlock, self).__init__()

        if comm.get_size("spatial") > 1:
            self.input_shape_loc = (forward_transform.nlat_local, forward_transform.nlon_local)
            self.output_shape_loc = (inverse_transform.nlat_local, inverse_transform.nlon_local)
        else:
            self.input_shape_loc = (forward_transform.nlat, forward_transform.nlon)
            self.output_shape_loc = (inverse_transform.nlat, inverse_transform.nlon)

        if inner_skip == "linear":
            self.inner_skip = nn.Conv2d(embed_dim, embed_dim, 1, 1, bias=False)
        elif inner_skip == "identity":
            self.inner_skip = nn.Identity()
        elif inner_skip == "none":
            pass
        else:
            raise ValueError(f"Unknown skip connection type {inner_skip}")

        # norm layer
        self.norm0 = norm_layer[0]()

        # convolution layer
        self.filter = SpectralFilterLayer(forward_transform,
                                          inverse_transform,
                                          embed_dim,
                                          filter_type,
                                          operator_type,
                                          sparsity_threshold,
                                          hidden_size_factor = mlp_ratio,
                                          factorization = factorization,
                                          rank = rank,
                                          separable = separable,
                                          complex_activation = complex_activation,
                                          spectral_layers = spectral_layers,
                                          bias = bias,
                                          drop_rate = path_drop_rate)

        if filter_type == "linear" or filter_type == "real linear":
            self.act_layer0 = act_layer()

        # norm layer
        self.norm1 = norm_layer[1]()
        
        if use_mlp == True:
            MLPH = DistributedMLP if (comm.get_size("matmul") > 1) else MLP
            mlp_hidden_dim = int(embed_dim * mlp_ratio)
            self.mlp = MLPH(in_features = embed_dim,
                            hidden_features = mlp_hidden_dim,
                            act_layer = act_layer,
                            drop_rate = mlp_drop_rate,
                            drop_type = "features",
                            comm_inp_name = comm_feature_inp_name,
                            comm_hidden_name = comm_feature_hidden_name,
                            checkpointing = checkpointing)

        # dropout
        self.drop_path = DropPath(path_drop_rate) if path_drop_rate > 0. else nn.Identity()

        if outer_skip == "linear":
            self.outer_skip = nn.Conv2d(embed_dim, embed_dim, 1, 1, bias=False)
        elif outer_skip == "identity":
            self.outer_skip = nn.Identity()
        elif outer_skip == "none":
            pass
        else:
            raise ValueError(f"Unknown skip connection type {outer_skip}")
        
        if final_activation:
            self.act_layer1 = act_layer()

    def forward(self, x):
        
        x_norm  = torch.zeros_like(x)
        x_norm[..., :self.input_shape_loc[0], :self.input_shape_loc[1]] = \
            self.norm0(x[..., :self.input_shape_loc[0], :self.input_shape_loc[1]])

        x, residual = self.filter(x_norm)
        
        if hasattr(self, 'inner_skip'):
            x = x + self.inner_skip(residual)

        if hasattr(self, 'act_layer0'):
            x = self.act_layer0(x)

        x_norm = torch.zeros_like(x)
        x_norm[..., :self.output_shape_loc[0], :self.output_shape_loc[1]] = \
            self.norm1(x[..., :self.output_shape_loc[0], :self.output_shape_loc[1]])
        x = x_norm

        if hasattr(self, 'mlp'):
            x = self.mlp(x)

        x = self.drop_path(x)

        if hasattr(self, 'outer_skip'):
            x = x + self.outer_skip(residual)

        if hasattr(self, "act_layer1"):
            x = self.act_layer1(x)

        return x


    # def forward(self, x):
    #     """Updated FNO block"""
        
    #     x, residual = self.filter(x)

    #     x_norm = torch.zeros_like(x)
    #     x_norm[..., :self.output_shape_loc[0], :self.output_shape_loc[1]] = \
    #         self.norm0(x[..., :self.output_shape_loc[0], :self.output_shape_loc[1]])
    #     x = x_norm

    #     if hasattr(self, "inner_skip"):
    #         x = x + self.inner_skip(residual)

    #     if hasattr(self, "act_layer0"):
    #         x = self.act_layer0(x)

    #     if hasattr(self, "mlp"):
    #         x = self.mlp(x)

    #     x_norm = torch.zeros_like(x)
    #     x_norm[..., :self.output_shape_loc[0], :self.output_shape_loc[1]] = \
    #         self.norm1(x[..., :self.output_shape_loc[0], :self.output_shape_loc[1]])
    #     x = x_norm

    #     x = self.drop_path(x)

    #     if hasattr(self, "outer_skip"):
    #         x = x + self.outer_skip(residual)

    #     if hasattr(self, "act_layer1"):
    #         x = self.act_layer1(x)

    #     return x

class SphericalFourierNeuralOperatorNet(nn.Module):
    """
    SFNO implementation as in Bonev et al.; Spherical Fourier Neural Operators: Learning Stable Dynamics on the Sphere
    """
    def __init__(
            self,
            spectral_transform = "sht",
            model_grid_type = "equiangular",
            sht_grid_type = "legendre-gauss",
            filter_type = "linear",
            operator_type = "dhconv",
            inp_shape = (721, 1440),
            out_shape = (721, 1440),
            scale_factor = 8,
            inp_chans = 2,
            out_chans = 2,
            embed_dim = 32,
            num_layers = 4,
            repeat_layers = 1,
            use_mlp = True,
            mlp_ratio = 2.,
            encoder_ratio = 1,
            decoder_ratio = 1,
            activation_function = "gelu",
            encoder_layers = 1,
            pos_embed = "none",
            pos_drop_rate = 0.,
            path_drop_rate = 0.,
            mlp_drop_rate = 0.,
            sparsity_threshold = 0.0,
            normalization_layer = "instance_norm",
            max_modes = None,
            hard_thresholding_fraction = 1.0,
            big_skip = True,
            rank = 1.0,
            factorization = None,
            separable = False,
            complex_activation = "real",
            spectral_layers = 3,
            bias = False,
            checkpointing = 0,
            **kwargs): 
        super(SphericalFourierNeuralOperatorNet, self).__init__()

        self.inp_shape = inp_shape
        self.out_shape = out_shape
        self.inp_chans = inp_chans
        self.out_chans = out_chans
        self.embed_dim = embed_dim
        self.repeat_layers = repeat_layers
        self.big_skip = big_skip
        self.checkpointing = checkpointing

        # compute the downscaled image size
        self.h = int(self.inp_shape[0] // scale_factor)
        self.w = int(self.inp_shape[1] // scale_factor)

        # Compute the maximum frequencies in h and in w
        if max_modes is not None:
            modes_lat, modes_lon = max_modes
        else:
            modes_lat = int(self.h * hard_thresholding_fraction)
            modes_lon = int((self.w // 2 + 1) * hard_thresholding_fraction)

        # prepare interpolation

        # prepare the spectral transforms
        if spectral_transform == "sht":
            sht_handle = th.RealSHT
            isht_handle = th.InverseRealSHT
            
            # parallelism
            if comm.get_size("spatial") > 1:
                polar_group = None if (comm.get_size("h") == 1) else comm.get_group("h")
                azimuth_group = None if (comm.get_size("w") == 1) else comm.get_group("w")
                thd.init(polar_group, azimuth_group)
                sht_handle = thd.DistributedRealSHT
                isht_handle = thd.DistributedInverseRealSHT
                
            # set up
            self.trans_down = sht_handle(*self.inp_shape, lmax=modes_lat, mmax=modes_lon, grid=model_grid_type).float()
            self.itrans_up  = isht_handle(*self.out_shape, lmax=modes_lat, mmax=modes_lon, grid=model_grid_type).float()
            self.trans      = sht_handle(self.h, self.w, lmax=modes_lat, mmax=modes_lon, grid=sht_grid_type).float()
            self.itrans     = isht_handle(self.h, self.w, lmax=modes_lat, mmax=modes_lon, grid=sht_grid_type).float()
 
        elif spectral_transform == "fft":
            fft_handle = RealFFT2
            ifft_handle = InverseRealFFT2

            # determine the global padding
            inp_dist_h = (self.inp_shape[0] + comm.get_size("h") - 1) // comm.get_size("h")
            inp_dist_w = (self.inp_shape[1] + comm.get_size("w") - 1) // comm.get_size("w")
            self.inp_padding = (inp_dist_h * comm.get_size("h") - self.inp_shape[0], inp_dist_w * comm.get_size("w") - self.inp_shape[1])
            out_dist_h = (self.out_shape[0] + comm.get_size("h") - 1) // comm.get_size("h")
            out_dist_w = (self.out_shape[1] + comm.get_size("w") - 1) // comm.get_size("w")
            self.out_padding = (out_dist_h * comm.get_size("h") - self.out_shape[0], out_dist_w * comm.get_size("w") - self.out_shape[1])

            # effective image size:
            self.inp_shape_eff = [ self.inp_shape[0] + self.inp_padding[0], self.inp_shape[1] + self.inp_padding[1] ]
            self.inp_shape_loc = [ self.inp_shape_eff[0] // comm.get_size("h"), self.inp_shape_eff[1] // comm.get_size("w")]
            self.out_shape_eff = [ self.out_shape[0] + self.out_padding[0], self.out_shape[1] + self.out_padding[1] ]
            self.out_shape_loc = [ self.out_shape_eff[0] // comm.get_size("h"), self.out_shape_eff[1] // comm.get_size("w")]

            if comm.get_size("spatial") > 1:
                fft_handle = DistributedRealFFT2
                ifft_handle = DistributedInverseRealFFT2
            
            self.trans_down = fft_handle(*self.inp_shape_eff, lmax=modes_lat, mmax=modes_lon).float()
            self.itrans_up  = ifft_handle(*self.out_shape_eff, lmax=modes_lat, mmax=modes_lon).float()
            self.trans      = fft_handle(self.h, self.w, lmax=modes_lat, mmax=modes_lon).float()
            self.itrans     = ifft_handle(self.h, self.w, lmax=modes_lat, mmax=modes_lon).float()
        else:
            raise(ValueError("Unknown spectral transform"))

        # use the SHT/FFT to compute the local, downscaled grid dimensions
        if comm.get_size("spatial") > 1:
            self.inp_shape_loc = (self.trans_down.nlat_local, self.trans_down.nlon_local)
            self.inp_shape_eff = [self.trans_down.nlat_local+self.trans_down.nlatpad_local, self.trans_down.nlon_local+self.trans_down.nlonpad_local]
            self.out_shape_loc = (self.itrans_up.nlat_local, self.itrans_up.nlon_local)
            self.h_loc = self.itrans.nlat_local
            self.w_loc = self.itrans.nlon_local
        else:
            self.inp_shape_loc = (self.trans_down.nlat, self.trans_down.nlon)
            self.inp_shape_eff = [self.trans_down.nlat, self.trans_down.nlon]
            self.out_shape_loc = (self.itrans_up.nlat, self.itrans_up.nlon)
            self.h_loc = self.itrans.nlat
            self.w_loc = self.itrans.nlon

        # determine activation function
        if activation_function == "relu":
            activation_function = nn.ReLU
        elif activation_function == "gelu":
            activation_function = nn.GELU
        elif activation_function == "silu":
            activation_function = nn.SiLU
        else:
            raise ValueError(f"Unknown activation function {activation_function}")

        # set up encoder
        if comm.get_size("matmul") > 1:
            self.encoder = DistributedEncoderDecoder(num_layers = encoder_layers,
                                                     input_dim = self.inp_chans,
                                                     output_dim = self.embed_dim,
                                                     hidden_dim = int(encoder_ratio * self.embed_dim),
                                                     act = activation_function,
                                                     comm_inp_name = "fin",
                                                     comm_out_name = "fout")
            fblock_mlp_inp_name = self.encoder.comm_out_name
            fblock_mlp_hidden_name = "fout" if (self.encoder.comm_out_name == "fin") else "fin"
        else:
            self.encoder = EncoderDecoder(num_layers = encoder_layers,
                                          input_dim = self.inp_chans,
                                          output_dim = self.embed_dim,
                                          hidden_dim = int(encoder_ratio  * self.embed_dim),
                                          act = activation_function)
            fblock_mlp_inp_name = "fin"
            fblock_mlp_hidden_name = "fout"

        # dropout
        self.pos_drop = nn.Dropout(p=pos_drop_rate) if pos_drop_rate > 0. else nn.Identity()
        dpr = [x.item() for x in torch.linspace(0, path_drop_rate, num_layers)]

        # pick norm layer
        if normalization_layer == "layer_norm":
            #if comm.get_size("spatial") > 1:
            norm_layer_inp =partial(DistributedLayerNorm,
                                    normalized_shape=(embed_dim),
                                    elementwise_affine=True,
                                    eps=1e-6)
            norm_layer_out = norm_layer_mid = norm_layer_inp
            #else:
            #    norm_layer_inp = partial(nn.LayerNorm, normalized_shape=(embed_dim, self.inp_shape_loc[0], self.inp_shape_loc[1]),
            #                             elementwise_affine=False,
            #                             eps=1e-6)
            #    norm_layer_mid = partial(nn.LayerNorm, normalized_shape=(embed_dim, self.h_loc, self.w_loc),
            #                             elementwise_affine=False,
            #                             eps=1e-6)
            #    norm_layer_out = partial(nn.LayerNorm, normalized_shape=(embed_dim, self.out_shape_loc[0], self.out_shape_loc[1]),
            #                             elementwise_affine=False,
            #                             eps=1e-6)
        elif normalization_layer == "instance_norm":
            if comm.get_size("spatial") > 1:
                norm_layer_inp = partial(DistributedInstanceNorm2d, num_features=embed_dim, eps=1e-6, affine=True)
            else:
                norm_layer_inp = partial(nn.InstanceNorm2d, num_features=embed_dim, eps=1e-6, affine=True, track_running_stats=False)
            norm_layer_out = norm_layer_mid = norm_layer_inp
        elif normalization_layer == "none":
            norm_layer_out = norm_layer_mid = norm_layer_inp = nn.Identity
        else:
            raise NotImplementedError(f"Error, normalization {normalization_layer} not implemented.")

        # FNO blocks
        self.blocks = nn.ModuleList([])
        for i in range(num_layers):

            first_layer = (i == 0)
            last_layer = (i == num_layers-1)

            forward_transform = self.trans_down if first_layer else self.trans
            inverse_transform = self.itrans_up if last_layer else self.itrans

            inner_skip = "linear"
            outer_skip = "identity"

            if first_layer:
                norm_layer = (norm_layer_inp, norm_layer_mid)
            elif last_layer:
                norm_layer = (norm_layer_mid, norm_layer_out)
            else:
                norm_layer = (norm_layer_mid, norm_layer_mid)


            block = FourierNeuralOperatorBlock(forward_transform,
                                               inverse_transform,
                                               embed_dim,
                                               filter_type = filter_type,
                                               operator_type = operator_type,
                                               mlp_ratio = mlp_ratio,
                                               mlp_drop_rate = mlp_drop_rate,
                                               path_drop_rate = dpr[i],
                                               act_layer = activation_function,
                                               norm_layer = norm_layer,
                                               sparsity_threshold = sparsity_threshold,
                                               inner_skip = inner_skip,
                                               outer_skip = outer_skip,
                                               use_mlp = use_mlp,
                                               comm_feature_inp_name = fblock_mlp_inp_name,
                                               comm_feature_hidden_name = fblock_mlp_hidden_name,
                                               rank = rank,
                                               factorization = factorization,
                                               separable = separable,
                                               complex_activation = complex_activation,
                                               spectral_layers = spectral_layers,
                                               bias = bias,
                                               checkpointing = checkpointing)

            self.blocks.append(block)

        # decoder takes the output of FNO blocks and the residual from the big skip connection
        if comm.get_size("matmul") > 1:
            comm_inp_name = fblock_mlp_inp_name
            comm_out_name = fblock_mlp_hidden_name
            self.decoder = DistributedEncoderDecoder(num_layers = encoder_layers,
                                                     input_dim = embed_dim,
                                                     output_dim = self.out_chans,
                                                     hidden_dim = int(decoder_ratio * embed_dim),
                                                     act = activation_function,
                                                     comm_inp_name = comm_inp_name,
                                                     comm_out_name = comm_out_name)
        else:
            self.decoder = EncoderDecoder(num_layers = encoder_layers,
                                          input_dim = embed_dim,
                                          output_dim = self.out_chans,
                                          hidden_dim = int(decoder_ratio  * embed_dim),
                                          act = activation_function)

        # output transform
        if self.big_skip:
            self.residual_transform = nn.Conv2d(self.out_chans, self.out_chans, 1, bias=False)


        # learned position embedding
        if pos_embed == "direct":
            # currently using deliberately a differently shape position embedding
            self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, self.inp_shape_loc[0], self.inp_shape_loc[1]))
            # information about how tensors are shared / sharded across ranks
            self.pos_embed.is_shared_mp = ["matmul"]
            self.pos_embed.sharded_dims_mp = [None, None, "h", "w"]
            self.pos_embed.type = "direct"
            with torch.no_grad():
                nn.init.trunc_normal_(self.pos_embed, std=.02)
        elif pos_embed == "frequency":
            if comm.get_size("spatial") > 1:
                lmax_loc = self.itrans_up.lmax_local
                mmax_loc = self.itrans_up.mmax_local
            else:
                lmax_loc = self.itrans_up.lmax
                mmax_loc = self.itrans_up.mmax

            rcoeffs = nn.Parameter(torch.tril(torch.randn(1, embed_dim, lmax_loc, mmax_loc), diagonal=0))
            ccoeffs = nn.Parameter(torch.tril(torch.randn(1, embed_dim, lmax_loc, mmax_loc-1), diagonal=-1))
            with torch.no_grad():
                nn.init.trunc_normal_(rcoeffs, std=.02)
                nn.init.trunc_normal_(ccoeffs, std=.02)
            self.pos_embed = nn.ParameterList([rcoeffs, ccoeffs])
            self.pos_embed.type = "frequency"


        elif pos_embed == "none" or pos_embed == "None" or pos_embed == None:
            pass
        else:
            raise ValueError("Unknown position embedding type")

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Helper routine for weight initialization"""
        if isinstance(m, DistributedMLP):
            nn.init.kaiming_normal_(m.fc1.weight, mode="fan_in", nonlinearity="relu")
            nn.init.kaiming_normal_(m.fc1.bias, mode="fan_in", nonlinearity="relu")
            nn.init.kaiming_normal_(m.fc2.weight, mode="fan_in", nonlinearity="=linear")
        elif isinstance(m, MLP) or isinstance(m, EncoderDecoder) or isinstance(m, DistributedEncoderDecoder):
            last = True
            for conv in reversed(m.fwd):
                if isinstance(conv, nn.Conv2d) and last:
                    nn.init.kaiming_normal_(conv.weight, mode="fan_in", nonlinearity="linear")
                elif isinstance(conv, nn.Conv2d):
                    nn.init.kaiming_normal_(conv.weight, mode="fan_in", nonlinearity="relu")
                    nn.init.kaiming_normal_(conv.bias, mode="fan_in", nonlinearity="relu")
                    last = False
        elif isinstance(m, FourierNeuralOperatorBlock):
            if hasattr(m, "inner_skip") and isinstance(m.inner_skip, nn.Conv2d):
                nn.init.kaiming_normal_(m.inner_skip.weight, mode="fan_in", nonlinearity="linear")
            if hasattr(m, "outer_skip") and isinstance(m.outer_skip, nn.Conv2d):
                nn.init.kaiming_normal_(m.outer_skip.weight, mode="fan_in", nonlinearity="linear")
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def _forward_features(self, x):
        for r in range(self.repeat_layers):
            for blk in self.blocks:
                if self.checkpointing >= 3:
                    x = checkpoint(blk, x)
                else:
                    x = blk(x)
                
        return x

    def forward(self, x):

        if comm.get_size("fin") > 1:
            x = scatter_to_parallel_region(x, "fin", 1)
        
        # save big skip
        if self.big_skip:
            # if output shape differs, use the spectral transforms to change resolution
            if self.out_shape != self.inp_shape:
                xtype = x.dtype
                # only take the predicted channels as residual
                residual = x[..., :self.out_chans, :, :].to(torch.float32)
                with amp.autocast(enabled=False):
                    residual = self.trans_down(residual)
                    residual = residual.contiguous()
                    residual = self.itrans_up(residual)
                    residual = residual.to(dtype=xtype)
            else:
                # only take the predicted channels
                residual = x[..., :self.out_chans, :, :].contiguous()

        if self.checkpointing >= 1:
            x = checkpoint(self.encoder, x)
        else:
            x = self.encoder(x)

        if hasattr(self, "pos_embed"):

            if self.pos_embed.type == "frequency":

                pos_embed = torch.stack([self.pos_embed[0], nn.functional.pad(self.pos_embed[1], (1,0), "constant", 0)], dim=-1)
                with amp.autocast(enabled=False):
                    pos_embed = self.itrans_up(torch.view_as_complex(pos_embed))
            else:
                pos_embed = self.pos_embed

            # old way of treating unequally shaped weights
            if self.pos_embed.type == "direct" and self.inp_shape_loc != self.inp_shape_eff:
                xp = torch.zeros_like(x)
                xp[..., :self.inp_shape_loc[0], :self.inp_shape_loc[1]] = x[..., :self.inp_shape_loc[0], :self.inp_shape_loc[1]] + pos_embed
                x = xp
            else:
                x = x + pos_embed

        # maybe clean the padding just in case
        x = self.pos_drop(x)

        # do the feature extraction
        x = self._forward_features(x)
        
        if self.checkpointing >= 1:
            x = checkpoint(self.decoder, x)
        else:
            x = self.decoder(x)
            
        if hasattr(self.decoder, "comm_out_name") and (comm.get_size(self.decoder.comm_out_name) > 1):
            x =	gather_from_parallel_region(x, self.decoder.comm_out_name, 1)

        if self.big_skip:
            x = x + self.residual_transform(residual)

        return x
