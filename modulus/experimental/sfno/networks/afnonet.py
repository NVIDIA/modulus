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
import torch.fft
from modulus.experimental.sfno.utils.img_utils import PeriodicPad2d

from modulus.experimental.sfno.networks.layers import DropPath


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AFNO2D(nn.Module):
    def __init__(self,
                 hidden_size,
                 num_blocks=8,
                 sparsity_threshold=0.01,
                 hard_thresholding_fraction=1,
                 hidden_size_factor=1):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def forward(self, x):
        bias = x

        dtype = x.dtype
        x = x.float()
        B, H, W, C = x.shape

        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        x = x.reshape(B, H, W // 2 + 1, self.num_blocks, self.block_size)

        o1_real = torch.zeros([B, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o1_imag = torch.zeros([B, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)


        total_modes = H // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        o1_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].real, self.w1[0]) - \
            torch.einsum('...bi,bio->...bo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].imag, self.w1[0]) + \
            torch.einsum('...bi,bio->...bo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].real, self.w1[1]) + \
            self.b1[1]
        )

        o2_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes]  = (
            torch.einsum('...bi,bio->...bo', o1_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[0]) - \
            torch.einsum('...bi,bio->...bo', o1_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[1]) + \
            self.b2[0]
        )

        o2_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes]  = (
            torch.einsum('...bi,bio->...bo', o1_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[0]) + \
            torch.einsum('...bi,bio->...bo', o1_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[1]) + \
            self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, H, W // 2 + 1, C)
        x = torch.fft.irfft2(x, s=(H, W), dim=(1,2), norm="ortho")
        x = x.type(dtype)

        return x + bias


class Block(nn.Module):
    def __init__(
            self,
            dim,
            mlp_ratio = 4.,
            drop = 0.,
            drop_path = 0.,
            act_layer = nn.GELU,
            norm_layer = nn.LayerNorm,
            double_skip = True,
            num_blocks = 8,
            sparsity_threshold = 0.01,
            hard_thresholding_fraction = 1.0
        ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.filter = AFNO2D(dim, num_blocks, sparsity_threshold, hard_thresholding_fraction) 
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        #self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.double_skip = double_skip

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.filter(x)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + residual
        return x

class PrecipNet(nn.Module):
    def __init__(self,
                 backbone,
                 patch_size = (16, 16),
                 inp_chans = 2,
                 out_chans = 2,
                 **kwargs):
        super().__init__()
        self.patch_size = patch_size
        self.inp_chans = inp_chans
        self.out_chans = out_chans
        self.backbone = backbone
        self.ppad = PeriodicPad2d(1)
        self.conv = nn.Conv2d(self.out_chans, self.out_chans, kernel_size=3, stride=1, padding=0, bias=True)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.backbone(x)
        x = self.ppad(x)
        x = self.conv(x)
        x = self.act(x)
        return x

class AdaptiveFourierNeuralOperatorNet(nn.Module):
    def __init__(
            self,
            inp_shape = (720, 1440),
            patch_size = (16, 16),
            inp_chans = 2,
            out_chans = 2,
            embed_dim = 768,
            num_layers = 12,
            mlp_ratio = 4.,
            drop_rate = 0.,
            drop_path_rate = 0.,
            num_blocks = 16,
            sparsity_threshold = 0.01,
            hard_thresholding_fraction = 1.0,
            **kwargs
        ):
        super(AdaptiveFourierNeuralOperatorNet, self).__init__()
        self.img_size = inp_shape
        self.patch_size = patch_size
        self.inp_chans = inp_chans
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=self.patch_size, inp_chans=self.inp_chans, embed_dim=self.embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]

        self.h = self.img_size[0] // self.patch_size[0]
        self.w = self.img_size[1] // self.patch_size[1]

        self.blocks = nn.ModuleList([
            Block(dim = self.embed_dim,
                  mlp_ratio = mlp_ratio,
                  drop = drop_rate,
                  drop_path = dpr[i],
                  norm_layer = norm_layer,
                  num_blocks = num_blocks,
                  sparsity_threshold = sparsity_threshold,
                  hard_thresholding_fraction = hard_thresholding_fraction) 
        for i in range(num_layers)])

        self.head = nn.Linear(self.embed_dim, self.out_chans*self.patch_size[0]*self.patch_size[1], bias=False)

        with torch.no_grad():
            nn.init.trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        x = x.reshape(B, self.h, self.w, self.embed_dim)
        for blk in self.blocks:
            x = blk(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        
        # rearrange
        b = x.shape[0]
        xv = x.view(b, self.h, self.w, self.patch_size[0], self.patch_size[1], -1)
        xvt = torch.permute(xv, (0, 5, 1, 3, 2, 4)).contiguous()
        x = xvt.view(b, -1, (self.h * self.patch_size[0]), (self.w * self.patch_size[1]))
        
        return x


class PatchEmbed(nn.Module):
    def __init__(self,
                 img_size = (224, 224),
                 patch_size = (16, 16),
                 inp_chans = 3,
                 embed_dim = 768):
        super().__init__()
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(inp_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
