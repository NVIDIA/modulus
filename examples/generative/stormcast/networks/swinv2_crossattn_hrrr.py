import logging
import math
from typing import Tuple, Optional, List, Union, Any, Type
from types import SimpleNamespace
from ruamel.yaml import YAML

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint, checkpoint_sequential

from timm.layers import DropPath, Mlp, ClassifierHead, to_2tuple, _assert


# Adapted from timm v0.9.2:
# https://github.com/huggingface/pytorch-image-models/blob/v0.9.2/timm/models/swin_transformer_v2_cr.py


def bchw_to_bhwc(x: torch.Tensor) -> torch.Tensor:
    """Permutes a tensor from the shape (B, C, H, W) to (B, H, W, C). """
    return x.permute(0, 2, 3, 1)


def bhwc_to_bchw(x: torch.Tensor) -> torch.Tensor:
    """Permutes a tensor from the shape (B, H, W, C) to (B, C, H, W). """
    return x.permute(0, 3, 1, 2)


def swin_from_yaml(fname, checkpoint_stages=False):
    yaml = YAML()
    with open(fname) as f:
        hparams = yaml.load(f)
    params = SimpleNamespace()
    for k,v in hparams.items():
        setattr(params, k, v)
    return swinv2net(params, checkpoint_stages=checkpoint_stages)


def swinv2crossnet(params, checkpoint_stages=False):
    return SwinTransformerV2Cr(
                  hrrr_img_size=params.hrrr_img_size,
                  era5_img_size=params.era5_img_size,
                  patch_size=params.patch_size,
                  depths = (params.depth,),   
                  num_heads=(params.num_heads,),
                  in_chans_hrrr=params.n_hrrr_channels,
                  in_chans_era5=params.n_era5_channels,
                  out_chans=params.n_hrrr_channels - 1 if params.add_zenith else params.n_hrrr_channels,
                  embed_dim=params.embed_dim,
                  img_window_ratio=params.window_ratio,
                  drop_path_rate=params.drop_path_rate,
                  full_pos_embed=params.full_pos_embed,
                  rel_pos=params.rel_pos,
                  mlp_ratio=params.mlp_ratio,
                  checkpoint_stages=checkpoint_stages,
                  boundary_padding_pixels=params.boundary_padding_pixels,
    )


def bchw_to_bhwc(x: torch.Tensor) -> torch.Tensor:
    """Permutes a tensor from the shape (B, C, H, W) to (B, H, W, C). """
    return x.permute(0, 2, 3, 1)


def bhwc_to_bchw(x: torch.Tensor) -> torch.Tensor:
    """Permutes a tensor from the shape (B, H, W, C) to (B, C, H, W). """
    return x.permute(0, 3, 1, 2)


def window_partition(x, window_size: Tuple[int, int]):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0], window_size[1], C)
    return windows



def window_reverse(windows, window_size: Tuple[int, int], img_size: Tuple[int, int]):
    """
    Args:
        windows: (num_windows * B, window_size[0], window_size[1], C)
        window_size (Tuple[int, int]): Window size
        img_size (Tuple[int, int]): Image size

    Returns:
        x: (B, H, W, C)
    """
    H, W = img_size
    C = windows.shape[-1]
    x = windows.view(-1, H // window_size[0], W // window_size[1], window_size[0], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H, W, C)
    return x


class WindowMultiHeadAttentionNoPos(nn.Module):
    r"""This class implements window-based Multi-Head-Attention with log-spaced continuous position bias.

    Args:
        dim (int): Number of input features
        window_size (int): Window size
        num_heads (int): Number of attention heads
        drop_attn (float): Dropout rate of attention map
        drop_proj (float): Dropout rate after projection
        meta_hidden_dim (int): Number of hidden features in the two layer MLP meta network
        sequential_attn (bool): If true sequential self-attention is performed
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int],
        drop_attn: float = 0.0,
        drop_proj: float = 0.0,
        sequential_attn: bool = False,
    ) -> None:
        super(WindowMultiHeadAttentionNoPos, self).__init__()
        assert dim % num_heads == 0, \
            "The number of input features (in_features) are not divisible by the number of heads (num_heads)."
        self.in_features: int = dim
        self.window_size: Tuple[int, int] = window_size
        self.num_heads: int = num_heads
        self.sequential_attn: bool = sequential_attn

#        self.qkv = nn.Linear(in_features=dim, out_features=dim * 3, bias=True)
        self.q = nn.Linear(in_features=dim, out_features=dim, bias=True)
        self.k = nn.Linear(in_features=dim, out_features=dim, bias=True)
        self.v = nn.Linear(in_features=dim, out_features=dim, bias=True)
        self.attn_drop = nn.Dropout(drop_attn)
        self.proj = nn.Linear(in_features=dim, out_features=dim, bias=True)
        self.proj_drop = nn.Dropout(drop_proj)
        # NOTE old checkpoints used inverse of logit_scale ('tau') following the paper, see conversion fn
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones(num_heads)))


    def update_input_size(self, new_window_size: int, **kwargs: Any) -> None:
        """Method updates the window size and so the pair-wise relative positions

        Args:
            new_window_size (int): New window size
            kwargs (Any): Unused
        """
        # Set new window size and new pair-wise relative positions
        self.window_size: int = new_window_size

    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ Forward pass.
        Args:
            x (torch.Tensor): Input tensor of the shape (B * windows, N, C)
            mask (Optional[torch.Tensor]): Attention mask for the shift case

        Returns:
            Output tensor of the shape [B * windows, N, C]
        """
        Bw, L, C = x.shape

#        qkv = self.qkv(x).view(Bw, L, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        if y is not None:
            y = torch.cat([x, y], dim=1) # y is hrrr + boundary
            _, Ly, _ = y.shape
            query = self.q(x).view(Bw, L, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).squeeze(2)
            key = self.k(y).view(Bw, Ly, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).squeeze(2)
            value = self.v(y).view(Bw, Ly, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).squeeze(2)
        else:
            Ly = L
            query = self.q(x).view(Bw, L, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).squeeze(2)
            key = self.k(x).view(Bw, L, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).squeeze(2)
            value = self.v(x).view(Bw, L, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).squeeze(2)
#        query, key, value = qkv.unbind(0)

        # compute attention map with scaled cosine attention
        attn = (F.normalize(query, dim=-1) @ F.normalize(key, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale.reshape(1, self.num_heads, 1, 1), max=math.log(1. / 0.01)).exp()
        attn = attn * logit_scale

        if mask is not None:
            # Apply mask if utilized
            num_win: int = mask.shape[0]
            attn = attn.view(Bw // num_win, num_win, self.num_heads, L, Ly)
            if Ly > L: # cross attention with global tokens, only mask out local
                attn[..., :L] = attn[..., :L] + mask.unsqueeze(1).unsqueeze(0)
            else:
                attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, L, Ly)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ value).transpose(1, 2).reshape(Bw, L, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class WindowMultiHeadAttention(nn.Module):
    r"""This class implements window-based Multi-Head-Attention with log-spaced continuous position bias.

    Args:
        dim (int): Number of input features
        window_size (int): Window size
        num_heads (int): Number of attention heads
        drop_attn (float): Dropout rate of attention map
        drop_proj (float): Dropout rate after projection
        meta_hidden_dim (int): Number of hidden features in the two layer MLP meta network
        sequential_attn (bool): If true sequential self-attention is performed
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int],
        drop_attn: float = 0.0,
        drop_proj: float = 0.0,
        meta_hidden_dim: int = 384,  # FIXME what's the optimal value?
        sequential_attn: bool = False,
    ) -> None:
        super(WindowMultiHeadAttention, self).__init__()
        assert dim % num_heads == 0, \
            "The number of input features (in_features) are not divisible by the number of heads (num_heads)."
        self.in_features: int = dim
        self.window_size: Tuple[int, int] = window_size
        self.num_heads: int = num_heads
        self.sequential_attn: bool = sequential_attn

        self.qkv = nn.Linear(in_features=dim, out_features=dim * 3, bias=True)
        self.attn_drop = nn.Dropout(drop_attn)
        self.proj = nn.Linear(in_features=dim, out_features=dim, bias=True)
        self.proj_drop = nn.Dropout(drop_proj)
        # meta network for positional encodings
        self.meta_mlp = Mlp(
            2,  # x, y
            hidden_features=meta_hidden_dim,
            out_features=num_heads,
            act_layer=nn.ReLU,
            drop=(0.125, 0.)  # FIXME should there be stochasticity, appears to 'overfit' without?
        )
        # NOTE old checkpoints used inverse of logit_scale ('tau') following the paper, see conversion fn
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones(num_heads)))
        self._make_pair_wise_relative_positions()

    def _make_pair_wise_relative_positions(self) -> None:
        """Method initializes the pair-wise relative positions to compute the positional biases."""
        device = self.logit_scale.device
        coordinates = torch.stack(torch.meshgrid([
            torch.arange(self.window_size[0], device=device),
            torch.arange(self.window_size[1], device=device)]), dim=0).flatten(1)
        relative_coordinates = coordinates[:, :, None] - coordinates[:, None, :]
        relative_coordinates = relative_coordinates.permute(1, 2, 0).reshape(-1, 2).float()
        relative_coordinates_log = torch.sign(relative_coordinates) * torch.log(
            1.0 + relative_coordinates.abs())
        self.register_buffer("relative_coordinates_log", relative_coordinates_log, persistent=False)

    def update_input_size(self, new_window_size: int, **kwargs: Any) -> None:
        """Method updates the window size and so the pair-wise relative positions

        Args:
            new_window_size (int): New window size
            kwargs (Any): Unused
        """
        # Set new window size and new pair-wise relative positions
        self.window_size: int = new_window_size
        self._make_pair_wise_relative_positions()

    def _relative_positional_encodings(self) -> torch.Tensor:
        """Method computes the relative positional encodings

        Returns:
            relative_position_bias (torch.Tensor): Relative positional encodings
            (1, number of heads, window size ** 2, window size ** 2)
        """
        window_area = self.window_size[0] * self.window_size[1]
        relative_position_bias = self.meta_mlp(self.relative_coordinates_log)
        relative_position_bias = relative_position_bias.transpose(1, 0).reshape(
            self.num_heads, window_area, window_area
        )
        relative_position_bias = relative_position_bias.unsqueeze(0)
        return relative_position_bias

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ Forward pass.
        Args:
            x (torch.Tensor): Input tensor of the shape (B * windows, N, C)
            mask (Optional[torch.Tensor]): Attention mask for the shift case

        Returns:
            Output tensor of the shape [B * windows, N, C]
        """
        Bw, L, C = x.shape

        qkv = self.qkv(x).view(Bw, L, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)

        # compute attention map with scaled cosine attention
        attn = (F.normalize(query, dim=-1) @ F.normalize(key, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale.reshape(1, self.num_heads, 1, 1), max=math.log(1. / 0.01)).exp()
        attn = attn * logit_scale
        attn = attn + self._relative_positional_encodings()

        if mask is not None:
            # Apply mask if utilized
            num_win: int = mask.shape[0]
            attn = attn.view(Bw // num_win, num_win, self.num_heads, L, L)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, L, L)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ value).transpose(1, 2).reshape(Bw, L, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerV2CrBlock(nn.Module):
    r"""This class implements the Swin transformer block.

    Args:
        dim (int): Number of input channels
        num_heads (int): Number of attention heads to be utilized
        feat_size (Tuple[int, int]): Input resolution
        window_size (Tuple[int, int]): Window size to be utilized
        shift_size (int): Shifting size to be used
        mlp_ratio (int): Ratio of the hidden dimension in the FFN to the input channels
        proj_drop (float): Dropout in input mapping
        drop_attn (float): Dropout rate of attention map
        drop_path (float): Dropout in main path
        extra_norm (bool): Insert extra norm on 'main' branch if True
        sequential_attn (bool): If true sequential self-attention is performed
        norm_layer (Type[nn.Module]): Type of normalization layer to be utilized
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        feat_size: Tuple[int, int],
        window_size: Tuple[int, int],
        shift_size: Tuple[int, int] = (0, 0),
        mlp_ratio: float = 4.0,
        init_values: Optional[float] = 0,
        proj_drop: float = 0.0,
        drop_attn: float = 0.0,
        drop_path: float = 0.0,
        extra_norm: bool = False,
        sequential_attn: bool = False,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        rel_pos: bool = True,
    ) -> None:
        super(SwinTransformerV2CrBlock, self).__init__()
        self.dim: int = dim
        self.feat_size: Tuple[int, int] = feat_size
        self.target_shift_size: Tuple[int, int] = to_2tuple(shift_size)
        self.window_size, self.shift_size = self._calc_window_shift(to_2tuple(window_size))
        self.window_area = self.window_size[0] * self.window_size[1]
        self.init_values: Optional[float] = init_values
        window_attn_block = WindowMultiHeadAttention if rel_pos else WindowMultiHeadAttentionNoPos

        # attn branch
        self.attn = window_attn_block(
            dim=dim,
            num_heads=num_heads,
            window_size=self.window_size,
            drop_attn=drop_attn,
            drop_proj=proj_drop,
            sequential_attn=sequential_attn,
        )
        self.norm1 = norm_layer(dim)
        self.drop_path1 = DropPath(drop_prob=drop_path) if drop_path > 0.0 else nn.Identity()

        # mlp branch
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            drop=proj_drop,
            out_features=dim,
        )
        self.norm2 = norm_layer(dim)
        self.drop_path2 = DropPath(drop_prob=drop_path) if drop_path > 0.0 else nn.Identity()

        # Extra main branch norm layer mentioned for Huge/Giant models in V2 paper.
        # Also being used as final network norm and optional stage ending norm while still in a C-last format.
        # Dropped for ERA5
        self.norm3 = nn.Identity()

        self._make_attention_mask()
        self.init_weights()

    def _calc_window_shift(self, target_window_size):
        window_size = [f if f <= w else w for f, w in zip(self.feat_size, target_window_size)]
        shift_size = [0 if f <= w else s for f, w, s in zip(self.feat_size, window_size, self.target_shift_size)]
        return tuple(window_size), tuple(shift_size)

    def _make_attention_mask(self) -> None:
        """Method generates the attention mask used in shift case."""
        # Make masks for shift case
        
        if any(self.shift_size):
            # calculate attention mask for SW-MSA
            H, W = self.feat_size
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            cnt = 0
            for h in (
                    slice(0, -self.window_size[0]),
                    slice(-self.window_size[0], -self.shift_size[0]),
                    slice(-self.shift_size[0], None)):
                for w in (
                        slice(0, -self.window_size[1]),
                        slice(-self.window_size[1], -self.shift_size[1]),
                        slice(-self.shift_size[1], None)):
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)  # num_windows, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_area)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        
        self.register_buffer("attn_mask", attn_mask, persistent=False)

    def init_weights(self):
        # extra, module specific weight init
        if self.init_values is not None:
            nn.init.constant_(self.norm1.weight, self.init_values)
            nn.init.constant_(self.norm2.weight, self.init_values)

    def update_input_size(self, new_window_size: Tuple[int, int], new_feat_size: Tuple[int, int]) -> None:
        """Method updates the image resolution to be processed and window size and so the pair-wise relative positions.

        Args:
            new_window_size (int): New window size
            new_feat_size (Tuple[int, int]): New input resolution
        """
        # Update input resolution
        self.feat_size: Tuple[int, int] = new_feat_size
        self.window_size, self.shift_size = self._calc_window_shift(to_2tuple(new_window_size))
        self.window_area = self.window_size[0] * self.window_size[1]
        self.attn.update_input_size(new_window_size=self.window_size)
        self._make_attention_mask()

    def _shifted_window_attn(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        B, H, W, C = x.shape
        # cyclic shift
        sh, sw = self.shift_size
        do_shift: bool = any(self.shift_size)
        if do_shift:
            x = torch.roll(x, shifts=(-sh, -sw), dims=(1, 2))
            if y is not None:
                y = torch.roll(y, shifts=(-sh, -sw), dims=(1, 2))

        # partition windows
        x_windows = window_partition(x, self.window_size)  # num_windows * B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size[0] * self.window_size[1], C)
        n_windows = H // self.window_size[0] * W // self.window_size[1]
        # assume y is full coarse era5 and hence no windows
        if y is not None:
            b, h, w, c = y.shape
            y_windows = y.unsqueeze(1) 
            y_windows = y_windows.expand((b, n_windows, h, w, c)) # replicate y over all windows
            y_windows = y_windows.view(-1, h * w, c)
        else:
            y_windows = None

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, y_windows, mask=self.attn_mask)  # num_windows * B, window_size * window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
        x = window_reverse(attn_windows, self.window_size, self.feat_size)  # B H' W' C

        # reverse cyclic shift
        if do_shift:
            x = torch.roll(x, shifts=(sh, sw), dims=(1, 2))
            if y is not None:
                y = torch.roll(y, shifts=(sh, sw), dims=(1, 2))

        return x

    def forward(self, xlist):
        """Forward pass.

        Args:
            xlist (x, y) interior and boundary

        Returns:
            output (torch.Tensor): Output tensor of the shape [B, C, H, W]
        """
        x = xlist[0]
        y = xlist[1]
        x = self._shifted_window_attn(x, y)
        x = x + self.drop_path1(self.norm1(x))
        B, H, W, C = x.shape
        x = x.reshape(B, -1, C)
        x = x + self.drop_path2(self.norm2(self.mlp(x)))
        x = self.norm3(x)  # main-branch norm enabled for some blocks / stages (every 6 for Huge/Giant)
        x = x.reshape(B, H, W, C)
        return x


class PatchMerging(nn.Module):
    """ This class implements the patch merging as a strided convolution with a normalization before.
    Args:
        dim (int): Number of input channels
        norm_layer (Type[nn.Module]): Type of normalization layer to be utilized.
    """

    def __init__(self, dim: int, norm_layer: Type[nn.Module] = nn.LayerNorm) -> None:
        super(PatchMerging, self).__init__()
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(in_features=4 * dim, out_features=2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass.
        Args:
            x (torch.Tensor): Input tensor of the shape [B, C, H, W]
        Returns:
            output (torch.Tensor): Output tensor of the shape [B, 2 * C, H // 2, W // 2]
        """
        B, H, W, C = x.shape
        x = x.reshape(B, H // 2, 2, W // 2, 2, C).permute(0, 1, 3, 4, 2, 5).flatten(3)
        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        x = self.proj(x)
        x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return x


class SwinTransformerV2CrStage(nn.Module):
    r"""This class implements a stage of the Swin transformer including multiple layers.

    Args:
        embed_dim (int): Number of input channels
        depth (int): Depth of the stage (number of layers)
        downscale (bool): If true input is downsampled (see Fig. 3 or V1 paper)
        feat_size (Tuple[int, int]): input feature map size (H, W)
        num_heads (int): Number of attention heads to be utilized
        window_size (int): Window size to be utilized
        mlp_ratio (int): Ratio of the hidden dimension in the FFN to the input channels
        proj_drop (float): Dropout in input mapping
        drop_attn (float): Dropout rate of attention map
        drop_path (float): Dropout in main path
        norm_layer (Type[nn.Module]): Type of normalization layer to be utilized. Default: nn.LayerNorm
        extra_norm_period (int): Insert extra norm layer on main branch every N (period) blocks
        extra_norm_stage (bool): End each stage with an extra norm layer in main branch
        sequential_attn (bool): If true sequential self-attention is performed
    """

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        downscale: bool,
        num_heads: int,
        feat_size: Tuple[int, int],
        window_size: Tuple[int, int],
        mlp_ratio: float = 4.0,
        init_values: Optional[float] = 0.0,
        proj_drop: float = 0.0,
        drop_attn: float = 0.0,
        drop_path: Union[List[float], float] = 0.0,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        extra_norm_period: int = 0,
        extra_norm_stage: bool = False,
        sequential_attn: bool = False,
        rel_pos: bool = True,
    ) -> None:
        super(SwinTransformerV2CrStage, self).__init__()
        self.downscale: bool = downscale
        self.grad_checkpointing: bool = False
        self.feat_size: Tuple[int, int] = (feat_size[0] // 2, feat_size[1] // 2) if downscale else feat_size

        if downscale:
            self.downsample = PatchMerging(embed_dim, norm_layer=norm_layer)
            embed_dim = embed_dim * 2
        else:
            self.downsample = nn.Identity()

        def _extra_norm(index):
            i = index + 1
            if extra_norm_period and i % extra_norm_period == 0:
                return True
            return i == depth if extra_norm_stage else False

        self.blocks = nn.Sequential(*[
            SwinTransformerV2CrBlock(
                dim=embed_dim,
                num_heads=num_heads,
                feat_size=self.feat_size,
                window_size=window_size,
                shift_size=tuple([0 if ((index % 2) == 0) else w // 2 for w in window_size]),
                mlp_ratio=mlp_ratio,
                init_values=init_values,
                proj_drop=proj_drop,
                drop_attn=drop_attn,
                drop_path=drop_path[index] if isinstance(drop_path, list) else drop_path,
                extra_norm=_extra_norm(index),
                sequential_attn=sequential_attn,
                norm_layer=norm_layer,
                rel_pos=rel_pos,
            )
            for index in range(depth)]
        )

    def update_input_size(self, new_window_size: int, new_feat_size: Tuple[int, int]) -> None:
        """Method updates the resolution to utilize and the window size and so the pair-wise relative positions.

        Args:
            new_window_size (int): New window size
            new_feat_size (Tuple[int, int]): New input resolution
        """
        self.feat_size: Tuple[int, int] = (
            (new_feat_size[0] // 2, new_feat_size[1] // 2) if self.downscale else new_feat_size
        )
        for block in self.blocks:
            block.update_input_size(new_window_size=new_window_size, new_feat_size=self.feat_size)

    def forward(self, xlist):
        """Forward pass.
        Args:
        xlist that contains
            x (torch.Tensor): Input regional tensor of the shape [B, C, H, W] or [B, L, C]
            y (torch.Tensor): Input boundary tensor of the shape [B, C, H, W] or [B, L, C]
        Returns:
            output (torch.Tensor): Output tensor of the shape [B, 2 * C, H // 2, W // 2]
        """
        x = xlist[0]
        y = xlist[1]
        x = bchw_to_bhwc(x)
        x = self.downsample(x)
        if y is not None:
            y = bchw_to_bhwc(y)
            y = self.downsample(y)
        for block in self.blocks:
            # Perform checkpointing if utilized
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint.checkpoint(block, [x,y])
            else:
                x = block([x,y])
        x = bhwc_to_bchw(x) 
        if y is not None:
           y = bhwc_to_bchw(y)
        return x

class SwinTransformerV2Cr(nn.Module):
    r""" Swin Transformer V2
        A PyTorch impl of : `Swin Transformer V2: Scaling Up Capacity and Resolution`  -
          https://arxiv.org/pdf/2111.09883

    Args:
        img_size: Input resolution.
        window_size: Window size. If None, img_size // window_div
        img_window_ratio: Window size to image size ratio.
        patch_size: Patch size.
        in_chans: Number of input channels.
        depths: Depth of the stage (number of layers).
        num_heads: Number of attention heads to be utilized.
        embed_dim: Patch embedding dimension.
        num_classes: Number of output classes.
        mlp_ratio:  Ratio of the hidden dimension in the FFN to the input channels.
        drop_rate: Dropout rate.
        proj_drop_rate: Projection dropout rate.
        attn_drop_rate: Dropout rate of attention map.
        drop_path_rate: Stochastic depth rate.
        norm_layer: Type of normalization layer to be utilized.
        extra_norm_period: Insert extra norm layer on main branch every N (period) blocks in stage
        extra_norm_stage: End each stage with an extra norm layer in main branch
        sequential_attn: If true sequential self-attention is performed.
    """

    def __init__(
        self,
        hrrr_img_size: Tuple[int, int] = (224, 224),
        era5_img_size: Tuple[int, int] = (224, 224),
        boundary_padding_pixels: int = 0,
        patch_size: int = 4,
        window_size: Optional[int] = None,
        img_window_ratio: int = 32,
        in_chans_hrrr: int = 5,
        out_chans: int = 5,
        in_chans_era5: int = 26,
        embed_dim: int = 96,
        depths: Tuple[int, ...] = (2, 2, 6, 2),
        num_heads: Tuple[int, ...] = (3, 6, 12, 24),
        mlp_ratio: float = 4.0,
        init_values: Optional[float] = 0.,
        drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        extra_norm_period: int = 0,
        extra_norm_stage: bool = False,
        sequential_attn: bool = False,
        global_pool: str = 'avg',
        weight_init='skip',
        full_pos_embed: bool = False,
        rel_pos: bool = True,
        checkpoint_stages: bool = False,
        **kwargs: Any
    ) -> None:
        super(SwinTransformerV2Cr, self).__init__()
        hrrr_img_size = to_2tuple(hrrr_img_size)
        era5_img_size = to_2tuple(era5_img_size)
        img_size = hrrr_img_size # this is the image size after patch merging etc.
        window_size = tuple([
            s // img_window_ratio for s in img_size]) if window_size is None else to_2tuple(window_size)

        self.patch_size: int = patch_size
        self.img_size: Tuple[int, int] = img_size
        self.window_size: int = window_size
        self.num_features: int = int(embed_dim)
        self.out_chans: int = out_chans
        self.feature_info = []
        self.full_pos_embed = full_pos_embed
        self.checkpoint_stages = checkpoint_stages
        self.depth = len(depths)
        self.boundary_padding_pixels = boundary_padding_pixels

        self.patch_embed_hrrr = PatchEmbed(
            img_size=hrrr_img_size,
            patch_size=patch_size,
            in_chans=in_chans_hrrr,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
        )

        self.patch_embed_era5 = PatchEmbed(
            img_size=era5_img_size,
            patch_size=patch_size,
            in_chans=in_chans_era5,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
        )
        patch_grid_size_era5: Tuple[int, int] = self.patch_embed_era5.grid_size
        patch_grid_size_hrrr: Tuple[int, int] = self.patch_embed_hrrr.grid_size
        patch_grid_size: Tuple[int, int] = patch_grid_size_hrrr

        dpr = [x.tolist() for x in torch.linspace(0, drop_path_rate, sum(depths)).split(depths)]
        stages = []
        in_dim = embed_dim
        in_scale = 1
        for stage_idx, (depth, num_heads) in enumerate(zip(depths, num_heads)):
            stages += [SwinTransformerV2CrStage(
                embed_dim=in_dim,
                depth=depth,
                downscale=False,
                feat_size=(
                    patch_grid_size[0] // in_scale,
                    patch_grid_size[1] // in_scale
                ),
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                init_values=init_values,
                proj_drop=proj_drop_rate,
                drop_attn=attn_drop_rate,
                drop_path=dpr[stage_idx],
                extra_norm_period=extra_norm_period,
                extra_norm_stage=extra_norm_stage or (stage_idx + 1) == len(depths),  # last stage ends w/ norm
                sequential_attn=sequential_attn,
                norm_layer=norm_layer,
                rel_pos=rel_pos,
            )]
            self.feature_info += [dict(num_chs=in_dim, reduction=4 * in_scale, module=f'stages.{stage_idx}')]

        self.stages = nn.Sequential(*stages)
        self.head = nn.Linear(embed_dim, self.out_chans*self.patch_size*self.patch_size, bias=False)

        if self.full_pos_embed:
            self.pos_embed_x = nn.Parameter(torch.randn(1, embed_dim, patch_grid_size_hrrr[0], patch_grid_size_hrrr[1]) * .02)
            self.pos_embed_y = nn.Parameter(torch.randn(1, embed_dim, patch_grid_size_era5[0], patch_grid_size_era5[1]) * .02)

        # current weight init skips custom init and uses pytorch layer defaults, seems to work well
        # FIXME more experiments needed
        if weight_init != 'skip':
            named_apply(init_weights, self)
    
#    def patch_merge(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
#        """ This function merges patches created from HRRR (x) and patches created from ERA5 (y). Improve this to implement spatial merging with differing numbers of patches for HRRR and ERA5. """
#
#        z = torch.cat((x, y), dim=1)
#
#        return z
#    
#    def patch_merge_with_boundary(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
#        """ This function merges patches created from HRRR (x) and patches created from ERA5 (y) when the boundary padding is not zero """
#
#        b, c, h, w = x.shape
#        B, C, H, W = y.shape
#
#        start_x = (H-h)//2
#        start_y = (W-w)//2
#
#        era5_interior = y[:, :, start_x:start_x+h, start_y:start_y+w].permute(0, 2, 3, 1) # B, H, W, C
#        era5_interior = self.merging_mlp(era5_interior).permute(0, 3, 1, 2) # B, C, H, W
#
#        y[:,:, start_x:start_x+h, start_y:start_y+w] = torch.cat( (era5_interior, x), dim=1)
#
#        return y

    def forward_features(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ''' x is hrrr/region and y is era5/boundary '''
        x = self.patch_embed_hrrr(x)
        y = self.patch_embed_era5(y)
        if self.full_pos_embed:
            x = x + self.pos_embed_x
            y = y + self.pos_embed_y
        
        if self.checkpoint_stages and not torch.jit.is_scripting():
            x = checkpoint(self.stages, [x,y], use_reentrant=False)
        else:
            x = self.stages([x,y])
        return x

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:
        B, _, h, w = x.shape
        x = bchw_to_bhwc(x)
        x = self.head(x)

        x = x.reshape(shape=(B, h, w, self.patch_size, self.patch_size, self.out_chans))
        x = torch.einsum("nhwpqc->nchpwq", x)
        x = x.reshape(shape=(B, self.out_chans, self.img_size[0], self.img_size[1]))
        if self.boundary_padding_pixels != 0:
            x = x[:,:, self.boundary_padding_pixels:-self.boundary_padding_pixels, self.boundary_padding_pixels:-self.boundary_padding_pixels]
        return x

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        z = self.forward_features(x, y)
        z = self.forward_head(z)
        return z

    def update_input_size(
            self,
            new_img_size: Optional[Tuple[int, int]] = None,
            new_window_size: Optional[int] = None,
            img_window_ratio: int = 32,
    ) -> None:
        """Method updates the image resolution to be processed and window size and so the pair-wise relative positions.

        Args:
            new_window_size (Optional[int]): New window size, if None based on new_img_size // window_div
            new_img_size (Optional[Tuple[int, int]]): New input resolution, if None current resolution is used
            img_window_ratio (int): divisor for calculating window size from image size
        """
        # Check parameters
        if new_img_size is None:
            new_img_size = self.img_size
        else:
            new_img_size = to_2tuple(new_img_size)
        if new_window_size is None:
            new_window_size = tuple([s // img_window_ratio for s in new_img_size])
        # Compute new patch resolution & update resolution of each stage
        new_patch_grid_size = (new_img_size[0] // self.patch_size, new_img_size[1] // self.patch_size)
        for index, stage in enumerate(self.stages):
            stage_scale = 2 ** max(index - 1, 0)
            stage.update_input_size(
                new_window_size=new_window_size,
                new_img_size=(new_patch_grid_size[0] // stage_scale, new_patch_grid_size[1] // stage_scale),
            )

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^patch_embed',  # stem and embed
            blocks=r'^stages\.(\d+)' if coarse else [
                (r'^stages\.(\d+).downsample', (0,)),
                (r'^stages\.(\d+)\.\w+\.(\d+)', None),
            ]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        for s in self.stages:
            s.grad_checkpointing = enable

    @torch.jit.ignore()
    def get_classifier(self) -> nn.Module:
        """Method returns the classification head of the model.
        Returns:
            head (nn.Module): Current classification head
        """
        return self.head.fc

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None) -> None:
        """Method results the classification head

        Args:
            num_classes (int): Number of classes to be predicted
            global_pool (str): Unused
        """
        self.num_classes = num_classes
        self.head.reset(num_classes, global_pool)


def init_weights(module: nn.Module, name: str = ''):
    # FIXME WIP determining if there's a better weight init
    if isinstance(module, nn.Linear):
        if 'qkv' in name:
            # treat the weights of Q, K, V separately
            val = math.sqrt(6. / float(module.weight.shape[0] // 3 + module.weight.shape[1]))
            nn.init.uniform_(module.weight, -val, val)
        elif 'head' in name:
            nn.init.zeros_(module.weight)
        else:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()

