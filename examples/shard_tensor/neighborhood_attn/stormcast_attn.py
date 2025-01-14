# modeled off of official FB implementation
# https://github.com/facebookresearch/DiT/blob/main/models.py#L16
# and Makani
from einops import rearrange
from enum import Enum
from typing import Optional, Tuple
from functools import partial

import math
import numpy as np

import torch.nn.functional as F
import torch
import torch.nn as nn

from natten.functional import na2d

class Attention(nn.Module):
    
    def __init__(
        self,
        dim,
        input_format="traditional",
        num_heads=8,
        qkv_bias=False,
        qk_norm=False,
        attn_drop_rate=0.0,
        proj_drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        attn_mask=None,
        activation_checkpointing: bool = False,
        attn_kernel: int = -1,
    ):
        super().__init__()
        self.activation_checkpointing = activation_checkpointing
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop_rate = attn_drop_rate

        self.proj = nn.Linear(dim, dim)
        self.attn_mask = attn_mask
        self.attn_kernel = attn_kernel

        if proj_drop_rate > 0:
            self.proj_drop = nn.Dropout(proj_drop_rate)
        else:
            self.proj_drop = nn.Identity()

    @torch.jit.ignore
    def checkpoint_forward_qkv(self, x):
        raise NotImplementedError("Activation checkpointing not implemented")
        # return checkpoint(self.qkv, x, use_reentrant=False)

    @torch.jit.ignore
    def checkpoint_forward_proj(self, x):
        raise NotImplementedError("Activation checkpointing not implemented")
        # return checkpoint(self.proj, x, use_reentrant=False)

    def forward(self, x, per_batch_attn_mask=None, latent_hw=None):

        if per_batch_attn_mask is not None:
            assert (
                self.attn_mask is None
            ), "Cannot pass per-batch attn mask with static attn mask"
            attn_mask = per_batch_attn_mask
        else:
            attn_mask = self.attn_mask
        B, N, C = x.shape
        if self.activation_checkpointing:
            qkv = self.checkpoint_forward_qkv(x)
        else:
            qkv = self.qkv(x)
            
        # For shard tensors, this becomes a local reshape:
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        # post shape is [3, B, num_heads, N, head_dim]
        q, k, v = qkv.unbind(0)

        # q shape is [B, num_heads, N, head_dim]
        q, k = self.q_norm(q), self.k_norm(k)

        # self.scale is already incorporated in the model
        if self.attn_kernel == -1:
            # Self-attn over whole sequence
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop_rate,
                attn_mask=attn_mask,
                scale=self.scale,
            )
            x = x.transpose(1, 2).reshape(B, N, C)
        else:
            # Windowed neighborhood self-attention
            assert (
                latent_hw is not None
            ), "NAT2D requires passing latent h, w dimensions to DiTBlock forward pass for attn reshape op"
            h, _ = latent_hw
            print(f"H is {h}")
            q, k, v = map(
                lambda x: rearrange(x, "b head (h w) c -> b h w head c", h=h), [q, k, v]
            )
            x = na2d(q, k, v, kernel_size=self.attn_kernel)
            x = rearrange(x, "b h w head c -> b (h w) (head c)")

        if self.activation_checkpointing:
            x = self.checkpoint_forward_proj(x)
        else:
            x = self.proj(x)

        x = self.proj_drop(x)
        return x
