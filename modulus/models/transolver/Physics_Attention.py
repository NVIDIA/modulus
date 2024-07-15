# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import torch
import torch.nn as nn
from einops import rearrange


class Physics_Attention_Irregular_Mesh(nn.Module):
    "for irregular meshes in 1D, 2D or 3D space"

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head**-0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)

        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l_i in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l_i.weight)  # use a principled initialization
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        # B N C
        B, N, C = x.shape

        ### (1) Slice
        fx_mid = (
            self.in_project_fx(x)
            .reshape(B, N, self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )  # B H N C
        x_mid = (
            self.in_project_x(x)
            .reshape(B, N, self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )  # B H N C
        slice_weights = self.softmax(
            self.in_project_slice(x_mid) / self.temperature
        )  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / (
            (slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head)
        )

        ### (2) Attention among slice tokens
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)  # B H G D

        ### (3) Deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, "b h n d -> b n (h d)")
        return self.to_out(out_x)


class Physics_Attention_Structured_Mesh_2D(nn.Module):
    "for structured mesh in 2D space"

    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        slice_num=64,
        H=101,
        W=31,
        kernel=3,
    ):  # kernel=3):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head**-0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.H = H
        self.W = W

        self.in_project_x = nn.Conv2d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_fx = nn.Conv2d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l_i in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l_i.weight)  # use a principled initialization
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        # B N C
        B, N, C = x.shape
        x = (
            x.reshape(B, self.H, self.W, C)
            .contiguous()
            .permute(0, 3, 1, 2)
            .contiguous()
        )  # B C H W

        ### (1) Slice
        fx_mid = (
            self.in_project_fx(x)
            .permute(0, 2, 3, 1)
            .contiguous()
            .reshape(B, N, self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )  # B H N C
        x_mid = (
            self.in_project_x(x)
            .permute(0, 2, 3, 1)
            .contiguous()
            .reshape(B, N, self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )  # B H N G
        slice_weights = self.softmax(
            self.in_project_slice(x_mid) / torch.clamp(self.temperature, min=0.1, max=5)
        )  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / (
            (slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head)
        )

        ### (2) Attention among slice tokens
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)  # B H G D

        ### (3) Deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, "b h n d -> b n (h d)")
        return self.to_out(out_x)


class Physics_Attention_Structured_Mesh_3D(nn.Module):
    "for structured mesh in 3D space"

    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.0,
        slice_num=32,
        H=32,
        W=32,
        D=32,
        kernel=3,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head**-0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        self.H = H
        self.W = W
        self.D = D

        self.in_project_x = nn.Conv3d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_fx = nn.Conv3d(dim, inner_dim, kernel, 1, kernel // 2)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        for l_i in [self.in_project_slice]:
            torch.nn.init.orthogonal_(l_i.weight)  # use a principled initialization
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        # B N C
        B, N, C = x.shape
        x = (
            x.reshape(B, self.H, self.W, self.D, C)
            .contiguous()
            .permute(0, 4, 1, 2, 3)
            .contiguous()
        )  # B C H W

        ### (1) Slice
        fx_mid = (
            self.in_project_fx(x)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .reshape(B, N, self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )  # B H N C
        x_mid = (
            self.in_project_x(x)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .reshape(B, N, self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .contiguous()
        )  # B H N G
        slice_weights = self.softmax(
            self.in_project_slice(x_mid) / torch.clamp(self.temperature, min=0.1, max=5)
        )  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / (
            (slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head)
        )

        ### (2) Attention among slice tokens
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)  # B H G D

        ### (3) Deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, "b h n d -> b n (h d)")
        return self.to_out(out_x)
