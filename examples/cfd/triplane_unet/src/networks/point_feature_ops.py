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

import enum
from typing import List, Optional, Tuple, Union

import numpy as np

# TODO(akamenev): migration
# import open3d as o3d
# import open3d.ml.torch as ml3d
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float, Int
from torch import Tensor

from .base_model import BaseModule
from .neighbor_ops import NeighborMLPConvLayer, neighbor_radius_search
from .net_utils import MLP, PositionalEncoding


def grid_init(bb_max, bb_min, resolution):
    # Define grid points
    grid = torch.meshgrid(
        torch.linspace(bb_min[0], bb_max[0], resolution[0]),
        torch.linspace(bb_min[1], bb_max[1], resolution[1]),
        torch.linspace(bb_min[2], bb_max[2], resolution[2]),
    )
    grid = torch.stack(grid, dim=-1)  # (n_x, n_y, n_z, 3)
    return grid


class PointFeatures:
    _shape_hint = None  # default value
    vertices: Float[Tensor, "N 3"]
    features: Float[Tensor, "N C"]
    num_channels: int = None

    def __class_getitem__(cls, item: str):
        # Create a new subclass with the shape hint set
        class _PointFeaturesSubclass(cls):
            _shape_hint = tuple(item.split())

        return _PointFeaturesSubclass

    def __init__(self, vertices, features):
        self.vertices = vertices
        self.features = features
        self.check()

    @property
    def device(self):
        return self.vertices.device

    def check(self):
        assert self.vertices.shape[0] == self.features.shape[0]
        assert self.vertices.shape[1] == 3
        assert len(self.vertices.shape) == 2
        assert len(self.features.shape) == 2
        self.num_channels = self.features.shape[-1]

    def __len__(self):
        return self.vertices.shape[0]

    def to(self, device):
        self.vertices = self.vertices.to(device)
        self.features = self.features.to(device)
        return self

    def voxel_down_sample(self, voxel_size: float):
        int_coords = torch.floor((self.vertices) / voxel_size).int()
        # use unique to get the index of the in coord torch.unique(int_coords, dim=0, return_inverse=True)
        _, unique_indices = np.unique(
            int_coords.cpu().numpy(), axis=0, return_index=True
        )
        unique_indices = torch.from_numpy(unique_indices).to(self.vertices.device)
        down_vertices = self.vertices[unique_indices]
        down_features = self.features[unique_indices]
        return PointFeatures(down_vertices, down_features)

    def __add__(self, other):
        assert len(self) == len(other)
        assert self.features.shape[1] == other.features.shape[1]
        return PointFeatures(self.vertices, self.features + other.features)

    def __mul__(self, other):
        assert len(self) == len(other)
        assert self.features.shape[1] == other.features.shape[1]
        return PointFeatures(self.vertices, self.features * other.features)

    def __repr__(self) -> str:
        return f"PointFeatures(vertices={self.vertices.shape}, features={self.features.shape})"

    def to_grid_features(self, resolution: List[int]):
        return GridFeatures(
            vertices=self.vertices.reshape(*resolution, 3),
            features=self.features.reshape(*resolution, self.features.shape[-1]),
        )


class GridFeaturesMemoryFormat(enum.Enum):
    """Memory format for grid features."""

    x_y_z_c = enum.auto()
    c_x_y_z = enum.auto()

    # flattened 3D memory format
    zc_x_y = enum.auto()
    xc_y_z = enum.auto()
    yc_x_z = enum.auto()


grid_mem_format2str_format = {
    GridFeaturesMemoryFormat.x_y_z_c: "x_y_z_c",
    GridFeaturesMemoryFormat.c_x_y_z: "c_x_y_z",
    GridFeaturesMemoryFormat.zc_x_y: "zc_x_y",
    GridFeaturesMemoryFormat.xc_y_z: "xc_y_z",
    GridFeaturesMemoryFormat.yc_x_z: "yc_x_z",
}


def convert_to_x_y_z_c(tensor, from_memory_format, num_channels):
    if from_memory_format == GridFeaturesMemoryFormat.zc_x_y:
        D_C, H, W = tensor.shape
        D, rem = divmod(D_C, num_channels)
        assert rem == 0, "Number of channels does not match."
        return tensor.reshape(D, num_channels, H, W).permute(2, 3, 0, 1)
    elif from_memory_format == GridFeaturesMemoryFormat.xc_y_z:
        H_C, W, D = tensor.shape
        H, rem = divmod(H_C, num_channels)
        assert rem == 0, "Number of channels does not match."
        return tensor.reshape(H, num_channels, W, D).permute(0, 2, 3, 1)
    elif from_memory_format == GridFeaturesMemoryFormat.yc_x_z:
        W_C, H, D = tensor.shape
        W, rem = divmod(W_C, num_channels)
        assert rem == 0, "Number of channels does not match."
        return tensor.reshape(W, num_channels, H, D).permute(2, 0, 3, 1)
    elif from_memory_format == GridFeaturesMemoryFormat.c_x_y_z:
        return tensor.permute(1, 2, 3, 0)
    else:
        raise ValueError(f"Unsupported memory format: {from_memory_format}")


def convert_from_x_y_z_c(tensor, to_memory_format):
    H, W, D, C = tensor.shape
    if to_memory_format == GridFeaturesMemoryFormat.zc_x_y:
        return tensor.permute(2, 3, 0, 1).reshape(D * C, H, W)
    elif to_memory_format == GridFeaturesMemoryFormat.xc_y_z:
        return tensor.permute(0, 3, 1, 2).reshape(H * C, W, D)
    elif to_memory_format == GridFeaturesMemoryFormat.yc_x_z:
        return tensor.permute(1, 3, 0, 2).reshape(W * C, H, D)
    elif to_memory_format == GridFeaturesMemoryFormat.c_x_y_z:
        return tensor.permute(3, 0, 1, 2)
    else:
        raise ValueError(f"Unsupported memory format: {to_memory_format}")


class GridFeatures(PointFeatures):
    memory_format: GridFeaturesMemoryFormat

    def __init__(
        self,
        vertices,
        features,
        memory_format: GridFeaturesMemoryFormat = GridFeaturesMemoryFormat.x_y_z_c,
        grid_shape: Optional[Tuple[int, int, int]] = None,
        num_channels: Optional[int] = None,
    ) -> None:
        self.memory_format = memory_format
        super().__init__(vertices, features)
        if memory_format == GridFeaturesMemoryFormat.x_y_z_c:
            H, W, D, C = self.features.shape
            self.grid_shape = (H, W, D)
            self.num_channels = C
        else:
            assert grid_shape is not None
            assert num_channels is not None

            self.grid_shape = grid_shape
            self.num_channels = num_channels
            self.memory_format = memory_format

    @staticmethod
    def from_conv_output(
        conv_output, vertices, memory_format, grid_shape, num_channels
    ):
        """Initialize GridFeatures from the output of a convolutional layer.

        Args:
            conv_output: Output tensor from a convolutional layer.
            vertices: The vertices to associate with the grid features.
            memory_format: The memory format of the grid features.
            num_spatial_dims: Tuple of spatial dimensions (H, W, D).
            num_channels: The number of output channels from the convolution.

        Returns:
            GridFeatures: A new GridFeatures object with the given memory format.
        """
        # Infer spatial dimensions based on the memory format
        rem = 0
        if memory_format == GridFeaturesMemoryFormat.zc_x_y:
            DC, H, W = conv_output.shape
            D, rem = divmod(DC, num_channels)
            assert D == grid_shape[2], "Spatial dimension D does not match."
        elif memory_format == GridFeaturesMemoryFormat.xc_y_z:
            HC, W, D = conv_output.shape
            H, rem = divmod(HC, num_channels)
            assert H == grid_shape[0], "Spatial dimension H does not match."
        elif memory_format == GridFeaturesMemoryFormat.yc_x_z:
            WC, H, D = conv_output.shape
            W, rem = divmod(WC, num_channels)
            assert W == grid_shape[1], "Spatial dimension W does not match."
        elif memory_format == GridFeaturesMemoryFormat.c_x_y_z:
            C, H, W, D = conv_output.shape
            assert C == num_channels, "Number of channels does not match."
        else:
            raise ValueError("Unsupported memory format.")
        assert rem == 0, "Number of channels does not match."

        # The channel dimension is now the last dimension after reshaping.
        return GridFeatures(
            vertices=vertices,
            features=conv_output,
            memory_format=memory_format,
            grid_shape=grid_shape,
            num_channels=num_channels,
        )

    def channel_size(self, memory_format: GridFeaturesMemoryFormat = None):
        if memory_format is None:
            memory_format = self.memory_format
        if memory_format == GridFeaturesMemoryFormat.x_y_z_c:
            return self.num_channels
        elif memory_format == GridFeaturesMemoryFormat.c_x_y_z:
            return self.num_channels
        elif memory_format == GridFeaturesMemoryFormat.xc_y_z:
            return self.num_channels * self.grid_shape[0]
        elif memory_format == GridFeaturesMemoryFormat.yc_x_z:
            return self.num_channels * self.grid_shape[1]
        elif memory_format == GridFeaturesMemoryFormat.zc_x_y:
            return self.num_channels * self.grid_shape[2]

    def check(self):
        if self.memory_format == GridFeaturesMemoryFormat.x_y_z_c:
            assert self.vertices.ndim == 4
            assert self.features.ndim == 4
            assert self.vertices.shape[0] == self.features.shape[0]
            assert self.vertices.shape[1] == self.features.shape[1]
            assert self.vertices.shape[2] == self.features.shape[2]
            assert self.vertices.shape[3] == 3
        elif self.memory_format == GridFeaturesMemoryFormat.c_x_y_z:
            assert self.vertices.ndim == 4
            assert self.features.ndim == 4
            # strided outputs
            # assert self.vertices.shape[0] == self.features.shape[1]
            # assert self.vertices.shape[1] == self.features.shape[2]
            # assert self.vertices.shape[2] == self.features.shape[3]
            assert self.vertices.shape[3] == 3
        else:
            pass

    @property
    def batch_features(self) -> Float[Tensor, "B C H W D"]:
        if self.memory_format == GridFeaturesMemoryFormat.x_y_z_c:
            return self.features.permute(3, 0, 1, 2).unsqueeze(0)
        elif self.memory_format == GridFeaturesMemoryFormat.c_x_y_z:
            return self.features.unsqueeze(0)
        else:
            raise ValueError("Unsupported memory format.")

    @property
    def point_features(self) -> PointFeatures:
        if self.memory_format == GridFeaturesMemoryFormat.c_x_y_z:
            permuted_features = self.features.permute(1, 2, 3, 0)
            # TODO: merge with the code below for x_y_z_c
            return PointFeatures(
                self.vertices.reshape(-1, 3), permuted_features.flatten(0, 2)
            )

        assert self.memory_format == GridFeaturesMemoryFormat.x_y_z_c
        # TODO: provide better fix
        # Crop the features to the grid_shape
        grid_shape = self.grid_shape
        if (
            self.features.shape[0] > grid_shape[0]
            or self.features.shape[1] > grid_shape[1]
            or self.features.shape[2] > grid_shape[2]
        ):
            self.features = self.features[
                : grid_shape[0], : grid_shape[1], : grid_shape[2]
            ]

        return PointFeatures(
            self.vertices.reshape(-1, 3),
            self.features.reshape(-1, self.features.shape[-1]),
        )

    @property
    def resolution(self) -> Tuple[int, int, int]:
        return self.vertices.shape[:3]

    def to(
        self, device=None, memory_format: GridFeaturesMemoryFormat = None
    ):  # -> GridFeatures
        assert device is not None or memory_format is not None
        if device is not None:
            self.vertices = self.vertices.to(device)
            self.features = self.features.to(device)

        if memory_format is not None:
            # Step 1: Convert to x_y_z_c format if not already in that format
            if self.memory_format != GridFeaturesMemoryFormat.x_y_z_c:
                self.features = convert_to_x_y_z_c(
                    self.features, self.memory_format, self.num_channels
                )

            # Step 2: Convert to the desired memory format from x_y_z_c
            if memory_format != GridFeaturesMemoryFormat.x_y_z_c:
                self.features = convert_from_x_y_z_c(self.features, memory_format)

            self.memory_format = memory_format  # Update the memory format
        return self

    def __repr__(self) -> str:
        return f"GridFeatures(vertices={self.vertices.shape}, features={self.features.shape})"

    def __add__(self, other):
        assert len(self) == len(other)
        assert self.features.shape == other.features.shape
        return GridFeatures(
            self.vertices,
            self.features + other.features,
            self.memory_format,
            grid_shape=self.grid_shape,
            num_channels=self.num_channels,
        )

    def strided_vertices(self, resolution: Tuple[int, int, int]):
        if self.vertices.shape[:3] == resolution:
            return self.vertices

        # Compute the stride
        if (
            self.vertices.shape[0] % resolution[0] == 0
            and self.vertices.shape[1] % resolution[1] == 0
            and self.vertices.shape[2] % resolution[2] == 0
        ):
            stride = (
                self.vertices.shape[0] // resolution[0],
                self.vertices.shape[1] // resolution[1],
                self.vertices.shape[2] // resolution[2],
            )
            vertices = self.vertices[:: stride[0], :: stride[1], :: stride[2]]
        else:
            # Use grid_sample 3D to interpolate the vertices
            # vertices have shape (H, W, D, 3)
            # create sample points using AABB
            grid_points = grid_init(
                bb_max=(1, 1, 1), bb_min=(-1, -1, -1), resolution=resolution
            )  # (res[0], res[1], res[2], 3)
            grid_points = grid_points.to(self.vertices.device)

            sampled_vertices = F.grid_sample(
                self.vertices.permute(3, 0, 1, 2).unsqueeze(0),
                grid_points.unsqueeze(0),
                align_corners=True,
            )
            vertices = sampled_vertices.squeeze().permute(1, 2, 3, 0)

        return vertices


class DownSampleLayer(BaseModule):
    def __init__(self, voxel_size):
        super().__init__()
        self.voxel_size = voxel_size

    def __repr__(self):
        return f"DownSampleLayer(voxel_size={self.voxel_size})"

    @torch.no_grad()
    def forward(self, vertices: Float[Tensor, "N 3"]) -> Float[Tensor, "M 3"]:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices.cpu().numpy())
        down_pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        down_vertices = torch.Tensor(np.array(down_pcd.points)).to(self.device)
        return down_vertices


class VerticesToPointFeatures(BaseModule):
    def __init__(
        self,
        embed_dim: int,
        out_features: Optional[int] = 32,
        use_mlp: Optional[bool] = True,
        pos_embed_range: Optional[float] = 2.0,
    ) -> None:
        super().__init__()
        self.pos_embed = PositionalEncoding(embed_dim, pos_embed_range)
        self.use_mlp = use_mlp
        if self.use_mlp:
            self.mlp = MLP([3 * embed_dim, out_features], torch.nn.GELU)

    def forward(self, vertices: Float[Tensor, "N 3"]) -> PointFeatures:
        vertices = vertices.to(self.device)
        vert_embed = self.pos_embed(vertices)
        if self.use_mlp:
            vert_embed = self.mlp(vert_embed)
        return PointFeatures(vertices, vert_embed)


class ToGrid(nn.Module):
    def __init__(
        self,
        resolution: Union[Int[Tensor, "3"], List[int]],
        voxel_size: float,
        num_channels: int,
        bbox_buffer_ratio: float = 0,
    ):
        super().__init__()
        resolution
        if isinstance(resolution, (list, tuple)):
            resolution = torch.tensor(resolution)
        self.resolution_list = resolution.int().tolist()
        self.resolution = resolution
        self.voxel_size = voxel_size
        self.bbox_buffer_ratio = bbox_buffer_ratio

        # Extents: a scalar or a torch tensor of shape 1 or 3
        self.extents = 2.0

        # Create an instance of the ContinuousConv layer
        self.conv = ml3d.layers.ContinuousConv(
            in_channels=num_channels, filters=num_channels, kernel_size=[3, 3, 3]
        )

    def forward(self, point_features: PointFeatures) -> GridFeatures:
        device = point_features.vertices.device
        vertices = point_features.vertices
        features = point_features.features
        # Define grid points
        bb_max = vertices.max(0)[0].cpu() * (1 + self.bbox_buffer_ratio)
        bb_min = vertices.min(0)[0].cpu() * (1 + self.bbox_buffer_ratio)
        bb_center = (bb_max + bb_min) / 2.0
        grid_size = self.resolution * self.voxel_size
        grid_min = bb_center - grid_size / 2.0
        grid_max = bb_center + grid_size / 2.0
        # Define grid points using grid_min, grid_max
        grid = torch.meshgrid(
            torch.linspace(grid_min[0], grid_max[0], self.resolution_list[0]).to(
                device
            ),
            torch.linspace(grid_min[1], grid_max[1], self.resolution_list[1]).to(
                device
            ),
            torch.linspace(grid_min[2], grid_max[2], self.resolution_list[2]).to(
                device
            ),
        )
        grid = torch.stack(grid, dim=-1)  # (n_x, n_y, n_z, 3)

        # Interpolate PointFeatures features to grid using contconv
        out_features = self.conv(
            features,
            vertices / self.voxel_size,  # normalize to voxel grid space
            grid.view(-1, 3) / self.voxel_size,  # normalize to voxel grid space
            self.extents,
        )
        out_features = out_features.view(*self.resolution_list, -1)
        return GridFeatures(grid, out_features)


class ToGridWithDist(nn.Module):
    def __init__(
        self,
        resolution: Union[Int[Tensor, "3"], List[int]],
        embed_dim: int = 32,
        hidden_channel: int = 64,
        bbox_buffer_ratio: float = 0,
    ) -> None:
        super().__init__()
        if isinstance(resolution, Tensor):
            resolution = resolution.tolist()
        self.resolution = resolution
        self.embed_dim = embed_dim
        self.bbox_buffer_ratio = bbox_buffer_ratio
        self.pos_embed = PositionalEncoding(embed_dim)
        self.conv = NeighborMLPConvLayer(
            mlp=MLP([hidden_channel + 4 * embed_dim, hidden_channel], torch.nn.GELU)
        )

    def forward(self, point_features: PointFeatures) -> GridFeatures:
        device = point_features.vertices.device
        vertices = point_features.vertices
        # Define grid points
        bb_max = vertices.max(0)[0] * (1 + self.bbox_buffer_ratio)
        bb_min = vertices.min(0)[0] * (1 + self.bbox_buffer_ratio)
        if self.training:
            # Add noise to bounding box
            bb_max += torch.randn_like(bb_max) * 0.01
            bb_min += torch.randn_like(bb_min) * 0.01

        # Define grid points
        grid = torch.meshgrid(
            torch.linspace(bb_min[0], bb_max[0], self.resolution[0]).to(device),
            torch.linspace(bb_min[1], bb_max[1], self.resolution[1]).to(device),
            torch.linspace(bb_min[2], bb_max[2], self.resolution[2]).to(device),
        )
        grid = torch.stack(grid, dim=-1)  # (n_x, n_y, n_z, 3)

        # Open3d point cloud
        grid_pcd = o3d.geometry.PointCloud()
        grid_pcd.points = o3d.utility.Vector3dVector(grid.view(-1, 3).cpu().numpy())

        # Compute distance to point cloud
        vertices_pcd = o3d.geometry.PointCloud()
        vertices_pcd.points = o3d.utility.Vector3dVector(vertices.cpu().numpy())

        vertices = vertices.to(device)
        grid = grid.to(device)

        # Compute distance from grid_pcd to vertices_pcd
        dists = torch.tensor(grid_pcd.compute_point_cloud_distance(vertices_pcd))
        dists = dists.to(device)
        # concatenate distance to grid points
        grid_coord_dists = torch.cat(
            (
                grid,
                dists.view(
                    self.resolution[0], self.resolution[1], self.resolution[2], 1
                ),
            ),
            dim=-1,
        )
        vertices_to_grid_nb = neighbor_radius_search(
            vertices, grid.view(-1, 3), radius=2 / self.resolution[0]
        )

        grid_coord_dist_embed = self.pos_embed(grid_coord_dists)
        grid_out = self.conv(
            point_features.features,
            vertices_to_grid_nb,
            grid_coord_dist_embed.view(-1, grid_coord_dist_embed.shape[-1]),
        )
        return GridFeatures(grid, grid_out.view(*self.resolution, -1))


class FromGrid(nn.Module):
    def __init__(self, voxel_size, num_channels):
        super().__init__()
        self.num_channels = num_channels
        self.voxel_size = voxel_size

        # Extents: a scalar or a torch tensor of shape 1 or 3
        self.extents = 2.0

        # Create an instance of the ContinuousConv layer
        self.conv = ml3d.layers.ContinuousConv(
            in_channels=num_channels, filters=num_channels, kernel_size=[3, 3, 3]
        )

    def forward(self, grid_features: PointFeatures, orig_point_features: PointFeatures):
        grid = grid_features.vertices
        features = grid_features.features
        orig_points = orig_point_features.vertices
        orig_features = orig_point_features.features

        # Interpolate PointFeatures features to grid using contconv
        out_features = self.conv(
            features,
            grid / self.voxel_size,  # normalize to voxel grid space
            orig_points / self.voxel_size,  # normalize to voxel grid space
            self.extents,
        )
        return PointFeatures(orig_points, orig_features + out_features)


class FromGridNeighborConv(nn.Module):
    def __init__(
        self, grid_channels: int, point_channels: int, out_channels: int, radius: float
    ):
        super().__init__()
        self.radius = radius
        self.conv = NeighborMLPConvLayer(
            mlp=MLP([grid_channels + point_channels, out_channels], torch.nn.GELU)
        )

    def forward(self, grid_features: PointFeatures, orig_point_features: PointFeatures):
        grid_vertices = grid_features.vertices
        grid_features = grid_features.features
        orig_points = orig_point_features.vertices
        orig_features = orig_point_features.features

        # Interpolate PointFeatures features to grid using contconv
        nb = neighbor_radius_search(grid_vertices, orig_points, radius=self.radius)
        out_features = self.conv(grid_features, nb, orig_features)
        return PointFeatures(orig_points, orig_features + out_features)
