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

import torch
import torch.nn as nn
import torch.nn.functional as F
import modulus.models.layers as layers
import modulus

from typing import Dict, List, Union, Tuple
from torch import Tensor
from dataclasses import dataclass
from ..meta import ModelMetaData
from ..module import Module
from ..mlp import FullyConnected

# ===================================================================
# ===================================================================
# 1D FNO
# ===================================================================
# ===================================================================


class FNO1DEncoder(nn.Module):
    """1D Spectral encoder for FNO

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels, by default 1
    num_fno_layers : int, optional
        Number of spectral convolutional layers, by default 4
    fno_layer_size : int, optional
        Latent features size in spectral convolutions, by default 32
    num_fno_modes : Union[int, List[int]], optional
        Number of Fourier modes kept in spectral convolutions, by default 16
    padding :  Union[int, List[int]], optional
        Domain padding for spectral convolutions, by default 8
    padding_type : str, optional
        Type of padding for spectral convolutions, by default "constant"
    activation_fn : nn.Module, optional
        Activation function, by default nn.GELU
    coord_features : bool, optional
        Use coordinate grid as additional feature map, by default True
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_fno_layers: int = 4,
        fno_layer_size: int = 32,
        num_fno_modes: Union[int, List[int]] = 16,
        padding: Union[int, List[int]] = 8,
        padding_type: str = "constant",
        activation_fn: nn.Module = nn.GELU(),
        coord_features: bool = True,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.num_fno_layers = num_fno_layers
        self.fno_width = fno_layer_size
        self.coord_features = coord_features
        # Spectral modes to have weights
        if isinstance(num_fno_modes, int):
            num_fno_modes = [num_fno_modes]
        # Add relative coordinate feature
        if self.coord_features:
            self.in_channels = self.in_channels + 1
        self.activation_fn = activation_fn

        self.spconv_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()

        # Initial lift network
        self.lift_network = torch.nn.Sequential()
        self.lift_network.append(
            layers.Conv1dFCLayer(self.in_channels, int(self.fno_width / 2))
        )
        self.lift_network.append(self.activation_fn)
        self.lift_network.append(
            layers.Conv1dFCLayer(int(self.fno_width / 2), self.fno_width)
        )

        # Build Neural Fourier Operators
        for _ in range(self.num_fno_layers):
            self.spconv_layers.append(
                layers.SpectralConv1d(self.fno_width, self.fno_width, num_fno_modes[0])
            )
            self.conv_layers.append(nn.Conv1d(self.fno_width, self.fno_width, 1))

        # Padding values for spectral conv
        if isinstance(padding, int):
            padding = [padding]
        self.pad = padding[:1]
        self.ipad = [-pad if pad > 0 else None for pad in self.pad]
        self.padding_type = padding_type

    def forward(self, x: Tensor) -> Tensor:
        if self.coord_features:
            coord_feat = self.meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)

        x = self.lift_network(x)
        # (left, right)
        x = F.pad(x, (0, self.pad[0]), mode=self.padding_type)
        # Spectral layers
        for k, conv_w in enumerate(zip(self.conv_layers, self.spconv_layers)):
            conv, w = conv_w
            if k < len(self.conv_layers) - 1:
                x = self.activation_fn(conv(x) + w(x))
            else:
                x = conv(x) + w(x)

        x = x[..., : self.ipad[0]]
        return x

    def meshgrid(self, shape: List[int], device: torch.device) -> Tensor:
        """Creates 1D meshgrid feature

        Parameters
        ----------
        shape : List[int]
            Tensor shape
        device : torch.device
            Device model is on

        Returns
        -------
        Tensor
            Meshgrid tensor
        """
        bsize, size_x = shape[0], shape[2]
        grid_x = torch.linspace(0, 1, size_x, dtype=torch.float32, device=device)
        grid_x = grid_x.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1)
        return grid_x


# ===================================================================
# ===================================================================
# 2D FNO
# ===================================================================
# ===================================================================


class FNO2DEncoder(nn.Module):
    """2D Spectral encoder for FNO

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels, by default 1
    num_fno_layers : int, optional
        Number of spectral convolutional layers, by default 4
    fno_layer_size : int, optional
        Latent features size in spectral convolutions, by default 32
    num_fno_modes : Union[int, List[int]], optional
        Number of Fourier modes kept in spectral convolutions, by default 16
    padding :  Union[int, List[int]], optional
        Domain padding for spectral convolutions, by default 8
    padding_type : str, optional
        Type of padding for spectral convolutions, by default "constant"
    activation_fn : nn.Module, optional
        Activation function, by default nn.GELU
    coord_features : bool, optional
        Use coordinate grid as additional feature map, by default True
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_fno_layers: int = 4,
        fno_layer_size: int = 32,
        num_fno_modes: Union[int, List[int]] = 16,
        padding: Union[int, List[int]] = 8,
        padding_type: str = "constant",
        activation_fn: nn.Module = nn.GELU(),
        coord_features: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_fno_layers = num_fno_layers
        self.fno_width = fno_layer_size
        self.coord_features = coord_features
        # Spectral modes to have weights
        if isinstance(num_fno_modes, int):
            num_fno_modes = [num_fno_modes, num_fno_modes]
        # Add relative coordinate feature
        if self.coord_features:
            self.in_channels = self.in_channels + 2
        self.activation_fn = activation_fn

        self.spconv_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()

        # Initial lift network
        self.lift_network = torch.nn.Sequential()
        self.lift_network.append(
            layers.Conv2dFCLayer(self.in_channels, int(self.fno_width / 2))
        )
        self.lift_network.append(self.activation_fn)
        self.lift_network.append(
            layers.Conv2dFCLayer(int(self.fno_width / 2), self.fno_width)
        )

        # Build Neural Fourier Operators
        for _ in range(self.num_fno_layers):
            self.spconv_layers.append(
                layers.SpectralConv2d(
                    self.fno_width, self.fno_width, num_fno_modes[0], num_fno_modes[1]
                )
            )
            self.conv_layers.append(nn.Conv2d(self.fno_width, self.fno_width, 1))

        # Padding values for spectral conv
        if isinstance(padding, int):
            padding = [padding, padding]
        padding = padding + [0, 0]  # Pad with zeros for smaller lists
        self.pad = padding[:2]
        self.ipad = [-pad if pad > 0 else None for pad in self.pad]
        self.padding_type = padding_type

    def forward(self, x: Tensor) -> Tensor:
        assert (
            x.dim() == 4
        ), "Only 4D tensors [batch, in_channels, grid_x, grid_y] accepted for 2D FNO"

        if self.coord_features:
            coord_feat = self.meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)

        x = self.lift_network(x)
        # (left, right, top, bottom)
        x = F.pad(x, (0, self.pad[1], 0, self.pad[0]), mode=self.padding_type)
        # Spectral layers
        for k, conv_w in enumerate(zip(self.conv_layers, self.spconv_layers)):
            conv, w = conv_w
            if k < len(self.conv_layers) - 1:
                x = self.activation_fn(conv(x) + w(x))
            else:
                x = conv(x) + w(x)

        # remove padding
        x = x[..., : self.ipad[0], : self.ipad[1]]

        return x

    def meshgrid(self, shape: List[int], device: torch.device) -> Tensor:
        """Creates 2D meshgrid feature

        Parameters
        ----------
        shape : List[int]
            Tensor shape
        device : torch.device
            Device model is on

        Returns
        -------
        Tensor
            Meshgrid tensor
        """
        bsize, size_x, size_y = shape[0], shape[2], shape[3]
        grid_x = torch.linspace(0, 1, size_x, dtype=torch.float32, device=device)
        grid_y = torch.linspace(0, 1, size_y, dtype=torch.float32, device=device)
        grid_x, grid_y = torch.meshgrid(grid_x, grid_y, indexing="ij")
        grid_x = grid_x.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1)
        grid_y = grid_y.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1)
        return torch.cat((grid_x, grid_y), dim=1)


# ===================================================================
# ===================================================================
# 3D FNO
# ===================================================================
# ===================================================================


class FNO3DEncoder(nn.Module):
    """3D Spectral encoder for FNO

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels, by default 1
    num_fno_layers : int, optional
        Number of spectral convolutional layers, by default 4
    fno_layer_size : int, optional
        Latent features size in spectral convolutions, by default 32
    num_fno_modes : Union[int, List[int]], optional
        Number of Fourier modes kept in spectral convolutions, by default 16
    padding :  Union[int, List[int]], optional
        Domain padding for spectral convolutions, by default 8
    padding_type : str, optional
        Type of padding for spectral convolutions, by default "constant"
    activation_fn : nn.Module, optional
        Activation function, by default nn.GELU
    coord_features : bool, optional
        Use coordinate grid as additional feature map, by default True
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_fno_layers: int = 4,
        fno_layer_size: int = 32,
        num_fno_modes: Union[int, List[int]] = 16,
        padding: Union[int, List[int]] = 8,
        padding_type: str = "constant",
        activation_fn: nn.Module = nn.GELU(),
        coord_features: bool = True,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.num_fno_layers = num_fno_layers
        self.fno_width = fno_layer_size
        self.coord_features = coord_features
        # Spectral modes to have weights
        if isinstance(num_fno_modes, int):
            num_fno_modes = [num_fno_modes, num_fno_modes, num_fno_modes]
        # Add relative coordinate feature
        if self.coord_features:
            self.in_channels = self.in_channels + 3
        self.activation_fn = activation_fn

        self.spconv_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()

        # Initial lift network
        self.lift_network = torch.nn.Sequential()
        self.lift_network.append(
            layers.Conv3dFCLayer(self.in_channels, int(self.fno_width / 2))
        )
        self.lift_network.append(self.activation_fn)
        self.lift_network.append(
            layers.Conv3dFCLayer(int(self.fno_width / 2), self.fno_width)
        )

        # Build Neural Fourier Operators
        for _ in range(self.num_fno_layers):
            self.spconv_layers.append(
                layers.SpectralConv3d(
                    self.fno_width,
                    self.fno_width,
                    num_fno_modes[0],
                    num_fno_modes[1],
                    num_fno_modes[2],
                )
            )
            self.conv_layers.append(nn.Conv3d(self.fno_width, self.fno_width, 1))

        # Padding values for spectral conv
        if isinstance(padding, int):
            padding = [padding, padding, padding]
        padding = padding + [0, 0, 0]  # Pad with zeros for smaller lists
        self.pad = padding[:3]
        self.ipad = [-pad if pad > 0 else None for pad in self.pad]
        self.padding_type = padding_type

    def forward(self, x: Tensor) -> Tensor:
        if self.coord_features:
            coord_feat = self.meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)

        x = self.lift_network(x)
        # (left, right, top, bottom, front, back)
        x = F.pad(
            x,
            (0, self.pad[2], 0, self.pad[1], 0, self.pad[0]),
            mode=self.padding_type,
        )
        # Spectral layers
        for k, conv_w in enumerate(zip(self.conv_layers, self.spconv_layers)):
            conv, w = conv_w
            if k < len(self.conv_layers) - 1:
                x = self.activation_fn(conv(x) + w(x))
            else:
                x = conv(x) + w(x)

        x = x[..., : self.ipad[0], : self.ipad[1], : self.ipad[2]]
        return x

    def meshgrid(self, shape: List[int], device: torch.device) -> Tensor:
        """Creates 3D meshgrid feature

        Parameters
        ----------
        shape : List[int]
            Tensor shape
        device : torch.device
            Device model is on

        Returns
        -------
        Tensor
            Meshgrid tensor
        """
        bsize, size_x, size_y, size_z = shape[0], shape[2], shape[3], shape[4]
        grid_x = torch.linspace(0, 1, size_x, dtype=torch.float32, device=device)
        grid_y = torch.linspace(0, 1, size_y, dtype=torch.float32, device=device)
        grid_z = torch.linspace(0, 1, size_z, dtype=torch.float32, device=device)
        grid_x, grid_y, grid_z = torch.meshgrid(grid_x, grid_y, grid_z, indexing="ij")
        grid_x = grid_x.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1, 1)
        grid_y = grid_y.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1, 1)
        grid_z = grid_z.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1, 1)
        return torch.cat((grid_x, grid_y, grid_z), dim=1)


# ===================================================================
# ===================================================================
# 4D FNO
# ===================================================================
# ===================================================================


class FNO4DEncoder(nn.Module):
    """4D Spectral encoder for FNO

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels, by default 1
    num_fno_layers : int, optional
        Number of spectral convolutional layers, by default 4
    fno_layer_size : int, optional
        Latent features size in spectral convolutions, by default 32
    num_fno_modes : Union[int, List[int]], optional
        Number of Fourier modes kept in spectral convolutions, by default 16
    padding :  Union[int, List[int]], optional
        Domain padding for spectral convolutions, by default 8
    padding_type : str, optional
        Type of padding for spectral convolutions, by default "constant"
    activation_fn : nn.Module, optional
        Activation function, by default nn.GELU
    coord_features : bool, optional
        Use coordinate grid as additional feature map, by default True
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_fno_layers: int = 4,
        fno_layer_size: int = 32,
        num_fno_modes: Union[int, List[int]] = 16,
        padding: Union[int, List[int]] = 8,
        padding_type: str = "constant",
        activation_fn: nn.Module = nn.GELU(),
        coord_features: bool = True,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.num_fno_layers = num_fno_layers
        self.fno_width = fno_layer_size
        self.coord_features = coord_features
        # Spectral modes to have weights
        if isinstance(num_fno_modes, int):
            num_fno_modes = [num_fno_modes, num_fno_modes, num_fno_modes, num_fno_modes]
        # Add relative coordinate feature
        if self.coord_features:
            self.in_channels = self.in_channels + 4
        self.activation_fn = activation_fn

        self.spconv_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()

        # Initial lift network
        self.lift_network = torch.nn.Sequential()
        self.lift_network.append(
            layers.ConvNdFCLayer(self.in_channels, int(self.fno_width / 2))
        )
        self.lift_network.append(self.activation_fn)
        self.lift_network.append(
            layers.ConvNdFCLayer(int(self.fno_width / 2), self.fno_width)
        )

        # Build Neural Fourier Operators
        for _ in range(self.num_fno_layers):
            self.spconv_layers.append(
                layers.SpectralConv4d(
                    self.fno_width,
                    self.fno_width,
                    num_fno_modes[0],
                    num_fno_modes[1],
                    num_fno_modes[2],
                    num_fno_modes[3],
                )
            )
            self.conv_layers.append(
                layers.ConvNdKernel1Layer(self.fno_width, self.fno_width)
            )

        # Padding values for spectral conv
        if isinstance(padding, int):
            padding = [padding, padding, padding, padding]
        padding = padding + [0, 0, 0, 0]  # Pad with zeros for smaller lists
        self.pad = padding[:4]
        self.ipad = [-pad if pad > 0 else None for pad in self.pad]
        self.padding_type = padding_type

    def forward(self, x: Tensor) -> Tensor:
        if self.coord_features:
            coord_feat = self.meshgrid(list(x.shape), x.device)
            x = torch.cat((x, coord_feat), dim=1)

        x = self.lift_network(x)
        # (left, right, top, bottom, front, back, past, future)
        x = F.pad(
            x,
            (0, self.pad[3], 0, self.pad[2], 0, self.pad[1], 0, self.pad[0]),
            mode=self.padding_type,
        )
        # Spectral layers
        for k, conv_w in enumerate(zip(self.conv_layers, self.spconv_layers)):
            conv, w = conv_w
            if k < len(self.conv_layers) - 1:
                x = self.activation_fn(conv(x) + w(x))
            else:
                x = conv(x) + w(x)

        x = x[..., : self.ipad[0], : self.ipad[1], : self.ipad[2], : self.ipad[3]]
        return x

    def meshgrid(self, shape: List[int], device: torch.device) -> Tensor:
        """Creates 4D meshgrid feature

        Parameters
        ----------
        shape : List[int]
            Tensor shape
        device : torch.device
            Device model is on

        Returns
        -------
        Tensor
            Meshgrid tensor
        """
        bsize, size_x, size_y, size_z, size_t = (
            shape[0],
            shape[2],
            shape[3],
            shape[4],
            shape[5],
        )
        grid_x = torch.linspace(0, 1, size_x, dtype=torch.float32, device=device)
        grid_y = torch.linspace(0, 1, size_y, dtype=torch.float32, device=device)
        grid_z = torch.linspace(0, 1, size_z, dtype=torch.float32, device=device)
        grid_t = torch.linspace(0, 1, size_t, dtype=torch.float32, device=device)
        grid_x, grid_y, grid_z, grid_t = torch.meshgrid(
            grid_x, grid_y, grid_z, grid_t, indexing="ij"
        )
        grid_x = grid_x.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1, 1, 1)
        grid_y = grid_y.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1, 1, 1)
        grid_z = grid_z.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1, 1, 1)
        grid_t = grid_t.unsqueeze(0).unsqueeze(0).repeat(bsize, 1, 1, 1, 1, 1)
        return torch.cat((grid_x, grid_y, grid_z, grid_t), dim=1)


# Functions for converting between point based and grid (image) representations
def _grid_to_points1d(value: Tensor) -> Tuple[Tensor, List[int]]:
    y_shape = list(value.size())
    output = torch.permute(value, (0, 2, 1))
    return output.reshape(-1, output.size(-1)), y_shape


def _points_to_grid1d(value: Tensor, shape: List[int]) -> Tensor:
    output = value.reshape(shape[0], shape[2], value.size(-1))
    return torch.permute(output, (0, 2, 1))


def _grid_to_points2d(value: Tensor) -> Tuple[Tensor, List[int]]:
    y_shape = list(value.size())
    output = torch.permute(value, (0, 2, 3, 1))
    return output.reshape(-1, output.size(-1)), y_shape


def _points_to_grid2d(value: Tensor, shape: List[int]) -> Tensor:
    output = value.reshape(shape[0], shape[2], shape[3], value.size(-1))
    return torch.permute(output, (0, 3, 1, 2))


def _grid_to_points3d(value: Tensor) -> Tuple[Tensor, List[int]]:
    y_shape = list(value.size())
    output = torch.permute(value, (0, 2, 3, 4, 1))
    return output.reshape(-1, output.size(-1)), y_shape


def _points_to_grid3d(value: Tensor, shape: List[int]) -> Tensor:
    output = value.reshape(shape[0], shape[2], shape[3], shape[4], value.size(-1))
    return torch.permute(output, (0, 4, 1, 2, 3))


def _grid_to_points4d(value: Tensor) -> Tuple[Tensor, List[int]]:
    y_shape = list(value.size())
    output = torch.permute(value, (0, 2, 3, 4, 5, 1))
    return output.reshape(-1, output.size(-1)), y_shape


def _points_to_grid4d(value: Tensor, shape: List[int]) -> Tensor:
    output = value.reshape(
        shape[0], shape[2], shape[3], shape[4], shape[5], value.size(-1)
    )
    return torch.permute(output, (0, 5, 1, 2, 3, 4))


# ===================================================================
# ===================================================================
# General FNO Model
# ===================================================================
# ===================================================================


@dataclass
class MetaData(ModelMetaData):
    name: str = "FourierNeuralOperator"
    # Optimization
    jit: bool = True
    cuda_graphs: bool = True
    amp: bool = False
    # Inference
    onnx_cpu: bool = False
    onnx_gpu: bool = False
    onnx_runtime: bool = False
    # Physics informed
    var_dim: int = 1
    func_torch: bool = False
    auto_grad: bool = False


class FNO(Module):
    """Fourier neural operator (FNO) model.

    Note
    ----
    The FNO architecture supports options for 1D, 2D, 3D and 4D fields which can
    be controlled using the `dimension` parameter.

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    decoder_layers : int, optional
        Number of decoder layers, by default 1
    decoder_layer_size : int, optional
        Number of neurons in decoder layers, by default 32
    decoder_activation_fn : str, optional
        Activation function for decoder, by default "silu"
    dimension : int
        Model dimensionality (supports 1, 2, 3).
    latent_channels : int, optional
        Latent features size in spectral convolutions, by default 32
    num_fno_layers : int, optional
        Number of spectral convolutional layers, by default 4
    num_fno_modes : Union[int, List[int]], optional
        Number of Fourier modes kept in spectral convolutions, by default 16
    padding : int, optional
        Domain padding for spectral convolutions, by default 8
    padding_type : str, optional
        Type of padding for spectral convolutions, by default "constant"
    activation_fn : str, optional
        Activation function, by default "gelu"
    coord_features : bool, optional
        Use coordinate grid as additional feature map, by default True

    Example
    -------
    >>> # define the 2d FNO model
    >>> model = modulus.models.fno.FNO(
    ...     in_channels=4,
    ...     out_channels=3,
    ...     decoder_layers=2,
    ...     decoder_layer_size=32,
    ...     dimension=2,
    ...     latent_channels=32,
    ...     num_fno_layers=2,
    ...     padding=0,
    ... )
    >>> input = torch.randn(32, 4, 32, 32) #(N, C, H, W)
    >>> output = model(input)
    >>> output.size()
    torch.Size([32, 3, 32, 32])

    Note
    ----
    Reference: Li, Zongyi, et al. "Fourier neural operator for parametric
    partial differential equations." arXiv preprint arXiv:2010.08895 (2020).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        decoder_layers: int = 1,
        decoder_layer_size: int = 32,
        decoder_activation_fn: str = "silu",
        dimension: int = 2,
        latent_channels: int = 32,
        num_fno_layers: int = 4,
        num_fno_modes: Union[int, List[int]] = 16,
        padding: int = 8,
        padding_type: str = "constant",
        activation_fn: str = "gelu",
        coord_features: bool = True,
    ) -> None:
        super().__init__(meta=MetaData())

        self.num_fno_layers = num_fno_layers
        self.num_fno_modes = num_fno_modes
        self.padding = padding
        self.padding_type = padding_type
        self.activation_fn = layers.get_activation(activation_fn)
        self.coord_features = coord_features

        # decoder net
        self.decoder_net = FullyConnected(
            in_features=latent_channels,
            layer_size=decoder_layer_size,
            out_features=out_channels,
            num_layers=decoder_layers,
            activation_fn=decoder_activation_fn,
        )

        if dimension == 1:
            FNOModel = FNO1DEncoder
            self.grid_to_points = _grid_to_points1d  # For JIT
            self.points_to_grid = _points_to_grid1d  # For JIT
        elif dimension == 2:
            FNOModel = FNO2DEncoder
            self.grid_to_points = _grid_to_points2d  # For JIT
            self.points_to_grid = _points_to_grid2d  # For JIT
        elif dimension == 3:
            FNOModel = FNO3DEncoder
            self.grid_to_points = _grid_to_points3d  # For JIT
            self.points_to_grid = _points_to_grid3d  # For JIT
        elif dimension == 4:
            FNOModel = FNO4DEncoder
            self.grid_to_points = _grid_to_points4d  # For JIT
            self.points_to_grid = _points_to_grid4d  # For JIT
        else:
            raise NotImplementedError(
                "Invalid dimensionality. Only 1D, 2D, 3D and 4D FNO implemented"
            )

        self.spec_encoder = FNOModel(
            in_channels,
            num_fno_layers=self.num_fno_layers,
            fno_layer_size=latent_channels,
            num_fno_modes=self.num_fno_modes,
            padding=self.padding,
            padding_type=self.padding_type,
            activation_fn=self.activation_fn,
            coord_features=self.coord_features,
        )

    def forward(self, x: Tensor) -> Tensor:
        # Fourier encoder
        y_latent = self.spec_encoder(x)

        # Reshape to pointwise inputs if not a conv FC model
        y_shape = y_latent.shape
        y_latent, y_shape = self.grid_to_points(y_latent)

        # Decoder
        y = self.decoder_net(y_latent)

        # Convert back into grid
        y = self.points_to_grid(y, y_shape)

        return y
