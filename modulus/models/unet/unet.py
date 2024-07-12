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

from dataclasses import dataclass
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..meta import ModelMetaData
from ..module import Module


class ConvBlock(nn.Module):
    """
    A convolutional block, followed by an optional normalization and activation.

    Parameters:
    ----------
        in_channels (int): Number of channels in the input.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int, tuple): Size of the convolving kernel. Default is 3.
        stride (int, tuple): Stride of the convolution. Default is 1.
        padding (int, tuple): Padding added to all sides of the input. Default is 1.
        dilation (int, tuple): Spacing between kernel elements. Default is 1.
        groups (int): Number of blocked connections from input channels to output channels. Default is 1.
        bias (bool): If True, adds a learnable bias to the output. Default is True.
        padding_mode (str): Padding mode to use. Default is 'zeros'.
        activation (Optional[str]): Type of activation to use. Default is 'relu'.
        normalization (Optional[str]): Type of normalization to use. Default is 'groupnorm'.
        normalization_args (dict): Arguments for the normalization layer.

    Returns:
    -------
        torch.Tensor: The processed output tensor.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple] = 3,
        stride: Union[int, tuple] = 1,
        padding: Union[int, tuple] = 1,
        dilation: Union[int, tuple] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        activation: Optional[str] = "relu",
        normalization: Optional[str] = "groupnorm",
        normalization_args: Optional[dict] = None,
    ):
        super().__init__()
        # Initialize convolution layer
        self.conv3d = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

        # Initialize activation function
        if activation:
            if hasattr(F, activation):
                self.activation = getattr(F, activation)
            else:
                raise ValueError(f"Activation type '{activation}' is not supported.")
        else:
            self.activation = nn.Identity()

        # Initialize normalization layer
        if normalization:
            if normalization == "groupnorm":
                default_args = {"num_groups": 1, "num_channels": out_channels}
                norm_args = {
                    **default_args,
                    **(normalization_args if normalization_args else {}),
                }
                self.norm = nn.GroupNorm(**norm_args)
            elif normalization == "batchnorm":
                self.norm = nn.BatchNorm3d(out_channels)
            else:
                raise ValueError(
                    f"Normalization type '{normalization}' is not supported."
                )
        else:
            self.norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv3d(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class ConvTranspose(nn.Module):
    """
    A transposed convolutional block, followed by an optional normalization and activation.

    Parameters:
    ----------
        in_channels (int): Number of channels in the input.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int, tuple): Size of the convolving kernel. Default is 3.
        stride (int, tuple): Stride of the convolution. Default is 2.
        padding (int, tuple): Padding added to all sides of the input. Default is 1.
        output_padding (int, tuple): Additional size added to one side of the output shape. Default is 1.
        groups (int): Number of blocked connections from input channels to output channels. Default is 1.
        bias (bool): If True, adds a learnable bias to the output. Default is True.
        dilation (int, tuple): Spacing between kernel elements. Default is 1.
        padding_mode (str): Padding mode to use. Default is 'zeros'.
        activation (Optional[str]): Type of activation to use. Default is None.
        normalization (Optional[str]): Type of normalization to use. Default is None.
        normalization_args (dict): Arguments for the normalization layer.

    Returns:
    -------
        torch.Tensor: The processed output tensor.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple] = 3,
        stride: Union[int, tuple] = 2,
        padding: Union[int, tuple] = 1,
        output_padding: Union[int, tuple] = 1,
        groups: int = 1,
        bias: bool = True,
        dilation: Union[int, tuple] = 1,
        padding_mode: str = "zeros",
        activation: Optional[str] = None,
        normalization: Optional[str] = None,
        normalization_args: Optional[dict] = None,
    ):
        super().__init__()
        # Initialize transposed convolution layer
        self.conv3d_transpose = nn.ConvTranspose3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
            padding_mode=padding_mode,
        )

        # Initialize activation function
        if activation:
            if hasattr(F, activation):
                self.activation = getattr(F, activation)
            else:
                raise ValueError(f"Activation type '{activation}' is not supported.")
        else:
            self.activation = nn.Identity()

        # Initialize normalization layer
        if normalization:
            if normalization == "groupnorm":
                default_args = {"num_groups": 1, "num_channels": out_channels}
                norm_args = {
                    **default_args,
                    **(normalization_args if normalization_args else {}),
                }
                self.norm = nn.GroupNorm(**norm_args)
            elif normalization == "batchnorm":
                self.norm = nn.BatchNorm3d(out_channels)
            else:
                raise ValueError(
                    f"Normalization type '{normalization}' is not supported."
                )
        else:
            self.norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv3d_transpose(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class Pool3d(nn.Module):
    """
    A pooling block that applies a specified 3D pooling operation over an input signal.

    Parameters:
    ----------
        pooling_type (str): Type of pooling operation ('AvgPool3d', 'MaxPool3d', or custom types if supported).
        kernel_size (int, tuple): Size of the window to take a pool over.
        stride (int, tuple, None): Stride of the pooling operation. Default is None (same as kernel_size).
        padding (int, tuple): Implicit zero padding to be added on both sides of the input. Default is 0.
        dilation (int, tuple): Control the spacing between the kernel points; useful for dilated pooling. Default is 1.
        ceil_mode (bool): When True, will use ceil instead of floor to compute the output shape. Default is False.
        count_include_pad (bool): Only used for AvgPool3d. If True, will include the zero-padding in the averaging calculation.

    Returns:
    -------
        torch.Tensor: The processed output tensor.
    """

    def __init__(
        self,
        pooling_type: str = "AvgPool3d",
        kernel_size: Union[int, tuple] = 2,
        stride: Optional[Union[int, tuple]] = None,
        padding: Union[int, tuple] = 0,
        dilation: Union[int, tuple] = 1,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ):
        super().__init__()

        # Validate pooling type and initialize pooling layer
        if pooling_type not in ["AvgPool3d", "MaxPool3d"]:
            raise ValueError(
                f"Invalid pooling_type '{pooling_type}'. Please choose from ['AvgPool3d', 'MaxPool3d'] or implement additional types."
            )

        # Initialize the corresponding pooling layer
        if pooling_type == "AvgPool3d":
            self.pooling = nn.AvgPool3d(
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                ceil_mode=ceil_mode,
                count_include_pad=count_include_pad,
            )
        elif pooling_type == "MaxPool3d":
            self.pooling = nn.MaxPool3d(
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                ceil_mode=ceil_mode,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pooling(x)


class EncoderBlock(nn.Module):
    """
    An encoder block that sequentially applies multiple convolutional blocks followed by a pooling operation, aggregating features at multiple scales.

    Parameters:
    ----------
        in_channels (int): Number of channels in the input.
        feature_map_channels (List[int]): List of the number of channels for each conv block within this encoder.
        model_depth (int): Number of times the conv-pool operation should be repeated.
        num_conv_blocks (int): Number of convolutional blocks per depth level.
        activation (Optional[str]): Type of activation to use. Default is 'relu'.
        pooling_type (str): Type of pooling to use ('AvgPool3d', 'MaxPool3d').
        pool_size (int): Size of the window for the pooling operation.

    Returns:
    -------
        torch.Tensor: The processed output tensor.
    """

    def __init__(
        self,
        in_channels: int,
        feature_map_channels: List[int],
        model_depth: int = 4,
        num_conv_blocks: int = 2,
        activation: Optional[str] = "relu",
        pooling_type: str = "AvgPool3d",
        pool_size: int = 2,
    ):
        super().__init__()

        if len(feature_map_channels) != model_depth * num_conv_blocks:
            raise ValueError(
                "The length of feature_map_channels should be equal to model_depth * num_conv_blocks"
            )

        self.layers = nn.ModuleList()
        current_channels = in_channels

        for depth in range(model_depth):
            for i in range(num_conv_blocks):
                self.layers.append(
                    ConvBlock(
                        in_channels=current_channels,
                        out_channels=feature_map_channels[depth * num_conv_blocks + i],
                        activation=activation,
                    )
                )
                current_channels = feature_map_channels[depth * num_conv_blocks + i]

            if (
                depth < model_depth - 1
            ):  # Add pooling between levels but not at the last level
                self.layers.append(
                    Pool3d(pooling_type=pooling_type, kernel_size=pool_size)
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class DecoderBlock(nn.Module):
    """
    A decoder block that sequentially applies multiple transposed convolutional blocks, optionally concatenating features from the corresponding encoder.

    Parameters:
    ----------
        in_channels (int): Number of channels in the input.
        feature_map_channels (List[int]): List of the number of channels for each deconv block within this decoder.
        model_depth (int): Number of times the deconv operation should be repeated.
        num_conv_blocks (int): Number of deconvolutional blocks per depth level.
        conv_activation (Optional[str]): Type of activation to usein conv layers. Default is 'relu'.
        conv_transpose_activation (Optional[str]): Type of activation to use in deconv layers. Default is None.

    Returns:
    -------
        torch.Tensor: The processed output tensor.
    """

    def __init__(
        self,
        out_channels: int,
        feature_map_channels: List[int],
        model_depth: int = 3,
        num_conv_blocks: int = 2,
        conv_activation: Optional[str] = "relu",
        conv_transpose_activation: Optional[str] = None,
    ):
        super().__init__()

        if len(feature_map_channels) != model_depth * num_conv_blocks + 1:
            raise ValueError(
                "The length of feature_map_channels in the decoder block should be equal to model_depth * num_conv_blocks + 1"
            )

        self.layers = nn.ModuleList()
        current_channels = feature_map_channels[0]
        feature_map_channels = feature_map_channels[1:]

        for depth in range(model_depth):
            for i in range(num_conv_blocks):
                if i == 0:
                    self.layers.append(
                        ConvTranspose(
                            in_channels=current_channels,
                            out_channels=current_channels,
                            activation=conv_transpose_activation,
                        )
                    )
                    current_channels += feature_map_channels[
                        depth * num_conv_blocks + i
                    ]

                self.layers.append(
                    ConvBlock(
                        in_channels=current_channels,
                        out_channels=feature_map_channels[depth * num_conv_blocks + i],
                        activation=conv_activation,
                    )
                )
                current_channels = feature_map_channels[depth * num_conv_blocks + i]

        # Final convolution
        self.layers.append(
            ConvBlock(
                in_channels=current_channels,
                out_channels=out_channels,
                activation=None,
                normalization=None,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


@dataclass
class MetaData(ModelMetaData):
    name: str = "UNet"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = True
    amp: bool = True
    # Inference
    onnx_cpu: bool = True
    onnx_gpu: bool = True
    onnx_runtime: bool = True
    # Physics informed
    var_dim: int = 1
    func_torch: bool = False
    auto_grad: bool = False


class UNet(Module):
    """
    U-Net model, featuring an encoder-decoder architecture with skip connections.
    Default parameters are set to replicate the architecture here: https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/.

    Parameters:
    ----------
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels in the output segmentation map.
        model_depth (int): Number of levels in the U-Net, not counting the bottleneck layer.
        feature_map_channels (List[int]): Number of channels for each conv block in the encoder and decoder.
        num_conv_blocks (int): Number of convolutional blocks per level in the encoder and decoder.
        conv_activation (Optional[str]): Type of activation to usein conv layers. Default is 'relu'.
        conv_transpose_activation (Optional[str]): Type of activation to use in deconv layers. Default is None.
        pooling_type (str): Type of pooling operation used in the encoder. Supports "AvgPool3d", "MaxPool3d".
        pool_size (int): Size of the window for the pooling operation.

    Returns:
    -------
        torch.Tensor: The processed output tensor.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        model_depth: int = 5,
        feature_map_channels: List[int] = [
            64,
            64,
            128,
            128,
            256,
            256,
            512,
            512,
            1024,
            1024,
        ],
        num_conv_blocks: int = 2,
        conv_activation: Optional[str] = "relu",
        conv_transpose_activation: Optional[str] = None,
        pooling_type: str = "MaxPool3d",
        pool_size: int = 2,
    ):
        super().__init__(meta=MetaData())

        # Construct the encoder
        self.encoder = EncoderBlock(
            in_channels=in_channels,
            feature_map_channels=feature_map_channels,
            model_depth=model_depth,
            num_conv_blocks=num_conv_blocks,
            activation=conv_activation,
            pooling_type=pooling_type,
            pool_size=pool_size,
        )

        # Construct the decoder
        decoder_feature_maps = feature_map_channels[::-1][
            1:
        ]  # Reverse and discard the first channel
        self.decoder = DecoderBlock(
            out_channels=out_channels,
            feature_map_channels=decoder_feature_maps,
            model_depth=model_depth - 1,
            num_conv_blocks=num_conv_blocks,
            conv_activation=conv_activation,
            conv_transpose_activation=conv_transpose_activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_features = []
        # Encoding path
        for layer in self.encoder.layers:
            if isinstance(layer, Pool3d):
                skip_features.append(x)
            x = layer(x)

        # Decoding path
        skip_features = skip_features[::-1]  # Reverse
        concats = 0  # keep track of the number of concats
        for layer in self.decoder.layers:
            if isinstance(layer, ConvTranspose):
                x = layer(x)
                x = torch.cat([x, skip_features[concats]], dim=1)
                concats += 1
            else:
                x = layer(x)

        return x


if __name__ == "__main__":
    inputs = torch.randn(1, 1, 96, 96, 96).cuda()
    print("The shape of inputs: ", inputs.shape)
    model = UNet(
        in_channels=1,
        out_channels=1,
        model_depth=5,
        feature_map_channels=[64, 64, 128, 128, 256, 256, 512, 512, 1024, 1024],
        num_conv_blocks=2,
    ).cuda()
    x = model(inputs)
    print("model: ", model)
    print("The shape of output: ", x.shape)
