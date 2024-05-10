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

from typing import List, Dict, Any, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # use non-interactive backend
import matplotlib.pyplot as plt

from .base_model import BaseModel
from .drivaer_base import DrivAerBase, drivaer_create_subplot

import unittest

from torch import Tensor
import torch
from torch import nn as nn
from torch.nn import functional as F

from .components.mlp import MLP
from src.utils.eval_funcs import eval_all_metrics
from src.utils.visualization import fig_to_numpy


class ChannelSELayer3D(nn.Module):
    """
    3D extension of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
        *Zhu et al., AnatomyNet, arXiv:arXiv:1808.05238*

    from https://github.com/ai-med/squeeze_and_excitation/blob/master/squeeze_and_excitation/squeeze_and_excitation_3D.py
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output tensor
        """
        batch_size, num_channels, D, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = self.avg_pool(input_tensor)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor.view(batch_size, num_channels)))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        output_tensor = torch.mul(
            input_tensor, fc_out_2.view(batch_size, num_channels, 1, 1, 1)
        )

        return output_tensor


class SpatialSELayer3D(nn.Module):
    """
    3D extension of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*

    from https://github.com/ai-med/squeeze_and_excitation/blob/master/squeeze_and_excitation/squeeze_and_excitation_3D.py
    """

    def __init__(self, num_channels):
        """
        :param num_channels: No of input channels

        """
        super(SpatialSELayer3D, self).__init__()
        self.conv = nn.Conv3d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        """
        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output_tensor
        """
        # channel squeeze
        batch_size, channel, D, H, W = input_tensor.size()

        if weights:
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)

        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        output_tensor = torch.mul(
            input_tensor, squeeze_tensor.view(batch_size, 1, D, H, W)
        )

        return output_tensor


class ChannelSpatialSELayer3D(nn.Module):
    """
    3D extension of concurrent spatial and channel squeeze & excitation:
    *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, arXiv:1803.02579*

    from https://github.com/ai-med/squeeze_and_excitation/blob/master/squeeze_and_excitation/squeeze_and_excitation_3D.py
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSpatialSELayer3D, self).__init__()
        self.cSE = ChannelSELayer3D(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer3D(num_channels)

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output_tensor
        """
        output_tensor = torch.max(self.cSE(input_tensor), self.sSE(input_tensor))
        return output_tensor


class UNetEncoderBlock(nn.Module):
    """
    A single U-Net encoder block
    """

    def __init__(
        self, in_channels: int, out_channels: int, dilation: int = 1, padding: int = 1
    ):
        super(UNetEncoderBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            ChannelSpatialSELayer3D(in_channels),
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                dilation=dilation,
                padding=padding,
            ),
            nn.MaxPool3d(kernel_size=2, stride=2),
            nn.BatchNorm3d(out_channels),
        )

    def forward(self, x):
        return self.block(x)


class UNetDecoderBlock(nn.Module):
    """
    A single U-Net decoder block
    """

    def __init__(self, in_channels: int, out_channels: int, padding: int = 1):
        super(UNetDecoderBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReLU(),
            ChannelSpatialSELayer3D(in_channels),
            nn.ConvTranspose3d(
                in_channels, out_channels, kernel_size=3, padding=padding, stride=2
            ),
            nn.BatchNorm3d(out_channels),
        )

    def forward(self, x):
        return self.block(x)


def pad_to_match(x, y):
    """
    Pad the input tensor x to match the spatial dimensions of the input tensor y
    """
    if x.size(2) == y.size(2) and x.size(3) == y.size(3) and x.size(4) == y.size(4):
        return x

    new_x = torch.zeros(
        x.size(0),
        x.size(1),
        y.size(2),
        y.size(3),
        y.size(4),
        device=x.device,
        dtype=x.dtype,
    )
    new_x[:, :, : x.size(2), : x.size(3), : x.size(4)] = x
    return new_x


class VWUNet(BaseModel):
    """
    Convolutional UNet proposed in https://arxiv.org/abs/2108.05798

    > A sequence of ReLU activation, squeeze & excitation block [26],
    > convolutional layer with dilations, max-pooling layer (with a stride of
    > two), and batch normalization layer is a single U-Net encoder block.
    > The convolutional and the max-pooling layer of the U-Net encoder
    > block are replaced by a transpose convolutional layer (with a stride of
    > two) for the velocity field decoder block.

    > The encoder has a convolutional layer followed by six U-Net encoder blocks.
    > We use concurrent spatial and channel
    > squeeze & excitation block [27], a variation of squeeze & excitation
    > block, which weighs the importance of spatial regions and the
    > importance of feature maps concurrently.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        out_decoder_channels: int = 1,
        encoder_channels: List[int] = [16, 32, 64, 128, 256, 512],
        decoder_channels: List[int] = [512, 256, 128, 64, 32, 16],
        drag_mlp_channels: List[int] = [128, 64, 32, 16],
    ):
        super(VWUNet, self).__init__()

        self.encoder = nn.ModuleList()
        self.encoder.append(
            nn.Conv3d(in_channels, encoder_channels[0], kernel_size=3, padding=1)
        )
        for i in range(1, len(encoder_channels)):
            self.encoder.append(
                UNetEncoderBlock(encoder_channels[i - 1], encoder_channels[i])
            )

        self.decoder = nn.ModuleList()
        for i in range(1, len(decoder_channels)):
            self.decoder.append(
                UNetDecoderBlock(decoder_channels[i - 1], decoder_channels[i])
            )

        self.final_decoder = nn.Sequential(
            nn.Conv3d(decoder_channels[-1], out_decoder_channels, kernel_size=1),
        )

        # MLP that extract features from the end of encoder
        self.mlp = MLP(encoder_channels[-1], out_channels, drag_mlp_channels)

    def forward(self, x):
        encoder_outs = [x]
        for encoder in self.encoder:
            x = encoder(x)
            encoder_outs.append(x)

        drag_x = F.adaptive_max_pool3d(x, 1).view(x.size(0), -1)
        drag_x = self.mlp(drag_x)

        for decoder, encoder_out in zip(self.decoder, encoder_outs[::-1]):
            x = pad_to_match(x, encoder_out)
            x = decoder(x + encoder_out)

        x = pad_to_match(x, encoder_outs[0])
        x = self.final_decoder(x)

        return x, drag_x


class VWUNetDrivAer(DrivAerBase, VWUNet):
    """
    Same as VWUNet but for DrivAer
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        out_decoder_channels: int = 1,
        encoder_channels: List[int] = [16, 32, 64, 128, 256, 512],
        decoder_channels: List[int] = [512, 256, 128, 64, 32, 16],
        drag_mlp_channels: List[int] = [128, 64, 32, 16],
        bbox_min: List[float] = [-1.0, -1.0, -1.0],
        bbox_max: List[float] = [1.0, 1.0, 1.0],
    ):
        DrivAerBase.__init__(self)
        VWUNet.__init__(
            self,
            in_channels,
            out_channels,
            out_decoder_channels,
            encoder_channels,
            decoder_channels,
            drag_mlp_channels,
        )

        self.bbox_min = torch.tensor(bbox_min)
        self.bbox_max = torch.tensor(bbox_max)
        self.vert_scaler = 2 / (self.bbox_max - self.bbox_min)

    def data_dict_to_input(self, data_dict) -> torch.Tensor:
        sdf = data_dict["sdf"].unsqueeze(1)  # add channel dimension to BxCxHxWxD
        return sdf.to(self.device)

    def forward_and_sample_pressure(
        self, data_dict, loss_fn=None, datamodule=None, **kwargs
    ) -> Tuple[Tensor, Tensor]:
        sdf = self.data_dict_to_input(data_dict)
        normalized_pred, drag_pred = self(sdf)

        # BxNx3 vertices
        # Assume these are centered
        vertices = data_dict["cell_centers"].float().to(self.device)  # BxNx3
        # Scale it to [-1, 1] for grid_sample
        vertices = (vertices - self.bbox_min.to(self.device)) * self.vert_scaler.to(
            self.device
        ) - 1
        # Convert to BxNx1x1x3
        vertices = vertices.unsqueeze(2).unsqueeze(2)

        # use tri-linear interpolation to sample the pressure field
        normalized_pred = F.grid_sample(
            normalized_pred,
            vertices,
            align_corners=True,
        )  # Bx1xNx1x1x1
        # Bx1xNx1x1x1 -> BxN
        normalized_pred = normalized_pred.view(normalized_pred.size(0), -1)
        return normalized_pred, drag_pred

    @torch.no_grad()
    def eval_dict(self, data_dict, loss_fn=None, datamodule=None, **kwargs) -> Dict:
        normalized_pred, drag_pred = self.forward_and_sample_pressure(
            data_dict, loss_fn, datamodule, **kwargs
        )
        if loss_fn is None:
            loss_fn = self.loss
        normalized_gt = (
            data_dict["time_avg_pressure_whitened"]
            .to(self.device)
            .view_as(normalized_pred)
        )
        out_dict = {"l2": loss_fn(normalized_pred, normalized_gt)}

        pred = datamodule.decode(normalized_pred.clone())
        gt = data_dict["time_avg_pressure"].to(self.device).view_as(pred)
        out_dict["l2_decoded"] = loss_fn(pred, gt)
        # Pressure evaluation
        out_dict.update(
            eval_all_metrics(normalized_gt, normalized_pred, prefix="norm_pressure")
        )
        # collect all drag outputs. All _ prefixed keys are collected in the meter
        gt_drag = data_dict["c_d"].float()
        out_dict["_gt_drag"] = gt_drag.cpu().flatten()
        out_dict["_pred_drag"] = drag_pred.detach().cpu().flatten()
        return out_dict

    def loss_dict(self, data_dict, loss_fn=None, datamodule=None, **kwargs) -> Dict:
        normalized_pred, drag_pred = self.forward_and_sample_pressure(
            data_dict, loss_fn, datamodule, **kwargs
        )
        normalized_gt = data_dict["time_avg_pressure_whitened"].to(self.device)
        normalized_pred = normalized_pred.view(normalized_pred.size(0), -1)

        return_dict = {}
        if loss_fn is None:
            loss_fn = self.loss

        return_dict["pressure_loss"] = loss_fn(normalized_pred, normalized_gt)

        # compute drag loss
        gt_drag = data_dict["c_d"].float().to(self.device)
        return_dict["drag_loss"] = loss_fn(drag_pred.view(-1), gt_drag.view(-1))

        return return_dict

    def image_pointcloud_dict(self, data_dict, datamodule) -> Tuple[Dict, Dict]:
        normalized_pred, _ = self.forward_and_sample_pressure(
            data_dict, datamodule=datamodule
        )
        normalized_pred = normalized_pred.detach().cpu()
        # denormalize
        pred = datamodule.decode(normalized_pred)
        gt_pressure = data_dict["time_avg_pressure"].cpu().view_as(pred)
        vertices = data_dict["cell_centers"].float().cpu().squeeze()

        # Plot
        fig = plt.figure(figsize=(21, 10))  # width, height in inches
        ax = fig.add_subplot(131, projection="3d")
        drivaer_create_subplot(ax, vertices, pred.numpy(), title="Pressure Prediction")
        ax = fig.add_subplot(132, projection="3d")
        drivaer_create_subplot(ax, vertices, gt_pressure.numpy(), title="GT Pressure")
        ax = fig.add_subplot(133, projection="3d")
        drivaer_create_subplot(
            ax, vertices, torch.abs(pred - gt_pressure).numpy(), title="Abs Difference"
        )

        # figure to numpy image
        fig.set_tight_layout(True)
        # set the background to white
        fig.patch.set_facecolor("white")
        im = fig_to_numpy(fig)
        return {"vis": im}, {}


class TestVWUNet(unittest.TestCase):
    """
    Test VWUNet on a simple case
    """

    def test_vwunet(self):
        vwunet = VWUNet()
        x = torch.randn(2, 3, 64, 64, 64)
        y = vwunet(x)
        self.assertEqual(y[0].shape, torch.Size([2, 3, 64, 64, 64]))
        self.assertEqual(y[1].shape, torch.Size([2, 1]))


if __name__ == "__main__":
    unittest.main()
