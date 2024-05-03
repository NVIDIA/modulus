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

from typing import Dict, Tuple

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")  # use non-interactive backend
import matplotlib.pyplot as plt

from .base_model import BaseModel

from src.networks.net_utils import SinusoidalEncoding
from src.utils.visualization import fig_to_numpy


def rel_l2(pred, gt):
    return torch.norm(pred - gt, p=2) / torch.norm(gt, p=2)


class AhmedBodyBase(BaseModel):
    """Ahmed body base class"""

    def __init__(
        self,
        use_uniformized_velocity: bool = True,
        velocity_pos_encoding: bool = False,
        normals_as_features: bool = True,
        vertices_as_features: bool = True,
        pos_encode_dim: int = 32,
        pos_encode_range: float = 1.0,
        random_purturb_train: bool = False,
        vertices_purturb_range: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        velocity_purturb_range: float = 0.05,  # 5% of the max velocity
    ) -> None:
        self.velocity_pos_encoding = velocity_pos_encoding
        self.use_uniformized_velocity = use_uniformized_velocity
        self.normals_as_features = normals_as_features
        self.vertices_as_features = vertices_as_features
        self.random_purturb_train = random_purturb_train
        self.velocity_purturb_range = velocity_purturb_range
        self.vertices_purturb_range = torch.tensor(vertices_purturb_range)
        self.vertices_purturb_offset = torch.tensor(vertices_purturb_range) / 2.0
        self.pos_encode_dim = pos_encode_dim
        self.pos_encode_range = pos_encode_range
        self.pos_encoder = SinusoidalEncoding(
            num_channels=self.pos_encode_dim, data_range=self.pos_encode_range
        )

    @property
    def ahmed_input_feature_dim(self):
        feature_dim = 1
        if self.velocity_pos_encoding:
            feature_dim += self.pos_encode_dim
        if self.vertices_as_features:
            feature_dim += 3
        if self.normals_as_features:
            feature_dim += 3
        return feature_dim

    def data_dict_to_input(self, data_dict):
        vertices = data_dict["cell_centers"]  # (B, N, 3)
        # If train and random_purturb_train, add random perturbation to the vertices
        if self.random_purturb_train and self.training:
            vertices += (
                torch.rand(3) * self.vertices_purturb_range
                - self.vertices_purturb_offset
            )

        vel = data_dict["velocity"]  # (B)
        if self.use_uniformized_velocity:
            vel = data_dict["uniformized_velocity"]  # (B)
        if self.random_purturb_train and self.training:
            vel = vel + (
                torch.rand(1) * self.velocity_purturb_range
                - self.velocity_purturb_range / 2.0
            )

        # replite the velocity to match the number of vertices
        features = torch.tile(vel.reshape(-1, 1, 1), (1, vertices.shape[1], 1))
        if self.velocity_pos_encoding:
            features = self.pos_encoder(features)
        if self.vertices_as_features:
            features = torch.cat((features, vertices), dim=-1)
        if self.normals_as_features:
            normals = data_dict["normals"]
            features = torch.cat((features, normals), dim=-1)
        return vertices.float().to(self.device), features.float().to(self.device)

    @torch.no_grad()
    def eval_dict(self, data_dict, loss_fn=None, datamodule=None, **kwargs):
        vertices, features = self.data_dict_to_input(data_dict)
        pred_press, pred_drag = self(vertices, features)

        if loss_fn is None:
            loss_fn = self.loss

        normalized_gt = data_dict["normalized_pressure"].to(self.device).reshape(1, -1)
        out_dict = {"l2": loss_fn(pred_press, normalized_gt)}
        # if the data_dict has normals, use it to compute the drag loss
        if "normals" in data_dict:
            normals = data_dict["normals"].squeeze(0)
            # compute the drag loss
            pred_drag = self.pressure_drag(pred_press, normals, data_dict["cell_areas"])
            gt_drag = self.pressure_drag(
                data_dict["normalized_pressure"], normals, data_dict["cell_areas"]
            )
            pred_drag = pred_drag.to(gt_drag)
            out_dict["pressure_drag_gt"] = gt_drag
            out_dict["pressure_drag_pred"] = pred_drag
            out_dict["pressure_drag_loss"] = loss_fn(pred_drag, gt_drag)

            # compute relative difference
            out_dict["drag_rel_diff"] = torch.abs(pred_drag - gt_drag) / torch.abs(
                gt_drag
            )
            out_dict["drag_rel_l2"] = rel_l2(pred_drag, gt_drag)

        return out_dict

    def loss_dict(self, data_dict, loss_fn=None, datamodule=None, **kwargs) -> Dict:
        vertices, features = self.data_dict_to_input(data_dict)
        pred_press, pred_drag = self(vertices, features)
        normalized_gt_pressure = data_dict["normalized_pressure"]

        return_dict = {}
        # if the data_dict has normals, use it to compute the drag loss
        if "normals" in data_dict:
            normals = data_dict["normals"].squeeze(0)
            # compute the drag loss
            pred_drag = self.pressure_drag(pred_press, normals, data_dict["cell_areas"])
            gt_drag = self.pressure_drag(
                data_dict["normalized_pressure"], normals, data_dict["cell_areas"]
            )
            pred_drag = pred_drag.to(gt_drag)
            return_dict["pressure_drag_loss"] = loss_fn(pred_drag, gt_drag)

        if loss_fn is None:
            loss_fn = self.loss

        return_dict["pressure_loss"] = loss_fn(
            pred_press.view(1, -1), normalized_gt_pressure.view(1, -1).to(self.device)
        )
        return return_dict

    def image_pointcloud_dict(self, data_dict, datamodule) -> Tuple[Dict, Dict]:
        vertices, features = self.data_dict_to_input(data_dict)
        pred_press, pred_drag = self(vertices, features).detach().cpu().squeeze()
        gt_pressure = data_dict["normalized_pressure"].squeeze()

        # Plot
        fig = plt.figure(figsize=(21, 10))  # width, height in inches

        def create_subplot(ax, vertices, data, title):
            # If data is pytorch tensor, move to cpu and numpy
            if isinstance(data, torch.Tensor):
                data = data.numpy()
            if isinstance(vertices, torch.Tensor):
                vertices = vertices.numpy()

            sc = ax.scatter(
                vertices[:, 0], vertices[:, 1], vertices[:, 2], c=data, cmap="viridis"
            )
            fig.colorbar(sc, ax=ax, shrink=0.5, aspect=5)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_aspect("equal")
            # remove all grid and ticks
            ax.grid(False)
            ax.set_title(title)

        vertices = vertices.detach().cpu()
        pred_press = pred_press.detach().cpu()
        ax = fig.add_subplot(131, projection="3d")
        create_subplot(ax, vertices, pred_press.numpy(), title="Pressure Prediction")
        ax = fig.add_subplot(132, projection="3d")
        create_subplot(ax, vertices, gt_pressure.numpy(), title="GT Pressure")
        ax = fig.add_subplot(133, projection="3d")
        create_subplot(
            ax, vertices, torch.abs(pred_press - gt_pressure).numpy(), title="Abs Difference"
        )

        # figure to numpy image
        fig.set_tight_layout(True)
        # set the background to white
        fig.patch.set_facecolor("white")
        im = fig_to_numpy(fig)

        # Point clouds
        # Normalize the pressure values using max and min values from both pred and gt_pressure
        pressures = torch.cat((pred_press.view(-1), gt_pressure.view(-1))).cpu()
        min_press = pressures.min()
        max_press = pressures.max()
        norm_pred = (pred_press - min_press) / (max_press - min_press)
        norm_gt = (gt_pressure - min_press) / (max_press - min_press)
        norm_diff = norm_pred - norm_gt

        # Map normalized pressures to colors
        colormap = plt.cm.viridis
        norm_pred_colors = colormap(norm_pred.numpy())
        norm_gt_colors = colormap(norm_gt.numpy())
        norm_diff_colors = colormap(np.abs(norm_diff.numpy()))

        # wrap it back with tensor
        norm_pred_colors = torch.from_numpy(norm_pred_colors)
        norm_gt_colors = torch.from_numpy(norm_gt_colors)
        norm_diff_colors = torch.from_numpy(norm_diff_colors)

        # Convert the colors to range 0-255 and remove alpha
        norm_pred_colors = norm_pred_colors[:, :3] * 255
        norm_gt_colors = norm_gt_colors[:, :3] * 255
        norm_diff_colors = norm_diff_colors[:, :3] * 255

        # concatenate vertices and norm pressures into Nx4 matrix
        pred_points = torch.cat((vertices, norm_pred.view(-1, 1)), dim=1)
        gt_points = torch.cat((vertices, norm_gt.view(-1, 1)), dim=1)
        diff_points = torch.cat((vertices, norm_diff.view(-1, 1)), dim=1)

        # concatenate vertices and colors into Nx6 matrix
        pred_points_color = torch.cat((vertices, norm_pred_colors), dim=1)
        gt_points_color = torch.cat((vertices, norm_gt_colors), dim=1)
        diff_points_color = torch.cat((vertices, norm_diff_colors), dim=1)

        return {"vis": im}, {
            "pred_pointcloud": pred_points,
            "gt_pointcloud": gt_points,
            "diff_pointcloud": diff_points,
            "pred_pointcloud_color": pred_points_color,
            "gt_pointcloud_color": gt_points_color,
            "diff_pointcloud_color": diff_points_color,
        }

    def pressure_drag(self, pressure, normals, areas, wall_shear_stress=None):
        weight = normals[:, 0].view(-1) * areas.view(-1)
        return torch.dot(pressure.view(-1), weight.to(pressure))


class AhmedBodyDragRegressionBase(AhmedBodyBase):
    """
    Base class for Ahmed body drag regression model.
    """

    def loss_dict(self, data_dict, loss_fn=None, datamodule=None, **kwargs) -> Dict:
        vertices, features = self.data_dict_to_input(data_dict)
        pred_drag = self(vertices, features)
        return {"drag_loss": loss_fn(pred_drag, data_dict["drag"])}
