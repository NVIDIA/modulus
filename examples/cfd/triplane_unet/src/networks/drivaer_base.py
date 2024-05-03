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
import torch

matplotlib.use("Agg")  # use non-interactive backend
import matplotlib.pyplot as plt

from .base_model import BaseModel

from src.utils.visualization import fig_to_numpy


def rel_l2(pred, gt):
    return torch.norm(pred - gt, p=2) / torch.norm(gt, p=2)


class DrivAerBase(BaseModel):
    dir_movement = torch.tensor([-1.0, 0.0, 0.0])

    def data_dict_to_input(self, data_dict) -> torch.Tensor:
        vertices = data_dict["cell_centers"].float()  # (n_in, 3)

        # Assume it is centered
        # center vertices
        # vertices_max = vertices.max(1)[0]
        # vertices_min = vertices.min(1)[0]
        # vertices_center = (vertices_max + vertices_min) / 2.0
        # vertices = vertices - vertices_center

        return vertices.to(self.device)

    def drag(self, pred_pressure, air_coeff, data_dict) -> float:
        coeff = (air_coeff / data_dict["proj_area_x"]).to(self.device)
        poly_normals = data_dict["cell_normals"].float().to(self.device)
        poly_area = data_dict["cell_areas"].float().to(self.device)
        poly_pressure = pred_pressure.float().to(self.device)
        poly_wss = data_dict["time_avg_wall_shear_stress"].float().to(self.device)

        # Compute coefficients
        # -x direction is the movement direction so
        c_p = -coeff * torch.sum(
            poly_normals[..., 0] * poly_area * poly_pressure.squeeze(-1), -1
        )
        c_f = coeff * torch.sum(poly_wss[..., 0] * poly_area, dim=-1)

        # Compute total drag coefficients
        return c_p + c_f

    @torch.no_grad()
    def eval_dict(self, data_dict, loss_fn=None, datamodule=None, **kwargs) -> Dict:
        vertices = self.data_dict_to_input(data_dict)
        normalized_pred, drag_pred = self(vertices)
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
        out_dict["mean_rel_l2_decoded"] = rel_l2(pred, gt)

        # compute drag loss
        gt_drag = data_dict["c_d"].float().to(self.device).view_as(drag_pred)
        out_dict["drag_loss"] = loss_fn(drag_pred, gt_drag)
        out_dict["drag_diff"] = torch.abs(drag_pred - gt_drag).mean()
        out_dict["drag_rel_diff"] = (
            torch.abs(drag_pred - gt_drag) / torch.abs(gt_drag)
        ).mean()
        return out_dict

    def loss_dict(self, data_dict, loss_fn=None, datamodule=None, **kwargs) -> Dict:
        vertices = self.data_dict_to_input(data_dict)
        normalized_pred, drag_pred = self(vertices)
        normalized_gt = (
            data_dict["time_avg_pressure_whitened"].to(self.device)
        )

        return_dict = {}
        if loss_fn is None:
            loss_fn = self.loss
        return_dict["pressure_loss"] = loss_fn(
            normalized_pred.view(1, -1), normalized_gt.view(1, -1).to(self.device)
        )

        # compute drag loss
        gt_drag = data_dict["c_d"].float().to(self.device)
        return_dict["drag_loss"] = loss_fn(drag_pred, gt_drag.view_as(drag_pred))

        return return_dict

    def image_pointcloud_dict(self, data_dict, datamodule) -> Tuple[Dict, Dict]:
        vertices = self.data_dict_to_input(data_dict)
        normalized_pred, _ = self(vertices)
        normalized_pred = normalized_pred.detach().cpu()
        # denormalize
        pred = datamodule.decode(normalized_pred)
        gt_pressure = data_dict["time_avg_pressure"].cpu().view_as(pred)
        vertices = vertices.cpu().squeeze()

        # Plot
        fig = plt.figure(figsize=(21, 10))  # width, height in inches

        def create_subplot(ax, vertices, data, title):
            # Flip along x axis
            vertices = vertices.clone()
            vertices[:, 0] = -vertices[:, 0]

            sc = ax.scatter(
                vertices[:, 0], vertices[:, 1], vertices[:, 2], c=data, cmap="viridis"
            )
            # Make the colorbar smaller
            # fig.colorbar(sc, ax=ax, shrink=0.25, aspect=5)
            # Show the numbers on the colorbar
            cbar = plt.colorbar(sc, ax=ax, shrink=0.25, aspect=5)
            cbar.set_label(title, rotation=270, labelpad=20)

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_aspect("equal")
            # remove grid and background
            ax.grid(False)
            # ax.xaxis.pane.set_edgecolor('black')
            # ax.yaxis.pane.set_edgecolor('black')
            # remove bounding wireframe
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False

            # remove all ticks
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])

            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False

            # ax.set_facecolor("white")
            # ax.set_title(title)

        ax = fig.add_subplot(131, projection="3d")
        create_subplot(ax, vertices, pred.numpy(), title="Pressure Prediction")
        ax = fig.add_subplot(132, projection="3d")
        create_subplot(ax, vertices, gt_pressure.numpy(), title="GT Pressure")
        ax = fig.add_subplot(133, projection="3d")
        create_subplot(
            ax, vertices, torch.abs(pred - gt_pressure).numpy(), title="Abs Difference"
        )

        # figure to numpy image
        fig.set_tight_layout(True)
        # set the background to white
        fig.patch.set_facecolor("white")
        im = fig_to_numpy(fig)
        return {"vis": im}, {}

        # Point cloud visualization
        # Normalize the pressure values using max and min values from both pred and gt_pressure
        # pressures = torch.cat((pred.view(-1), gt_pressure.view(-1))).cpu()
        # min_press = pressures.min()
        # max_press = pressures.max()
        # norm_pred = (pred - min_press) / (max_press - min_press)
        # norm_gt = (gt_pressure - min_press) / (max_press - min_press)

        # # Map normalized pressures to colors
        # colormap = plt.cm.viridis
        # norm_pred_colors = colormap(norm_pred.numpy())
        # norm_gt_colors = colormap(norm_gt.numpy())
        # norm_diff_colors = colormap(np.abs((norm_pred - norm_gt).numpy()))

        # # wrap it back with tensor
        # norm_pred_colors = torch.from_numpy(norm_pred_colors)
        # norm_gt_colors = torch.from_numpy(norm_gt_colors)
        # norm_diff_colors = torch.from_numpy(norm_diff_colors)

        # # Convert the colors to range 0-255 and remove alpha
        # norm_pred_colors = norm_pred_colors[:, :3] * 255
        # norm_gt_colors = norm_gt_colors[:, :3] * 255
        # norm_diff_colors = norm_diff_colors[:, :3] * 255

        # # concatenate vertices and colors into Nx6 matrix
        # pred_points = torch.cat((vertices, norm_pred_colors), dim=1)
        # gt_points = torch.cat((vertices, norm_gt_colors), dim=1)
        # diff_points = torch.cat((vertices, norm_diff_colors), dim=1)

        # return {"vis": im}, {"pred": pred_points, "gt": gt_points, "diff": diff_points}


class DrivAerDragRegressionBase(DrivAerBase):
    """
    Base class for drag regression networks
    """

    def loss_dict(self, data_dict, loss_fn=None, datamodule=None, **kwargs) -> Dict:
        vertices = self.data_dict_to_input(data_dict)
        pred_drag = self(vertices)
        gt_drag = data_dict["c_d_computed"].float().to(self.device)
        return {"drag_loss": loss_fn(pred_drag, gt_drag)}

    def image_pointcloud_dict(self, data_dict, datamodule):
        return {}
