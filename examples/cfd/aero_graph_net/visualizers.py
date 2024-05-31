# @package _global_

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

import pyvista as pv

import numpy as np

from dgl import DGLGraph
from torch import Tensor

from loggers import ExperimentLogger


class MeshVisualizer:
    """Mesh visualizer.

    Visualizes mesh in 3x3 grid where each row contains 3 images,
    (GT, prediction, abs error), using different camera positions.
    """

    def __init__(self, tag: str, camera_positions: list) -> None:
        self.tag = tag
        self.camera_positions = camera_positions

    def __call__(
        self,
        graph: DGLGraph,
        pred: Tensor,
        gt: Tensor,
        step: int,
        elogger: ExperimentLogger,
    ) -> None:
        # Assuming `pos` goes first.
        vertices = graph.ndata["x"][:, :3].cpu().numpy()
        p_gt = gt[:, :1].cpu().numpy()
        p_pred = pred[:, :1].cpu().numpy()
        abs_err = np.abs(p_pred - p_gt)

        plotter = pv.Plotter(shape=(3, 3), off_screen=True)

        # TODO(akamenev): this is currently plotting point clouds as
        # opposed to meshes. This limitation is due to DGLGraph not storing faces.
        def plot_point_cloud(
            scalar, pc, cam_pos, clim=(-600, 400), show_bar=False, text=None
        ):
            data = pv.PolyData(vertices)
            data[scalar] = pc
            plotter.add_points(
                data,
                scalars=scalar,
                cmap="jet",
                clim=clim,
                point_size=5,
                show_scalar_bar=show_bar,
            )
            plotter.camera_position = cam_pos
            if text is not None:
                plotter.add_text(text, position="upper_left")

        def plot_column(col, scalar, data, text, clim=(-600, 400)):
            num_rows = 3
            for row in range(num_rows):
                plotter.subplot(row, col)
                text = text if row == 0 else None
                show_bar = row == (num_rows - 1)
                plot_point_cloud(
                    scalar,
                    data,
                    self.camera_positions[row],
                    clim=clim,
                    show_bar=show_bar,
                    text=text,
                )

        plot_column(0, "p_gt", p_gt, "GT Pressure")
        plot_column(1, "p_pred", p_pred, "Predicted Pressure")
        plot_column(2, "abs_err", abs_err, "Abs Error", clim=(-10, 100))

        img = plotter.screenshot()
        elogger.log_image(self.tag, img, step)
