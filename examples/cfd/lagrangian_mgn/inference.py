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

import os
import time
import hydra
from hydra.utils import to_absolute_path

import dgl
from dgl.dataloading import GraphDataLoader
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import tri as mtri
from matplotlib.patches import Rectangle
import matplotlib  #

matplotlib.use("TkAgg")  # for plotting

import numpy as np
from networkx import radius
from omegaconf import DictConfig
import torch

from modulus.models.meshgraphnet import MeshGraphNet
from modulus.datapipes.gnn.lagrangian_dataset import LagrangianDataset, graph_update
from modulus.launch.logging import PythonLogger
from modulus.launch.utils import load_checkpoint


class MGNRollout:
    def __init__(self, cfg: DictConfig, logger: PythonLogger):
        self.num_test_samples = cfg.num_test_samples
        self.num_test_time_steps = cfg.num_test_time_steps
        self.dim = cfg.num_output_features
        self.frame_skip = cfg.frame_skip
        self.num_history = 5
        self.num_node_type = 6
        self.plotting_index = 0
        self.radius = cfg.radius

        # set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using {self.device} device")

        # instantiate dataset
        self.dataset = LagrangianDataset(
            name="Water",
            data_dir=to_absolute_path(cfg.data_dir),
            split="valid",
            num_samples=cfg.num_test_samples,
            num_steps=cfg.num_test_time_steps,
            radius=cfg.radius,
        )
        self.dim = self.dataset.dim
        self.dt = self.dataset.dt
        self.bound = self.dataset.bound

        self.dataset.set_normalizer_device(device=self.device)
        self.time_integrator = self.dataset.time_integrator
        self.compute_boundary_feature = self.dataset.compute_boundary_feature
        self.boundary_clamp = self.dataset.boundary_clamp

        self.gravity = torch.zeros(self.dim, device=self.device)
        self.gravity[-1] = -9.8 * self.dt**2
        self.gravity = self.dataset.normalize_acceleration(self.gravity)

        # instantiate dataloader
        self.dataloader = GraphDataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )

        # instantiate the model
        self.model = MeshGraphNet(
            cfg.num_input_features,
            cfg.num_edge_features,
            cfg.num_output_features,
            cfg.processor_size,
            mlp_activation_fn=cfg.activation,
            do_concat_trick=cfg.do_concat_trick,
            num_processor_checkpoint_segments=cfg.num_processor_checkpoint_segments,
            recompute_activation=cfg.recompute_activation,
        )
        if cfg.jit:
            self.model = torch.jit.script(self.model).to(self.device)
        else:
            self.model = self.model.to(self.device)

        # enable train mode
        self.model.eval()

        # load checkpoint
        load_checkpoint(
            to_absolute_path(cfg.ckpt_path),
            models=self.model,
            device=self.device,
        )

    def predict(self):
        self.pred = []
        self.exact = []
        self.node_type = []

        for i, (graph, mask) in enumerate(self.dataloader):

            graph = graph.to(self.device)
            if graph.ndata["t"][0].item() == 0:
                # initialize
                self.pred = []
                self.exact = []
                self.node_type = []
                position = graph.ndata["mesh_pos"][..., : self.dim]
                position_zero = torch.zeros_like(position)
                history = graph.ndata["x"][
                    ..., self.dim : self.dim + self.dim * self.num_history
                ]
                node_type = graph.ndata["x"][..., -self.num_node_type :].clone()
                # boundary_mask = mask.reshape(-1, 1).repeat(1, self.dim).to(self.device)

            # inference step
            boundary_feature = self.compute_boundary_feature(
                position, radius=self.radius, bound=self.bound
            )
            graph.ndata["x"] = torch.cat(
                [position, history, boundary_feature, node_type], dim=-1
            )
            acceleration = self.model(
                graph.ndata["x"], graph.edata["x"], graph
            ).detach()  # predict
            # acceleration = acceleration + self.gravity

            # update the inputs using the prediction from previous iteration
            position, velocity = self.time_integrator(
                position=position,
                velocity=history[..., : self.dim],
                acceleration=acceleration,
                dt=self.dt,
            )
            position = self.boundary_clamp(position, bound=self.bound)
            graph.ndata["mesh_pos"] = position
            velocity = self.dataset.normalize_velocity(velocity)
            history = torch.cat([velocity, history[..., : -self.dim]], dim=-1)

            # do not update the "wall_boundary"  nodes
            # pred_i = torch.where(boundary_mask, pred_i, 0)

            self.pred.append(position.cpu())
            self.exact.append(graph.ndata["y"][..., : self.dim].cpu())
            self.node_type.append(node_type.cpu())

    def unit_test_example(self, n=1000, dim=2, t=200):
        # unit test for 2d cases

        # Create linspace for the grid
        n = int(n ** (1 / dim))
        x = torch.linspace(-1, 1, n)
        xs = torch.meshgrid(
            [
                x,
            ]
            * dim
        )
        points = torch.stack(xs, dim=-1)
        points = points.reshape(-1, dim)

        # Filter points that are within the unit circle
        radius = torch.sum(points**2, dim=1)
        points = points[radius <= 1]
        # a ball centered at (0.5, 0.5)
        points = points * 0.15 + 0.5
        num_nodes = points.shape[0]

        # create graph
        position = points
        history = torch.zeros(num_nodes, self.dim * self.num_history, dtype=torch.float)
        history[:, ::2] = 10  # initial velocity
        boundary_feature = self.compute_boundary_feature(
            position, radius=self.radius, bound=self.bound
        )
        node_type = torch.ones((num_nodes,), dtype=torch.long) * 5
        node_type = torch.nn.functional.one_hot(node_type, num_classes=6)
        graph = dgl.graph(([], []), num_nodes=num_nodes)
        print(position.shape, history.shape, boundary_feature.shape, node_type.shape)
        graph.ndata["mesh_pos"] = position
        graph.ndata["x"] = torch.cat(
            [position, history, boundary_feature, node_type], dim=-1
        )
        graph = graph_update(graph, radius=self.radius)

        # inference
        self.pred = [position]
        self.node_type = [node_type]
        graph = graph.to(self.device)
        history = history.to(self.device)
        node_type = node_type.to(self.device)
        position = position.to(self.device)
        for i in range(t):
            acceleration = self.model(
                graph.ndata["x"], graph.edata["x"], graph
            ).detach()  # predict
            position, velocity = self.time_integrator(
                position=position,
                velocity=history[..., : self.dim],
                acceleration=acceleration,
                dt=self.dt,
            )
            position = self.boundary_clamp(position, bound=self.bound)
            graph.ndata["mesh_pos"] = position
            velocity = self.dataset.normalize_velocity(velocity)
            history = torch.cat([velocity, history[..., : -self.dim]], dim=-1)
            boundary_feature = self.compute_boundary_feature(
                position, radius=self.radius, bound=self.bound
            )
            graph.ndata["x"] = torch.cat(
                [position, history, boundary_feature, node_type], dim=-1
            )
            graph = graph_update(graph, radius=self.radius)

            self.pred.append(position.cpu())
            self.node_type.append(node_type.cpu())
            print(f"step {i}")
        self.exact = self.pred

    def init_animation2d(self, index=0):
        # fig configs
        self.plotting_index = index
        plt.rcParams["image.cmap"] = "inferno"
        self.fig, self.ax = plt.subplots(1, 2, figsize=(16, 9))

        # make animations dir
        if not os.path.exists("./animations"):
            os.makedirs("./animations")

    def animate2d(self, num):
        num *= self.frame_skip
        num = num + self.plotting_index * self.num_test_time_steps
        node_type = self.node_type[num]
        node_type = (
            torch.argmax(node_type, dim=1).numpy() / self.num_node_type
        )  # from one-hot to index
        y_pred = self.pred[num].numpy()
        y_exact = self.exact[num].numpy()

        self.ax[0].cla()
        self.ax[0].set_aspect("equal")
        self.ax[0].scatter(1 - y_pred[:, 0], y_pred[:, 1], c=node_type)
        self.ax[0].set_xlim(self.bound[0], self.bound[1])
        self.ax[0].set_ylim(self.bound[0], self.bound[1])
        self.ax[0].set_title("Modulus MeshGraphNet Prediction", color="black")

        self.ax[1].cla()
        self.ax[1].set_aspect("equal")
        self.ax[1].scatter(1 - y_exact[:, 0], y_exact[:, 1], c=node_type)
        self.ax[1].set_xlim(self.bound[0], self.bound[1])
        self.ax[1].set_ylim(self.bound[0], self.bound[1])
        self.ax[1].set_title("Ground Truth", color="black")

        self.fig.subplots_adjust(
            left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.1, hspace=0.2
        )
        return self.fig

    def init_animation3d(self, index=0):
        # fig configs
        self.plotting_index = index
        plt.rcParams["image.cmap"] = "inferno"
        self.fig = plt.figure(figsize=(16, 9))
        ax0 = self.fig.add_subplot(121, projection="3d")
        ax1 = self.fig.add_subplot(122, projection="3d")
        self.ax = [ax0, ax1]

        # make animations dir
        if not os.path.exists("./animations"):
            os.makedirs("./animations")

    def animate3d(self, num):
        num *= self.frame_skip
        num = num + self.plotting_index * self.num_test_time_steps
        node_type = self.node_type[num]
        node_type = (
            torch.argmax(node_type, dim=1).numpy() / self.num_node_type
        )  # from one-hot to index
        y_pred = self.pred[num].numpy()
        y_exact = self.exact[num].numpy()

        self.ax[0].cla()
        self.ax[0].set_aspect("equal")
        self.ax[0].scatter(y_pred[:, 2], y_pred[:, 0], y_pred[:, 1], c=node_type)
        self.ax[0].set_xlim(self.bound[0], self.bound[1])
        self.ax[0].set_ylim(self.bound[0], self.bound[1])
        self.ax[0].set_zlim(self.bound[0], self.bound[1])
        self.ax[0].set_title("Modulus MeshGraphNet Prediction", color="black")

        self.ax[1].cla()
        self.ax[1].set_aspect("equal")
        self.ax[1].scatter(y_exact[:, 2], y_exact[:, 0], y_exact[:, 1], c=node_type)
        self.ax[1].set_xlim(self.bound[0], self.bound[1])
        self.ax[1].set_ylim(self.bound[0], self.bound[1])
        self.ax[1].set_zlim(self.bound[0], self.bound[1])
        self.ax[1].set_title("Ground Truth", color="black")

        self.fig.subplots_adjust(
            left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.1, hspace=0.2
        )
        return self.fig

    def plot_error(self, pred, target):
        # pred, target (time, num_nodes, 2)
        time_step = pred.shape[0]
        loss = torch.nn.functional.mse_loss(
            pred.reshape(time_step, -1), target.reshape(time_step, -1), reduction="none"
        )
        loss = torch.mean(loss, dim=1)
        plt.figure(figsize=(10, 6))
        plt.plot(loss.numpy(), marker="o", linestyle="-", color="b")
        plt.title("Lagranigian MeshGraphNet")
        plt.xlabel("time steps")
        plt.ylabel("MSE error")
        plt.grid(True)
        return plt


@hydra.main(version_base="1.3", config_path="conf", config_name="config_2d")
def main(cfg: DictConfig) -> None:
    logger = PythonLogger("main")  # General python logger
    logger.file_logging()
    logger.info("Rollout started...")
    rollout = MGNRollout(cfg, logger)

    # test on dataset
    rollout.predict()

    # unit test
    # rollout.unit_test_example(t=cfg.num_test_time_steps)

    # compute the roll out loss
    pred = torch.stack([tensor.reshape(-1) for tensor in rollout.pred], dim=0)
    target = torch.stack([tensor.reshape(-1) for tensor in rollout.exact], dim=0)
    loss = torch.nn.functional.mse_loss(pred, target)
    print(f"the rollout loss is {loss}")

    # plot the roll out loss
    error_plt = rollout.plot_error(pred, target)
    error_plt.savefig("animations/error.png")

    # plot
    if cfg.dim == 2:
        rollout.init_animation2d(index=0)
        ani = animation.FuncAnimation(
            rollout.fig,
            rollout.animate2d,
            frames=(cfg.num_test_time_steps - 5) // cfg.frame_skip,
            interval=cfg.frame_interval,
        )
    elif cfg.dim == 3:
        rollout.init_animation3d(index=0)
        ani = animation.FuncAnimation(
            rollout.fig,
            rollout.animate3d,
            frames=(cfg.num_test_time_steps - 5) // cfg.frame_skip,
            interval=cfg.frame_interval,
        )

    ani.save("animations/animation.gif")
    logger.info(f"Created animation")


if __name__ == "__main__":
    main()
