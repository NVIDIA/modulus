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

import logging
import os

import hydra
from hydra.utils import instantiate, to_absolute_path

import dgl
from dgl.dataloading import GraphDataLoader

import matplotlib
from matplotlib import animation
from matplotlib import pyplot as plt

matplotlib.use("TkAgg")  # for plotting

from omegaconf import DictConfig, OmegaConf

import torch

from modulus.datapipes.gnn.lagrangian_dataset import graph_update
from modulus.launch.utils import load_checkpoint

from loggers import get_gpu_info, init_python_logging


logger = logging.getLogger("lmgn")


class MGNRollout:
    def __init__(self, cfg: DictConfig):
        self.num_steps = cfg.data.test.num_steps
        self.dim = cfg.dim
        self.frame_skip = cfg.inference.frame_skip
        self.num_history = cfg.data.test.num_history
        self.num_node_type = cfg.data.test.num_node_types
        self.plotting_index = 0

        # set device
        self.device = cfg.test.device
        logger.info(f"Using {self.device} device")

        # instantiate dataset
        logger.info("Loading the test dataset...")
        self.dataset = instantiate(cfg.data.test)
        logger.info(f"Using {len(self.dataset)} test samples.")

        self.dim = self.dataset.dim
        self.radius = self.dataset.radius
        self.dt = self.dataset.dt
        self.bounds = self.dataset.bounds

        self.time_integrator = self.dataset.time_integrator
        self.compute_boundary_feature = self.dataset.compute_boundary_feature
        self.boundary_clamp = self.dataset.boundary_clamp

        # instantiate dataloader
        self.dataloader = GraphDataLoader(
            self.dataset,
            **cfg.test.dataloader,
        )

        # instantiate the model
        logger.info("Creating the model...")
        # instantiate the model
        self.model = instantiate(cfg.model)

        if cfg.compile.enabled:
            self.model = torch.compile(self.model, **cfg.compile.args)
        self.model = self.model.to(self.device)

        # enable train mode
        self.model.eval()

        # load checkpoint
        load_checkpoint(
            to_absolute_path(cfg.resume_dir),
            models=self.model,
            device=self.device,
        )

    @torch.inference_mode()
    def predict(self):
        self.pred = []
        self.exact = []
        self.node_type = []

        for i, graph in enumerate(self.dataloader):
            graph = graph.to(self.device)
            if graph.ndata["t"][0].item() == 0:
                # initialize
                self.pred = []
                self.exact = []
                self.node_type = []
                position = graph.ndata["pos"][..., : self.dim]
                history = graph.ndata["x"][
                    ..., self.dim : self.dim + self.dim * self.num_history
                ]
                node_type = graph.ndata["x"][..., -self.num_node_type :].clone()

            # inference step
            boundary_feature = self.compute_boundary_feature(
                position, radius=self.radius, bounds=self.bounds
            )
            graph.ndata["x"] = torch.cat(
                [position, history, boundary_feature, node_type], dim=-1
            )
            acceleration = self.model(
                graph.ndata["x"], graph.edata["x"], graph
            )  # predict

            # update the inputs using the prediction from previous iteration
            position, velocity = self.time_integrator(
                position=position,
                velocity=history[..., -self.dim :],
                acceleration=acceleration,
                dt=self.dt,
            )
            position = self.boundary_clamp(position, bounds=self.bounds)
            graph.ndata["pos"] = position
            velocity = self.dataset.normalize_velocity(velocity)
            history = torch.cat([history[..., self.dim :], velocity], dim=-1)

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
            position, radius=self.radius, bounds=self.bounds
        )
        node_type = torch.ones((num_nodes,), dtype=torch.long) * 5
        node_type = torch.nn.functional.one_hot(node_type, num_classes=6)
        graph = dgl.graph(([], []), num_nodes=num_nodes)
        print(position.shape, history.shape, boundary_feature.shape, node_type.shape)
        graph.ndata["pos"] = position
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
            position = self.boundary_clamp(position, bounds=self.bounds)
            graph.ndata["pos"] = position
            velocity = self.dataset.normalize_velocity(velocity)
            history = torch.cat([velocity, history[..., : -self.dim]], dim=-1)
            boundary_feature = self.compute_boundary_feature(
                position, radius=self.radius, bounds=self.bounds
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
        num = num + self.plotting_index * self.num_steps
        node_type = self.node_type[num]
        node_type = (
            torch.argmax(node_type, dim=1).numpy() / self.num_node_type
        )  # from one-hot to index
        y_pred = self.pred[num].numpy()
        y_exact = self.exact[num].numpy()

        self.ax[0].cla()
        self.ax[0].set_aspect("equal")
        self.ax[0].scatter(1 - y_pred[:, 0], y_pred[:, 1], c=node_type)
        self.ax[0].set_xlim(self.bounds[0], self.bounds[1])
        self.ax[0].set_ylim(self.bounds[0], self.bounds[1])
        self.ax[0].set_title("Modulus MeshGraphNet Prediction", color="black")

        self.ax[1].cla()
        self.ax[1].set_aspect("equal")
        self.ax[1].scatter(1 - y_exact[:, 0], y_exact[:, 1], c=node_type)
        self.ax[1].set_xlim(self.bounds[0], self.bounds[1])
        self.ax[1].set_ylim(self.bounds[0], self.bounds[1])
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
        num = num + self.plotting_index * self.num_steps
        node_type = self.node_type[num]
        node_type = (
            torch.argmax(node_type, dim=1).numpy() / self.num_node_type
        )  # from one-hot to index
        y_pred = self.pred[num].numpy()
        y_exact = self.exact[num].numpy()

        self.ax[0].cla()
        self.ax[0].set_aspect("equal")
        self.ax[0].scatter(y_pred[:, 2], y_pred[:, 0], y_pred[:, 1], c=node_type)
        self.ax[0].set_xlim(self.bounds[0], self.bounds[1])
        self.ax[0].set_ylim(self.bounds[0], self.bounds[1])
        self.ax[0].set_zlim(self.bounds[0], self.bounds[1])
        self.ax[0].set_title("Modulus MeshGraphNet Prediction", color="black")

        self.ax[1].cla()
        self.ax[1].set_aspect("equal")
        self.ax[1].scatter(y_exact[:, 2], y_exact[:, 0], y_exact[:, 1], c=node_type)
        self.ax[1].set_xlim(self.bounds[0], self.bounds[1])
        self.ax[1].set_ylim(self.bounds[0], self.bounds[1])
        self.ax[1].set_zlim(self.bounds[0], self.bounds[1])
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
        plt.title("Lagrangian MeshGraphNet")
        plt.xlabel("time steps")
        plt.ylabel("Position MSE error")
        plt.grid(True)
        return plt


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    init_python_logging(cfg, base_filename="inference")
    logger.info(f"Config summary:\n{OmegaConf.to_yaml(cfg, sort_keys=True)}")
    logger.info(get_gpu_info())

    logger.info("Rollout started...")
    rollout = MGNRollout(cfg)

    # test on dataset
    rollout.predict()

    # unit test
    # rollout.unit_test_example(t=cfg.num_steps)

    # compute the roll out loss
    pred = torch.stack([tensor.reshape(-1) for tensor in rollout.pred], dim=0)
    target = torch.stack([tensor.reshape(-1) for tensor in rollout.exact], dim=0)
    loss = torch.nn.functional.mse_loss(pred, target)
    logger.info(f"The rollout loss is {loss:.5f}")

    # plot the roll out loss
    error_plt = rollout.plot_error(pred, target)
    out_dir = os.path.join(cfg.output, "animations")
    os.makedirs(out_dir, exist_ok=True)
    error_plt.savefig(os.path.join(out_dir, "error.png"))

    # plot
    if cfg.dim == 2:
        rollout.init_animation2d(index=0)
        ani = animation.FuncAnimation(
            rollout.fig,
            rollout.animate2d,
            frames=(rollout.num_steps - rollout.num_history - 1) // rollout.frame_skip,
            interval=cfg.inference.frame_interval,
        )
    elif cfg.dim == 3:
        rollout.init_animation3d(index=0)
        ani = animation.FuncAnimation(
            rollout.fig,
            rollout.animate3d,
            frames=(cfg.data.test.num_steps - 5) // cfg.inference.frame_skip,
            interval=cfg.inference.frame_interval,
        )

    ani.save(os.path.join(out_dir, "animation.gif"))
    logger.info(f"Created animation")


if __name__ == "__main__":
    main()
