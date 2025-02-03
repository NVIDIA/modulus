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

# ruff: noqa: S101
import functools
import json
import logging
import os
from collections.abc import Sequence
from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F

try:
    import tensorflow.compat.v1 as tf
except ImportError:
    raise ImportError(
        "Mesh Graph Net Datapipe requires the Tensorflow library. "
        'Install: pip install "tensorflow<=2.17.1"'
    )

try:
    import dgl
    from dgl.data import DGLDataset
except ImportError:
    raise ImportError(
        "Mesh Graph Net Datapipe requires the DGL library. Install the "
        + "desired CUDA version at: https://www.dgl.ai/pages/start.html"
    )

from .lagrangian_reading_utils import parse_serialized_simulation_example

# Hide GPU from visible devices for TF
tf.config.set_visible_devices([], "GPU")

logger = logging.getLogger("lmgn")


def compute_edge_index(pos, radius):
    # compute the graph connectivity using pairwise distance
    distances = torch.cdist(pos, pos, p=2)
    mask = distances < radius  # & (distances > 0) # include self-edge
    edge_index = torch.nonzero(mask).t().contiguous()
    return edge_index


def compute_edge_attr(graph, radius=0.015):
    # compute the displacement and distance per edge
    edge_index = graph.edges()
    displacement = graph.ndata["pos"][edge_index[1]] - graph.ndata["pos"][edge_index[0]]
    distance = torch.pairwise_distance(
        graph.ndata["pos"][edge_index[0]],
        graph.ndata["pos"][edge_index[1]],
        keepdim=True,
    )
    # direction = displacement / distance
    distance = torch.exp(-(distance**2) / radius**2)
    graph.edata["x"] = torch.cat((displacement, distance), dim=-1)
    return


def graph_update(graph, radius):
    # remove all previous edges and re-construct the graph using pair-wise distance
    # TODO: use more efficient graph construction method
    num_edges = graph.num_edges()
    if num_edges > 0:
        graph.remove_edges(torch.arange(num_edges, device=graph.device))
    pos = graph.ndata["pos"]
    edge_index = compute_edge_index(pos, radius)
    graph.add_edges(edge_index[0], edge_index[1])
    compute_edge_attr(graph)
    return graph


class LagrangianDataset(DGLDataset):
    """In-memory MeshGraphNet Dataset for Lagrangian mesh.
    Notes:
        - This dataset prepares and processes the data available in MeshGraphNet's repo:
            https://github.com/google-deepmind/deepmind-research/tree/master/learning_to_simulate

    Parameters
    ----------
    name : str, optional
        Name of the dataset, by default "dataset"
    data_dir : _type_, optional
        Specifying the directory that stores the raw data in .TFRecord format., by default None
    split : str, optional
        Dataset split ["train", "valid", "test"], by default "train"
    num_sequences : int, optional
        Number of sequences, by default 1000
    num_history : int, optional.
        Number of velocities, including the current, to include in the history, by default 5.
    num_steps : int, optional
        Number of time steps in each sequence, by default is set from the dataset metadata.
    noise_std : float, optional
        The standard deviation of the noise added to the "train" split, by default 0.0003
    radius : float, optional
        Connectivity radius, by default is set from the dataset metadata.
    dt : float, optional
        Time step increment, by default is set from the dataset metadata.
    bounds :
        Domain bounds, by default is set from the dataset metadata.
    force_reload : bool, optional
        force reload, by default False
    verbose : bool, optional
        verbose, by default False
    """

    def __init__(
        self,
        name: str = "dataset",
        data_dir: Optional[str] = None,
        split: str = "train",
        num_sequences: int = 1000,
        num_history: int = 5,
        num_steps: Optional[int] = None,
        noise_std: float = 0.0003,
        radius: Optional[float] = None,
        dt: Optional[float] = None,
        bounds: Optional[Sequence[tuple[float, float]]] = None,
        num_node_types: int = 6,
        force_reload: bool = False,
        verbose: bool = False,
    ):
        super().__init__(
            name=name,
            force_reload=force_reload,
            verbose=verbose,
        )
        self.data_dir = data_dir
        self.split = split
        self.num_sequences = num_sequences
        self.num_history = num_history
        self.noise_std = noise_std
        self.num_node_types = num_node_types

        path_metadata = os.path.join(data_dir, "metadata.json")
        with open(path_metadata, "r", encoding="utf-8") as file:
            metadata = json.load(file)
        # Note: DM datasets contain sequence_length + 1 time steps for each sequence.
        self.num_steps = (
            (metadata["sequence_length"] + 1) if num_steps is None else num_steps
        )
        self.dt = metadata["dt"] if dt is None else dt
        self.radius = (
            metadata["default_connectivity_radius"] if radius is None else radius
        )
        # Assuming bounds are the same for all dimensions.
        self.bounds = metadata["bounds"][0] if bounds is None else bounds[0]
        self.dim = metadata["dim"]

        self.vel_mean = torch.tensor(metadata["vel_mean"]).reshape(1, self.dim)
        self.vel_std = torch.tensor(metadata["vel_std"]).reshape(1, self.dim)
        self.acc_mean = torch.tensor(metadata["acc_mean"]).reshape(1, self.dim)
        self.acc_std = torch.tensor(metadata["acc_std"]).reshape(1, self.dim)

        # Create the node features.
        logger.info(f"Preparing the {split} dataset...")
        dataset_iterator = self._load_tf_data(self.data_dir, self.split)
        self.node_type = []
        self.rollout_mask = []
        self.node_features = []
        for i in range(self.num_sequences):
            data_np = dataset_iterator.get_next()

            position = torch.from_numpy(
                data_np[1]["position"][: self.num_steps].numpy()
            )  # (t, num_particles, 2)
            assert position.shape[0] == self.num_steps, f"{self.num_steps=}, {i=}"

            node_type = torch.from_numpy(
                data_np[0]["particle_type"].numpy()
            )  # (num_particles,)
            assert node_type.shape[0] == position.shape[1], f"{i=}"

            # noise_mask.append(torch.eq(node_type, torch.zeros_like(node_type)))

            # if self.split != "train":
            #     self.rollout_mask.append(self._get_rollout_mask(node_type))

            features = {}
            # velocity = self.compute_velocity(position, dt=self.dt)
            # acceleration = self.compute_acceleration(position, dt=self.dt)
            # velocity = self.normalize_velocity(velocity)
            # acceleration = self.normalize_acceleration(acceleration)

            features["position"] = position[: self.num_steps]
            # features["velocity"] = velocity[: self.num_steps + self.num_history]
            # features["acceleration"] = acceleration[: self.num_steps + self.num_history]

            self.node_type.append(F.one_hot(node_type, num_classes=self.num_node_types))
            self.node_features.append(features)

        # For each sequence, there are (num_steps - num_history - 1) values
        # with velocity and acceleration.
        self.num_samples_per_sequence = self.num_steps - self.num_history - 1
        self.length = num_sequences * self.num_samples_per_sequence

        logger.info("Finished dataset preparation.")

    def __getitem__(self, idx):
        if not (0 <= idx < self.length):
            raise IndexError(f"Invalid index {idx}, must be in [0, {self.length})")

        # graph and time step indices.
        gidx, tidx = divmod(idx, self.num_samples_per_sequence)

        # Current time step.
        t = tidx + self.num_history
        pos = self.node_features[gidx]["position"][tidx : t + 2]
        assert len(pos) == self.num_history + 2
        # Current position at t.
        pos_t = pos[-2]
        # Velocities.
        vel = self.time_diff(pos)
        # Target acceleration.
        acc = self.time_diff(vel)
        # Boundary features for the current position.
        boundary_features = self.compute_boundary_feature(
            pos_t, self.radius, bounds=self.bounds
        )

        # Add noise.
        # TODO(akamenev)

        # Normalize velocity and acceleration.
        vel = self.normalize_velocity(vel)
        acc = self.normalize_acceleration(acc)

        # Create graph node features.
        # (t, num_particles, dimension) -> (num_particles, t * dimension)
        vel_history = vel[:-1].permute(1, 0, 2).flatten(start_dim=1)

        node_features = torch.cat(
            (pos_t, vel_history, boundary_features, self.node_type[gidx]), dim=-1
        )

        target_pos = pos[-1]
        target_vel = vel[-1]
        target_acc = acc[-1]

        node_targets = torch.cat((target_pos, target_vel, target_acc), dim=-1)

        graph = dgl.graph(([], []), num_nodes=node_features.shape[0])
        graph.ndata["x"] = node_features
        graph.ndata["y"] = node_targets
        graph.ndata["pos"] = pos_t
        graph.ndata["t"] = torch.tensor([tidx]).repeat(
            node_features.shape[0]
        )  # just to track the start
        graph_update(graph, radius=self.radius)

        return graph

        # mesh_pos = self.node_features[gidx]["position"][tidx + self.num_history - 1]
        # history = self.node_features[gidx]["velocity"][
        #     tidx : tidx + self.num_history
        # ].flip(0)
        # history = torch.flatten(
        #     history.permute(1, 0, 2), start_dim=1
        # )  # (n_node, num_history * dimension)

        # if self.split == "train":
        #     # mesh_pos += torch.std(mesh_pos) * self.noise_std * torch.randn_like(mesh_pos)
        #     history += torch.std(history) * self.noise_std * torch.randn_like(history)

        # boundary_features = self.compute_boundary_feature(
        #     mesh_pos, self.radius, bound=self.bound
        # )
        # node_features = torch.cat(
        #     (mesh_pos, history, boundary_features, self.node_type[gidx]), dim=-1
        # )
        # # node_features[..., :self.dim] = 0 # position-invariance
        # target_pos = self.node_features[gidx]["position"][tidx + self.num_history]
        # target_vel = self.node_features[gidx]["velocity"][tidx + self.num_history]
        # target_acc = self.node_features[gidx]["acceleration"][
        #     tidx + self.num_history - 1
        # ]
        # node_targets = torch.cat((target_pos, target_vel, target_acc), dim=-1)

        # graph = dgl.graph(([], []), num_nodes=node_features.shape[0])
        # # edge_index = self.edge_index[gidx][tidx]
        # # graph = dgl.graph((edge_index[0], edge_index[1]), num_nodes=node_features.shape[0])
        # graph.ndata["x"] = node_features
        # graph.ndata["y"] = node_targets
        # graph.ndata["mesh_pos"] = mesh_pos
        # graph.ndata["t"] = torch.tensor([tidx]).repeat(
        #     node_features.shape[0]
        # )  # just to track the start
        # # compute_edge_attr(graph)
        # graph_update(graph, radius=self.radius)

        # if self.split == "train":
        #     return graph
        # else:
        #     rollout_mask = self.rollout_mask[gidx]
        #     return graph, rollout_mask

    def set_normalizer_device(self, device):
        pass
        # self.vel_mean = self.vel_mean.to(device)
        # self.vel_std = self.vel_std.to(device)
        # self.acc_mean = self.acc_mean.to(device)
        # self.acc_std = self.acc_std.to(device)

    def normalize_velocity(self, velocity):
        velocity = velocity - self.vel_mean.to(velocity.device)
        velocity = velocity / self.vel_std.to(velocity.device)
        return velocity

    def denormalize_velocity(self, velocity):
        velocity = velocity * self.vel_std.to(velocity.device)
        velocity = velocity + self.vel_mean.to(velocity.device)
        return velocity

    def normalize_acceleration(self, acceleration):
        acceleration = acceleration - self.acc_mean.to(acceleration.device)
        acceleration = acceleration / self.acc_std.to(acceleration.device)
        return acceleration

    def denormalize_acceleration(self, acceleration):
        acceleration = acceleration * self.acc_std.to(acceleration.device)
        acceleration = acceleration + self.acc_mean.to(acceleration.device)
        return acceleration

    def time_integrator(self, position, velocity, acceleration, dt, denormalize=True):
        # given the position x(t), velocity v(t), and acceleration a(t)
        # output x(t+1)
        if denormalize:
            velocity = self.denormalize_velocity(velocity)
            acceleration = self.denormalize_acceleration(acceleration)

        velocity_next = velocity + acceleration  # * dt
        position_next = position + velocity_next  # * dt
        return position_next, velocity_next

    def unpack_inputs(self, graph: dgl.DGLGraph):
        """Unpacks the graph inputs into position and velocity.

        Returns:
        --------
        Tuple
            position, velocity inputs. Velocity is normalized.
        """
        ndata = graph.ndata["x"]
        pos = ndata[..., : self.dim]
        vel = ndata[..., self.dim : self.dim + self.dim * self.num_history]
        # (num_particles, t * dimension) -> (t, num_particles, dimension)
        vel = vel.reshape(-1, self.num_history, self.dim).permute(1, 0, 2)
        return pos, vel

    def unpack_targets(self, graph: dgl.DGLGraph):
        """Unpacks the graph targets into position, velocity and acceleration.

        Returns:
        --------
        Tuple
            position, velocity, acceleration targets. Velocity and acceleration are normalized.
        """
        ndata = graph.ndata["y"]
        pos = ndata[..., : self.dim]
        vel = ndata[..., self.dim : 2 * self.dim]
        acc = ndata[..., 2 * self.dim : 3 * self.dim]
        return pos, vel, acc

    @staticmethod
    def time_diff(x: Tensor):
        return x[1:] - x[:-1]

    @staticmethod
    def compute_velocity(x, dt):
        # compute the derivative using finite difference
        v = torch.zeros_like(x)
        v[1:] = x[1:] - x[:-1]  # / dt
        v[0] = v[1]
        return v

    @staticmethod
    def compute_acceleration(x, dt):
        # compute the derivative using finite difference
        a = torch.zeros_like(x)
        a[1:-1] = x[2:] - 2 * x[1:-1] + x[:-2]  # / dt**2
        a[0] = a[1]
        a[-1] = a[-2]
        return a

    @staticmethod
    def compute_boundary_feature(position, radius=0.015, bounds=[0.1, 0.9]):
        distance = torch.cat([position - bounds[0], bounds[1] - position], dim=-1)
        features = torch.exp(-(distance**2) / radius**2)
        features[distance > radius] = 0
        return features

    @staticmethod
    def boundary_clamp(position, bounds=[0.1, 0.9], eps=0.001):
        return torch.clamp(position, min=bounds[0] + eps, max=bounds[1] - eps)

    def __len__(self):
        return self.length

    def _load_tf_data(self, path, split):
        """
        Utility for loading the .tfrecord dataset in DeepMind's MeshGraphNet repo:
        https://github.com/google-deepmind/deepmind-research/tree/master/learning_to_simulate
        Follow the instructions provided in that repo to download the .tfrecord files.
        """
        dataset = self._load_dataset(path, split)
        dataset_iterator = tf.data.make_one_shot_iterator(dataset)
        return dataset_iterator

    def _load_dataset(self, path, split):
        with open(os.path.join(path, "metadata.json"), "r") as fp:
            meta = json.loads(fp.read())
        dataset = tf.data.TFRecordDataset(os.path.join(path, split + ".tfrecord"))
        return dataset.map(
            functools.partial(parse_serialized_simulation_example, metadata=meta),
            num_parallel_calls=8,
        ).prefetch(tf.data.AUTOTUNE)

    @staticmethod
    def _get_rollout_mask(node_type):
        mask = torch.logical_or(
            torch.eq(node_type, torch.zeros_like(node_type)),
            torch.eq(
                node_type,
                torch.zeros_like(node_type) + 5,
            ),
        )
        return mask

    @staticmethod
    def _add_noise(features, targets, noise_std, noise_mask):
        noise = torch.normal(mean=0, std=noise_std, size=features.size())
        noise_mask = noise_mask.expand(features.size()[0], -1, features.size()[2])
        noise = torch.where(noise_mask, noise, torch.zeros_like(noise))
        features += noise * features
        targets -= noise * targets
        return features, targets
