# ignore_header_test
# ruff: noqa: E402

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

# © Copyright 2023 HP Development Company, L.P.
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


import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Embedding, Linear, ReLU

try:
    from torch_scatter import scatter
except ImportError:
    raise ImportError(
        "VFGN pipeline requires the PyTorch_Geometric library. Install the "
        + "package at: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html"
    )

from dataclasses import dataclass

from torch.utils.checkpoint import checkpoint

from ..meta import ModelMetaData

STD_EPSILON = 1e-8


@dataclass
class MetaData(ModelMetaData):
    name: str = "VFGN"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = True
    amp_cpu: bool = False  # Reflect padding not supported in bfloat16
    amp_gpu: bool = False
    # Inference
    onnx_cpu: bool = False
    onnx_gpu: bool = False
    onnx_runtime: bool = False
    # Physics informed
    var_dim: int = 1
    func_torch: bool = False
    auto_grad: bool = False


class MLPNet(torch.nn.Module):
    """
    A Multilayer Perceptron (MLP) network implemented in PyTorch, configurable with
    a variable number of hidden layers and layer normalization.
    """

    def __init__(
        self, mlp_hidden_size, mlp_num_hidden_layers, output_size, layer_norm=True
    ):
        super(MLPNet, self).__init__()
        self.mlp_hidden_size = mlp_hidden_size
        self.lins = []
        if mlp_num_hidden_layers > 1:
            for i in range(mlp_num_hidden_layers - 1):
                self.lins.append(Linear(mlp_hidden_size, mlp_hidden_size))
        self.lins = torch.nn.ModuleList(self.lins)

        self.lin_e = Linear(mlp_hidden_size, output_size)
        self.layer_norm = layer_norm
        self.relu = ReLU()

    def dynamic(self, name: str, module_class, *args, **kwargs):
        if not hasattr(self, name):
            print("call dynamic, MLPnet")
            self.add_module(name, module_class(*args, **kwargs))
        return getattr(self, name)

    def forward(self, x):
        # origin_device=x.device

        lin_s = self.dynamic("lin_s", Linear, x.shape[-1], self.mlp_hidden_size)
        # lin_s = lin_s.to(origin_device)

        x = lin_s(x)
        x = self.relu(x)

        for lin_i in self.lins:
            x = lin_i(x)
            x = self.relu(x)

        x = self.lin_e(x)
        if self.layer_norm:
            x = F.layer_norm(x, x.shape[1:])
        return x


class EncoderNet(torch.nn.Module):
    """
    Encoder network.
    """

    def __init__(self, mlp_hidden_size, mlp_num_hidden_layers, latent_size):
        super(EncoderNet, self).__init__()

        self._mlp_hidden_size = mlp_hidden_size
        self._mlp_num_hidden_layers = mlp_num_hidden_layers

        self.edge_mlp = MLPNet(mlp_hidden_size, mlp_num_hidden_layers, latent_size)
        self.node_mlp = MLPNet(mlp_hidden_size, mlp_num_hidden_layers, latent_size)

    def forward(self, x, edge_attr):
        x = self.node_mlp(x)
        edge_attr = self.edge_mlp(edge_attr)

        return x, edge_attr


class EdgeBlock(torch.nn.Module):
    """Edge block"""

    def __init__(
        self,
        mlp_hidden_size,
        mlp_num_hidden_layers,
        latent_size,
        node_dim=0,
        use_receiver_nodes=True,
        use_sender_nodes=True,
    ):
        super(EdgeBlock, self).__init__()
        self.node_dim = node_dim
        self._edge_model = MLPNet(mlp_hidden_size, mlp_num_hidden_layers, latent_size)

        self.use_receiver_nodes = use_receiver_nodes
        self.use_sender_nodes = use_sender_nodes

    def forward(self, x, edge_attr, receivers, senders):
        edges_to_collect = []
        edges_to_collect.append(edge_attr)

        if self.use_receiver_nodes:
            receivers_edge = x[receivers, :]
            edges_to_collect.append(receivers_edge)

        if self.use_sender_nodes:
            senders_edge = x[senders, :]
            edges_to_collect.append(senders_edge)

        collected_edges = torch.cat(edges_to_collect, axis=-1)

        updated_edges = self._edge_model(collected_edges)

        return x, updated_edges, receivers, senders


class NodeBlock(torch.nn.Module):
    """Node block"""

    def __init__(
        self,
        mlp_hidden_size,
        mlp_num_hidden_layers,
        latent_size,
        aggr="add",
        node_dim=0,
        use_received_edges=True,
        use_sent_edges=False,
    ):
        super(NodeBlock, self).__init__()
        self.aggr = aggr
        self.node_dim = node_dim

        self.use_received_edges = use_received_edges
        self.use_sent_edges = use_sent_edges

        self._node_model = MLPNet(mlp_hidden_size, mlp_num_hidden_layers, latent_size)

    def forward(self, x, edge_attr, receivers, senders):
        nodes_to_collect = []
        nodes_to_collect.append(x)

        dim_size = x.shape[self.node_dim]

        # aggregate received edges
        if self.use_received_edges:
            receivers_edge = scatter(
                dim=self.node_dim,
                dim_size=dim_size,
                index=receivers,
                src=edge_attr,
                reduce=self.aggr,
            )
            nodes_to_collect.append(receivers_edge)

        # aggregate sent edges
        if self.use_sent_edges:
            senders_edge = scatter(
                dim=self.node_dim,
                dim_size=dim_size,
                index=senders,
                src=edge_attr,
                reduce=self.aggr,
            )
            nodes_to_collect.append(senders_edge)

        collected_nodes = torch.cat(nodes_to_collect, axis=-1)

        updated_nodes = self._node_model(collected_nodes)

        return updated_nodes, edge_attr, receivers, senders


class InteractionNet(torch.nn.Module):
    """Interaction network"""

    def __init__(
        self,
        mlp_hidden_size,
        mlp_num_hidden_layers,
        latent_size,
        aggr="add",
        node_dim=0,
    ):
        super(InteractionNet, self).__init__()
        self._edge_block = EdgeBlock(
            mlp_hidden_size, mlp_num_hidden_layers, latent_size, aggr, node_dim
        )
        self._node_block = NodeBlock(
            mlp_hidden_size, mlp_num_hidden_layers, latent_size, aggr, node_dim
        )

    def forward(self, x, edge_attr, receivers, senders):
        if not (x.shape[-1] == edge_attr.shape[-1]):
            raise ValueError("node feature size should equal to edge feature size")

        return self._node_block(*self._edge_block(x, edge_attr, receivers, senders))


class ResInteractionNet(torch.nn.Module):
    """Residual interaction network"""

    def __init__(
        self,
        mlp_hidden_size,
        mlp_num_hidden_layers,
        latent_size,
        aggr="add",
        node_dim=0,
    ):
        super(ResInteractionNet, self).__init__()
        self.itn = InteractionNet(
            mlp_hidden_size, mlp_num_hidden_layers, latent_size, aggr, node_dim
        )

    def forward(self, x, edge_attr, receivers, senders):
        x_res, edge_attr_res, receivers, senders = self.itn(
            x, edge_attr, receivers, senders
        )

        x_new = x + x_res
        edge_attr_new = edge_attr + edge_attr_res

        return x_new, edge_attr_new, receivers, senders


class DecoderNet(torch.nn.Module):
    """decoder network"""

    def __init__(self, mlp_hidden_size, mlp_num_hidden_layers, output_size):
        super(DecoderNet, self).__init__()
        self.mlp = MLPNet(
            mlp_hidden_size, mlp_num_hidden_layers, output_size, layer_norm=False
        )

    def forward(self, x):
        # number of layer is important, or the network will overfit
        x = self.mlp(x)
        return x


class EncodeProcessDecode(torch.nn.Module):
    """Module that assembles encoder, processor, and decoder"""

    def __init__(
        self,
        latent_size,
        mlp_hidden_size,
        mlp_num_hidden_layers,
        num_message_passing_steps,
        output_size,
        device_list=None,
    ):
        super(EncodeProcessDecode, self).__init__()
        if device_list is None:
            self.device_list = ["cpu"]
        else:
            self.device_list = device_list

        self._encoder_network = EncoderNet(
            mlp_hidden_size, mlp_num_hidden_layers, latent_size
        )

        self._processor_networks = []
        for _ in range(num_message_passing_steps):
            self._processor_networks.append(
                InteractionNet(mlp_hidden_size, mlp_num_hidden_layers, latent_size)
            )
        self._processor_networks = torch.nn.ModuleList(self._processor_networks)

        self._decoder_network = DecoderNet(
            mlp_hidden_size, mlp_num_hidden_layers, output_size
        )

    def set_device(self, device_list):
        """sets the device"""
        self.device_list = device_list

    def forward(self, x, edge_attr, receivers, senders):
        x, edge_attr = self._encoder_network(x, edge_attr)
        # print(
        #     "Check shape of EncodeProcessDecode, x/ edge_attr: ",
        #     x.shape,
        #     edge_attr.shape,
        # )

        pre_x = x
        pre_edge_attr = edge_attr

        n_steps = len(self._processor_networks)
        n_inter = int(n_steps / len(self.device_list))

        i = 0
        j = 0

        origin_device = x.device

        for processor_network_k in self._processor_networks:
            p_device = self.device_list[j]
            processor_network_k = processor_network_k.to(p_device)
            pre_x = pre_x.to(p_device)
            pre_edge_attr = pre_edge_attr.to(p_device)
            receivers = receivers.to(p_device)
            senders = senders.to(p_device)

            diff_x, diff_edge_attr, receivers, senders = checkpoint(
                processor_network_k, pre_x, pre_edge_attr, receivers, senders
            )

            pre_x = x.to(p_device) + diff_x
            pre_edge_attr = edge_attr.to(p_device) + diff_edge_attr
            i += 1
            if i % n_inter == 0:
                j += 1

        x = self._decoder_network(pre_x.to(origin_device))

        return x


class LearnedSimulator(torch.nn.Module):
    """The VFGN architecture"""

    # setting DGCNN (dynamic graph computation)
    def __init__(
        self,
        num_dimensions,
        num_seq,
        boundaries,
        num_particle_types,
        particle_type_embedding_size,
        normalization_stats,
        graph_mode="radius",
        connectivity_param=0.015,
    ):
        super(LearnedSimulator, self).__init__()
        # network parameters
        self._latent_size = 128
        self._mlp_hidden_size = 128
        self._mlp_num_hidden_layers = 2
        self._num_message_passing_steps = 10
        self._num_dimensions = num_dimensions
        self._num_seq = num_seq

        # graph parameters
        self._connectivity_param = connectivity_param  # either knn or radius
        self._boundaries = boundaries
        self._normalization_stats = normalization_stats

        self.graph_mode = graph_mode

        self._graph_network = EncodeProcessDecode(
            self._latent_size,
            self._mlp_hidden_size,
            self._mlp_num_hidden_layers,
            self._num_message_passing_steps,
            self._num_dimensions,
        )

        # positional embedding with different particle types
        self._num_particle_types = num_particle_types
        self.embedding = Embedding(
            self._num_particle_types + 1, particle_type_embedding_size
        )
        self.message_passing_devices = []

    def setMessagePassingDevices(self, devices):
        """
        setts the devices to be used for message passing in the neural network model.
        """
        self.message_passing_devices = devices

    def to(self, device):
        """Device transfer"""
        new_self = super(LearnedSimulator, self).to(device)
        new_self._boundaries = self._boundaries.to(device)
        for key in self._normalization_stats:
            new_self._normalization_stats[key].to(device)
        if device != "cpu":
            self._graph_network.set_device(self.message_passing_devices)
        return new_self

    def time_diff(self, input_seq):
        """
        Calculates the difference between consecutive elements in a sequence, effectively computing the discrete time derivative.
        """
        return input_seq[:, 1:] - input_seq[:, :-1]

    def _compute_connectivity_for_batch(
        self, senders_list, receivers_list, n_node, n_edge
    ):
        senders_per_graph_list = np.split(senders_list, np.cumsum(n_edge[:-1]), axis=0)
        receivers_per_graph_list = np.split(
            receivers_list, np.cumsum(n_edge[:-1]), axis=0
        )

        receivers_list = []
        senders_list = []
        n_edge_list = []
        num_nodes_in_previous_graphs = 0

        n = n_node.shape[0]

        drop_out_rate = 0.6

        # Compute connectivity for each graph in the batch.
        for i in range(n):
            total_num_edges_graph_i = len(senders_per_graph_list[i])

            random_num = random.choice([True, False])

            if random_num:
                choiced_indices = random.choices(
                    [j for j in range(total_num_edges_graph_i)],
                    k=int(total_num_edges_graph_i * drop_out_rate),
                )
                choiced_indices = sorted(choiced_indices)

                senders_graph_i = senders_per_graph_list[i][choiced_indices]
                receivers_graph_i = receivers_per_graph_list[i][choiced_indices]
            else:
                senders_graph_i = senders_per_graph_list[i]
                receivers_graph_i = receivers_per_graph_list[i]

            num_edges_graph_i = len(senders_graph_i)
            n_edge_list.append(num_edges_graph_i)

            # Because the inputs will be concatenated, we need to add offsets to the
            # sender and receiver indices according to the number of nodes in previous
            # graphs in the same batch.
            receivers_list.append(receivers_graph_i + num_nodes_in_previous_graphs)
            senders_list.append(senders_graph_i + num_nodes_in_previous_graphs)

            num_nodes_graph_i = n_node[i]
            num_nodes_in_previous_graphs += num_nodes_graph_i

        # Concatenate all of the results.
        senders = np.concatenate(senders_list, axis=0).astype(np.int32)
        receivers = np.concatenate(receivers_list, axis=0).astype(np.int32)

        return senders, receivers

    def get_random_walk_noise_for_position_sequence(
        self, position_sequence, noise_std_last_step
    ):
        """Returns random-walk noise in the velocity applied to the position."""

        velocity_sequence = self.time_diff(position_sequence)

        # We want the noise scale in the velocity at the last step to be fixed.
        # Because we are going to compose noise at each step using a random_walk:
        # std_last_step**2 = num_velocities * std_each_step**2
        # so to keep `std_last_step` fixed, we apply at each step:
        # std_each_step `std_last_step / np.sqrt(num_input_velocities)`
        # TODO(alvarosg): Make sure this is consistent with the value and
        # description provided in the paper.
        num_velocities = velocity_sequence.shape[1]
        velocity_sequence_noise = torch.empty(
            velocity_sequence.shape, dtype=velocity_sequence.dtype
        ).normal_(
            mean=0, std=noise_std_last_step / num_velocities**0.5
        )  # float

        # Apply the random walk
        velocity_sequence_noise = torch.cumsum(velocity_sequence_noise, dim=1)

        # Integrate the noise in the velocity to the positions, assuming
        # an Euler intergrator and a dt = 1, and adding no noise to the very first
        # position (since that will only be used to calculate the first position
        # change).
        position_sequence_noise = torch.cat(
            [
                torch.zeros(
                    velocity_sequence_noise[:, 0:1].shape, dtype=velocity_sequence.dtype
                ),
                torch.cumsum(velocity_sequence_noise, axis=1),
            ],
            axis=1,
        )

        return position_sequence_noise

    def EncodingFeature(
        self,
        position_sequence,
        n_node,
        n_edge,
        senders_list,
        receivers_list,
        global_context,
        particle_types,
    ):
        """Feature encoder"""
        # aggregate all features
        most_recent_position = position_sequence[:, -1]
        velocity_sequence = self.time_diff(position_sequence)
        acceleration_sequence = self.time_diff(velocity_sequence)

        # dynamically updage the graph
        senders, receivers = self._compute_connectivity_for_batch(
            senders_list.cpu().detach().numpy(),
            receivers_list.cpu().detach().numpy(),
            n_node.cpu().detach().numpy(),
            n_edge.cpu().detach().numpy(),
        )
        senders = torch.LongTensor(senders).to(position_sequence.device)
        receivers = torch.LongTensor(receivers).to(position_sequence.device)

        # 1. Node features
        node_features = []
        velocity_stats = self._normalization_stats["velocity"]
        normalized_velocity_sequence = (
            velocity_sequence - velocity_stats.mean
        ) / velocity_stats.std
        normalized_velocity_sequence = normalized_velocity_sequence[:, -1]

        flat_velocity_sequence = normalized_velocity_sequence.reshape(
            [normalized_velocity_sequence.shape[0], -1]
        )
        node_features.append(flat_velocity_sequence)

        acceleration_stats = self._normalization_stats["acceleration"]
        normalized_acceleration_sequence = (
            acceleration_sequence - acceleration_stats.mean
        ) / acceleration_stats.std

        flat_acceleration_sequence = normalized_acceleration_sequence.reshape(
            [normalized_acceleration_sequence.shape[0], -1]
        )
        node_features.append(flat_acceleration_sequence)

        if self._num_particle_types > 1:
            particle_type_embedding = self.embedding(particle_types)
            node_features.append(particle_type_embedding)

        # 2. Edge features
        edge_features = []
        # Relative displacement and distances normalized to radius
        normalized_relative_displacements = (
            most_recent_position.index_select(0, senders)
            - most_recent_position.index_select(0, receivers)
        ) / self._connectivity_param
        edge_features.append(normalized_relative_displacements)
        normalized_relative_distances = torch.norm(
            normalized_relative_displacements, dim=-1, keepdim=True
        )
        edge_features.append(normalized_relative_distances)

        # 3. Normalized the global context.
        if global_context is not None:
            context_stats = self._normalization_stats["context"]
            # Context in some datasets are all zero, so add an epsilon for numerical
            global_context = (global_context - context_stats.mean) / torch.maximum(
                context_stats.std,
                torch.FloatTensor([STD_EPSILON]).to(context_stats.std.device),
            )

            global_features = []
            for i in range(global_context.shape[0]):
                global_context_ = torch.unsqueeze(global_context[i], 0)
                context_i = torch.repeat_interleave(
                    global_context_, n_node[i].to(torch.long), dim=0
                )

                global_features.append(context_i)

            global_features = torch.cat(global_features, 0)
            global_features = global_features.reshape(global_features.shape[0], -1)

            node_features.append(global_features)

        x = torch.cat(node_features, -1)
        edge_attr = torch.cat(edge_features, -1)

        #  cast from double to float as the input of network
        x = x.float()
        edge_attr = edge_attr.float()

        return x, edge_attr, senders, receivers

    def DecodingFeature(
        self, normalized_accelerations, position_sequence, predict_length
    ):
        """Feature decoder"""
        #  cast from float to double as the output of network
        normalized_accelerations = normalized_accelerations.double()

        # model works on the normal space - need to invert it to the original space
        acceleration_stats = self._normalization_stats["acceleration"]
        normalized_accelerations = normalized_accelerations.reshape(
            [-1, predict_length, 3]
        )

        accelerations = (
            normalized_accelerations * acceleration_stats.std
        ) + acceleration_stats.mean
        velocity_changes = torch.cumsum(
            accelerations, axis=1, dtype=accelerations.dtype
        )

        most_recent_velocity = position_sequence[:, -1] - position_sequence[:, -2]
        most_recent_velocity = torch.unsqueeze(most_recent_velocity, axis=1)
        most_recent_velocities = torch.tile(
            most_recent_velocity, [1, predict_length, 1]
        )
        velocities = most_recent_velocities + velocity_changes

        position_changes = torch.cumsum(velocities, axis=1, dtype=velocities.dtype)

        most_recent_position = position_sequence[:, -1]
        most_recent_position = torch.unsqueeze(most_recent_position, axis=1)
        most_recent_positions = torch.tile(most_recent_position, [1, predict_length, 1])

        new_positions = most_recent_positions + position_changes

        return new_positions

    def _inverse_decoder_postprocessor(self, next_positions, position_sequence):
        """Inverse of `_decoder_postprocessor`."""
        most_recent_positions = position_sequence[:, -2:]
        previous_positions = torch.cat(
            [most_recent_positions, next_positions[:, :-1]], axis=1
        )

        positions = torch.cat(
            [torch.unsqueeze(position_sequence[:, -1], axis=1), next_positions], axis=1
        )

        velocities = positions - previous_positions
        accelerations = velocities[:, 1:] - velocities[:, :-1]

        acceleration_stats = self._normalization_stats["acceleration"]
        normalized_accelerations = (
            accelerations - acceleration_stats.mean
        ) / acceleration_stats.std

        normalized_accelerations = normalized_accelerations.reshape(
            [-1, self._num_dimensions]
        )

        #  cast from double to float as the input of network
        normalized_accelerations = normalized_accelerations.float()
        return normalized_accelerations

    def inference(
        self,
        position_sequence,
        n_particles_per_example,
        n_edges_per_example,
        senders,
        receivers,
        predict_length,
        global_context=None,
        particle_types=None,
    ):
        """
        Encoder & Decoder processes with graph neural network
        """
        input_graph = self.EncodingFeature(
            position_sequence,
            n_particles_per_example,
            n_edges_per_example,
            senders,
            receivers,
            global_context,
            particle_types,
        )

        predicted_normalized_accelerations = self._graph_network(*input_graph)

        next_position = self.DecodingFeature(
            predicted_normalized_accelerations, position_sequence, predict_length
        )

        return next_position

    def forward(
        self,
        next_positions,
        position_sequence_noise,
        position_sequence,
        n_particles_per_example,
        n_edges_per_example,
        senders,
        receivers,
        predict_length,
        global_context=None,
        particle_types=None,
    ):
        """
        # PyTorch version main training module

        Original TF implementation explanation
        Produces normalized and predicted acceleration targets.

        Args:
        next_position: Tensor of shape [num_particles_in_batch, num_dimensions]
            with the positions the model should output given the inputs.
        position_sequence_noise: Tensor of the same shape as `position_sequence`
            with the noise to apply to each particle.
        position_sequence, n_node, global_context, particle_types: Inputs to the
            model as defined by `_build`.

        Returns:
        Tensors of shape [num_particles_in_batch, num_dimensions] with the
            predicted and target normalized accelerations.
        """

        # Add noise to the input position sequence.
        noisy_position_sequence = position_sequence + position_sequence_noise

        # Perform the forward pass with the noisy position sequence.
        input_graph = self.EncodingFeature(
            noisy_position_sequence,
            n_particles_per_example,
            n_edges_per_example,
            senders,
            receivers,
            global_context,
            particle_types,
        )

        predicted_normalized_accelerations = self._graph_network(*input_graph)

        # Calculate the target acceleration, using an `adjusted_next_position `that
        # is shifted by the noise in the last input position.
        most_recent_noise = position_sequence_noise[:, -1]

        most_recent_noise = torch.unsqueeze(most_recent_noise, axis=1)

        most_recent_noises = torch.tile(most_recent_noise, [1, predict_length, 1])

        next_position_adjusted = next_positions + most_recent_noises
        # print("next_position_adjusted: ", next_position_adjusted, next_position_adjusted.shape)

        target_normalized_acceleration = self._inverse_decoder_postprocessor(
            next_position_adjusted, noisy_position_sequence
        )
        # As a result the inverted Euler update in the `_inverse_decoder` produces:
        # * A target acceleration that does not explicitly correct for the noise in
        #   the input positions, as the `next_position_adjusted` is different
        #   from the true `next_position`.
        # * A target acceleration that exactly corrects noise in the input velocity
        #   since the target next velocity calculated by the inverse Euler update
        #   as `next_position_adjusted - noisy_position_sequence[:,-1]`
        #   matches the ground truth next velocity (noise cancels out).
        # print("predicted_normalized_accelerations: ", predicted_normalized_accelerations, predicted_normalized_accelerations.shape)
        # print("target_normalized_acceleration: ", target_normalized_acceleration, target_normalized_acceleration.shape)
        # #for both:  torch.Size([71424, 3])
        return predicted_normalized_accelerations, target_normalized_acceleration

    def get_normalized_acceleration(self, acceleration, predict_length):
        """
        Normalizes the acceleration data using predefined statistics and
        replicates it across a specified prediction length.
        """
        acceleration_stats = self._normalization_stats["acceleration"]
        normalized_acceleration = (
            acceleration - acceleration_stats.mean
        ) / acceleration_stats.std
        normalized_acceleration = torch.tile(normalized_acceleration, [predict_length])
        return normalized_acceleration
