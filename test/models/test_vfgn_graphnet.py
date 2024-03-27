# ignore_header_test
# ruff: noqa: E402

# © Copyright 2023 HP Development Company, L.P.
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


import random

import numpy as np
import pytest
import torch

# can be commented out when test in Modulus installation
# import sys
# sys.path.append("/home/chenle/codes/sintering/modulus24/")
from modulus.models.vfgn.graph_network_modules import (
    EncodeProcessDecode,
    LearnedSimulator,
    MLPNet,
)

from . import common


@pytest.mark.skip(reason="Skipping, these tests need better dependency management")
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_mlpnet_forward(device):
    """Test MLP-NET forward pass"""
    torch.manual_seed(0)
    # Construct MLP-NET model

    model = MLPNet(
        mlp_hidden_size=128,
        mlp_num_hidden_layers=2,
        output_size=128,
    ).to(device)

    bsize = 2
    node_size = 1000
    feat_size = 3
    invar = torch.randn(bsize, node_size, feat_size).to(device)
    assert common.validate_forward_accuracy(
        model, (invar,), file_name="vfgn_mlp_output.pth", atol=1e-4
    )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_mlpnet_optims(device):
    def setup_model():
        """Setups up fresh MLP-NET model and inputs for optim test"""
        model = MLPNet(
            mlp_hidden_size=128,
            mlp_num_hidden_layers=2,
            output_size=3,
        ).to(device)

        bsize = 2
        node_size = random.randint(10, 10000)
        feat_size = random.randint(10, 100)
        invar = torch.randn(bsize, node_size, feat_size).to(device)
        return model, invar

    # Ideally always check graphs first
    model, invar = setup_model()
    assert common.validate_cuda_graphs(model, (invar,))

    # Check JIT
    model, invar = setup_model()
    assert common.validate_jit(model, (invar,))
    # Check AMP
    model, invar = setup_model()
    assert common.validate_amp(model, (invar,))
    # Check Combo
    model, invar = setup_model()
    assert common.validate_combo_optims(model, (invar,))


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_mlpnet_checkpoint(device):
    """Test MLP-NET checkpoint save/load"""
    # Construct MLP-NET models
    model_1 = MLPNet(
        mlp_hidden_size=128,
        mlp_num_hidden_layers=2,
        output_size=3,
    ).to(device)

    model_2 = MLPNet(
        mlp_hidden_size=128,
        mlp_num_hidden_layers=2,
        output_size=3,
    ).to(device)

    # bsize = random.randint(1, 4)
    # invar = torch.randn(bsize, 2, 8).to(device)
    # bsize = 2
    # node_size = random.randint(10, 10000)
    # feat_size = random.randint(10, 100)
    bsize = 2
    node_size = 1000
    feat_size = 3
    invar = torch.randn(bsize, node_size, feat_size).to(device)

    assert common.validate_checkpoint(model_1, model_2, (invar,))


@common.check_ort_version()
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_mlpnet_deploy(device):
    """Test MLP-NET  deployment support"""
    model = MLPNet(
        mlp_hidden_size=128,
        mlp_num_hidden_layers=2,
        output_size=3,
    ).to(device)

    bsize = random.randint(1, 2)
    node_size = random.randint(10, 10000)
    feat_size = random.randint(10, 100)
    invar = torch.randn(bsize, node_size, feat_size).to(device)

    assert common.validate_onnx_export(model, (invar,))
    assert common.validate_onnx_runtime(model, (invar,))


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_encodeProcessDecode_forward(device):
    """Test EncodeProcessDecode forward pass, model load, save"""

    torch.manual_seed(0)
    # Construct EncodeProcessDecode model

    model = EncodeProcessDecode(
        latent_size=128,
        mlp_hidden_size=128,
        mlp_num_hidden_layers=2,
        num_message_passing_steps=10,
        output_size=3,
    ).to(device)

    # random init node attribute tensor, edge attribute tensor
    node_cnt, node_feat_size = 5000, 61
    edge_cnt, edge_feat_size = 7700, 5

    invar_node_attr = torch.randn(node_cnt, node_feat_size).to(device)
    invar_edge_attr = torch.randn(edge_cnt, edge_feat_size).to(device)

    # random init sender, receiver index list: int
    invar_receivers = torch.randint(node_cnt, (edge_cnt,)).to(device)
    invar_senders = torch.randint(node_cnt, (edge_cnt,)).to(device)

    invar = (invar_node_attr, invar_edge_attr, invar_receivers, invar_senders)

    assert common.validate_forward_accuracy(model, (*invar,), atol=1e-4)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_encodeProcessDecode_constructor(device):
    """Test EncodeProcessDecode constructor options - in/out dimensions match"""

    # Define dictionary of constructor args
    arg_list = []
    arg_list.append(
        {
            "latent_size": 128,
            "mlp_hidden_size": 128,
            "mlp_num_hidden_layers": 2,
            "num_message_passing_steps": random.randint(1, 11),
            "output_size": 3,
        }
    )

    for kw_args in arg_list:
        # Construct FC model
        model = EncodeProcessDecode(**kw_args).to(device)

        node_cnt, node_feat_size = random.randint(1, 10000), random.randint(1, 100)
        edge_cnt, edge_feat_size = random.randint(1, 10000), random.randint(1, 100)
        invar_node_attr = torch.randn(node_cnt, node_feat_size).to(device)
        invar_edge_attr = torch.randn(edge_cnt, edge_feat_size).to(device)

        invar_receivers = torch.randint(node_cnt, (edge_cnt,)).to(device)
        invar_senders = torch.randint(node_cnt, (edge_cnt,)).to(device)

        invar = (invar_node_attr, invar_edge_attr, invar_receivers, invar_senders)

        outvar = model(*invar)
        assert outvar.shape == (node_cnt, 3)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_encodeProcessDecode_optims(device):
    def setup_model():
        """Setups up fresh EncodeProcessDecode model and inputs for optim test"""
        model = EncodeProcessDecode(
            latent_size=128,
            mlp_hidden_size=128,
            mlp_num_hidden_layers=2,
            num_message_passing_steps=10,
            output_size=3,
        ).to(device)

        node_cnt, node_feat_size = random.randint(1, 10000), random.randint(1, 100)
        edge_cnt, edge_feat_size = random.randint(1, 10000), random.randint(1, 100)

        invar_node_attr = torch.randn(node_cnt, node_feat_size).to(device)
        invar_edge_attr = torch.randn(edge_cnt, edge_feat_size).to(device)
        invar_receivers = torch.randint(node_cnt, (edge_cnt,)).to(device)
        invar_senders = torch.randint(node_cnt, (edge_cnt,)).to(device)

        invar = (invar_node_attr, invar_edge_attr, invar_receivers, invar_senders)

        return model, invar

    # Ideally always check graphs first
    model, invar = setup_model()
    assert common.validate_cuda_graphs(model, (*invar,))

    # Check JIT
    model, invar = setup_model()
    assert common.validate_jit(model, (*invar,))
    # Check AMP
    model, invar = setup_model()
    assert common.validate_amp(model, (*invar,))
    # Check Combo
    model, invar = setup_model()
    assert common.validate_combo_optims(model, (*invar,))


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_encodeProcessDecode_checkpoint(device):
    """Test EncodeProcessDecode checkpoint save/load"""
    # Construct MLP-NET models
    model_1 = EncodeProcessDecode(
        latent_size=128,
        mlp_hidden_size=128,
        mlp_num_hidden_layers=2,
        num_message_passing_steps=10,
        output_size=3,
    ).to(device)

    model_2 = EncodeProcessDecode(
        latent_size=128,
        mlp_hidden_size=128,
        mlp_num_hidden_layers=2,
        num_message_passing_steps=10,
        output_size=3,
    ).to(device)

    node_cnt, node_feat_size = random.randint(1, 10000), random.randint(1, 100)
    edge_cnt, edge_feat_size = random.randint(1, 10000), random.randint(1, 100)
    invar_node_attr = torch.randn(node_cnt, node_feat_size).to(device)
    invar_edge_attr = torch.randn(edge_cnt, edge_feat_size).to(device)

    invar_receivers = torch.randint(node_cnt, (edge_cnt,)).to(device)
    invar_senders = torch.randint(node_cnt, (edge_cnt,)).to(device)

    invar = (invar_node_attr, invar_edge_attr, invar_receivers, invar_senders)

    assert common.validate_checkpoint(model_1, model_2, (*invar,))


@common.check_ort_version()
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_encodeProcessDecode_deploy(device):
    """Test EncodeProcessDecode deployment support"""
    model = EncodeProcessDecode(
        latent_size=128,
        mlp_hidden_size=128,
        mlp_num_hidden_layers=2,
        num_message_passing_steps=10,
        output_size=3,
    ).to(device)

    node_cnt, node_feat_size = random.randint(1, 10000), random.randint(1, 100)
    edge_cnt, edge_feat_size = random.randint(1, 10000), random.randint(1, 100)
    invar_node_attr = torch.randn(node_cnt, node_feat_size).to(device)
    invar_edge_attr = torch.randn(edge_cnt, edge_feat_size).to(device)

    invar_receivers = torch.randint(node_cnt, (edge_cnt,)).to(device)
    invar_senders = torch.randint(node_cnt, (edge_cnt,)).to(device)

    invar = (invar_node_attr, invar_edge_attr, invar_receivers, invar_senders)

    assert common.validate_onnx_runtime(model, (*invar,))


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_simulator_forward(device):
    """Test VFGN simulator forward pass"""
    torch.manual_seed(0)
    # Construct VFGN simulator model

    class Stats:
        def __init__(self, mean, std):
            self.mean = mean
            self.std = std

        def to(self, device):
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)
            return self

    dummy_metadata = [
        -0.008786813221081134,
        -0.004392095496219808,
        -0.00173827297612587,
    ]
    dummy_context_meta = [747.9937656176759]
    dummy_stats = Stats(
        torch.DoubleTensor(np.array(dummy_metadata, dtype=np.float64)),
        torch.DoubleTensor(np.array(dummy_metadata, dtype=np.float64)),
    )
    dummy_context_stats = Stats(
        torch.DoubleTensor(np.array(dummy_context_meta, dtype=np.float64)),
        torch.DoubleTensor(np.array(dummy_context_meta, dtype=np.float64)),
    )
    normalization_stats = {
        "acceleration": dummy_stats,
        "velocity": dummy_stats,
        "context": dummy_context_stats,
    }
    model = LearnedSimulator(
        num_dimensions=3,
        num_seq=5,  # the default implementation on INPUT_SEQUENCE_LENGTH >= 3
        boundaries=torch.DoubleTensor(
            [[-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0]]
        ),  # torch.DoubleTensor(metadata['bounds'])
        num_particle_types=3,
        particle_type_embedding_size=16,
        normalization_stats=normalization_stats,
    ).to(device)

    # initialize node number, simulation steps
    bsize = 2
    node_cnt = 1024
    edge_cnt = 4096
    sim_steps = 5
    k = 4

    invar_next_positions = torch.randn(bsize * node_cnt, 1, 3).to(
        device
    )  # target to predict
    invar_position_sequence_noise = torch.randn(bsize * node_cnt, sim_steps, 3).to(
        device
    )  # noise augmentation
    invar_position_sequence = torch.randn(bsize * node_cnt, sim_steps, 3).to(
        device
    )  # input sequence

    invar_n_particles = torch.tensor([node_cnt] * bsize).to(
        device
    )  # node size per batch
    invar_n_edges = torch.tensor([edge_cnt] * bsize).to(device)  # edge size per batch

    # identity graph with repeatation
    invar_receivers = (
        torch.arange(node_cnt).repeat(bsize).repeat_interleave(k).to(device)
    )
    invar_senders = torch.arange(node_cnt).repeat(bsize).repeat_interleave(k).to(device)

    invar_predict_length = 2  # random.randint(1, 5)
    invar_global_context = torch.randn(bsize, sim_steps, 1).to(device)
    invar_particle_types = torch.randint(3, (bsize * node_cnt,)).to(device)

    print("invar_global_context: ", invar_global_context.shape)
    invar = (
        invar_next_positions,
        invar_position_sequence_noise,
        invar_position_sequence,
        invar_n_particles,
        invar_n_edges,
        invar_senders,
        invar_receivers,
        invar_predict_length,
        invar_global_context,
        invar_particle_types,
    )

    assert common.validate_forward_accuracy(
        model,
        (*invar,),
        # file_name=f"vfgn_ls_output.pth",
        atol=1e-4,
    )


# def test_simulator_forward_batch(device):
#     """Test VFGN simulator forward pass"""
#     torch.manual_seed(0)
#     # Construct VFGN simulator model
#
#     class Stats:
#         def __init__(self, mean, std):
#             self.mean = mean
#             self.std = std
#
#         def to(self, device):
#             self.mean = self.mean.to(device)
#             self.std = self.std.to(device)
#             return self
#
#     cast = lambda v: np.array(v, dtype=np.float64)
#     dummy_metadata = [-0.008786813221081134, -0.004392095496219808, -0.00173827297612587]
#     dummy_stats = Stats(torch.DoubleTensor(cast(dummy_metadata)),
#                         torch.DoubleTensor(cast(dummy_metadata)))
#
#     normalization_stats = {'acceleration': dummy_stats, 'velocity': dummy_stats, 'context': dummy_stats}
#     model = LearnedSimulator(
#         num_dimensions=3,
#         num_seq=5,  # the default implementation on INPUT_SEQUENCE_LENGTH >= 3
#         boundaries=torch.DoubleTensor([[-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0]]), # torch.DoubleTensor(metadata['bounds'])
#         num_particle_types=3,
#         particle_type_embedding_size=16,
#         normalization_stats=normalization_stats,
#     ).to(device)
#
#     # initialize node number, simulation steps s
#     bsize = 2
#     node_cnt = random.randint(1, 10000)
#     edge_cnt = random.randint(1, 10000)
#     sim_steps = random.randint(1, 100)
#
#     invar_next_positions = torch.randn(bsize, node_cnt, 3).to(device)
#     invar_position_sequence_noise = torch.randn(bsize, node_cnt, sim_steps, 3).to(device)
#     invar_position_sequence = torch.randn(bsize, node_cnt, sim_steps, 3).to(device)
#
#     invar_n_particles = torch.tensor([node_cnt]).to(device)
#     invar_n_edges = torch.tensor([edge_cnt]).to(device)
#
#     invar_receivers = torch.randint(node_cnt, (bsize, edge_cnt)).to(device)
#     invar_senders = torch.randint(node_cnt, (bsize, edge_cnt)).to(device)
#
#     invar_predict_length = random.randint(1, 5)
#     invar_global_context = torch.randn(bsize, sim_steps, 1).to(device)
#     invar_particle_types = torch.randint(3, (bsize, node_cnt)).to(device)
#
#     invar = (invar_next_positions, invar_position_sequence_noise, invar_position_sequence,
#              invar_n_particles, invar_n_edges, invar_senders, invar_receivers, invar_predict_length,
#              invar_global_context, invar_particle_types)
#
#     assert common.validate_forward_accuracy(model, (*invar,))


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_simulator_optims(device):
    def setup_model():
        """Setups up fresh VFGN simulator model and inputs for optim test"""

        class Stats:
            def __init__(self, mean, std):
                self.mean = mean
                self.std = std

            def to(self, device):
                self.mean = self.mean.to(device)
                self.std = self.std.to(device)
                return self

        dummy_metadata = [
            -0.008786813221081134,
            -0.004392095496219808,
            -0.00173827297612587,
        ]
        dummy_stats = Stats(
            torch.DoubleTensor(np.array(dummy_metadata, dtype=np.float64)),
            torch.DoubleTensor(np.array(dummy_metadata, dtype=np.float64)),
        )

        normalization_stats = {
            "acceleration": dummy_stats,
            "velocity": dummy_stats,
            "context": dummy_stats,
        }
        model = LearnedSimulator(
            num_dimensions=3,
            num_seq=5,  # the implementation on INPUT_SEQUENCE_LENGTH >= 3
            boundaries=torch.DoubleTensor([[-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0]]),
            # torch.DoubleTensor(metadata['bounds'])
            num_particle_types=3,
            particle_type_embedding_size=16,
            normalization_stats=normalization_stats,
        ).to(device)

        bsize = 2
        node_cnt = 1024
        edge_cnt = 4096
        sim_steps = 5
        k = 4

        invar_next_positions = torch.randn(bsize * node_cnt, 1, 3).to(
            device
        )  # target to predict
        invar_position_sequence_noise = torch.randn(bsize * node_cnt, sim_steps, 3).to(
            device
        )  # noise augmentation
        invar_position_sequence = torch.randn(bsize * node_cnt, sim_steps, 3).to(
            device
        )  # input sequence

        invar_n_particles = torch.tensor([node_cnt] * bsize).to(
            device
        )  # node size per batch
        invar_n_edges = torch.tensor([edge_cnt] * bsize).to(
            device
        )  # edge size per batch

        # identity graph with repeatation
        invar_receivers = (
            torch.arange(node_cnt).repeat(bsize).repeat_interleave(k).to(device)
        )
        invar_senders = (
            torch.arange(node_cnt).repeat(bsize).repeat_interleave(k).to(device)
        )

        invar_predict_length = torch.LongTensor([random.randint(1, 5)]).squeeze()
        invar_global_context = torch.randn(bsize, sim_steps, 1).to(device)
        invar_particle_types = torch.randint(3, (bsize * node_cnt,)).to(device)

        invar = (
            invar_next_positions,
            invar_position_sequence_noise,
            invar_position_sequence,
            invar_n_particles,
            invar_n_edges,
            invar_senders,
            invar_receivers,
            invar_predict_length,
            invar_global_context,
            invar_particle_types,
        )

        return model, invar

    # Ideally always check graphs first
    model, invar = setup_model()
    assert common.validate_cuda_graphs(model, (*invar,))

    # Check JIT
    model, invar = setup_model()
    assert common.validate_jit(model, (*invar,))
    # Check AMP
    model, invar = setup_model()
    assert common.validate_amp(model, (*invar,))
    # Check Combo
    model, invar = setup_model()
    assert common.validate_combo_optims(model, (*invar,))


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_simulator_checkpoint(device):
    """Test VFGN simulator checkpoint save/load"""
    # Construct VFGN simulator model models
    class Stats:
        def __init__(self, mean, std):
            self.mean = mean
            self.std = std

        def to(self, device):
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)
            return self

    dummy_metadata = [
        -0.008786813221081134,
        -0.004392095496219808,
        -0.00173827297612587,
    ]
    dummy_context_meta = [747.9937656176759]
    dummy_stats = Stats(
        torch.DoubleTensor(np.array(dummy_metadata, dtype=np.float64)),
        torch.DoubleTensor(np.array(dummy_metadata, dtype=np.float64)),
    )
    dummy_context_stats = Stats(
        torch.DoubleTensor(np.array(dummy_context_meta, dtype=np.float64)),
        torch.DoubleTensor(np.array(dummy_context_meta, dtype=np.float64)),
    )

    normalization_stats = {
        "acceleration": dummy_stats,
        "velocity": dummy_stats,
        "context": dummy_context_stats,
    }

    model_1 = LearnedSimulator(
        num_dimensions=3,
        num_seq=5,  # the implementation on INPUT_SEQUENCE_LENGTH >= 3
        boundaries=torch.DoubleTensor([[-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0]]),
        # boundaries=[[-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0]],
        num_particle_types=3,
        particle_type_embedding_size=16,
        normalization_stats=normalization_stats,
    ).to(device)

    model_2 = LearnedSimulator(
        num_dimensions=3,
        num_seq=5,  # the implementation on INPUT_SEQUENCE_LENGTH >= 3
        boundaries=torch.DoubleTensor([[-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0]]),
        num_particle_types=3,
        particle_type_embedding_size=16,
        normalization_stats=normalization_stats,
    ).to(device)

    bsize = 2
    node_cnt = 1024
    edge_cnt = 4096
    sim_steps = 5
    k = 4

    invar_next_positions = torch.randn(bsize * node_cnt, 1, 3).to(
        device
    )  # target to predict
    invar_position_sequence_noise = torch.randn(bsize * node_cnt, sim_steps, 3).to(
        device
    )  # noise augmentation
    invar_position_sequence = torch.randn(bsize * node_cnt, sim_steps, 3).to(
        device
    )  # input sequence

    invar_n_particles = torch.tensor([node_cnt] * bsize).to(
        device
    )  # node size per batch
    invar_n_edges = torch.tensor([edge_cnt] * bsize).to(device)  # edge size per batch

    # identity graph with repeatation
    invar_receivers = (
        torch.arange(node_cnt).repeat(bsize).repeat_interleave(k).to(device)
    )
    invar_senders = torch.arange(node_cnt).repeat(bsize).repeat_interleave(k).to(device)

    invar_predict_length = random.randint(1, 5)
    invar_global_context = torch.randn(bsize, sim_steps, 1).to(device)
    invar_particle_types = torch.randint(3, (bsize * node_cnt,)).to(device)

    invar = (
        invar_next_positions,
        invar_position_sequence_noise,
        invar_position_sequence,
        invar_n_particles,
        invar_n_edges,
        invar_senders,
        invar_receivers,
        invar_predict_length,
        invar_global_context,
        invar_particle_types,
    )

    assert common.validate_checkpoint(model_1, model_2, (*invar,))


@common.check_ort_version()
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_simulator_deploy(device):
    """Test VFGN simulator deployment support"""

    class Stats:
        def __init__(self, mean, std):
            self.mean = mean
            self.std = std

        def to(self, device):
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)
            return self

    dummy_metadata = [
        -0.008786813221081134,
        -0.004392095496219808,
        -0.00173827297612587,
    ]
    dummy_stats = Stats(
        torch.DoubleTensor(np.array(dummy_metadata, dtype=np.float64)),
        torch.DoubleTensor(np.array(dummy_metadata, dtype=np.float64)),
    )

    normalization_stats = {
        "acceleration": dummy_stats,
        "velocity": dummy_stats,
        "context": dummy_stats,
    }
    model = LearnedSimulator(
        num_dimensions=3,
        num_seq=5,  # the implementation on INPUT_SEQUENCE_LENGTH >= 3
        boundaries=torch.DoubleTensor([[-5.0, 5.0], [-5.0, 5.0], [-5.0, 5.0]]),
        # torch.DoubleTensor(metadata['bounds'])
        num_particle_types=3,
        particle_type_embedding_size=16,
        normalization_stats=normalization_stats,
    ).to(device)

    bsize = 2
    node_cnt = 1024
    edge_cnt = 4096
    sim_steps = 5
    k = 4

    invar_next_positions = torch.randn(bsize * node_cnt, 1, 3).to(
        device
    )  # target to predict
    invar_position_sequence_noise = torch.randn(bsize * node_cnt, sim_steps, 3).to(
        device
    )  # noise augmentation
    invar_position_sequence = torch.randn(bsize * node_cnt, sim_steps, 3).to(
        device
    )  # input sequence

    invar_n_particles = torch.tensor([node_cnt] * bsize).to(
        device
    )  # node size per batch
    invar_n_edges = torch.tensor([edge_cnt] * bsize).to(device)  # edge size per batch

    # identity graph with repeatation
    invar_receivers = (
        torch.arange(node_cnt).repeat(bsize).repeat_interleave(k).to(device)
    )
    invar_senders = torch.arange(node_cnt).repeat(bsize).repeat_interleave(k).to(device)

    invar_predict_length = torch.LongTensor([random.randint(1, 5)]).squeeze()
    invar_global_context = torch.randn(bsize, sim_steps, 1).to(device)
    invar_particle_types = torch.randint(3, (bsize * node_cnt,)).to(device)

    invar = (
        invar_next_positions,
        invar_position_sequence_noise,
        invar_position_sequence,
        invar_n_particles,
        invar_n_edges,
        invar_senders,
        invar_receivers,
        invar_predict_length,
        invar_global_context,
        invar_particle_types,
    )

    assert common.validate_onnx_export(model, (invar,))
    assert common.validate_onnx_runtime(model, (invar,))
