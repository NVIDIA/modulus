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

import sys, os

script_path = os.path.abspath(__file__)
sys.path.append(os.path.join(os.path.dirname(script_path), ".."))

import common
import pytest
import torch

from utils import fix_random_seeds
from modulus.models.sfno.sfnonet import SphericalFourierNeuralOperatorNet


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("checkpointing", [0, 2])
def test_sfno_forward(device, checkpointing):
    """Test sfno forward pass with & without checkpointing"""

    in_chans = 2
    h, w = 8, 16
    params = {}

    fix_random_seeds()
    x = torch.randn(1, in_chans, h, w)
    x = x.to(device)

    # Construct sfno model
    model = SphericalFourierNeuralOperatorNet(
        params,
        img_shape=(h, w),
        scale_factor=4,
        in_chans=in_chans,
        out_chans=in_chans,
        embed_dim=16,
        num_layers=2,
        encoder_layers=1,
        num_blocks=4,
        spectral_layers=2,
        checkpointing=checkpointing,
    ).to(device)

    assert common.validate_forward_accuracy(model, (x,), rtol=1e-3)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize(
    "filter_type, operator_type, use_mlp, activation_function, pos_embed, \
    normalization_layer, use_complex_kernels, factorization, separable, \
    complex_network",
    [
        (
            "non-linear",
            "diagonal",
            True,
            "relu",
            True,
            "layer_norm",
            True,
            None,
            False,
            True,
        ),
        (
            "linear",
            "diagonal",
            False,
            "gelu",
            False,
            "instance_norm",
            True,
            "dense",
            True,
            False,
        ),
        (
            "non-linear",
            "diagonal",
            False,
            "silu",
            True,
            "none",
            False,
            "cp",
            False,
            True,
        ),
    ],
)
def test_sfno_constructor(
    device,
    filter_type,
    operator_type,
    use_mlp,
    activation_function,
    pos_embed,
    normalization_layer,
    use_complex_kernels,
    factorization,
    separable,
    complex_network,
):
    """Test sfno constructor options"""
    # Define dictionary of constructor args

    in_chans = 2
    h, w = 8, 16
    batch_size = 2

    arg_list = [
        {
            "params": {},
            "img_shape": (h, w),
            "scale_factor": 4,
            "in_chans": in_chans,
            "out_chans": in_chans,
            "embed_dim": 16,
            "num_layers": 2,
            "encoder_layers": 1,
            "num_blocks": 4,
            "spectral_layers": 2,
            "checkpointing": 0,
            "filter_type": filter_type,
            "operator_type": operator_type,
            "use_mlp": use_mlp,
            "activation_function": activation_function,
            "pos_embed": pos_embed,
            "normalization_layer": normalization_layer,
            "use_complex_kernels": use_complex_kernels,
            "factorization": factorization,
            "separable": separable,
            "complex_network": complex_network,
        },
        {
            "params": {},
            "img_shape": (h, w),
            "scale_factor": 4,
            "in_chans": in_chans,
            "out_chans": in_chans,
            "embed_dim": 16,
            "num_layers": 2,
            "encoder_layers": 1,
            "num_blocks": 4,
            "spectral_layers": 2,
            "checkpointing": 0,
            "filter_type": filter_type,
            "operator_type": operator_type,
            "use_mlp": use_mlp,
            "activation_function": activation_function,
            "pos_embed": pos_embed,
            "normalization_layer": normalization_layer,
            "use_complex_kernels": use_complex_kernels,
            "factorization": factorization,
            "separable": separable,
            "complex_network": complex_network,
        },
    ]
    for kw_args in arg_list:
        # Construct sfno model
        model = SphericalFourierNeuralOperatorNet(**kw_args).to(device)

        fix_random_seeds()
        x = torch.randn(batch_size, in_chans, h, w)
        x = x.to(device)

        outvar = model(x)
        assert outvar.shape == (batch_size, in_chans, h, w)


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_sfno_optims(device):
    """Test sfno optimizations"""

    def setup_model():
        """Set up fresh model and inputs for each optim test"""

        in_chans = 2
        h, w = 8, 16
        fix_random_seeds()
        x = torch.randn(1, in_chans, h, w)
        x = x.to(device)

        model_kwds = {
            "params": {},
            "img_shape": (h, w),
            "scale_factor": 4,
            "in_chans": in_chans,
            "out_chans": in_chans,
            "embed_dim": 16,
            "num_layers": 2,
            "encoder_layers": 1,
            "num_blocks": 4,
            "spectral_layers": 1,
            "checkpointing": 0,
        }

        # Construct SFNO model
        model = SphericalFourierNeuralOperatorNet(**model_kwds).to(device)

        return model, (x,)

    # # Ideally always check graphs first
    model, invar = setup_model()
    assert common.validate_cuda_graphs(model, (*invar,))
    # # Check JIT
    model, invar = setup_model()
    assert common.validate_jit(model, (*invar,))
    # # Check AMP
    model, invar = setup_model()
    assert common.validate_amp(model, (*invar,))
    # # Check Combo
    model, invar = setup_model()
    assert common.validate_combo_optims(model, (*invar,))


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_sfno_checkpoint(device):
    """Test sfno checkpoint save/load"""

    in_chans = 4
    h, w = 8, 16
    fix_random_seeds()
    x = torch.randn(1, in_chans, h, w)
    x = x.to(device)

    model_kwds = {
        "params": {},
        "img_shape": (h, w),
        "scale_factor": 3,
        "in_chans": in_chans,
        "out_chans": in_chans,
        "embed_dim": 16,
        "num_layers": 4,
        "encoder_layers": 1,
        "num_blocks": 4,
        "spectral_layers": 3,
        "checkpointing": 0,
    }

    # Construct sfno model
    model_1 = SphericalFourierNeuralOperatorNet(**model_kwds).to(device)
    model_2 = SphericalFourierNeuralOperatorNet(**model_kwds).to(device)

    x = torch.randn(1, in_chans, h, w)
    x = x.to(device)

    assert common.validate_checkpoint(
        model_1,
        model_2,
        (x,),
    )


@common.check_ort_version()
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_sfno_deploy(device):
    """Test sfno deployment support"""

    in_chans = 3
    h, w = 8, 16
    fix_random_seeds()
    x = torch.randn(1, in_chans, h, w)
    x = x.to(device)

    model_kwds = {
        "params": {},
        "img_shape": (h, w),
        "scale_factor": 3,
        "in_chans": in_chans,
        "out_chans": in_chans,
        "embed_dim": 16,
        "num_layers": 4,
        "encoder_layers": 1,
        "num_blocks": 4,
        "spectral_layers": 3,
        "checkpointing": 0,
    }

    # Construct SFNO model
    model = SphericalFourierNeuralOperatorNet(**model_kwds).to(device)

    assert common.validate_onnx_export(model, x)
    assert common.validate_onnx_runtime(model, x)
