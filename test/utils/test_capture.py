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


import logging
import pytest
import torch
import torch.nn as nn

from modulus.models.mlp import FullyConnected
from modulus.utils import StaticCaptureTraining, StaticCaptureEvaluateNoGrad
from modulus.utils.capture import _StaticCapture

try:
    from apex import optimizers
except:
    pass

Tensor = torch.Tensor
logger = logging.getLogger("__name__")


@pytest.fixture
def model():
    return FullyConnected(2, 64, 2)


@pytest.fixture
def logger():
    logger = logging.getLogger("launch")
    formatter = logging.Formatter(
        "[%(asctime)s - %(name)s - %(levelname)s] %(message)s", datefmt="%H:%M:%S"
    )
    streamhandler = logging.StreamHandler()
    streamhandler.setFormatter(formatter)
    streamhandler.setLevel(logging.INFO)
    logger.addHandler(streamhandler)
    return logger


@pytest.mark.parametrize(
    "optim_type, device",
    [("pytorch", "cuda:0"), ("apex", "cuda:0"), ("pytorch", "cpu")],
)
@pytest.mark.parametrize("use_graphs", [True, False])
@pytest.mark.parametrize(
    "use_amp, amp_type",
    [(True, torch.float16), (True, torch.bfloat16), (False, torch.float16)],
)
def test_capture_training(
    model,
    logger,
    device,
    optim_type,
    use_graphs,
    use_amp,
    amp_type,
):

    model = model.to(device)
    input = torch.rand(8, 2).to(device)
    output = torch.rand(8, 2).to(device)
    # Set up optimizer
    if optim_type == "pytorch":
        optim = torch.optim.Adam(model.parameters(), lr=0.001)
    else:
        if optimizers:
            optim = optimizers.FusedAdam(model.parameters(), lr=0.001)
        else:
            logger.warn("Apex not installed, skipping fused Adam tests")
            return

    # Create training step function with optimization wrapper
    @StaticCaptureTraining(
        model=model,
        optim=optim,
        logger=logger,
        use_graphs=use_graphs,
        use_amp=use_amp,
        cuda_graph_warmup=1,
        amp_type=amp_type,
    )
    def training_step(invar, outvar):
        predvar = model(invar)
        loss = torch.sum(torch.pow(predvar - outvar, 2))
        return loss

    # Sample training loop
    for i in range(3):
        loss = training_step(input, output)
        input.copy_(torch.rand(8, 2).to(device))
        assert loss > 0, "MSE loss should always be larger than zero"

    # Test control via meta data
    model.meta.cuda_graphs = use_graphs
    model.meta.amp_gpu = use_amp
    model.meta.amp_cpu = use_amp
    # Create training step function with optimization wrapper
    @StaticCaptureTraining(
        model=model,
        optim=optim,
        logger=logger,
        cuda_graph_warmup=1,
    )
    def training_step(invar, outvar):
        predvar = model(invar)
        loss = torch.sum(torch.pow(predvar - outvar, 2))
        return loss

    # Sample training loop
    for i in range(3):
        loss = training_step(input, output)
        input.copy_(torch.rand(8, 2).to(device))
        assert loss > 0, "MSE loss should always be larger than zero"


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("use_graphs", [True, False])
@pytest.mark.parametrize(
    "use_amp, amp_type",
    [(True, torch.float16), (True, torch.bfloat16), (False, torch.float16)],
)
def test_capture_evaluate(
    model,
    logger,
    device,
    use_graphs,
    use_amp,
    amp_type,
):

    model = model.to(device)
    input = torch.rand(8, 2).to(device)
    # Create eval step function with optimization wrapper
    @StaticCaptureEvaluateNoGrad(
        model=model,
        logger=logger,
        use_graphs=use_graphs,
        use_amp=use_amp,
        cuda_graph_warmup=1,
        amp_type=amp_type,
    )
    def eval_step(invar):
        predvar = model(invar)
        return predvar

    # Sample eval loop
    for i in range(3):
        predvar = eval_step(input)
        input.copy_(torch.rand(8, 2).to(device))
        assert predvar.shape == torch.Size((8, 2))


def test_capture_errors():
    # Test fail cases when capture should error
    model = torch.nn.Sequential(
        nn.Linear(2, 16),
        nn.ReLU(),
        nn.Linear(16, 2),
    )
    try:
        StaticCaptureEvaluateNoGrad(model=model)
        raise AssertionError(
            "Static capture should error if model is not Modulus.Module"
        )
    except ValueError:
        pass


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_capture_scaler_checkpointing(model, device):
    # Testing the class variables of AMP grad scaler for checkpointing
    #
    model = model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    # Test if it can ignore invalid scalar dicts
    _StaticCapture.scaler_dict = {"phoo": 0}
    capture = StaticCaptureTraining(model=model, optim=optim)
    assert not "phoo" in capture.scaler.state_dict()

    # Test capture will load from singleton state dict
    # Needed when loading a checkpoint
    scaler_dict = _StaticCapture.scaler_singleton.state_dict()
    _StaticCapture.scaler_dict = scaler_dict

    capture = StaticCaptureTraining(model=model, optim=optim)
    assert scaler_dict == capture.scaler.state_dict()
