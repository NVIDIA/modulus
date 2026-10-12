# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
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
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

import train
from utils.nn import TimeScaledModel, build_condition, get_dit, sample_flow
from utils.trainer import Trainer

from physicsnemo.core import Module
from physicsnemo.diffusion.metrics.losses import MSEDSMLoss
from physicsnemo.diffusion.noise_schedulers import RectifiedFlowNoiseScheduler
from physicsnemo.distributed import DistributedManager

DistributedManager.initialize()

# Training and sampling reach t = 1, where the loss and the sampling flow stay finite
pytestmark = pytest.mark.filterwarnings(
    "ignore:RectifiedFlowNoiseScheduler was constructed with t_max=1.0"
)

CHECKPOINT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "checkpoints",
    "corrdiff_era5_hrrr.mdlus",
)


def _load_config(config_name: str) -> DictConfig:
    """Load a configuration through Hydra."""
    with initialize(version_base=None, config_path="config", job_name="test_training"):
        return compose(config_name=config_name)


@pytest.fixture
def cfg() -> DictConfig:
    """Return the mock-data test configuration."""
    return _load_config(config_name="test_diffusion")


def _setup_rundir(tmp_path: Path) -> str:
    """Return a temporary run directory path that all test ranks share."""
    rundir = str(tmp_path / "rundir")
    dist = DistributedManager()
    if dist.world_size > 1:
        output_list = [None]
        torch.distributed.scatter_object_list(
            output_list, [rundir] * dist.world_size, src=0
        )
        rundir = output_list[0]
    return rundir


@pytest.mark.parametrize("fp_optimizations", ["fp32", "amp-bf16"])
@pytest.mark.parametrize("lr_decay", ["cosine", "constant"])
def test_training(
    tmp_path: Path, cfg: DictConfig, fp_optimizations: str, lr_decay: str
):
    """Test that training runs and writes checkpoints and validation plots."""
    rundir = _setup_rundir(tmp_path)
    cfg.training.rundir = rundir
    cfg.training.batch_size = 2 * DistributedManager().world_size
    cfg.training.perf.fp_optimizations = fp_optimizations
    cfg.training.scheduler.lr_decay = lr_decay

    train.main(cfg)

    checkpoint_path = os.path.join(rundir, "checkpoints")
    assert os.path.isfile(os.path.join(checkpoint_path, "TimeScaledModel.0.10.mdlus"))
    assert os.path.isfile(os.path.join(checkpoint_path, "checkpoint.0.10.pt"))
    assert os.path.isfile(os.path.join(rundir, "images", "state_0_000000010.png"))


def test_checkpointing(tmp_path: Path, cfg: DictConfig):
    """Test that training resumes from the latest checkpoint."""
    rundir = _setup_rundir(tmp_path)
    cfg.training.rundir = rundir
    cfg.training.batch_size = 2 * DistributedManager().world_size

    # Train for 10 steps, then continue the same run to 20 steps
    train.main(cfg)
    cfg.training.total_train_steps = 20
    trainer = Trainer(cfg)
    assert trainer.total_steps == 10
    assert trainer.scheduler.last_epoch == 10
    trainer.train()

    checkpoint_path = os.path.join(rundir, "checkpoints")
    assert os.path.isfile(os.path.join(checkpoint_path, "TimeScaledModel.0.20.mdlus"))


@pytest.mark.parametrize("t", [None, [0.0, 1.0]])
@pytest.mark.parametrize("amp", [False, True])
def test_loss_equals_x0_mse(
    tmp_path: Path, cfg: DictConfig, t: list[float] | None, amp: bool
):
    """Test that the training loss equals the x0-prediction loss of MSEDSMLoss.
    It covers sampled times and the endpoints t = 0 and 1, with and without
    bf16 autocast."""
    cfg.training.rundir = _setup_rundir(tmp_path)
    cfg.training.batch_size = 2 * DistributedManager().world_size
    trainer = Trainer(cfg)
    x0_mse = MSEDSMLoss(
        trainer.train_net,
        trainer.noise_scheduler,
        prediction_type="x0",
        reduction="none",
    )

    device = trainer.device
    (h, w) = cfg.dataset.image_size
    x0 = torch.randn(2, cfg.dataset.num_state_channels, h, w, device=device)
    background = torch.randn(
        2, cfg.dataset.num_background_channels, h, w, device=device
    )
    scalar_conditions = torch.randn(
        2, cfg.dataset.num_scalar_cond_channels, device=device
    )
    condition = build_condition(background, scalar_conditions)
    if t is not None:
        t = torch.tensor(t, device=device)

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        torch.manual_seed(0)
        loss = trainer.loss_fn(x0, torch.ones_like(x0), t=t, condition=condition)
        torch.manual_seed(0)  # same times and noise as in the training loss
        expected = x0_mse(x0, t=t, condition=condition)
    assert torch.isfinite(loss).all()
    assert torch.equal(loss, expected)


class GaussianX0Predictor(torch.nn.Module):
    """Exact x0-predictor E[x0 | x_t] for data x0 ~ N(mean, std^2) on the
    rectified-flow path x_t = (1 - t) x0 + t eps."""

    def __init__(self, mean: float, std: float):
        super().__init__()
        self.mean = mean
        self.var = std**2
        self.unused = torch.nn.Parameter(torch.zeros(()))  # sets the device

    def forward(self, x, t, condition=None):
        alpha = (1 - t).reshape(-1, 1, 1, 1)
        sigma = t.reshape(-1, 1, 1, 1)
        gain = alpha * self.var / (alpha**2 * self.var + sigma**2)
        return self.mean + gain * (x - alpha * self.mean)


@pytest.mark.parametrize(
    "solver, num_steps, time_shift",
    [("heun", 32, 1.0), ("heun", 64, 4.0), ("euler", 128, 1.0)],
)
def test_sampling_from_pure_noise(solver: str, num_steps: int, time_shift: float):
    """Test that sampling from pure noise (t = 1) with the exact x0-predictor
    of Gaussian data reproduces the data distribution."""
    (mean, std) = (2.0, 0.5)
    model = GaussianX0Predictor(mean, std).to(DistributedManager().device)
    torch.manual_seed(0)
    x = sample_flow(
        model,
        None,
        (4, 1, 64, 64),
        RectifiedFlowNoiseScheduler(t_max=1.0),
        num_steps=num_steps,
        solver=solver,
        time_shift=time_shift,
    )
    assert torch.isfinite(x).all()
    assert abs(x.mean().item() - mean) < 0.02
    assert abs(x.std().item() - std) < 0.02


@pytest.mark.skipif(
    not os.path.isfile(CHECKPOINT), reason="released checkpoint not available"
)
def test_released_checkpoint():
    """Test that the released weights match the model config and that
    sampling from pure noise works."""
    assert isinstance(Module.from_checkpoint(CHECKPOINT), TimeScaledModel)

    device = DistributedManager().device
    model_cfg = _load_config(config_name="diffusion").model
    net = get_dit(
        target_channels=99,
        conditional_channels=30,
        scalar_condition_channels=4,
        img_resolution=(256, 256),
        **OmegaConf.to_container(model_cfg.hyperparameters, resolve=True),
    )
    net.load(CHECKPOINT)  # strict: all weights must match the architecture
    net = net.to(device).eval()

    torch.manual_seed(0)
    condition = build_condition(
        torch.randn(1, 30, 256, 256, device=device),
        torch.tensor([[0.0, 1.0, 1.0, 1.0]], device=device),
    )
    with torch.autocast("cuda", dtype=torch.bfloat16):
        x = sample_flow(
            net,
            condition,
            (1, 99, 256, 256),
            RectifiedFlowNoiseScheduler(t_max=1.0),
            num_steps=4,
        )
    assert x.shape == (1, 99, 256, 256)
    assert torch.isfinite(x).all()
