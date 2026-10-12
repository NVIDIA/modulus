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

"""End-to-end training, resume, metric, and inference acceptance tests."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from data import build_dataset
from hydra import compose, initialize_config_dir
from infer import predict
from make_synthetic_data import generate
from omegaconf import OmegaConf
from train import backward_step, evaluate, joint_loss, primary_metric_terms, run

from physicsnemo import Module
from physicsnemo.datapipes import DataLoader
from physicsnemo.distributed import DistributedManager
from physicsnemo.experimental.models.smart import SMART


@pytest.fixture
def config(tmp_path):
    """Build a small, isolated recipe and original analytic data."""
    torch.set_num_threads(2)
    generate(tmp_path / "data", train_cases=4, val_cases=3)
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[1] / "conf"), version_base="1.3"
    ):
        cfg = compose(config_name="smoke")
    cfg.data.train_path = str(tmp_path / "data/train")
    cfg.data.val_path = str(tmp_path / "data/val")
    cfg.output_dir = str(tmp_path / "run")
    return cfg


@pytest.fixture(autouse=True)
def cleanup_manager():
    """Release the distributed singleton between example tests."""
    yield
    if DistributedManager.is_initialized():
        DistributedManager.cleanup()


def test_relative_loss_channels_and_zero_gradient():
    """Verify channel balancing and finite gradients for zero fields."""
    target = torch.tensor([[[1.0, 100.0], [2.0, 200.0]]])
    prediction = (target * torch.tensor([2.0, 1.5])).requires_grad_()
    loss = joint_loss((prediction, None), {"surface_targets": target})
    torch.testing.assert_close(loss, torch.tensor(0.75))
    loss.backward()
    assert torch.isfinite(prediction.grad).all()

    prediction = torch.zeros_like(target, requires_grad=True)
    joint_loss(
        (prediction, None), {"surface_targets": torch.zeros_like(target)}
    ).backward()
    assert torch.isfinite(prediction.grad).all()


def test_relative_loss_restores_target_origin():
    """Relative errors depend on the physical zero, unlike standardized MSE."""
    target = torch.tensor([[[1.0], [2.0]]])
    prediction = torch.tensor([[[2.0], [4.0]]], requires_grad=True)
    factors = {"surface": {"mean": torch.tensor([10.0]), "std": torch.tensor([2.0])}}
    expected = torch.linalg.vector_norm(
        torch.tensor([2.0, 4.0])
    ) / torch.linalg.vector_norm(torch.tensor([12.0, 14.0]))
    loss = joint_loss(
        (prediction, None), {"surface_targets": target}, field_factors=factors
    )
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    # The reference-comparison recipe's standardized MSE is intentionally unchanged.
    torch.testing.assert_close(
        joint_loss(
            (prediction, None),
            {"surface_targets": target},
            kind="mse",
            field_factors=factors,
        ),
        torch.tensor(2.5),
    )


def test_primary_loss_physical_case_weighting_and_auxiliary_channels():
    """Use analytic errors with unequal case scales, rotated signs and offsets."""
    factors = {
        "surface": {
            "mean": torch.tensor([10.0, 1.0, 2.0, 3.0]),
            "std": torch.tensor([2.0, 3.0, 4.0, 5.0]),
        },
        "volume": {
            "mean": torch.tensor([1.0, 2.0, 3.0, 10.0, 0.1]),
            "std": torch.tensor([2.0, 3.0, 4.0, 5.0, 0.2]),
        },
    }
    predictions, batch = [], {}
    for field, base, pressure, vector in (
        ("surface", [2.0, 1.0, 2.0, 2.0], 0, slice(1, 4)),
        ("volume", [1.0, 2.0, 2.0, 3.0, 0.2], 3, slice(0, 3)),
    ):
        target = torch.tensor(base).repeat(2, 3, 1)
        target[1] *= 10
        prediction = target.clone()
        for case, magnitude in enumerate((1.1, 0.8)):
            prediction[case, :, vector] *= -magnitude
            prediction[case, :, pressure] *= (
                (1.2, 1.4) if field == "surface" else (1.3, 1.5)
            )[case]
            if field == "volume":
                prediction[case, :, 4] *= 2 + case
        mean, std = factors[field]["mean"], factors[field]["std"]
        batch[f"{field}_targets"] = (target - mean) / std
        predictions.append(((prediction - mean) / std).requires_grad_())
    terms = primary_metric_terms(tuple(predictions), batch, field_factors=factors)
    expected = {
        "surface_pressure_relative_l1": 0.3,
        "surface_wall_shear_magnitude_relative_l1": 0.15,
        "volume_pressure_relative_l1": 0.4,
        "volume_velocity_magnitude_relative_l1": 0.15,
    }
    for name, value in expected.items():
        torch.testing.assert_close(terms[name], torch.tensor(value))
    loss = joint_loss(
        tuple(predictions), batch, kind="primary_metric_l1", field_factors=factors
    )
    # Mean relative channel errors: surface 1.5375, volume 1.55.
    torch.testing.assert_close(loss, torch.tensor(1.771875))
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in predictions)
    assert torch.count_nonzero(predictions[1].grad[..., 4]) > 0


def test_primary_loss_zero_fields_and_channel_contract():
    """Zero norms must have finite gradients; the paper's 1/3 layout is distinct."""
    predictions = tuple(torch.zeros(2, 3, n, requires_grad=True) for n in (4, 5))
    batch = {
        f"{field}_targets": p.detach().clone()
        for field, p in zip(("surface", "volume"), predictions)
    }
    loss = joint_loss(predictions, batch, kind="primary_metric_l1")
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in predictions)
    with pytest.raises(ValueError, match="aligned"):
        joint_loss((predictions[0][..., :1], None), batch, kind="primary_metric_l1")
    with pytest.raises(ValueError, match="No active"):
        joint_loss((None, None), {}, kind="primary_metric_l1")


def test_primary_training_resume_and_recipe_guards(config, tmp_path):
    """Exercise the normal trainer, checkpointed objective and exact Adam resume."""
    for path in (tmp_path / "data").glob("*/*.npz"):
        with np.load(path) as saved:
            sample = {key: saved[key].copy() for key in saved.files}
        sample["surface_targets"] = np.concatenate(
            [sample["surface_targets"], 0.1 * sample["surface_queries"] + 0.2], axis=-1
        )
        sample["volume_targets"] = np.concatenate(
            [
                sample["volume_targets"],
                1 + 0.15 * sample["volume_queries"][:, :1],
                0.1 + 0.02 * sample["volume_queries"][:, 1:2] ** 2,
            ],
            axis=-1,
        )
        np.savez(path, **sample)
    config.model.surface_channels, config.model.volume_channels = 4, 5
    OmegaConf.update(config, "training.loss", "primary_metric_l1", force_add=True)
    config.training.epochs = 2
    full_history = run(config)
    assert config.training.metric_channel_l2_weight == 0.25
    full = Module.from_checkpoint(
        Path(config.output_dir) / "checkpoints/SMART.0.2.mdlus"
    )
    full_optimizer = torch.load(
        Path(config.output_dir) / "checkpoints/checkpoint.0.2.pt",
        map_location="cpu",
        weights_only=False,
    )["optimizer_state_dict"]
    config.output_dir = str(tmp_path / "primary-resumed")
    config.training.epochs = 1
    run(config)
    config.training.epochs = 2
    config.training.resume = True
    resumed_history = run(config)
    resumed = Module.from_checkpoint(
        Path(config.output_dir) / "checkpoints/SMART.0.2.mdlus"
    )
    assert resumed_history == full_history[1:]
    for key, value in full.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
    checkpoint = Path(config.output_dir) / "checkpoints/checkpoint.0.2.pt"
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    for index, state in full_optimizer["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(
                value,
                saved["optimizer_state_dict"]["state"][index][key],
                rtol=0,
                atol=0,
            )
    for key, value, message in (
        ("relative_l2_space", "standardized", "physical field"),
        ("metric_channel_l2_weight", 1.0, "metric_channel_l2_weight"),
    ):
        previous = config.training[key]
        config.training[key] = value
        with pytest.raises(ValueError, match=message):
            run(config)
        config.training[key] = previous
    # Reject an incompatible objective recorded in the actual checkpoint too.
    saved["metadata"]["config"]["training"]["metric_channel_l2_weight"] = 1.0
    torch.save(saved, checkpoint)
    with pytest.raises(
        ValueError, match="Resume configuration differs in metric_channel_l2_weight"
    ):
        run(config)


def test_inference_float32_overrides_saved_attention(config, tmp_path, monkeypatch):
    """An explicit float32 request must reach attention despite saved FP16 settings."""
    from physicsnemo.experimental.models.smart import SMART

    OmegaConf.update(config, "model.attention_precision", "float16", force_add=True)
    OmegaConf.update(config, "model.qk_normalization", True, force_add=True)
    OmegaConf.update(config, "training.matmul_precision", "high", force_add=True)
    run_dir = Path(config.output_dir)
    run_dir.mkdir()
    OmegaConf.save(config, run_dir / "config.yaml")
    model = SMART(**OmegaConf.to_container(config.model))
    model.save(run_dir / "best.mdlus")
    original = torch.nn.functional.scaled_dot_product_attention
    dtypes = []

    def inspect_attention(query, key, value, *args, **kwargs):
        """Record attention arithmetic dtypes while delegating to PyTorch."""
        dtypes.append(query.dtype)
        return original(query, key, value, *args, **kwargs)

    monkeypatch.setattr(
        torch.nn.functional, "scaled_dot_product_attention", inspect_attention
    )
    previous = torch.get_float32_matmul_precision()
    try:
        predict(
            run_dir,
            Path(config.data.val_path),
            tmp_path / "fp32-prediction",
            device="cpu",
            precision="float32",
        )
        assert dtypes and set(dtypes) == {torch.float32}
        assert torch.get_float32_matmul_precision() == "highest"
    finally:
        torch.set_float32_matmul_precision(previous)


@pytest.mark.parametrize("use_muon", [False, True])
@pytest.mark.parametrize("external_checkpoint", [False, True])
def test_training_resume_is_exact(config, tmp_path, use_muon, external_checkpoint):
    """Compare every resumed weight and metric against uninterrupted training."""
    if use_muon:
        config.optimizer = OmegaConf.create(
            {"_target_": "optimizers.muon_adamw", "lr": 0.001, "weight_decay": 0.0001}
        )
        OmegaConf.update(
            config,
            "scheduler",
            {
                "_target_": "torch.optim.lr_scheduler.StepLR",
                "step_size": 1,
                "gamma": 0.5,
            },
            force_add=True,
        )
        OmegaConf.update(config, "training.loss", "mse", force_add=True)
    config.training.epochs = 3
    uninterrupted = run(config)
    assert uninterrupted[-1]["train_loss"] < uninterrupted[0]["train_loss"]
    full = Module.from_checkpoint(
        str(Path(config.output_dir) / "checkpoints/SMART.0.3.mdlus")
    )
    config.output_dir = str(tmp_path / "resumed")
    config.training.epochs = 1
    run(config)
    config.training.epochs = 3
    config.training.resume = True
    if external_checkpoint:
        OmegaConf.update(
            config,
            "training.resume_from",
            str(Path(config.output_dir) / "checkpoints"),
            force_add=True,
        )
        config.output_dir = str(tmp_path / "continued")
    resumed_history = run(config)
    resumed = Module.from_checkpoint(
        str(Path(config.output_dir) / "checkpoints/SMART.0.3.mdlus")
    )
    for name, tensor in full.state_dict().items():
        torch.testing.assert_close(tensor, resumed.state_dict()[name], rtol=0, atol=0)
    assert resumed_history == uninterrupted[1:]
    config.data.coordinate_scale = [2.0, 2.0, 2.0]
    with pytest.raises(ValueError, match="differs in data"):
        run(config)


def assert_exact_optimizer_state(actual, expected):
    """Compare tensor bytes and non-tensor optimizer metadata recursively."""
    assert type(actual) is type(expected)
    if torch.is_tensor(actual):
        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert torch.equal(
            actual.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
            expected.detach().cpu().contiguous().reshape(-1).view(torch.uint8),
        )
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_exact_optimizer_state(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            assert_exact_optimizer_state(left, right)
    else:
        assert actual == expected


@pytest.mark.parametrize("use_muon", [False, True])
def test_weights_only_initialization_and_exact_resume(
    config, tmp_path, monkeypatch, use_muon
):
    """Start fresh histories from trained weights, then resume without the source."""
    import train

    config.optimizer = OmegaConf.create(
        {
            "_target_": "optimizers.muon_adamw" if use_muon else "torch.optim.AdamW",
            "lr": 1e-3,
            "weight_decay": 1e-4,
        }
    )
    OmegaConf.update(
        config,
        "scheduler",
        {"_target_": "torch.optim.lr_scheduler.StepLR", "step_size": 1, "gamma": 0.5},
        force_add=True,
    )
    OmegaConf.update(config, "training.loss", "mse", force_add=True)
    config.training.epochs = 1
    run(config)
    source = Path(config.output_dir) / "checkpoints/SMART.0.1.mdlus"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    source_state = Module.from_checkpoint(source).state_dict()
    parent_optimizer = source.parent / "checkpoint.0.1.pt"
    parent_hash = hashlib.sha256(parent_optimizer.read_bytes()).hexdigest()
    backward = train.backward_step
    fresh_starts = []

    def observe_start(loss, model, optimizer, scaler, gradient_clip):
        """Observe actual weights and empty optimizer states before the first update."""
        state = optimizer.state_dict()
        if all(not child["state"] for child in state.get("optimizers", [state])):
            # Check exact weight values independently of checkpoint storage devices.
            actual = {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            }
            torch.testing.assert_close(actual, source_state, rtol=0, atol=0)
            assert all(group["lr"] == 2e-3 for group in optimizer.param_groups)
            fresh_starts.append(True)
        return backward(loss, model, optimizer, scaler, gradient_clip)

    monkeypatch.setattr(train, "backward_step", observe_start)
    config.seed = 99
    config.optimizer.lr = 2e-3
    config.training.initial_weights = str(source)
    config.output_dir = str(tmp_path / "continuous")
    config.training.epochs = 2
    continuous = run(config)
    continuous_dir = Path(config.output_dir)
    config.output_dir = str(tmp_path / "split")
    config.training.epochs = 1
    split = run(config)
    split_dir = Path(config.output_dir)
    provenance = json.loads((split_dir / "initialization.json").read_text())
    assert provenance["kind"] == "weights_only"
    assert provenance["source_sha256"] == source_hash
    assert provenance["verified_ranks"] == 1
    assert not any(
        provenance[key]
        for key in ["optimizer_restored", "scheduler_restored", "scaler_restored"]
    )
    timestamp = (split_dir / "initialization.json").stat().st_mtime_ns
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    # A real continuation uses the new checkpoint and must not reopen/reload
    # the original initialization file, which may have since been archived.
    archived = source.with_suffix(".archived")
    source.rename(archived)
    config.training.epochs = 2
    config.training.resume = True
    split += run(config)
    assert split == continuous
    assert len(fresh_starts) == 2
    assert (split_dir / "initialization.json").stat().st_mtime_ns == timestamp
    final_models = [
        Module.from_checkpoint(directory / "checkpoints/SMART.0.2.mdlus")
        for directory in [continuous_dir, split_dir]
    ]
    torch.testing.assert_close(
        final_models[0].state_dict(), final_models[1].state_dict(), rtol=0, atol=0
    )
    final_states = [
        torch.load(
            directory / "checkpoints/checkpoint.0.2.pt",
            map_location="cpu",
            weights_only=False,
        )
        for directory in [continuous_dir, split_dir]
    ]
    for key in ["optimizer_state_dict", "scheduler_state_dict"]:
        assert_exact_optimizer_state(final_states[0][key], final_states[1][key])
    for state in final_states:
        assert state["metadata"]["initialization"] == provenance
        optimizer_state = state["optimizer_state_dict"]
        steps = {
            int(history["step"])
            for child in optimizer_state.get("optimizers", [optimizer_state])
            for history in child["state"].values()
            if "step" in history
        }
        assert steps == {8}
    assert hashlib.sha256(archived.read_bytes()).hexdigest() == source_hash
    assert hashlib.sha256(parent_optimizer.read_bytes()).hexdigest() == parent_hash
    config.training.initial_weights = str(archived)
    with pytest.raises(ValueError, match="differs in training.initial_weights"):
        run(config)


@pytest.mark.parametrize("invalid", ["missing", "architecture", "nonfinite"])
def test_initial_weights_reject_invalid_source(config, tmp_path, invalid):
    """Require an existing, structurally compatible and finite source checkpoint."""
    from resume import load_initial_weights

    model = SMART(**config.model)
    source = tmp_path / "source.mdlus"
    if invalid == "missing":
        with pytest.raises(FileNotFoundError, match="does not exist"):
            load_initial_weights(model, source)
        return
    options = dict(config.model)
    if invalid == "architecture":
        options["hidden_dim"] *= 2
    parent = SMART(**options)
    if invalid == "nonfinite":
        with torch.no_grad():
            next(parent.parameters()).fill_(float("nan"))
    parent.save(source)
    with pytest.raises(RuntimeError if invalid == "architecture" else ValueError):
        load_initial_weights(model, source)


def test_full_validation_is_chunk_invariant(config):
    """Ensure validation reduces full-field norms rather than chunk ratios."""
    dataset = build_dataset(config.data.val_path, config, training=False)
    model = SMART(**config.model)
    small = evaluate(model, dataset, torch.device("cpu"), 11, "float32")
    full = evaluate(model, dataset, torch.device("cpu"), 1000, "float32")
    for key in full:
        assert small[key] == pytest.approx(full[key], rel=1e-5, abs=1e-6)


def test_validation_l1_restores_fields_and_distinguishes_vectors():
    """Check case averaging, physical means, and magnitude versus direction error."""
    factors = {
        "surface": {
            "mean": torch.tensor([10.0, 0.0, 0.0, 0.0]),
            "std": torch.tensor([2.0, 3.0, 4.0, 5.0]),
        },
        "volume": {
            "mean": torch.tensor([1.0, 2.0, 3.0, 10.0, 0.0]),
            "std": torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0]),
        },
    }
    samples, predictions = [], []
    for case in range(2):
        targets = {
            "surface": torch.tensor(
                [[10.0, 1.0, 0.0, 0.0], [20.0, 2.0, 0.0, 0.0], [30.0, 3.0, 0.0, 0.0]]
            )
            * (1 + 9 * case),
            "volume": torch.tensor(
                [[1.0, 0.0, 0.0, 40.0, 0.1], [2.0, 0.0, 0.0, 50.0, 0.2]]
            )
            * (1 + 9 * case),
        }
        sample = {"geometry": torch.full((3, 3), float(case))}
        predicted = {}
        for field, target in targets.items():
            actual = target.clone()
            if field == "surface":
                actual[:, 0] *= 1.1 + 0.2 * case
                actual[:, 1:4] *= -1
            else:
                actual[:, 3] *= 1.2 + 0.2 * case
                actual[:, :3] *= -1
            mean, std = factors[field]["mean"], factors[field]["std"]
            sample[f"{field}_queries"] = (
                torch.arange(len(target)).float()[:, None].repeat(1, 3)
            )
            sample[f"{field}_targets"] = (target - mean) / std
            predicted[field] = (actual - mean) / std
        samples.append(sample)
        predictions.append(predicted)

    class Dataset:
        """Two cases with different target norms, in standardized units."""

        reader = SimpleNamespace(field_factors=factors)

        def __len__(self):
            return len(samples)

        def __getitem__(self, index):
            return samples[index], {}

    class Model:
        """Return predetermined predictions indexed by query coordinates."""

        surface_channels, volume_channels = 4, 5

        def eval(self):
            """Keep predetermined outputs unchanged."""
            return self

        def encode_geometry(self, geometry, parameters):
            """Use the synthetic geometry coordinate as the case identifier."""
            return (int(geometry[0, 0, 0]),)

        def decode_queries(self, queries, contexts, parameters, *, field):
            """Select predictions for each requested point index."""
            return predictions[contexts[0]][field][queries[0, :, 0].long()].unsqueeze(0)

    values = evaluate(Model(), Dataset(), torch.device("cpu"), 2, "float32")
    assert values["surface_pressure_relative_l1"] == pytest.approx(0.2)
    assert values["volume_pressure_relative_l1"] == pytest.approx(0.3)
    for field, name in (("surface", "wall_shear"), ("volume", "velocity")):
        assert values[f"{field}_{name}_magnitude_relative_l1"] == pytest.approx(
            0, abs=1e-7
        )
        assert values[f"{field}_{name}_vector_relative_l1"] == pytest.approx(2)
    assert values["loss"] == pytest.approx(1.01)


def test_inference_without_targets_matches_direct(config, tmp_path):
    """Compare streamed disk outputs with direct inference without target arrays."""
    config.training.epochs = 1
    run(config)
    inference_data = tmp_path / "inference"
    inference_data.mkdir()
    for path in Path(config.data.val_path).glob("*.npz"):
        with np.load(path) as sample:
            np.savez(
                inference_data / path.name,
                **{k: sample[k] for k in sample.files if not k.endswith("targets")},
            )
    destination = tmp_path / "predictions"
    manifest = predict(
        Path(config.output_dir),
        inference_data,
        destination,
        chunk_size=17,
        device="cpu",
    )
    model = Module.from_checkpoint(str(Path(config.output_dir) / "best.mdlus")).eval()
    dataset = build_dataset(inference_data, config, training=False, targets=False)
    assert len(manifest) == 3
    with torch.no_grad():
        for i, record in enumerate(manifest):
            sample, _ = dataset[i]
            surface, volume = model(
                sample["geometry"][None],
                sample["surface_queries"][None],
                sample["volume_queries"][None],
                sample["parameters"][None],
            )
            for name, value in (("surface", surface), ("volume", volume)):
                actual = torch.from_numpy(np.load(destination / record[name]))
                torch.testing.assert_close(actual, value[0], atol=2e-6, rtol=2e-5)


def test_explicit_inference_checkpoint_and_latent_budget(config, tmp_path):
    """Stream the chosen weights and sampling budget, preserving saved artifacts."""
    run_dir = Path(config.output_dir)
    run_dir.mkdir()
    OmegaConf.save(config, run_dir / "config.yaml")
    model = SMART(**config.model).eval()
    selected = tmp_path / "selected.mdlus"
    model.save(selected)
    checkpoint_hash = hashlib.sha256(selected.read_bytes()).hexdigest()
    config_bytes = (run_dir / "config.yaml").read_bytes()
    # Make the default alias observably different from the explicit selection.
    with torch.no_grad():
        model.surface_head.layers[-1].bias.add_(10)
    model.save(run_dir / "best.mdlus")
    destination = tmp_path / "selected-predictions"
    manifest = predict(
        run_dir,
        Path(config.data.val_path),
        destination,
        checkpoint=selected,
        num_latents=16,
        chunk_size=17,
        device="cpu",
    )
    expected = Module.from_checkpoint(selected).eval()
    expected.num_latents = 16
    dataset = build_dataset(config.data.val_path, config, training=False, targets=False)
    try:
        with torch.no_grad():
            for index, row in enumerate(manifest):
                sample, _ = dataset[index]
                inputs = {
                    name: sample[name][None]
                    for name in (
                        "geometry",
                        "surface_queries",
                        "volume_queries",
                        "parameters",
                    )
                }
                for field, prediction in zip(("surface", "volume"), expected(**inputs)):
                    actual = torch.from_numpy(np.load(destination / row[field]))
                    torch.testing.assert_close(
                        actual, prediction[0], atol=2e-6, rtol=2e-5
                    )
            old_budget = expected(**inputs)[0]
            expected.num_latents = 8
            assert not torch.equal(old_budget, expected(**inputs)[0])
    finally:
        dataset.close()
    identity = json.loads((destination / "inference.json").read_text())
    assert identity["checkpoint"] == str(selected.resolve())
    assert identity["checkpoint_sha256"] == checkpoint_hash
    assert identity["config_sha256"] == hashlib.sha256(config_bytes).hexdigest()
    assert identity["trained_num_latents"] == 8
    assert identity["inference_num_latents"] == 16
    assert identity["geometry_points_per_layer"] == 16
    assert len(identity["cases"]) == 3
    assert all(row["coarse_latents"] == 16 for row in identity["cases"])
    assert hashlib.sha256(selected.read_bytes()).hexdigest() == checkpoint_hash
    assert (run_dir / "config.yaml").read_bytes() == config_bytes
    with pytest.raises(FileExistsError, match="must be empty"):
        predict(run_dir, Path(config.data.val_path), destination, device="cpu")
    config.model.num_heads = 2
    OmegaConf.save(config, run_dir / "config.yaml")
    with pytest.raises(ValueError, match="model.num_heads"):
        predict(
            run_dir,
            Path(config.data.val_path),
            tmp_path / "mismatched",
            checkpoint=selected,
            device="cpu",
        )


@pytest.mark.parametrize("num_latents", [0, -1, True, 1.5])
def test_inference_rejects_invalid_latent_count(tmp_path, num_latents):
    """Reject invalid latent overrides before opening any checkpoint or dataset."""
    with pytest.raises(ValueError, match="positive integer"):
        predict(tmp_path, tmp_path, tmp_path / "output", num_latents=num_latents)


def test_data_sampling_alignment_and_repeatability(config):
    """Check coordinated target sampling and epoch-addressed randomness."""
    # Encode a checkable point id into both coordinates and targets.
    dataset = build_dataset(config.data.train_path, config, training=True)
    sample, _ = dataset.reader[0]
    sample["surface_targets"] = sample["surface_queries"][:, :1].clone()
    a = dataset.transforms(sample.clone())
    torch.testing.assert_close(a["surface_targets"], a["surface_queries"][:, :1])
    loader = DataLoader(dataset, seed=5, prefetch_factor=0, use_streams=False)
    loader.set_epoch(2)
    first = next(iter(loader))
    loader.set_epoch(2)
    second = next(iter(loader))
    torch.testing.assert_close(first["surface_queries"], second["surface_queries"])


def test_invalid_data_and_split(config):
    """Reject overlapping splits and invalid spatial normalization."""
    config.data.val_path = config.data.train_path
    with pytest.raises(ValueError, match="separate"):
        run(config)
    config.data.coordinate_scale = [1.0, 0.0, 1.0]
    with pytest.raises(ValueError, match="positive scale"):
        build_dataset(config.data.train_path, config, training=True)


def test_resume_missing_checkpoint_fails(config):
    """Fail explicitly when resume has no complete checkpoint pair."""
    config.training.resume = True
    with pytest.raises(FileNotFoundError, match="No complete"):
        run(config)


def test_resume_validation_sampling_is_fixed(config):
    """Chunking may change; the held-out sample used to choose best weights may not."""
    config.training.epochs = 1
    run(config)
    config.training.resume = True
    config.training.epochs = 2
    config.validation.chunk_size = 17
    run(config)
    OmegaConf.update(config, "validation.points", 10, force_add=True)
    config.training.epochs = 3
    with pytest.raises(ValueError, match="validation sampling"):
        run(config)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA scaler required")
def test_attention_fp16_scaler_resume(config, tmp_path):
    """Resume attention-only fp16 training with calibrated gradient scaler state."""
    OmegaConf.update(config, "model.attention_precision", "float16", force_add=True)
    OmegaConf.update(config, "training.resume_precision_change", True, force_add=True)
    OmegaConf.update(config, "training.grad_scaler_init_scale", 32.0, force_add=True)
    config.training.epochs = 3
    run(config)
    full_path = Path(config.output_dir) / "checkpoints"
    full = Module.from_checkpoint(full_path / "SMART.0.3.mdlus")
    full_state = torch.load(
        full_path / "checkpoint.0.3.pt", map_location="cpu", weights_only=False
    )
    config.output_dir = str(tmp_path / "attention-resumed")
    config.training.epochs = 1
    run(config)
    config.training.epochs = 3
    config.training.resume = True
    run(config)
    resumed_path = Path(config.output_dir) / "checkpoints"
    resumed = Module.from_checkpoint(resumed_path / "SMART.0.3.mdlus")
    resumed_state = torch.load(
        resumed_path / "checkpoint.0.3.pt", map_location="cpu", weights_only=False
    )
    assert full_state["scaler_state_dict"]
    assert full_state["scaler_state_dict"] == resumed_state["scaler_state_dict"]
    for name, value in full.state_dict().items():
        torch.testing.assert_close(
            value, resumed.state_dict()[name], rtol=1e-5, atol=1e-6
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA scaler required")
def test_overflow_skip_is_visible_and_atomic():
    """Finite forward loss with infinite gradients must skip both optimizers."""
    from optimizers import muon_adamw

    model = torch.nn.Linear(2, 2, device="cuda")
    optimizer = muon_adamw(model.parameters())
    scaler = torch.amp.GradScaler("cuda", init_scale=32.0)
    original = {name: p.detach().clone() for name, p in model.named_parameters()}
    hook = model.weight.register_hook(lambda grad: torch.full_like(grad, float("inf")))
    loss = model(torch.ones(1, 2, device="cuda")).square().sum()
    assert torch.isfinite(loss)
    norm, skipped = backward_step(loss, model, optimizer, scaler, 1.0)
    assert skipped and not torch.isfinite(norm)
    assert scaler.get_scale() == 16.0
    for name, p in model.named_parameters():
        torch.testing.assert_close(p, original[name], rtol=0, atol=0)
    hook.remove()
    optimizer.zero_grad(set_to_none=True)
    norm, skipped = backward_step(
        model(torch.ones(1, 2, device="cuda")).square().sum(),
        model,
        optimizer,
        scaler,
        1.0,
    )
    assert not skipped and torch.isfinite(norm)
    assert not torch.equal(model.weight, original["weight"])


def test_update_budget_schedule_and_resume(config, tmp_path, monkeypatch):
    """A non-epoch-aligned endpoint and update LR survive an interrupted epoch boundary."""
    import train as training

    config.training.epochs = 1  # Explicit updates override this legacy endpoint.
    config.training.max_updates = 6
    config.training.scheduler_interval = "update"
    OmegaConf.update(config, "training.loss", "mse", force_add=True)
    OmegaConf.update(
        config,
        "scheduler",
        {
            "_target_": "torch.optim.lr_scheduler.CosineAnnealingLR",
            "T_max": 6,
            "eta_min": 0.0001,
        },
        force_add=True,
    )
    uninterrupted = run(config)
    assert [r["optimization"]["optimizer_updates"] for r in uninterrupted] == [4, 6]
    assert [r["optimization"]["optimizer_steps"] for r in uninterrupted] == [4, 2]
    full_path = Path(config.output_dir) / "checkpoints"
    full = Module.from_checkpoint(full_path / "SMART.0.2.mdlus")
    full_state = torch.load(
        full_path / "checkpoint.0.2.pt", map_location="cpu", weights_only=False
    )
    assert full_state["metadata"]["optimizer_updates"] == 6
    assert full_state["metadata"]["epoch_complete"] is False
    assert full_state["scheduler_state_dict"]["last_epoch"] == 6
    assert full_state["optimizer_state_dict"]["param_groups"][0]["lr"] == 0.0001

    config.output_dir = str(tmp_path / "update-resumed")
    original_save = training._save_training_checkpoint

    def interrupted_save(*args, **kwargs):
        """Simulate an interruption after a complete checkpoint is durable."""
        original_save(*args, **kwargs)
        raise RuntimeError("simulated interruption after durable checkpoint")

    with monkeypatch.context() as patch:
        patch.setattr(training, "_save_training_checkpoint", interrupted_save)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            run(config)
    config.training.resume = True
    resumed_history = run(config)
    assert resumed_history == uninterrupted[1:]
    resumed_path = Path(config.output_dir) / "checkpoints"
    resumed = Module.from_checkpoint(resumed_path / "SMART.0.2.mdlus")
    for key, value in full.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
    resumed_state = torch.load(
        resumed_path / "checkpoint.0.2.pt", map_location="cpu", weights_only=False
    )
    assert resumed_state["scheduler_state_dict"] == full_state["scheduler_state_dict"]
    for index, state in full_state["optimizer_state_dict"]["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(
                value,
                resumed_state["optimizer_state_dict"]["state"][index][key],
                rtol=0,
                atol=0,
            )
    assert run(config) == []  # A finished partial epoch cannot consume extra updates.
    config.training.max_updates = 7
    with pytest.raises(ValueError, match="max_updates"):
        run(config)


@pytest.mark.parametrize("field", ["surface", "volume"])
def test_field_only_loss_and_evaluation_without_opposite_data(field):
    """Single-field metrics need no opposite query/target tensors or head."""
    channels = 4 if field == "surface" else 5
    model = SMART(
        hidden_dim=24,
        num_heads=4,
        num_layers=2,
        num_latents=4,
        geometry_subsample=8,
        surface_channels=channels if field == "surface" else 0,
        volume_channels=channels if field == "volume" else 0,
    )
    sample = {
        "geometry": torch.randn(12, 3),
        f"{field}_queries": torch.randn(7, 3),
        f"{field}_targets": torch.ones(7, channels),
    }

    class Dataset:
        """Supply a controlled validation sample for the regression test."""

        reader = SimpleNamespace(field_factors={})

        def __len__(self):
            return 1

        def __getitem__(self, index):
            return sample, {}

    result = evaluate(model, Dataset(), torch.device("cpu"), 3, "float32")
    other = "volume" if field == "surface" else "surface"
    assert result and not any(key.startswith(other) for key in result)
    target = sample[f"{field}_targets"].unsqueeze(0)
    prediction = (target + 2).requires_grad_()
    predictions = (prediction, None) if field == "surface" else (None, prediction)
    loss = joint_loss(predictions, {f"{field}_targets": target}, kind="mse")
    torch.testing.assert_close(loss, torch.tensor(4.0))
    loss.backward()
    assert torch.isfinite(prediction.grad).all()


def test_cfd_feature_evaluation_chunks_and_reuses_mesh(monkeypatch):
    """Native query features are computed per chunk against one stable mesh allocation."""
    import sys
    import types

    calls = []

    def augment(batch, cfg):
        """Populate only the query features requested by the current chunk."""
        query = batch["surface_queries"]
        calls.append((query.shape[1], batch["cfd_mesh_vertices"].data_ptr()))
        batch["surface_query_features"] = torch.ones_like(query)
        return batch

    monkeypatch.setitem(
        sys.modules,
        "cfd_features",
        types.SimpleNamespace(augment_query_features=augment),
    )
    model = SMART(
        hidden_dim=24,
        num_heads=4,
        num_layers=2,
        num_latents=4,
        geometry_subsample=8,
        surface_channels=4,
        volume_channels=0,
        surface_query_feature_channels=3,
    )
    sample = {
        "geometry": torch.randn(12, 3),
        "surface_queries": torch.randn(7, 3),
        "surface_targets": torch.ones(7, 4),
        "cfd_mesh_vertices": torch.randn(10, 3),
        "cfd_mesh_faces": torch.tensor([[0, 1, 2]]),
        "cfd_query_center": torch.zeros(3),
        "cfd_query_scale": torch.ones(3),
    }

    class Dataset:
        """Supply a controlled validation sample for the regression test."""

        reader = SimpleNamespace(field_factors={})

        def __len__(self):
            return 1

        def __getitem__(self, index):
            return sample, {}

    result = evaluate(
        model, Dataset(), torch.device("cpu"), 3, "float32", cfg=OmegaConf.create({})
    )
    assert result["surface_relative_l2"] > 0
    assert [size for size, _ in calls] == [3, 3, 1]
    assert len({pointer for _, pointer in calls}) == 1


def test_update_scheduler_does_not_count_overflow_skips(config, monkeypatch):
    """Skipped mixed-precision steps cannot consume LR decay or the update budget."""
    import train as training

    config.training.max_updates = 4
    config.training.scheduler_interval = "update"
    OmegaConf.update(
        config,
        "scheduler",
        {
            "_target_": "torch.optim.lr_scheduler.StepLR",
            "step_size": 2,
            "gamma": 0.5,
        },
        force_add=True,
    )
    original = training.backward_step
    calls = 0

    def skip_once(loss, model, optimizer, scaler, gradient_clip):
        """Simulate one gradient overflow without advancing optimizer state."""
        nonlocal calls
        calls += 1
        if calls == 1:
            return torch.tensor(0.0), True
        return original(loss, model, optimizer, scaler, gradient_clip)

    monkeypatch.setattr(training, "backward_step", skip_once)
    history = run(config)
    assert calls == 5
    assert [entry["optimization"]["optimizer_updates"] for entry in history] == [3, 4]
    assert sum(entry["optimization"]["skipped_steps"] for entry in history) == 1
    saved = torch.load(
        Path(config.output_dir) / "checkpoints/checkpoint.0.2.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert saved["scheduler_state_dict"]["last_epoch"] == 4
    assert saved["optimizer_state_dict"]["param_groups"][0]["lr"] == 0.003 * 0.25


def test_interrupted_checkpoint_is_not_published(tmp_path, monkeypatch):
    """An interrupted state write cannot hide the last complete checkpoint pair."""
    import train

    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "SMART.0.1.mdlus").write_bytes(b"previous model")
    (checkpoints / "checkpoint.0.1.pt").write_bytes(b"previous optimizer")

    def partial_save(path, **kwargs):
        """Simulate a termination after an incomplete optimizer file appears."""
        (Path(path) / "SMART.0.2.mdlus").write_bytes(b"new model")
        (Path(path) / "checkpoint.0.2.pt").write_bytes(b"truncated")
        raise RuntimeError("interrupted write")

    monkeypatch.setattr(train, "save_checkpoint", partial_save)
    with pytest.raises(RuntimeError, match="interrupted write"):
        train._save_training_checkpoint(checkpoints)
    assert {p.name for p in checkpoints.iterdir()} == {
        "SMART.0.1.mdlus",
        "checkpoint.0.1.pt",
    }


def test_training_closes_readers_after_failure(config, monkeypatch):
    """An optimizer construction failure still releases both data readers."""
    import train

    real_build = train.build_dataset
    closed = []

    def tracked_build(*args, **kwargs):
        """Record each reader shutdown while delegating dataset behavior."""
        dataset = real_build(*args, **kwargs)
        original = dataset.close

        def close():
            """Release the actual reader and record this cleanup event."""
            closed.append(dataset)
            original()

        dataset.close = close
        return dataset

    def fail_optimizer(*args, **kwargs):
        """Fail after both datasets have been opened."""
        raise RuntimeError("optimizer failure")

    monkeypatch.setattr(train, "build_dataset", tracked_build)
    monkeypatch.setattr(train, "instantiate", fail_optimizer)
    with pytest.raises(RuntimeError, match="optimizer failure"):
        run(config)
    assert len(closed) == 2
    assert closed[0] is not closed[1]
