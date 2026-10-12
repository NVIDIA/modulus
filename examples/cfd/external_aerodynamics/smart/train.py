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

"""Joint or independent SMART field training with PhysicsNeMo checkpoint APIs."""

from __future__ import annotations

import json
import logging
import random
import time
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
import numpy as np
import torch
import torch.distributed as distributed
from data import build_dataset, model_inputs
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from resume import (
    _state_hash,
    load_initial_weights,
)
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from physicsnemo.datapipes import DataLoader
from physicsnemo.distributed import DistributedManager
from physicsnemo.experimental.models.smart import SMART
from physicsnemo.metrics.general.relative_error import relative_mse
from physicsnemo.utils.checkpoint import load_checkpoint, save_checkpoint

log = logging.getLogger(__name__)


def validate_resume_data(previous: dict, current: dict, *, repacked: bool) -> None:
    """Allow I/O tuning, and explicitly verified native-to-Zarr repacking only."""
    previous, current = dict(previous), dict(current)
    for key in (
        "workers",
        "prefetch",
        "io_threads",
        "pin_memory",
        "cuda_streams",
        "cache_validation",
        "to_device",
        "torch_num_threads",
    ):
        previous.pop(key, None)
        current.pop(key, None)
    if previous != current and repacked:
        if (previous.get("format"), current.get("format")) != (
            "drivaerml_mesh",
            "drivaerml_zarr",
        ):
            raise ValueError("Resume repacking only supports native DrivAerML to Zarr")
        for split in ("train_path", "val_path"):
            sources = {
                str(p.resolve())
                for p in Path(previous[split]).glob("run_*/domain_*.pdmsh")
            }
            attributes = [
                json.loads(p.read_text())
                for p in Path(current[split]).glob("*.zarr/.zattrs")
            ]
            if not sources or len(attributes) != len(sources):
                raise ValueError("Resume repacking changes the case split")
            prepared = {a.get("preparation", {}).get("source") for a in attributes}
            if prepared != sources:
                raise ValueError("Resume repacking source identities differ")
            for attributes_for_case in attributes:
                preparation = attributes_for_case["preparation"]
                for name, config_name in (
                    ("density", "reference_density"),
                    ("speed", "reference_speed"),
                ):
                    if preparation[name] != current.get(config_name):
                        raise ValueError("Resume repacking changes field scaling")
            previous.pop(split)
            current.pop(split)
        previous.pop("format")
        current.pop("format")
    if previous != current:
        raise ValueError(
            "Resume configuration differs in data; start a new output directory"
        )


def seed_everything(seed: int) -> None:
    """Seed model sampling and user transforms at each epoch boundary."""
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)


def autocast_context(device: torch.device, precision: str):
    """Select autocast while keeping parameters and loss accumulation in fp32."""
    if precision not in ("float32", "float16", "bfloat16"):
        raise ValueError("precision must be float32, float16, or bfloat16")
    if precision == "float16" and device.type != "cuda":
        raise ValueError("float16 training requires CUDA; use float32 on CPU")
    return torch.autocast(
        device.type, dtype=getattr(torch, precision), enabled=precision != "float32"
    )


def primary_metric_terms(
    predictions: tuple,
    batch,
    *,
    field_factors: dict | None = None,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """DrivAerML per-case pressure and vector-magnitude relative L1 errors.

    Surface channels are pressure followed by three shear components. Volume
    channels are three velocity components, pressure and turbulent viscosity.
    Undo standardization before measuring physical magnitudes and target norms.
    Each case has equal weight, regardless of its target magnitude.
    """
    terms = {}
    layouts = (
        ("surface", 4, 0, slice(1, 4), "wall_shear"),
        ("volume", 5, 3, slice(0, 3), "velocity"),
    )
    for prediction, (field, channels, pressure, vector, name) in zip(
        predictions, layouts
    ):
        if prediction is None:
            continue
        prediction = prediction.float()
        target = batch[f"{field}_targets"].float()
        if (
            prediction.ndim != 3
            or prediction.shape != target.shape
            or prediction.shape[-1] != channels
        ):
            raise ValueError(
                f"primary_metric_l1 requires aligned (batch, points, {channels}) "
                f"{field} predictions and targets"
            )
        if field_factors and field in field_factors:
            factors = field_factors[field]
            std, mean = factors["std"].to(prediction), factors["mean"].to(prediction)
            prediction, target = prediction * std + mean, target * std + mean
        p, t = prediction[..., pressure], target[..., pressure]
        terms[f"{field}_pressure_relative_l1"] = (
            (p - t).abs().sum(1) / t.abs().sum(1).clamp_min(eps)
        ).mean()
        p = torch.linalg.vector_norm(prediction[..., vector], dim=-1)
        t = torch.linalg.vector_norm(target[..., vector], dim=-1)
        terms[f"{field}_{name}_magnitude_relative_l1"] = (
            (p - t).abs().sum(1) / t.sum(1).clamp_min(eps)
        ).mean()
    if not terms:
        raise ValueError("No active prediction heads")
    return terms


def joint_loss(
    predictions: tuple,
    batch,
    eps: float = 1e-8,
    *,
    kind: str = "relative_l2",
    field_factors: dict | None = None,
) -> torch.Tensor:
    """Compute relative L2, standardized MSE, or the DrivAerML metric objective.

    Reductions are fp32. A tiny numerator floor defines a finite zero-error
    gradient; the denominator floor also handles all-zero target channels.
    For relative L2, undo field standardization when factors are supplied:
    subtracting a mean changes the target norm and therefore the objective.
    MSE intentionally stays in the standardized space used by the data reader.
    ``primary_metric_l1`` sums physical pressure and magnitude errors, plus
    0.25 times the original channel L2 loss to retain direction and viscosity
    supervision. This optional objective is an extension of the SMART paper.
    """
    if kind == "primary_metric_l1":
        terms = primary_metric_terms(
            predictions, batch, field_factors=field_factors, eps=eps
        )
        channel_loss = joint_loss(
            predictions, batch, eps, kind="relative_l2", field_factors=field_factors
        )
        return sum(terms.values()) + 0.25 * channel_loss
    if kind not in ("relative_l2", "mse"):
        raise ValueError("loss must be relative_l2, mse, or primary_metric_l1")
    losses = []
    for field, prediction in zip(("surface", "volume"), predictions):
        if prediction is not None:
            if kind == "mse":
                losses.append(
                    torch.nn.functional.mse_loss(
                        prediction.float(), batch[f"{field}_targets"].float()
                    )
                )
                continue
            prediction = prediction.float()
            target = batch[f"{field}_targets"].float()
            if field_factors and field in field_factors:
                factors = field_factors[field]
                std = factors["std"].to(device=prediction.device, dtype=torch.float32)
                mean = factors["mean"].to(device=prediction.device, dtype=torch.float32)
                prediction = prediction * std + mean
                target = target * std + mean
            ratio = relative_mse(prediction, target, dim=1, eps=eps)
            losses.append(ratio.clamp_min(torch.finfo(ratio.dtype).tiny).sqrt().mean())
    if not losses:
        raise ValueError("No active prediction heads")
    return sum(losses)


@torch.no_grad()
def evaluate(
    model: SMART,
    dataset,
    device: torch.device,
    chunk_size: int,
    precision: str,
    rank: int = 0,
    world_size: int = 1,
    cfg: DictConfig | None = None,
) -> dict[str, float]:
    """Evaluate every validation point without padding or duplicate distributed samples.

    Accumulate squared errors over all chunks before taking norms. Averaging
    chunk-wise relative errors would change the metric with chunk size.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    model.eval()
    totals = torch.zeros(3, dtype=torch.float64, device=device)
    channel_totals = {
        field: torch.zeros(
            getattr(model, f"{field}_channels"), dtype=torch.float64, device=device
        )
        for field in ("surface", "volume")
    }
    channel_l1_totals = {
        field: torch.zeros_like(value) for field, value in channel_totals.items()
    }
    # Keep magnitude errors distinct from vector errors. The GeoTransolver
    # example reports errors of velocity/shear magnitudes for its scalar metric.
    vector_slices = {"surface": slice(1, 4), "volume": slice(0, 3)}
    vector_totals = torch.zeros((2, 2), dtype=torch.float64, device=device)
    field_factors = getattr(dataset.reader, "field_factors", {})
    for index in range(rank, len(dataset), world_size):
        sample, _ = dataset[index]
        geometry = sample["geometry"].unsqueeze(0).to(device)
        parameters = sample.get("parameters", None)
        parameters = None if parameters is None else parameters.unsqueeze(0).to(device)
        mesh_inputs = {
            key: sample[key].unsqueeze(0).to(device)
            for key in (
                "cfd_mesh_vertices",
                "cfd_mesh_faces",
                "cfd_query_center",
                "cfd_query_scale",
            )
            if key in sample
        }
        with autocast_context(device, precision):
            contexts = model.encode_geometry(geometry, parameters)
        for field_index, field in enumerate(("surface", "volume")):
            if not getattr(model, f"{field}_channels"):
                continue
            queries, targets = sample[f"{field}_queries"], sample[f"{field}_targets"]
            numerator = torch.zeros(
                targets.shape[-1], dtype=torch.float64, device=device
            )
            denominator = torch.zeros_like(numerator)
            l1_numerator = torch.zeros_like(numerator)
            l1_denominator = torch.zeros_like(numerator)
            vector_sums = torch.zeros(3, dtype=torch.float64, device=device)
            for start in range(0, len(queries), chunk_size):
                query = queries[start : start + chunk_size].unsqueeze(0).to(device)
                target = targets[start : start + chunk_size].to(
                    device, dtype=torch.float64
                )
                with autocast_context(device, precision):
                    features = sample.get(f"{field}_query_features")
                    feature_args = (
                        {
                            "query_features": features[start : start + chunk_size]
                            .unsqueeze(0)
                            .to(device)
                        }
                        if features is not None
                        else {}
                    )
                    if mesh_inputs:
                        if cfg is None:
                            raise ValueError(
                                "CFD feature evaluation requires its saved configuration"
                            )
                        from cfd_features import augment_query_features

                        feature_batch = {**mesh_inputs, f"{field}_queries": query}
                        augment_query_features(feature_batch, cfg)
                        feature_args["query_features"] = feature_batch.get(
                            f"{field}_query_features"
                        )
                    prediction = model.decode_queries(
                        query, contexts, parameters, field=field, **feature_args
                    )[0]
                if field in field_factors:
                    mean = field_factors[field]["mean"].to(
                        device=device, dtype=torch.float64
                    )
                    std = field_factors[field]["std"].to(
                        device=device, dtype=torch.float64
                    )
                    prediction = prediction.double() * std + mean
                    target = target * std + mean
                numerator += (prediction.double() - target).square().sum(0)
                denominator += target.square().sum(0)
                l1_numerator += (prediction.double() - target).abs().sum(0)
                l1_denominator += target.abs().sum(0)
                vector_slice = vector_slices[field]
                if target.shape[-1] >= vector_slice.stop:
                    pred_vector = prediction.double()[:, vector_slice]
                    target_vector = target[:, vector_slice]
                    pred_magnitude = torch.linalg.vector_norm(pred_vector, dim=-1)
                    target_magnitude = torch.linalg.vector_norm(target_vector, dim=-1)
                    vector_sums[0] += (pred_magnitude - target_magnitude).abs().sum()
                    vector_sums[1] += torch.linalg.vector_norm(
                        pred_vector - target_vector, dim=-1
                    ).sum()
                    vector_sums[2] += target_magnitude.sum()
            per_channel = (numerator / denominator.clamp_min(1e-8)).sqrt()
            channel_totals[field] += per_channel
            channel_l1_totals[field] += l1_numerator / l1_denominator.clamp_min(1e-8)
            vector_totals[field_index] += vector_sums[:2] / vector_sums[2].clamp_min(
                1e-8
            )
            totals[field_index] += per_channel.mean()
        totals[2] += 1
    if world_size > 1:
        distributed.all_reduce(totals)
        distributed.all_reduce(vector_totals)
        for value in (*channel_totals.values(), *channel_l1_totals.values()):
            if value.numel():
                distributed.all_reduce(value)
    if totals[2] == 0 or not torch.isfinite(totals).all():
        raise FloatingPointError(
            "Validation is empty or produced non-finite predictions"
        )
    result = {
        f"{field}_relative_l2": float(totals[i] / totals[2])
        for i, field in enumerate(("surface", "volume"))
        if getattr(model, f"{field}_channels")
    }
    result["loss"] = sum(result.values())
    for field, values in channel_totals.items():
        for index, value in enumerate(values):
            result[f"{field}_channel_{index}_relative_l2"] = float(value / totals[2])
    for field, values in channel_l1_totals.items():
        for index, value in enumerate(values):
            result[f"{field}_channel_{index}_relative_l1"] = float(value / totals[2])
    for field_index, field in enumerate(("surface", "volume")):
        if getattr(model, f"{field}_channels") >= vector_slices[field].stop:
            name = "wall_shear" if field == "surface" else "velocity"
            for index, kind in enumerate(("magnitude", "vector")):
                result[f"{field}_{name}_{kind}_relative_l1"] = float(
                    vector_totals[field_index, index] / totals[2]
                )
    if model.surface_channels:
        result["surface_pressure_relative_l1"] = result["surface_channel_0_relative_l1"]
    if model.volume_channels >= 4:
        result["volume_pressure_relative_l1"] = result["volume_channel_3_relative_l1"]
    return result


def backward_step(loss, model, optimizer, scaler, gradient_clip: float):
    """Apply one atomic scaled update and expose overflow skips and gradient norm.

    A finite loss does not imply finite gradients. GradScaler may skip an entire
    optimizer update; callers must record this instead of silently counting it
    as successful training. The returned norm is measured before clipping.
    """
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        gradient_clip or float("inf"),
        error_if_nonfinite=not scaler.is_enabled(),
    )
    old_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    return norm.detach(), scaler.get_scale() < old_scale


def _save_training_checkpoint(path: Path, **kwargs) -> None:
    """Publish the optimizer file last so only complete checkpoint pairs resume."""
    path.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".pending-", dir=path) as temporary:
        save_checkpoint(temporary, **kwargs)
        staged = Path(temporary)
        for model_file in staged.glob("*.mdlus"):
            model_file.replace(path / model_file.name)
        for state_file in staged.glob("checkpoint.*.pt"):
            state_file.replace(path / state_file.name)


def run(cfg: DictConfig) -> list[dict]:
    """Train or resume, closing data readers even after a failed update."""
    with ExitStack() as resources:
        if any(
            cfg.model.get(f"{field}_query_feature_channels", 0)
            for field in ("surface", "volume")
        ):
            from cfd_features import clear_feature_cache

            resources.callback(clear_feature_cache)
        return _run(cfg, resources)


def _run(cfg: DictConfig, resources: ExitStack) -> list[dict]:
    """Run the epoch loop with its readers registered for exception-safe cleanup."""
    if (
        cfg.training.epochs <= 0
        or cfg.training.batch_size <= 0
        or cfg.training.gradient_clip < 0
    ):
        raise ValueError(
            "epochs/batch_size must be positive and gradient_clip nonnegative"
        )
    if (
        cfg.data.get("format", "numpy").startswith("drivaerml")
        and cfg.training.batch_size != 1
    ):
        raise ValueError(
            "DrivAerML requires batch_size=1 per rank for variable mesh sizes"
        )
    max_updates = cfg.training.get("max_updates")
    if max_updates is not None and (
        isinstance(max_updates, bool)
        or not isinstance(max_updates, int)
        or max_updates <= 0
    ):
        raise ValueError("training.max_updates must be a positive integer or null")
    scheduler_interval = cfg.training.get("scheduler_interval", "epoch")
    if scheduler_interval not in ("epoch", "update"):
        raise ValueError("training.scheduler_interval must be epoch or update")
    if (
        max_updates is not None
        and cfg.get("scheduler")
        and scheduler_interval != "update"
    ):
        raise ValueError("An update-budgeted run requires scheduler_interval=update")
    if not DistributedManager.is_initialized():
        DistributedManager.initialize()
    manager = DistributedManager()
    device = manager.device
    torch.set_float32_matmul_precision(cfg.training.get("matmul_precision", "highest"))
    if cfg.data.get("torch_num_threads"):
        torch.set_num_threads(cfg.data.torch_num_threads)
    autocast_context(device, cfg.training.precision)
    loss_kind = cfg.training.get("loss", "relative_l2")
    initial_weights = cfg.training.get("initial_weights")
    if initial_weights is not None and (
        not isinstance(initial_weights, str) or not initial_weights.strip()
    ):
        raise ValueError("training.initial_weights must be a checkpoint path or null")
    loss_space = cfg.training.get("relative_l2_space", "physical")
    if loss_space not in ("physical", "standardized"):
        raise ValueError("relative_l2_space must be physical or standardized")
    if loss_kind == "primary_metric_l1":
        if loss_space != "physical":
            raise ValueError("primary_metric_l1 requires physical field scaling")
        if cfg.training.get("metric_channel_l2_weight", 0.25) != 0.25:
            raise ValueError("primary_metric_l1 uses metric_channel_l2_weight=0.25")
        for field, channels in (("surface", 4), ("volume", 5)):
            if cfg.model[f"{field}_channels"] not in (0, channels):
                raise ValueError(
                    f"primary_metric_l1 requires {channels} {field} channels"
                )
        OmegaConf.update(cfg, "training.metric_channel_l2_weight", 0.25, force_add=True)
        OmegaConf.update(cfg, "training.relative_l2_space", "physical", force_add=True)
    if loss_kind == "relative_l2":
        # Persist the objective's origin so an older standardized-field run
        # cannot silently resume under the corrected physical-field objective.
        OmegaConf.update(cfg, "training.relative_l2_space", loss_space, force_add=True)
    seed_everything(cfg.seed)
    train_path, val_path = (
        Path(to_absolute_path(cfg.data[key])).resolve()
        for key in ("train_path", "val_path")
    )
    if (
        train_path == val_path
        or train_path in val_path.parents
        or val_path in train_path.parents
    ):
        raise ValueError(
            "Train and validation paths must be separate, non-nested directories"
        )
    train_data = build_dataset(train_path, cfg, training=True)
    resources.callback(train_data.close)
    factors = getattr(train_data.reader, "field_factors", {})
    if factors:
        OmegaConf.update(
            cfg,
            "data.field_statistics",
            {
                field: {key: value.tolist() for key, value in stats.items()}
                for field, stats in factors.items()
            },
            force_add=True,
        )
    val_data = build_dataset(val_path, cfg, training=False)
    resources.callback(val_data.close)
    # Use epoch-addressable ordering on one rank as well: RandomSampler advances
    # its private generator and would change the order after a resumed run.
    sampler = DistributedSampler(
        train_data,
        num_replicas=manager.world_size,
        rank=manager.rank,
        seed=cfg.seed,
        shuffle=True,
    )
    loader = DataLoader(
        train_data,
        batch_size=cfg.training.batch_size,
        sampler=sampler,
        shuffle=False,
        use_streams=cfg.data.get("cuda_streams", False),
        prefetch_factor=cfg.data.prefetch,
        seed=cfg.seed + manager.rank,
    )
    if not len(loader):
        raise ValueError("Training dataset must provide at least one batch per rank")
    model = SMART(**OmegaConf.to_container(cfg.model)).to(device)
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    scheduler = (
        instantiate(cfg.scheduler, optimizer=optimizer)
        if cfg.get("scheduler")
        else None
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda"
        and (
            cfg.training.precision == "float16"
            or cfg.model.get("attention_precision") == "float16"
        ),
        init_scale=cfg.training.get("grad_scaler_init_scale", 65536.0),
    )
    output = Path(to_absolute_path(cfg.output_dir))
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output / "checkpoints"
    start_epoch = 0
    optimizer_updates = 0
    best = float("inf")
    resolved = OmegaConf.to_container(cfg, resolve=True)
    metadata = {}
    initialization = None
    if cfg.training.resume:
        # Require complete pairs; PhysicsNeMo's loader otherwise tolerates missing files.
        load_dir = checkpoint_dir
        if not any(checkpoint_dir.glob("checkpoint.0.*.pt")) and cfg.training.get(
            "resume_from"
        ):
            load_dir = Path(to_absolute_path(cfg.training.resume_from))
        epochs = [
            int(p.stem.split(".")[-1])
            for p in load_dir.glob("checkpoint.0.*.pt")
            if (load_dir / f"SMART.0.{p.stem.split('.')[-1]}.mdlus").exists()
        ]
        if not epochs:
            raise FileNotFoundError(f"No complete training checkpoint in {load_dir}")
        start_epoch = load_checkpoint(
            load_dir,
            models=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None
            if cfg.training.get("resume_precision_change", False)
            else scaler,
            epoch=max(epochs),
            metadata_dict=metadata,
            device=device,
        )
        previous = metadata.get("config", {})
        previous_model = dict(previous.get("model", {}))
        current_model = dict(resolved["model"])
        previous_model.setdefault("qk_normalization", False)
        current_model.setdefault("qk_normalization", False)
        previous_model.setdefault("attention_precision", None)
        current_model.setdefault("attention_precision", None)
        previous_model.setdefault("attention_precision_scope", "attention")
        current_model.setdefault("attention_precision_scope", "attention")
        if cfg.training.get("resume_precision_change", False):
            previous_model.pop("attention_precision")
            current_model.pop("attention_precision")
            previous_model.pop("attention_precision_scope")
            current_model.pop("attention_precision_scope")
        if previous_model != current_model:
            raise ValueError("Resume configuration differs in model")
        for key in ("optimizer", "scheduler", "seed"):
            if previous.get(key) != resolved.get(key):
                raise ValueError(
                    f"Resume configuration differs in {key}; start a new output directory"
                )
        validate_resume_data(
            previous.get("data", {}),
            resolved["data"],
            repacked=cfg.training.get("resume_repacked_data", False),
        )
        previous_validation = dict(previous.get("validation", {}))
        current_validation = dict(resolved.get("validation", {}))
        for validation_cfg in (previous_validation, current_validation):
            validation_cfg.pop("chunk_size", None)
            validation_cfg.setdefault("points", None)
        if previous_validation != current_validation:
            raise ValueError("Resume configuration differs in validation sampling")
        if loss_kind == "primary_metric_l1":
            old_training = previous.get("training", {})
            if old_training.get("relative_l2_space", "physical") != "physical":
                raise ValueError("Resume configuration differs in relative_l2_space")
            if old_training.get("metric_channel_l2_weight", 0.25) != 0.25:
                raise ValueError(
                    "Resume configuration differs in metric_channel_l2_weight"
                )
        if loss_kind == "relative_l2":
            old_space = previous.get("training", {}).get(
                "relative_l2_space",
                "standardized"
                if previous.get("data", {}).get("field_statistics")
                else "physical",
            )
            if old_space != loss_space:
                raise ValueError("Resume configuration differs in relative_l2_space")
        for key in (
            "precision",
            "batch_size",
            "gradient_clip",
            "loss",
            "initial_weights",
        ):
            if previous.get("training", {}).get(key) != resolved["training"].get(key):
                if (
                    key == "precision"
                    and cfg.training.get("resume_precision_change", False)
                    and {previous["training"][key], resolved["training"][key]}
                    <= {"float32", "bfloat16"}
                ):
                    continue
                raise ValueError(f"Resume configuration differs in training.{key}")
        for key, default in (("max_updates", None), ("scheduler_interval", "epoch")):
            if previous.get("training", {}).get(key, default) != cfg.training.get(
                key, default
            ):
                raise ValueError(f"Resume configuration differs in training.{key}")
        optimizer_updates = metadata.get("optimizer_updates", 0)
        if scheduler_interval == "update" and "optimizer_updates" not in metadata:
            raise ValueError(
                "Update scheduling requires a checkpoint with optimizer_updates"
            )
        if (
            scheduler_interval == "update"
            and scheduler is not None
            and scheduler.last_epoch != optimizer_updates
        ):
            raise ValueError(
                "Scheduler update counter differs from checkpoint optimizer_updates"
            )
        for key, default in (
            ("matmul_precision", "highest"),
            ("compile_blocks", False),
        ):
            if previous.get("training", {}).get(key, default) != cfg.training.get(
                key, default
            ) and not cfg.training.get("resume_precision_change", False):
                raise ValueError(f"Resume configuration differs in training.{key}")
        if metadata.get("world_size") != manager.world_size:
            raise ValueError(
                "Resume requires the original world size for repeatable sampling"
            )
        previous_scaled = (
            previous.get("training", {}).get("precision") == "float16"
            or previous.get("model", {}).get("attention_precision") == "float16"
        )
        if (
            cfg.training.get("resume_precision_change", False)
            and scaler.is_enabled()
            and previous_scaled
        ):
            # A newly enabled scaler starts fresh when migrating from fp32.
            # Subsequent resumes restore its calibrated state from the new run.
            load_checkpoint(load_dir, scaler=scaler, epoch=start_epoch, device=device)
        best = metadata["best_validation_loss"]
        initialization = metadata.get("initialization")
        if any(
            previous.get("data", {}).get(key) != resolved["data"].get(key)
            for key in ("format", "train_path", "val_path")
        ):
            best = float("inf")
        if manager.rank == 0:
            (output / "resume.json").write_text(
                json.dumps(
                    {
                        "checkpoint_directory": str(load_dir),
                        "epoch": start_epoch,
                        "previous_configuration": previous,
                        "configuration": resolved,
                    },
                    indent=2,
                )
            )
    elif (output / "config.yaml").exists():
        raise FileExistsError(
            "Output directory already contains a run; set training.resume=true or choose a new directory"
        )
    elif initial_weights is not None:
        initialization = load_initial_weights(
            model, Path(to_absolute_path(initial_weights))
        )
        if manager.world_size > 1:
            initializations = [None] * manager.world_size
            distributed.all_gather_object(initializations, initialization)
            if any(record != initialization for record in initializations):
                raise RuntimeError("Initial model weights differ across ranks")
        initialization["verified_ranks"] = manager.world_size
    else:
        initialization = {
            "kind": "seeded_fresh",
            "seed": cfg.seed,
            "loaded_model_state_sha256": _state_hash(model.state_dict()),
            "optimizer_restored": False,
            "scheduler_restored": False,
            "scaler_restored": False,
        }
        if manager.world_size > 1:
            initializations = [None] * manager.world_size
            distributed.all_gather_object(initializations, initialization)
            if any(record != initialization for record in initializations):
                raise RuntimeError("Fresh model initialization differs across ranks")
        initialization["verified_ranks"] = manager.world_size
    if manager.rank == 0:
        if initialization is not None:
            initialization_path = output / "initialization.json"
            if (
                initialization_path.exists()
                and json.loads(initialization_path.read_text()) != initialization
            ):
                raise ValueError("Conflicting weights-only initialization record")
            if not initialization_path.exists():
                initialization_path.write_text(
                    json.dumps(initialization, indent=2) + "\n"
                )
        OmegaConf.save(cfg, output / "config.yaml", resolve=True)
        factors = getattr(train_data.reader, "field_factors", {})
        if factors:
            np.savez(
                output / "normalization.npz",
                **{
                    f"{field}_{key}": value.numpy()
                    for field, stats in factors.items()
                    for key, value in stats.items()
                },
            )
        log.info(
            "Training %s parameters on %s cases, validating %s cases, world size %s",
            sum(p.numel() for p in model.parameters()),
            len(train_data),
            len(val_data),
            manager.world_size,
        )
    if cfg.training.get("compile_blocks", False):
        # Reuse compiled graphs across equal-shaped layers while keeping Module
        # checkpoint names and the stochastic geometry sampler unchanged.
        for block in (model.position, *model.geometry_attention, *model.decoder):
            block.compile()
        if not model.share_weights:
            for block in model.encoder_refinement:
                block.compile()
    trained = (
        DistributedDataParallel(
            model,
            device_ids=[manager.local_rank] if device.type == "cuda" else None,
            broadcast_buffers=False,
        )
        if manager.world_size > 1
        else model
    )
    history = []
    epoch = start_epoch
    while (
        optimizer_updates < max_updates
        if max_updates is not None
        else epoch < cfg.training.epochs
    ):
        # Epoch seeding makes interrupted epoch-boundary resumes match uninterrupted runs.
        seed_everything(cfg.seed + 1000003 * epoch + manager.rank)
        loader.set_epoch(epoch)
        trained.train()
        total = torch.zeros(2, device=device, dtype=torch.float64)
        gradient_stats = torch.zeros(4, device=device, dtype=torch.float64)
        gradient_max = torch.zeros((), device=device, dtype=torch.float64)
        epoch_start = time.monotonic()
        data_wait = 0.0
        previous_step_end = epoch_start
        learning_rate = optimizer.param_groups[0]["lr"]
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        for step, host_batch in enumerate(loader):
            data_wait += time.monotonic() - previous_step_end
            batch = host_batch.to(device)
            if "cfd_mesh_vertices" in batch:
                from cfd_features import augment_query_features

                augment_query_features(batch, cfg)
            optimizer.zero_grad(set_to_none=True)
            inputs = model_inputs(batch)
            with autocast_context(device, cfg.training.precision):
                predictions = trained(**inputs)
            loss = joint_loss(
                predictions,
                batch,
                kind=loss_kind,
                field_factors=factors if loss_space == "physical" else None,
            )
            valid = torch.isfinite(loss).to(torch.int32)
            if manager.world_size > 1:
                distributed.all_reduce(valid, op=distributed.ReduceOp.MIN)
            if not valid.item():
                raise FloatingPointError(
                    "Non-finite training loss; check input scaling and precision"
                )
            gradient_norm, skipped = backward_step(
                loss, model, optimizer, scaler, cfg.training.gradient_clip
            )
            if not skipped:
                optimizer_updates += 1
                if scheduler is not None and scheduler_interval == "update":
                    scheduler.step()
            finite_norm = torch.isfinite(gradient_norm)
            safe_norm = torch.where(finite_norm, gradient_norm, 0.0).double()
            gradient_stats[0] += safe_norm
            gradient_stats[1] += finite_norm
            gradient_stats[2] += skipped
            gradient_stats[3] += 1
            gradient_max = torch.maximum(gradient_max, safe_norm)
            count = batch["geometry"].shape[0]
            total[0] += loss.detach().double() * count
            total[1] += count
            interval = cfg.training.get("log_interval", 0)
            if interval and step % interval == 0 and manager.rank == 0:
                log.info(
                    "epoch=%s step=%s loss=%.6g elapsed=%.1fs",
                    epoch + 1,
                    step,
                    float(loss.detach()),
                    time.monotonic() - epoch_start,
                )
            previous_step_end = time.monotonic()
            if max_updates is not None and optimizer_updates >= max_updates:
                break
        if manager.world_size > 1:
            distributed.all_reduce(total)
            distributed.all_reduce(gradient_stats)
            distributed.all_reduce(gradient_max, op=distributed.ReduceOp.MAX)
        train_seconds = time.monotonic() - epoch_start
        validation_start = time.monotonic()
        validation = evaluate(
            model,
            val_data,
            device,
            cfg.validation.chunk_size,
            cfg.training.precision,
            manager.rank,
            manager.world_size,
            cfg,
        )
        validation_seconds = time.monotonic() - validation_start
        record = {
            "epoch": epoch + 1,
            "train_loss": float(total[0] / total[1]),
            "learning_rate": learning_rate,
            "validation": validation,
            "optimization": {
                "skipped_steps": int(gradient_stats[2] / manager.world_size),
                "optimizer_steps": int(
                    (gradient_stats[3] - gradient_stats[2]) / manager.world_size
                ),
                "optimizer_updates": optimizer_updates,
                "gradient_norm_mean": float(
                    gradient_stats[0] / gradient_stats[1].clamp_min(1)
                ),
                "gradient_norm_max": float(gradient_max),
                "grad_scale": scaler.get_scale(),
            },
        }
        if scheduler is not None and scheduler_interval == "epoch":
            scheduler.step()
        history.append(record)
        is_best = validation["loss"] < best
        best = min(best, validation["loss"])
        if manager.rank == 0:
            log.info("%s", json.dumps(record))
            if record["optimization"]["skipped_steps"]:
                log.warning(
                    "Gradient overflow skipped %s optimizer updates in epoch %s; "
                    "current scale=%s. Check attention precision and gradient norms.",
                    record["optimization"]["skipped_steps"],
                    epoch + 1,
                    scaler.get_scale(),
                )
            timing = dict(
                epoch=epoch + 1,
                train_seconds=train_seconds,
                validation_seconds=validation_seconds,
                data_wait_seconds=data_wait,
                seconds_per_update=train_seconds
                / max(1, record["optimization"]["optimizer_steps"]),
                peak_allocated_bytes=(
                    torch.cuda.max_memory_allocated(device)
                    if device.type == "cuda"
                    else 0
                ),
                peak_reserved_bytes=(
                    torch.cuda.max_memory_reserved(device)
                    if device.type == "cuda"
                    else 0
                ),
            )
            log.info("Epoch timing: %s", json.dumps(timing))
            with (output / "timings.jsonl").open("a") as stream:
                stream.write(json.dumps(timing) + "\n")
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            if (
                epoch == start_epoch
                or (epoch + 1) % cfg.training.get("save_interval", 1) == 0
                or epoch + 1 == cfg.training.epochs
                or (max_updates is not None and optimizer_updates >= max_updates)
            ):
                _save_training_checkpoint(
                    checkpoint_dir,
                    models=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    epoch=epoch + 1,
                    metadata={
                        "config": resolved,
                        "world_size": manager.world_size,
                        "best_validation_loss": best,
                        "initialization": initialization,
                        "optimizer_updates": optimizer_updates,
                        "batches_seen_in_epoch": step + 1,
                        "epoch_complete": step + 1 == len(loader),
                    },
                )
            if is_best:
                temporary_best = output / ".best.pending.mdlus"
                model.save(str(temporary_best))
                temporary_best.replace(output / "best.mdlus")
        if manager.world_size > 1:
            distributed.barrier()
        epoch += 1
    return history


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra CLI, also launchable with torchrun for distributed training."""
    try:
        run(cfg)
    finally:
        if DistributedManager.is_initialized():
            DistributedManager.cleanup()


if __name__ == "__main__":
    main()
