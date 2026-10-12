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

"""Stream SMART predictions into NumPy files with bounded GPU query memory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from data import build_dataset
from omegaconf import OmegaConf
from train import autocast_context

from physicsnemo import Module


def file_sha256(path: Path) -> str:
    """Hash a saved artifact without holding another checkpoint copy in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.inference_mode()
def predict(
    run_dir: Path,
    data_dir: Path,
    output_dir: Path,
    *,
    chunk_size: int = 16384,
    device: str = "auto",
    precision: str = "float32",
    checkpoint: Path | None = None,
    num_latents: int | None = None,
) -> list[dict]:
    """Load saved preprocessing and a trained checkpoint, then stream all cases.

    Inputs follow the saved NumPy, native CFD mesh, or prepared Zarr format.
    Targets are unnecessary. Output .npy files are written as memory maps, so
    output storage does not grow GPU memory or require concatenation in RAM.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if num_latents is not None and (
        isinstance(num_latents, bool)
        or not isinstance(num_latents, int)
        or num_latents <= 0
    ):
        raise ValueError("num_latents must be a positive integer")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Prediction output directory must be empty")
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device == "auto"
        else torch.device(device)
    )
    autocast_context(device, precision)
    config_bytes = (run_dir / "config.yaml").read_bytes()
    cfg = OmegaConf.create(config_bytes.decode("utf-8"))
    checkpoint = Path(checkpoint) if checkpoint is not None else run_dir / "best.mdlus"
    checkpoint_hash = file_sha256(checkpoint)
    torch.set_float32_matmul_precision(
        "highest"
        if precision == "float32"
        else cfg.training.get("matmul_precision", "highest")
    )
    # The caller's precision must also control nested attention autocast saved
    # by the training preset. Otherwise --precision=float32 can still run FP16.
    model = (
        Module.from_checkpoint(
            str(checkpoint), override_args={"attention_precision": None}
        )
        .to(device)
        .eval()
    )
    if file_sha256(checkpoint) != checkpoint_hash:
        raise RuntimeError(
            "Checkpoint changed while loading; choose a stable checkpoint"
        )
    for key, expected in cfg.model.items():
        if key in ("attention_precision", "attention_precision_scope"):
            continue
        if getattr(model, key, None) != expected:
            raise ValueError(
                f"Checkpoint differs from run configuration in model.{key}"
            )
    trained_num_latents = model.num_latents
    if num_latents is not None:
        model.num_latents = num_latents
    identity = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "trained_num_latents": trained_num_latents,
        "inference_num_latents": model.num_latents,
        "geometry_points_per_layer": model.geometry_subsample,
        "geometry_source": cfg.data.get(
            "geometry_source",
            "cfd_surface"
            if cfg.data.get("format", "numpy").startswith("drivaerml")
            else "provided",
        ),
        "geometry_pool_cap": cfg.data.get("geometry_points"),
        "coordinate_frame": cfg.data.get("coordinate_frame", "provided"),
        "geometry_seed": model.eval_seed,
        "precision": precision,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "chunk_size": chunk_size,
        "device": str(device),
        "cases": [],
    }
    native = cfg.data.get("format", "numpy").startswith("drivaerml_")
    if native:
        cfg.validation.points = None  # Inference evaluates every supplied query.
    dataset = build_dataset(data_dir, cfg, training=False, targets=False)
    try:
        paths = dataset.reader._groups if native else sorted(data_dir.glob("*.npz"))
        if not paths or len(paths) != len(dataset):
            raise ValueError("data_dir must contain one supported file/group per case")
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest = []
        for index, path in enumerate(paths):
            sample, _ = dataset[index]
            geometry = sample["geometry"].unsqueeze(0).to(device)
            identity["cases"].append(
                {
                    "input": str(path.resolve()),
                    "geometry_points": geometry.shape[1],
                    "coarse_latents": min(model.num_latents, geometry.shape[1]),
                }
            )
            parameters = sample.get("parameters", None)
            parameters = (
                None if parameters is None else parameters.unsqueeze(0).to(device)
            )
            with autocast_context(device, precision):
                contexts = model.encode_geometry(geometry, parameters)
            feature_mesh = {
                key: sample[key].unsqueeze(0).to(device)
                for key in (
                    "cfd_mesh_vertices",
                    "cfd_mesh_faces",
                    "cfd_query_center",
                    "cfd_query_scale",
                )
                if key in sample
            }
            case_name = (
                path.parent.name
                if cfg.data.get("format") == "drivaerml_mesh"
                else path.stem
            )
            record = {"input": str(path.resolve())}
            for field in ("surface", "volume"):
                channels = getattr(model, f"{field}_channels")
                if not channels:
                    continue
                queries = sample[f"{field}_queries"]
                final = output_dir / f"{case_name}.{field}.npy"
                temporary = final.with_suffix(".partial.npy")
                output = np.lib.format.open_memmap(
                    temporary,
                    mode="w+",
                    dtype=np.float32,
                    shape=(len(queries), channels),
                )
                for start in range(0, len(queries), chunk_size):
                    query = queries[start : start + chunk_size].unsqueeze(0).to(device)
                    query_features = None
                    if getattr(model, f"{field}_query_feature_channels", 0):
                        if f"{field}_query_features" in sample:
                            query_features = (
                                sample[f"{field}_query_features"][
                                    start : start + chunk_size
                                ]
                                .unsqueeze(0)
                                .to(device)
                            )
                        else:
                            from cfd_features import augment_query_features

                            feature_batch = dict(feature_mesh)
                            feature_batch[f"{field}_queries"] = query
                            augment_query_features(feature_batch, cfg)
                            query_features = feature_batch[f"{field}_query_features"]
                    with autocast_context(device, precision):
                        values = model.decode_queries(
                            query,
                            contexts,
                            parameters,
                            field=field,
                            query_features=query_features,
                        )
                    values = values[0].float().cpu()
                    factors = getattr(dataset.reader, "field_factors", {}).get(field)
                    if factors:
                        values = values * factors["std"] + factors["mean"]
                    if not torch.isfinite(values).all():
                        raise FloatingPointError(
                            f"Non-finite {field} predictions for {path}"
                        )
                    output[start : start + chunk_size] = values.numpy()
                output.flush()
                del output
                temporary.replace(final)
                record[field] = final.name
            manifest.append(record)
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (output_dir / "inference.json").write_text(
            json.dumps(identity, indent=2) + "\n"
        )
        return manifest
    finally:
        dataset.close()
        if any(
            cfg.model.get(f"{field}_query_feature_channels", 0)
            for field in ("surface", "volume")
        ):
            from cfd_features import clear_feature_cache

            clear_feature_cache()


def main() -> None:
    """Parse inference paths and optional device/precision settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=16384)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Explicit model file; defaults to best.mdlus in --run-dir",
    )
    parser.add_argument(
        "--num-latents",
        type=int,
        help="Inference coarse-latent count; defaults to the saved training value",
    )
    parser.add_argument(
        "--precision", choices=["float32", "float16", "bfloat16"], default="float32"
    )
    predict(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
