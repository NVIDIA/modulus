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

"""Prepare complete, shuffled DrivAerML fields for sequential TensorStore reads.

Every native surface cell and volume point is retained. Only point ordering and
storage layout change; no CFD-derived features are added to model geometry.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import tensorstore as ts
import torch

from physicsnemo.datapipes import DomainMeshReader
from physicsnemo.datapipes._rng import derive_seed


def write_array(path: Path, values: torch.Tensor, chunk_points: int) -> None:
    """Write losslessly compressed float32 Zarr v2 chunks using TensorStore."""
    array = values.detach().float().cpu().numpy()
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(f"Expected a finite two-dimensional array: {path}")
    store = ts.open(
        {
            "driver": "zarr",
            "kvstore": {"driver": "file", "path": str(path)},
            "metadata": {
                "shape": list(array.shape),
                "chunks": [min(chunk_points, len(array)), array.shape[1]],
                "dtype": "<f4",
                "compressor": {
                    "id": "blosc",
                    "cname": "lz4",
                    "clevel": 1,
                    "shuffle": 1,
                    "blocksize": 0,
                },
            },
        },
        create=True,
    ).result()
    store.write(array).result()
    # Verify scattered rows through the actual storage decoder after writing.
    ids = np.unique(np.linspace(0, len(array) - 1, min(257, len(array))).astype(int))
    np.testing.assert_array_equal(store[ids].read().result(), array[ids])


def prepare_case(
    domain,
    output: Path,
    *,
    source: str,
    density: float = 1.205,
    speed: float = 30.0,
    seed: int = 42,
    chunk_points: int = 1000000,
    device: str = "cpu",
) -> dict:
    """Preserve complete point/field pairs in one atomically published Zarr group."""
    if (
        not math.isfinite(density)
        or not math.isfinite(speed)
        or density <= 0
        or speed <= 0
        or chunk_points <= 0
    ):
        raise ValueError("Density, speed, and chunk_points must be positive")
    output = Path(output)
    identity = dict(
        format_version=1,
        source=source,
        density=density,
        speed=speed,
        seed=seed,
        chunk_points=chunk_points,
    )
    identity["geometry_source"] = "cfd_surface"
    if output.exists():
        previous = json.loads((output / ".zattrs").read_text())
        if previous.get("preparation") != identity:
            raise ValueError(f"Existing preparation differs: {output}")
        return previous
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial-" + uuid.uuid4().hex)
    partial.mkdir()
    started = time.monotonic()
    try:
        (partial / ".zgroup").write_text('{"zarr_format": 2}')
        points = domain.boundaries["vehicle"].points.as_subclass(torch.Tensor)
        length = (points.amax(0) - points.amin(0)).amax()
        if not torch.isfinite(length) or length <= 0:
            raise ValueError("Source geometry has invalid length")
        counts = {}
        for field_index, field in enumerate(("surface", "volume")):
            mesh = (
                domain.boundaries["vehicle"] if field == "surface" else domain.interior
            )
            count = mesh.n_cells if field == "surface" else mesh.n_points
            counts[field] = count
            rng = torch.Generator(device=device).manual_seed(
                derive_seed(seed, field_index)
            )
            order = torch.randperm(count, device=device, generator=rng)
            if field == "surface":
                vertices = mesh.points.to(device)
                cells = mesh.cells.to(device)
                coordinates = vertices[cells].mean(1)
                del vertices, cells
                data = mesh.cell_data
                values = torch.cat(
                    (
                        data["pMeanTrim"].to(device).reshape(-1, 1),
                        data["wallShearStressMeanTrim"].to(device),
                    ),
                    1,
                )
                values = values / (density * speed**2)
            else:
                coordinates = mesh.points.to(device)
                data = mesh.point_data
                values = torch.cat(
                    (
                        data["UMeanTrim"].to(device) / speed,
                        data["pMeanTrim"].to(device).reshape(-1, 1)
                        / (density * speed**2),
                        data["nutMeanTrim"].to(device).reshape(-1, 1)
                        / (speed * float(length)),
                    ),
                    1,
                )
            write_array(
                partial / f"{field}_mesh_centers", coordinates[order], chunk_points
            )
            write_array(partial / f"{field}_fields", values[order], chunk_points)
            del order, coordinates, values
        metadata = dict(
            air_density=density,
            stream_velocity=speed,
            preparation=identity,
            point_counts=counts,
            geometry_points=counts["surface"],
            turbulence_reference_length=float(length),
            preparation_seconds=time.monotonic() - started,
        )
        (partial / ".zattrs").write_text(json.dumps(metadata, indent=2))
        partial.rename(output)
        return metadata
    except BaseException:
        shutil.rmtree(partial)
        raise


def main() -> None:
    """Prepare disjoint case partitions under torchrun or a single process."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--density", type=float, default=1.205)
    parser.add_argument("--speed", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-points", type=int, default=1000000)
    args = parser.parse_args()
    torch.set_num_threads(4)
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    device = args.device
    if device == "cuda":
        device = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
        torch.cuda.set_device(device)
    domains = DomainMeshReader(args.input, pattern="run_*/domain_*.pdmsh")
    for index in range(rank, len(domains), world):
        source = domains._paths[index]
        domain, _ = domains[index]
        report = prepare_case(
            domain,
            args.output / (source.parent.name + ".zarr"),
            source=str(source.resolve()),
            density=args.density,
            speed=args.speed,
            seed=derive_seed(args.seed, index),
            chunk_points=args.chunk_points,
            device=device,
        )
        print(
            json.dumps(dict(rank=rank, case=source.parent.name, **report)), flush=True
        )


if __name__ == "__main__":
    main()
