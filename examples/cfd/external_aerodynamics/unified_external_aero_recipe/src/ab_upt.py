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

"""AB-UPT adapter and bounded, independently sampled DomainMesh reader.

Only the paper and PhysicsNeMo APIs were used for this integration. Sampling
streams are independent of query counts, access order, worker count, and rank.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import torch
from output_normalize import split_concat_by_target
from tensordict import TensorDict

from physicsnemo import Module
from physicsnemo.datapipes.keys import as_nested_key
from physicsnemo.datapipes.readers import DomainMeshReader
from physicsnemo.experimental.models.ab_upt import ABUPT
from physicsnemo.mesh import DomainMesh, Mesh
from physicsnemo.mesh.calculus.measure import cell_measures, set_cell_measures

SURFACE_TARGETS = {"surface.pressure": "scalar", "surface.wss": "vector"}
VOLUME_TARGETS = {
    "volume.velocity": "vector",
    "volume.pressure": "scalar",
    "volume.nut": "scalar",
}


def _modes(mode: str) -> tuple[bool, bool]:
    """Validate the supervised branches."""
    if mode not in ("surface", "volume", "joint"):
        raise ValueError("mode must be 'surface', 'volume', or 'joint'")
    return mode != "volume", mode != "surface"


class ABUPTRecipe(Module):
    """Map the independent AB-UPT operator to named recipe targets.

    A context is built once per forward. Decoder chunks share it, including
    during training; chunking does not detach gradients or resample anchors.
    Smaller chunks bound individual attention operations, not the full
    training autograd graph. This DrivAerML adapter predicts velocity,
    pressure, and eddy viscosity in the volume, rather than the paper's
    velocity/pressure/vorticity channel layout.
    """

    def __init__(
        self,
        mode: str = "joint",
        query_chunk_size: int = 4096,
        coordinate_scale: float = 1000.0,
        hidden_dim: int = 192,
        num_heads: int = 3,
        num_interaction_blocks: int = 2,
        num_shared_layers: int = 1,
        num_decoder_layers: int = 6,
        pooling_radius: float = 0.25,
        pooling_chunk_size: int = 4096,
        radius_search_backend: str | None = None,
    ):
        super().__init__()
        surface, volume = _modes(mode)
        if (
            isinstance(query_chunk_size, bool)
            or not isinstance(query_chunk_size, int)
            or query_chunk_size < 1
        ):
            raise ValueError("query_chunk_size must be a positive integer")
        if not math.isfinite(coordinate_scale) or coordinate_scale <= 0:
            raise ValueError("coordinate_scale must be finite and positive")
        self.coordinate_scale = coordinate_scale
        self.mode = mode
        self.query_chunk_size = query_chunk_size
        self.operator = ABUPT(
            surface_channels=4 if surface else 0,
            volume_channels=5 if volume else 0,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_interaction_blocks=num_interaction_blocks,
            num_shared_layers=num_shared_layers,
            num_decoder_layers=num_decoder_layers,
            pooling_radius=pooling_radius * coordinate_scale,
            pooling_chunk_size=pooling_chunk_size,
            parameter_channels=3,
            radius_search_backend=radius_search_backend,
        )
        self.meta = replace(self.operator.meta, name="ABUPTRecipe")

    def forward(
        self,
        geometry,
        supernodes,
        surface_anchors,
        volume_anchors,
        parameters,
        surface_queries=None,
        volume_queries=None,
    ):
        """Return named batched fields; surface and volume lengths may differ."""
        if parameters.ndim == 3 and parameters.shape[1] == 1:
            parameters = parameters[:, 0]
        # Velocity magnitude remains available to distinguish flow conditions.
        context = self.operator.encode(
            geometry * self.coordinate_scale,
            supernodes * self.coordinate_scale,
            surface_anchors * self.coordinate_scale,
            volume_anchors * self.coordinate_scale,
            parameters,
        )
        output = TensorDict({}, batch_size=[geometry.shape[0]], device=geometry.device)
        for field, enabled, queries, targets in (
            (
                "surface",
                self.operator.surface_channels,
                surface_queries,
                SURFACE_TARGETS,
            ),
            ("volume", self.operator.volume_channels, volume_queries, VOLUME_TARGETS),
        ):
            if not enabled:
                if queries is not None:
                    raise ValueError(f"Queries supplied for disabled {field} head")
                continue
            if queries is None:
                raise ValueError(f"Explicit {field}_queries are required by the recipe")
            chunks = queries.split(self.query_chunk_size, dim=1)
            prediction = torch.cat(
                [
                    self.operator.decode_queries(
                        q * self.coordinate_scale, context, field
                    )
                    for q in chunks
                ],
                dim=1,
            )
            branch = split_concat_by_target(prediction, targets)
            for name in targets:
                key = as_nested_key(name)
                output.set(key, branch[key])
        return output


class ABUPTDomainReader(DomainMeshReader):
    """Sample geometry, anchors, and targets independently from native domains.

    This reader gathers only selected fields and coordinates from memory-mapped
    TensorDict stores before device transfer. Zarr loading uses the base
    reader's full-domain path, so its host-memory use can be larger. Geometry
    uses surface vertices, supernodes are a subset of that cloud, and surface
    anchors/queries use cell centroids. Both anchor streams are always present.
    Uniform sampling without replacement is performed on CPU with independent
    NumPy SeedSequence streams. Validation must use a distinct reader at epoch 0.

    ``surface_queries`` and ``volume_queries`` may be None for all locations.
    Counts above the available size use all locations without replacement.
    Surface integration measures use inverse inclusion probabilities; sampled
    force integrals are Monte Carlo estimates, not full-surface integrals.
    """

    def __init__(
        self,
        path,
        *,
        pattern="run_*/domain_*.pdmsh",
        mode="joint",
        geometry_points=65536,
        num_supernodes=16384,
        num_surface_anchors=16384,
        num_volume_anchors=16384,
        surface_queries=16384,
        volume_queries=16384,
        pin_memory=False,
    ):
        super().__init__(path, pattern=pattern, pin_memory=pin_memory)
        self.surface, self.volume = _modes(mode)
        for name, count in (
            ("geometry_points", geometry_points),
            ("num_supernodes", num_supernodes),
            ("num_surface_anchors", num_surface_anchors),
            ("num_volume_anchors", num_volume_anchors),
            ("surface_queries", surface_queries),
            ("volume_queries", volume_queries),
        ):
            if count is None and name.endswith("queries"):
                continue
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ValueError(f"{name} must be a positive integer")
        if num_supernodes > geometry_points:
            raise ValueError("num_supernodes must not exceed geometry_points")
        self.geometry_points, self.num_supernodes = geometry_points, num_supernodes
        self.num_surface_anchors, self.num_volume_anchors = (
            num_surface_anchors,
            num_volume_anchors,
        )
        self.surface_queries, self.volume_queries = surface_queries, volume_queries

    def close(self):
        """Release reader resources (loads are scoped to each sample)."""

    def _indices(self, total, count, index, stream):
        """Select sorted indices with a stream independent of other draws."""
        if total == 0:
            raise ValueError("AB-UPT requires nonempty surface and volume meshes")
        if count is None or count >= total:
            return torch.arange(total)
        rng = np.random.default_rng(
            np.random.SeedSequence([self._seed_base or 0, self._epoch, index, stream])
        )
        return torch.from_numpy(np.sort(rng.choice(total, count, replace=False)))

    def __getitem__(self, index):
        """Load one case and return small geometry/anchor/query meshes."""
        domain = self._load_sample(index)
        if "vehicle" not in domain.boundary_names:
            raise ValueError("AB-UPT requires a 'vehicle' surface boundary")
        vehicle, interior = domain.boundaries["vehicle"], domain.interior
        if vehicle.cells.shape[-1] != 3:
            raise ValueError("AB-UPT recipe requires a triangulated vehicle surface")
        gi = self._indices(vehicle.n_points, self.geometry_points, index, 0)
        geometry = vehicle.points[gi].clone()
        si = self._indices(len(gi), self.num_supernodes, index, 1)
        sa = self._indices(vehicle.n_cells, self.num_surface_anchors, index, 2)
        va = self._indices(interior.n_points, self.num_volume_anchors, index, 3)
        # For inactive output heads retain only anchors, without loading labels.
        sq = (
            self._indices(vehicle.n_cells, self.surface_queries, index, 4)
            if self.surface
            else sa
        )
        vq = (
            self._indices(interior.n_points, self.volume_queries, index, 5)
            if self.volume
            else va
        )

        def centroids(ids):
            return vehicle.points[vehicle.cells[ids].long()].mean(dim=1)

        surface_data, volume_data = TensorDict({}, [len(sq)]), TensorDict({}, [len(vq)])
        if self.surface:
            for src, dst in (
                ("pMeanTrim", "surface.pressure"),
                ("wallShearStressMeanTrim", "surface.wss"),
            ):
                surface_data.set(as_nested_key(dst), vehicle.cell_data[src][sq].clone())
        if self.volume:
            for src, dst in (
                ("UMeanTrim", "volume.velocity"),
                ("pMeanTrim", "volume.pressure"),
                ("nutMeanTrim", "volume.nut"),
            ):
                volume_data.set(
                    as_nested_key(dst), interior.point_data[src][vq].clone()
                )
        # Remap only selected triangles; avoid a full-mesh point-index map.
        vertices, cells = torch.unique(
            vehicle.cells[sq].long(), sorted=True, return_inverse=True
        )
        sampled_vehicle = Mesh(
            points=vehicle.points[vertices].clone(), cells=cells.reshape(-1, 3)
        )
        # Preserve explicit source measures when present, otherwise geometric areas.
        if "_effective_measure" in vehicle.cell_data:
            measures = vehicle.cell_data["_effective_measure"][sq].clone()
        else:
            measures = cell_measures(sampled_vehicle)
        set_cell_measures(sampled_vehicle, measures * (vehicle.n_cells / len(sq)))
        sampled_vehicle.cell_data["normals"] = sampled_vehicle.cell_normals
        result = DomainMesh(
            interior=Mesh(points=interior.points[vq].clone(), point_data=volume_data),
            boundaries={
                "vehicle": sampled_vehicle,
                "abupt_surface": Mesh(points=centroids(sq), point_data=surface_data),
                "abupt_geometry": Mesh(points=geometry),
                "abupt_supernodes": Mesh(points=geometry[si].clone()),
                "abupt_surface_anchors": Mesh(points=centroids(sa)),
                "abupt_volume_anchors": Mesh(points=interior.points[va].clone()),
            },
            global_data=domain.global_data.clone(),
        )
        if self.pin_memory:
            result = result.pin_memory()
        return result, {"source_path": str(self._paths[index]), "index": index}
