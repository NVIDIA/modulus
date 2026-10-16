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

"""CFD-only geometry and independent online query sampling for SMART.

These components are instantiated only by the SMART dataset configurations.
They preserve the existing recipe's DomainMesh target/force contract.
"""

from __future__ import annotations

import torch
from merge_global_data import MeshReaderWithGlobalData

from physicsnemo.datapipes import DomainMeshReader
from physicsnemo.datapipes._rng import derive_seed
from physicsnemo.datapipes.transforms.mesh import CenterMesh as _CenterMesh
from physicsnemo.datapipes.transforms.mesh import MeshToDomainMesh, SubsampleMesh
from physicsnemo.datapipes.transforms.mesh.base import MeshTransform
from physicsnemo.mesh import DomainMesh, Mesh


class _SMARTSeeds:
    """Attach stateless case/epoch seeds before device transfer and transforms."""

    def close(self):
        """Release cached store references when the dataset is closed."""
        getattr(self, "_zarr_groups", {}).clear()

    def _load_sample(self, index):
        sample = super()._load_sample(index)
        global_data = sample.global_data.clone()
        for stream, name in enumerate(("smart_query_seed", "smart_geometry_seed")):
            if name in global_data:
                raise ValueError(f"Reserved SMART metadata already exists: {name}")
            seed = derive_seed(self._seed_base or 0, self._epoch, index, stream)
            global_data[name] = torch.tensor(seed & ((1 << 63) - 1), dtype=torch.int64)
        if isinstance(sample, DomainMesh):
            return DomainMesh(
                interior=sample.interior,
                boundaries=sample.boundaries,
                global_data=global_data,
            )
        return sample.with_data(global_data=global_data)


class SMARTSurfaceReader(_SMARTSeeds, MeshReaderWithGlobalData):
    """Read a full CFD surface and supply independent query/geometry seeds."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.subsample_n_cells is not None or self.subsample_n_points is not None:
            raise ValueError(
                "SMART surface geometry must not be subsampled by the reader"
            )


class SMARTVolumeReader(_SMARTSeeds, DomainMeshReader):
    """Read sampled volume fields with a full CFD boundary and geometry seed."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.drop_in_file_boundaries or any(
            n is not None for n in self._boundary_subsample_sizes()
        ):
            raise ValueError("SMART requires the complete in-file CFD boundary")
        if self._extra_boundaries:
            raise ValueError(
                "SMART uses CFD geometry; extra boundaries are not supported"
            )


class SMARTSurfaceMeshToDomainMesh(MeshToDomainMesh):
    """Retain all CFD centroids while selecting aligned surface queries.

    The sampled vehicle boundary stays aligned with interior targets, normals
    and effective areas, so the existing force and native export paths work.
    A separate geometry-only boundary holds the complete centroid cloud.
    Target names are injected by the existing dataset builder.
    """

    def __init__(self, n_cells: int, **kwargs):
        super().__init__(**kwargs)
        if isinstance(n_cells, bool) or not isinstance(n_cells, int) or n_cells <= 0:
            raise ValueError("n_cells must be a positive integer")
        if self._interior_points != "cell_centroids":
            raise ValueError("SMART surface queries must use cell_centroids")
        if self._boundary_name == "smart_geometry":
            raise ValueError("smart_geometry is reserved for the full geometry cloud")
        self.n_cells = n_cells

    def __call__(self, mesh: Mesh) -> DomainMesh:
        """Sample all query-associated arrays together, retaining full geometry."""
        geometry = Mesh(points=mesh.cell_centroids)
        # A fresh generator per case avoids worker/prefetch-order dependence.
        sampler = SubsampleMesh(n_cells=self.n_cells)
        sampler.set_generator(
            torch.Generator(device=mesh.points.device).manual_seed(
                int(mesh.global_data["smart_query_seed"].item())
            )
        )
        domain = super().__call__(sampler(mesh))
        return DomainMesh(
            interior=domain.interior,
            boundaries={
                **{
                    # Native CPU export must not retain source memmap tensors.
                    name: boundary.clone()
                    for name, boundary in domain.boundaries.items()
                },
                "smart_geometry": geometry,
            },
            global_data=domain.global_data,
        )


class SMARTVolumeGeometry(MeshTransform):
    """Expose the full CFD boundary centroids without embedding the cloud."""

    def __call__(self, mesh: Mesh) -> Mesh:
        """Require a domain with a CFD vehicle boundary."""
        raise TypeError("SMART volume geometry requires a DomainMesh")

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:
        """Retain the CFD surface and add only its geometry coordinates."""
        if "smart_geometry" in domain.boundaries:
            raise ValueError("smart_geometry is reserved for the full geometry cloud")
        surface = domain.boundaries["vehicle"]
        if surface.n_cells == 0:
            raise ValueError("SMART requires a CFD vehicle boundary with cells")
        return DomainMesh(
            interior=domain.interior,
            boundaries={
                **{
                    # Native CPU export must not retain source memmap tensors.
                    name: boundary.clone()
                    for name, boundary in domain.boundaries.items()
                },
                "smart_geometry": Mesh(points=surface.cell_centroids),
            },
            global_data=domain.global_data,
        )


class CenterMesh(_CenterMesh):
    """Center SMART domains on the full CFD boundary, independent of queries.

    The class name retains the recipe's augmentation insertion convention.
    Surface Mesh inputs use the standard PhysicsNeMo centering operation.
    """

    def apply_to_domain(self, domain: DomainMesh) -> DomainMesh:
        """Apply one boundary-derived translation to geometry and queries."""
        return domain.translate(-self._compute_com(domain.boundaries["vehicle"]))
