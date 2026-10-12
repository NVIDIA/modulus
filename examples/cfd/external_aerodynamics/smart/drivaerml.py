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

"""DrivAerML Zarr input matching the GeoTransolver data configuration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from hydra.utils import to_absolute_path

from physicsnemo.datapipes import (
    Dataset,
    DomainMeshReader,
    Reader,
    TensorStoreZarrReader,
)
from physicsnemo.datapipes._rng import spawn_generator
from physicsnemo.nn.functional import weighted_multinomial


class DrivAerMLReader(TensorStoreZarrReader):
    """Read paired fields with bounded volume I/O and per-case deterministic sampling.

    Uses the complete vehicle CFD surface-cell-center cloud for geometry and
    its bounding-box center for coordinate translation. Optional mesh inputs support
    geometry-only
    query features, computed after transfer to the model device. A cyclic
    five-million-point volume block precedes exact query
    sampling, matching the reference recipe's 25x disk-read multiplier.
    """

    def __init__(self, path, cfg, *, training: bool, targets: bool = True):
        fields, disk_sampling = self._configure(cfg, training=training, targets=targets)
        super().__init__(
            path,
            fields=fields,
            coordinated_subsampling=disk_sampling,
            data_copy_concurrency=cfg.data.io_threads,
            file_io_concurrency=cfg.data.io_threads,
            pin_memory=cfg.data.get("pin_memory", False),
        )
        self.set_generator(torch.Generator().manual_seed(cfg.seed))

    def _configure(self, cfg, *, training: bool, targets: bool = True):
        """Validate shared sampling, coordinate, and field-normalization settings."""
        self.cfg = cfg
        self.training = training
        self.targets = targets
        self._sample_cache = (
            {}
            if not training
            and cfg.data.get("cache_validation", False)
            and cfg.validation.points is not None
            else None
        )
        self.counts = {
            field: cfg.data[f"{field}_points"] if training else cfg.validation.points
            for field in ("surface", "volume")
            if cfg.model[f"{field}_channels"]
        }
        self.query_feature_channels = {
            field: cfg.model.get(f"{field}_query_feature_channels", 0)
            for field in self.counts
        }
        if any(
            width not in (0, 3 if field == "surface" else 4)
            for field, width in self.query_feature_channels.items()
        ):
            raise ValueError("CFD surface/volume query features require 3/4 channels")
        self.cfd_features = any(self.query_feature_channels.values())
        self.geometry_points = cfg.data.geometry_points
        self.geometry_source = cfg.data.get("geometry_source", "cfd_surface")
        if self.geometry_source != "cfd_surface":
            raise ValueError("This recipe requires geometry_source=cfd_surface")
        self.coordinate_frame = cfg.data.get("coordinate_frame", "cfd_surface_bbox")
        if self.coordinate_frame != "cfd_surface_bbox":
            raise ValueError("This recipe requires coordinate_frame=cfd_surface_bbox")
        self.geometry_rng_streams = cfg.data.get("geometry_rng_streams", True)
        if self.geometry_rng_streams is not True:
            raise ValueError("CFD geometry requires independent geometry RNG streams")
        self.scale = torch.tensor(list(cfg.data.reference_scale), dtype=torch.float32)
        if (
            self.scale.shape != (3,)
            or not torch.isfinite(self.scale).all()
            or (self.scale <= 0).any()
        ):
            raise ValueError(
                "reference_scale must contain three finite positive values"
            )
        if cfg.model.parameter_channels != 2:
            raise ValueError(
                "DrivAerML requires two conditions: density and stream speed"
            )
        if any(count is not None and count <= 0 for count in self.counts.values()) or (
            self.geometry_points is not None
            and (
                isinstance(self.geometry_points, bool)
                or not isinstance(self.geometry_points, int)
                or self.geometry_points <= 0
            )
        ):
            raise ValueError(
                "Point counts must be positive (or null for full validation)"
            )
        # Empty defaults preserve the native density/speed convention. Explicit
        # parameter normalization must be honored for both native and Zarr data.
        means = list(cfg.data.get("parameter_mean", []))
        stds = list(cfg.data.get("parameter_std", []))
        self.parameter_mean = torch.tensor(
            means if means else [0.0, 0.0], dtype=torch.float32
        )
        self.parameter_std = torch.tensor(
            stds if stds else [1.0, 1.0], dtype=torch.float32
        )
        if (
            bool(means) != bool(stds)
            or self.parameter_mean.shape != (2,)
            or self.parameter_std.shape != (2,)
            or not torch.isfinite(self.parameter_mean).all()
            or not torch.isfinite(self.parameter_std).all()
            or (self.parameter_std <= 0).any()
        ):
            raise ValueError(
                "Parameter mean/std require two finite values and positive std"
            )
        fields = ["air_density", "stream_velocity"]
        self.field_factors = {}
        normalization = Path(to_absolute_path(cfg.data.normalization_dir))
        for field in self.counts:
            fields.append(f"{field}_mesh_centers")
            if targets:
                fields.append(f"{field}_fields")
            if cfg.data.get("field_statistics"):
                mean = torch.tensor(
                    list(cfg.data.field_statistics[field].mean), dtype=torch.float32
                )
                std = torch.tensor(
                    list(cfg.data.field_statistics[field].std), dtype=torch.float32
                )
            else:
                with np.load(
                    normalization / f"{field}_fields_normalization.npz"
                ) as arrays:
                    mean = torch.from_numpy(arrays["mean"].copy()).float().reshape(-1)
                    std = torch.from_numpy(arrays["std"].copy()).float().reshape(-1)
            if (
                mean.shape != (cfg.model[f"{field}_channels"],)
                or std.shape != mean.shape
            ):
                raise ValueError(
                    f"{field} normalization does not match output channels"
                )
            if (
                not torch.isfinite(mean).all()
                or not torch.isfinite(std).all()
                or (std <= 0).any()
            ):
                raise ValueError(
                    f"{field} normalization must be finite with positive std"
                )
            self.field_factors[field] = {"mean": mean, "std": std}
        volume_count = self.counts.get("volume")
        # Even a volume-only model must read the complete vehicle surface pool.
        # No pressure/shear labels are required to construct CFD geometry inputs.
        if (
            self.geometry_source == "cfd_surface"
            and "surface_mesh_centers" not in fields
        ):
            fields.append("surface_mesh_centers")
        disk_sampling = None
        if volume_count is not None:
            disk_sampling = {
                "n_points": cfg.data.volume_read_multiplier * volume_count,
                "target_keys": [key for key in fields if key.startswith("volume_")],
            }
        return fields, disk_sampling

    def _read_attributes(self, group_path):
        """Read only the small attribute file, avoiding a recursive chunk listing."""
        if (group_path / ".zattrs").exists():
            attributes = json.loads((group_path / ".zattrs").read_text())
        elif (group_path / "zarr.json").exists():
            attributes = json.loads((group_path / "zarr.json").read_text()).get(
                "attributes", {}
            )
        else:
            attributes = {}
        return {
            key: torch.as_tensor(attributes[key], dtype=torch.float32)
            for key in ("air_density", "stream_velocity")
            if key in attributes
        }

    def close(self):
        """Release sampled validation tensors when the reader is closed."""
        if self._sample_cache is not None:
            self._sample_cache.clear()
        super().close()

    def set_epoch(self, epoch):
        """Invalidate memoized validation samples when their sampling seed changes."""
        Reader.set_epoch(self, epoch)
        if self._sample_cache is not None:
            self._sample_cache.clear()

    def set_generator(self, generator):
        """Keep memoized samples consistent with the caller's generator."""
        Reader.set_generator(self, generator)
        if self._sample_cache is not None:
            self._sample_cache.clear()

    def _load_sample(self, index):
        """Reuse the shared async reader, then map sampled arrays to SMART inputs."""
        if self._sample_cache is not None and index in self._sample_cache:
            return self._sample_cache[index]
        raw = super()._load_sample(index)
        return self._prepare(raw, index)

    def geometry_pool(self, raw):
        """Return the complete source cloud before any field-query sampling."""
        return (
            raw["geometry_coordinates"]
            if "geometry_coordinates" in raw
            else raw["surface_mesh_centers"]
        )

    def coordinate_center(self, raw):
        """Use the same deterministic CFD bounding-box frame for geometry and fields."""
        if "smart_coordinate_center" not in raw:
            points = self.geometry_pool(raw)
            raw["smart_coordinate_center"] = (
                points.amin(dim=0) + points.amax(dim=0)
            ) * 0.5
        return raw["smart_coordinate_center"]

    def _feature_mesh(self, index):
        """Resolve the native CFD vehicle mesh without reading CAD or target data.

        Native Zarr samples are shuffled independently of native mesh indexing;
        query features use geometric triangle lookup, never index matching.
        The native mesh supplies connectivity that the field cache omits.
        """
        from physicsnemo.mesh import DomainMesh

        root = self.cfg.data.get("cfd_feature_mesh_root")
        if not root:
            raise ValueError("CFD query features on Zarr require cfd_feature_mesh_root")
        case = Path(self._groups[index]).name.removesuffix(".zarr")
        candidates = sorted(
            (Path(to_absolute_path(root)) / case).glob("domain_*.pdmsh")
        )
        if len(candidates) != 1:
            raise ValueError(
                f"Expected one native CFD domain for {case}, got {len(candidates)}"
            )
        domain = DomainMesh.load(candidates[0])
        return domain.boundaries["vehicle"]

    def _prepare(self, raw, index):
        """Map raw sampled fields to normalized SMART tensors."""
        generator = self._index_generator(index)
        geometry_generator = (
            spawn_generator(generator.initial_seed(), 0)
            if self.geometry_rng_streams
            else generator
        )
        center = self.coordinate_center(raw)
        geometry = self.geometry_pool(raw)
        if self.geometry_points is not None and len(geometry) < self.geometry_points:
            raise ValueError(
                f"{self._groups[index]} has fewer than {self.geometry_points} geometry points"
            )
        ids = (
            slice(None)
            if self.geometry_points is None
            else weighted_multinomial(
                len(geometry),
                self.geometry_points,
                strategy="poisson_gap",
                generator=geometry_generator,
            )
        )
        result = {
            "geometry": (geometry[ids] - center) / self.scale,
            "parameters": torch.stack(
                [raw["air_density"].reshape(()), raw["stream_velocity"].reshape(())]
            ),
        }
        if self.cfd_features:
            if "cfd_mesh_vertices" in raw:
                vertices, faces = raw["cfd_mesh_vertices"], raw["cfd_mesh_faces"]
            else:
                mesh = self._feature_mesh(index)
                vertices, faces = mesh.points, mesh.cells
            result.update(
                cfd_mesh_vertices=vertices.as_subclass(torch.Tensor),
                cfd_mesh_faces=faces.as_subclass(torch.Tensor).to(torch.int32),
                cfd_query_center=center,
                cfd_query_scale=self.scale,
            )
        result["parameters"] = (
            result["parameters"] - self.parameter_mean
        ) / self.parameter_std
        if self.training and self.geometry_rng_streams:
            # A separate per-case/per-epoch stream drives latent and layer draws.
            # Removing the outer pool never advances the CFD sampling generator.
            result["geometry_seed"] = torch.tensor(
                spawn_generator(generator.initial_seed(), 1).initial_seed() % (2**63),
                dtype=torch.int64,
            )
        for field, count in self.counts.items():
            queries = raw[f"{field}_mesh_centers"]
            if count is not None and len(queries) < count:
                raise ValueError(f"{field} has fewer than {count} query points")
            ids = (
                slice(None)
                if count is None
                else weighted_multinomial(
                    len(queries), count, strategy="exact", generator=generator
                )
            )
            result[f"{field}_queries"] = (queries[ids] - center) / self.scale
            if self.targets:
                values = raw[f"{field}_fields"]
                if values.shape != (len(queries), self.cfg.model[f"{field}_channels"]):
                    raise ValueError(
                        f"Invalid {field} field shape: {tuple(values.shape)}"
                    )
                factors = self.field_factors[field]
                result[f"{field}_targets"] = (values[ids] - factors["mean"]) / factors[
                    "std"
                ]
        if not all(torch.isfinite(value).all() for value in result.values()):
            raise ValueError(f"Non-finite data in {self._groups[index]}")
        if self._sample_cache is not None:
            self._sample_cache[index] = result
        return result

    def _get_field_names(self):
        """Advertise transformed SMART keys rather than raw Zarr array names."""
        keys = ["geometry", "parameters"]
        if self.cfd_features:
            keys.extend(
                (
                    "cfd_mesh_vertices",
                    "cfd_mesh_faces",
                    "cfd_query_center",
                    "cfd_query_scale",
                )
            )
        if self.training and self.geometry_rng_streams:
            keys.append("geometry_seed")
        for field in self.counts:
            keys.append(f"{field}_queries")
            if self.targets:
                keys.append(f"{field}_targets")
        return keys


class DrivAerMLMeshReader(DrivAerMLReader):
    """Read native memory-mapped DrivAerML without copying the dataset.

    Mesh and DomainMesh loading reuse PhysicsNeMo readers. Query indices cover
    the complete native field with the library's low-memory Poisson-gap sampler;
    this avoids assuming the native volume's disk order was pre-shuffled.
    """

    def __init__(self, path, cfg, *, training: bool, targets: bool = True):
        self._configure(cfg, training=training, targets=targets)
        Reader.__init__(self, pin_memory=cfg.data.get("pin_memory", False))
        self.domains = DomainMeshReader(path, pattern="run_*/domain_*.pdmsh")
        self._groups = self.domains._paths
        self._length = len(self._groups)
        if len({p.parent.name for p in self._groups}) != self._length:
            raise ValueError("Expected one CFD domain mesh per case directory")
        self.set_generator(torch.Generator().manual_seed(cfg.seed))

    def _load_sample(self, index):
        """Gather query coordinates and physical fields before normalization."""
        if self._sample_cache is not None and index in self._sample_cache:
            return self._sample_cache[index]
        domain, _ = self.domains[index]
        generator = self._index_generator(index)
        rho = torch.as_tensor(
            self.cfg.data.get("reference_density")
            if self.cfg.data.get("reference_density") is not None
            else domain.global_data["rho_inf"].reshape(()),
            dtype=torch.float32,
        )
        speed = torch.as_tensor(
            self.cfg.data.get("reference_speed")
            if self.cfg.data.get("reference_speed") is not None
            else domain.global_data["U_inf"].norm(),
            dtype=torch.float32,
        )
        if (
            not torch.isfinite(rho)
            or not torch.isfinite(speed)
            or rho <= 0
            or speed <= 0
        ):
            raise ValueError("Freestream density and speed must be finite and positive")
        # Match Curator PNM-25.11-Compatible preprocessing, including U*L for nut.
        field_scale = rho * speed.square()
        length_points = domain.boundaries["vehicle"].points
        length = (length_points.amax(dim=0) - length_points.amin(dim=0)).amax()
        if length <= 0 or not torch.isfinite(length):
            raise ValueError("Geometry must have a finite positive length scale")
        raw = {
            "air_density": rho,
            "stream_velocity": speed,
        }
        surface = domain.boundaries["vehicle"]
        raw["geometry_coordinates"] = surface.points[surface.cells].mean(dim=1)
        if self.cfd_features:
            raw["cfd_mesh_vertices"] = surface.points
            raw["cfd_mesh_faces"] = surface.cells
        for field, count in self.counts.items():
            mesh = (
                domain.boundaries["vehicle"] if field == "surface" else domain.interior
            )
            total = mesh.n_cells if field == "surface" else mesh.n_points
            if count is not None and total < count:
                raise ValueError(f"{field} has fewer than {count} query points")
            ids = (
                slice(None)
                if count is None
                else weighted_multinomial(
                    total, count, strategy="poisson_gap", generator=generator
                )
            )
            if field == "surface":
                # CFD query samples are independent of the encoder geometry pool.
                raw["surface_mesh_centers"] = mesh.points[mesh.cells[ids]].mean(dim=1)
                if self.targets:
                    pressure = mesh.cell_data["pMeanTrim"][ids].reshape(-1, 1)
                    shear = mesh.cell_data["wallShearStressMeanTrim"][ids]
                    raw["surface_fields"] = (
                        torch.cat([pressure, shear], dim=1) / field_scale
                    )
            else:
                raw["volume_mesh_centers"] = mesh.points[ids]
                if self.targets:
                    velocity = mesh.point_data["UMeanTrim"][ids] / speed
                    pressure = (
                        mesh.point_data["pMeanTrim"][ids].reshape(-1, 1) / field_scale
                    )
                    nut = mesh.point_data["nutMeanTrim"][ids].reshape(-1, 1) / (
                        speed * length
                    )
                    raw["volume_fields"] = torch.cat([velocity, pressure, nut], dim=1)
        return self._prepare(raw, index)

    def _get_sample_metadata(self, index):
        """Identify the original case for prediction manifests."""
        return {
            "source_file": str(self._groups[index]),
            "source_filename": self._groups[index].parent.name,
        }


def build_drivaerml_dataset(path, cfg, *, training: bool, targets: bool = True):
    """Build the standard PhysicsNeMo Dataset with the DrivAerML reader."""
    reader_type = (
        DrivAerMLMeshReader if cfg.data.format == "drivaerml_mesh" else DrivAerMLReader
    )
    return Dataset(
        reader_type(path, cfg, training=training, targets=targets),
        num_workers=cfg.data.workers,
        device="auto" if training and cfg.data.get("to_device", False) else None,
    )
