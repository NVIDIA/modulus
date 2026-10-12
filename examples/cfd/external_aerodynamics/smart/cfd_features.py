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

"""Geometry-only query features from the native CFD vehicle triangle mesh.

Distances are exact nearest-triangle *unsigned* distances in source length units
(metres for DrivAerML). Directions point from the closest triangle point to the
query. Normals preserve the native triangle winding. We deliberately do not use
CAD-derived cache arrays or infer a watertight sign from an unverified CFD mesh.
Only requested queries are processed. One mesh acceleration structure is retained
per process; full volume features are never materialized or cached.
"""

from __future__ import annotations

import threading

import torch
import warp as wp

from physicsnemo.core.function_spec import FunctionSpec
from physicsnemo.nn.functional.geometry.sdf import _bvh_query_distance


class CFDQueryFeatures:
    """Reusable exact CFD triangle queries on the input tensor's CPU/GPU device.

    Reuses PhysicsNeMo's Warp nearest-triangle kernel while retaining its BVH
    across query chunks. Instances hold geometry tensors only, never targets.
    """

    def __init__(self, vertices: torch.Tensor, faces: torch.Tensor):
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
            raise ValueError("CFD mesh vertices must be nonempty (N,3)")
        if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
            raise ValueError("CFD vehicle surface must contain triangle faces")
        if faces.dtype not in (torch.int32, torch.int64):
            raise ValueError("CFD mesh connectivity must be integer")
        if not vertices.is_floating_point():
            raise ValueError("CFD vertices must be floating-point coordinates")
        if vertices.device != faces.device:
            raise ValueError("CFD vertices and faces must be on the same device")
        if not torch.isfinite(vertices).all():
            raise ValueError("CFD vertices must be finite")
        if faces.min() < 0 or faces.max() >= len(vertices):
            raise ValueError("CFD connectivity references a missing vertex")
        self.vertices = vertices.detach().to(torch.float32).contiguous()
        self.faces = faces.detach().to(torch.int32).contiguous()
        launch_device, stream = FunctionSpec.warp_launch_context(self.vertices)
        with FunctionSpec.warp_stream_scope(stream):
            wp.init()
            self._wp_vertices = wp.from_torch(self.vertices, dtype=wp.vec3f)
            self._wp_faces = wp.from_torch(self.faces.reshape(-1), dtype=wp.int32)
            self.mesh = wp.Mesh(
                points=self._wp_vertices,
                indices=self._wp_faces,
                support_winding_number=False,
            )
        self.device = launch_device

    @torch.no_grad()
    def closest(self, queries: torch.Tensor):
        """Return exact closest points and triangle indices, preserving order."""
        if queries.ndim != 2 or queries.shape[1] != 3:
            raise ValueError("CFD queries must have shape (N,3)")
        if not queries.is_floating_point():
            raise ValueError("CFD queries must be floating-point coordinates")
        if queries.device != self.vertices.device or not torch.isfinite(queries).all():
            raise ValueError("CFD queries must be finite and share the mesh device")
        points = queries.to(torch.float32).contiguous()
        distances = torch.full(
            (len(points),), float("nan"), device=points.device, dtype=torch.float32
        )
        closest = torch.full_like(points, float("nan"))
        face_ids = torch.full(
            (len(points),), -1, device=points.device, dtype=torch.int64
        )
        if not len(points):
            return closest, face_ids
        device, stream = FunctionSpec.warp_launch_context(points)
        with FunctionSpec.warp_stream_scope(stream):
            wp.launch(
                _bvh_query_distance,
                dim=len(points),
                inputs=[
                    self.mesh.id,
                    wp.from_torch(points, dtype=wp.vec3f),
                    float("inf"),
                    wp.from_torch(distances, dtype=wp.float32),
                    wp.from_torch(closest, dtype=wp.vec3f),
                    wp.from_torch(face_ids, dtype=wp.int64),
                    False,
                ],
                device=device,
                stream=stream,
            )
        if (face_ids < 0).any() or not torch.isfinite(closest).all():
            raise ValueError(
                "CFD nearest-triangle query failed; inspect degenerate faces"
            )
        return closest, face_ids

    def volume(self, queries: torch.Tensor):
        """Return [distance in metres, unit closest-surface-to-query direction]."""
        closest, _ = self.closest(queries)
        delta = queries.float() - closest
        distance = torch.linalg.vector_norm(delta, dim=-1, keepdim=True)
        # At surface points float32 closest-point reconstruction can leave a
        # sub-micrometre residual; do not amplify it into a spurious direction.
        direction = torch.where(distance > 1e-6, delta / distance.clamp_min(1e-30), 0)
        return torch.cat((distance, direction), dim=-1)

    def surface(self, queries: torch.Tensor, tolerance: float = 1e-4):
        """Return unit native triangle normals for surface-centroid queries."""
        closest, ids = self.closest(queries)
        if (
            torch.linalg.vector_norm(queries.float() - closest, dim=-1) > tolerance
        ).any():
            raise ValueError("Surface queries are not on the supplied CFD vehicle mesh")
        triangles = self.vertices[self.faces[ids].long()]
        normals = torch.linalg.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0], dim=-1
        )
        lengths = torch.linalg.vector_norm(normals, dim=-1, keepdim=True)
        if (lengths == 0).any() or not torch.isfinite(lengths).all():
            raise ValueError("Degenerate CFD triangle cannot provide a unit normal")
        return normals / lengths


# Bounded cache protects streamed native inference without retaining all cases.
# Strong source references prevent an allocator reusing a pointer while cached.
_cache = None
_cache_lock = threading.RLock()


def clear_feature_cache():
    """Release the single retained CFD mesh and BVH."""
    global _cache
    with _cache_lock:
        _cache = None


def _identity(value):
    # Inference tensors intentionally have no version counter. The reader's
    # geometry is immutable and held strongly for the lifetime of this cache.
    version = None if torch.is_inference(value) else value._version
    return (value.data_ptr(), tuple(value.shape), value.device, value.dtype, version)


def _get_geometry(vertices, faces):
    global _cache
    key = (_identity(vertices), _identity(faces))
    with _cache_lock:
        if _cache is None or _cache[0] != key:
            _cache = None
            query = CFDQueryFeatures(vertices, faces)
            _cache = (key, query, vertices, faces)
        return _cache[1]


@torch.no_grad()
def augment_query_features(batch, cfg):
    """Add requested query features to a batched mutable mapping in place.

    Call after transferring the selected query tensors and CFD mesh to the
    training/inference device. For native inference pass one query chunk and
    keep mesh tensors stable across chunks. Coordinate normalization is undone
    before geometric queries; normals and distances use the physical frame.
    Geometry RNG and all other RNG streams remain untouched.
    """
    active = {
        field: cfg.model.get(f"{field}_query_feature_channels", 0)
        for field in ("surface", "volume")
    }
    requested = [
        field
        for field, channels in active.items()
        if channels
        and f"{field}_queries" in batch
        and f"{field}_query_features" not in batch
    ]
    if not requested:
        return batch
    if any(active[f] != (3 if f == "surface" else 4) for f in requested):
        raise ValueError("CFD surface/volume query features require 3/4 channels")
    keys = (
        "cfd_mesh_vertices",
        "cfd_mesh_faces",
        "cfd_query_center",
        "cfd_query_scale",
    )
    if any(key not in batch for key in keys):
        raise ValueError(
            "CFD feature inputs require the native vehicle mesh and coordinate frame"
        )
    vertices, faces, center, scale = (batch[key] for key in keys)
    if (
        vertices.ndim != 3
        or faces.ndim != 3
        or center.shape != (len(vertices), 3)
        or scale.shape != center.shape
    ):
        raise ValueError(
            "CFD feature hook requires batched mesh and [B,3] coordinate frame"
        )
    if (
        (scale <= 0).any()
        or not torch.isfinite(scale).all()
        or not torch.isfinite(center).all()
    ):
        raise ValueError("Invalid CFD query coordinate frame")
    outputs = {field: [] for field in requested}
    for index in range(len(vertices)):
        geometry = _get_geometry(vertices[index], faces[index])
        for field in requested:
            query = batch[f"{field}_queries"][index] * scale[index] + center[index]
            features = (
                geometry.surface(query)
                if field == "surface"
                else geometry.volume(query)
            )
            outputs[field].append(features)
    for field, values in outputs.items():
        batch[f"{field}_query_features"] = torch.stack(values)
    return batch
