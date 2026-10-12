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

"""CFD-only feature geometry, reproducibility, and actual reader acceptance tests."""

import copy

import cfd_features
import pytest
import torch
from cfd_features import CFDQueryFeatures, augment_query_features, clear_feature_cache
from data import model_inputs
from drivaerml import DrivAerMLMeshReader, DrivAerMLReader
from omegaconf import OmegaConf
from prepare_drivaerml import prepare_case

from physicsnemo.mesh import DomainMesh, Mesh


@pytest.fixture(autouse=True)
def empty_feature_cache():
    """Release retained meshes before and after each feature test."""
    clear_feature_cache()
    yield
    clear_feature_cache()


@pytest.fixture
def triangle_geometry():
    # Two orthogonal faces with opposite principal normal directions.
    """Create two orthogonal triangles with known outward directions."""
    vertices = torch.tensor(
        [[0.0, 0, 0], [2.0, 0, 0], [0.0, 2, 0], [4.0, 0, 0], [4.0, 0, 2], [4.0, 2, 0]]
    )
    faces = torch.tensor([[0, 1, 2], [3, 4, 5]])
    return vertices, faces


@pytest.fixture
def queries():
    """Provide on-surface and off-surface points with analytic distances."""
    return torch.tensor(
        [[0.3, 0.4, 2.0], [0.1, 0.2, 0.0], [5.0, 0.2, 0.2], [-1.0, 0.0, 0.0]]
    )


@pytest.fixture
def feature_config():
    """Enable both old SMART query feature projections."""
    return OmegaConf.create(
        {
            "model": {
                "surface_query_feature_channels": 3,
                "volume_query_feature_channels": 4,
            }
        }
    )


def _batch(vertices, faces, queries):
    center = torch.tensor([1.7, -2.0, 0.8])
    scale = torch.tensor([12.0, 4.5, 3.25])
    return {
        "cfd_mesh_vertices": vertices[None],
        "cfd_mesh_faces": faces[None],
        "cfd_query_center": center[None],
        "cfd_query_scale": scale[None],
        "surface_queries": ((vertices[faces].mean(1) - center) / scale)[None],
        "volume_queries": ((queries - center) / scale)[None],
    }


def test_exact_triangle_distance_and_near_surface_direction(triangle_geometry, queries):
    """Compare triangle distances and directions with analytic expectations."""
    geometry = CFDQueryFeatures(*triangle_geometry)
    expected = torch.tensor(
        [
            [2.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, -1.0, 0.0, 0.0],
        ]
    )
    # Interior-face closest points distinguish triangles from a centroid/vertex cloud.
    torch.testing.assert_close(geometry.volume(queries), expected, atol=1e-6, rtol=1e-6)


def test_surface_normals_preserve_native_winding(triangle_geometry):
    """Check unit normals and reversal under reversed connectivity."""
    vertices, faces = triangle_geometry
    queries = vertices[faces].mean(1)
    expected = Mesh(points=vertices, cells=faces).cell_normals
    torch.testing.assert_close(
        CFDQueryFeatures(vertices, faces).surface(queries), expected
    )
    reversed_faces = faces.flip(-1)
    torch.testing.assert_close(
        CFDQueryFeatures(vertices, reversed_faces).surface(queries), -expected
    )


def test_query_order_and_chunk_invariance(triangle_geometry, queries):
    """Feature values must depend on coordinates rather than query order."""
    geometry = CFDQueryFeatures(*triangle_geometry)
    whole = geometry.volume(queries)
    permutation = torch.tensor([3, 0, 2, 1])
    torch.testing.assert_close(
        geometry.volume(queries[permutation]), whole[permutation], atol=0, rtol=0
    )
    chunked = torch.cat(
        [geometry.volume(queries[i : i + 1]) for i in range(len(queries))]
    )
    torch.testing.assert_close(chunked, whole, atol=0, rtol=0)


def test_physical_frame_rng_and_bvh_reuse(triangle_geometry, queries, feature_config):
    """Undo coordinate scaling without advancing RNG or rebuilding the same BVH."""
    vertices, faces = triangle_geometry
    batch = _batch(vertices, faces, queries)
    state = torch.random.get_rng_state().clone()
    augment_query_features(batch, feature_config)
    assert torch.equal(state, torch.random.get_rng_state())
    torch.testing.assert_close(
        batch["volume_query_features"][0],
        CFDQueryFeatures(vertices, faces).volume(queries),
        atol=2e-6,
        rtol=2e-6,
    )
    cached = cfd_features._cache[1]
    chunk = {key: value for key, value in batch.items() if key.startswith("cfd_")}
    chunk["volume_queries"] = batch["volume_queries"][:, 2:]
    augment_query_features(chunk, feature_config)
    assert cfd_features._cache[1] is cached
    torch.testing.assert_close(
        chunk["volume_query_features"],
        batch["volume_query_features"][:, 2:],
        atol=0,
        rtol=0,
    )


def test_cache_invalidation_and_release(triangle_geometry, queries, feature_config):
    """A replaced mesh gets a new BVH and explicit cleanup drops cached references."""
    vertices, faces = triangle_geometry
    batch = _batch(vertices.clone(), faces, queries)
    augment_query_features(batch, feature_config)
    first = cfd_features._cache[1]
    # Replace mesh tensors: previous acceleration structure must never be reused.
    other = _batch(vertices.clone(), faces.clone(), queries)
    augment_query_features(other, feature_config)
    assert cfd_features._cache[1] is not first
    clear_feature_cache()
    assert cfd_features._cache is None


def test_inference_mode_tensors(triangle_geometry, queries, feature_config):
    """Cache geometry safely when tensors have no version counters."""
    expected = _batch(*triangle_geometry, queries)
    augment_query_features(expected, feature_config)
    with torch.inference_mode():
        batch = {
            key: value.clone()
            for key, value in expected.items()
            if not key.endswith("_query_features")
        }
        assert torch.is_inference(batch["cfd_mesh_vertices"])
        augment_query_features(batch, feature_config)
        torch.testing.assert_close(
            batch["volume_query_features"], expected["volume_query_features"]
        )


def test_legacy_inputs_omit_absent_feature_keywords(triangle_geometry):
    """Coordinate-only inputs retain the older model call signature."""
    vertices, faces = triangle_geometry
    inputs = model_inputs(
        {
            "geometry": vertices[None],
            "surface_queries": vertices[faces].mean(1)[None],
            "parameters": torch.ones(1, 2),
        }
    )
    assert "surface_query_features" not in inputs
    assert "volume_query_features" not in inputs


@pytest.fixture
def prepared_case(tmp_path, triangle_geometry):
    """Save an original CFD mesh and aligned Zarr fields for reader tests."""
    vertices, faces = triangle_geometry
    volume_points = torch.tensor(
        [[0.2, 0.3, 0.6], [0.4, 0.1, 1.1], [0.1, 0.1, 2.0], [0.5, 0.5, 0.5]]
    )
    surface = Mesh(
        points=vertices,
        cells=faces,
        cell_data={
            "pMeanTrim": torch.tensor([1.0, 2.0]),
            "wallShearStressMeanTrim": torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        },
    )
    volume = Mesh(
        points=volume_points,
        cells=torch.empty((0, 4), dtype=torch.int64),
        point_data={
            "UMeanTrim": volume_points * 0.3,
            "pMeanTrim": volume_points[:, 2],
            "nutMeanTrim": volume_points[:, 0] * 0.001,
        },
    )
    domain = DomainMesh(
        interior=volume,
        boundaries={"vehicle": surface},
        global_data={
            "rho_inf": torch.tensor(1.205),
            "U_inf": torch.tensor([30.0, 0.0, 0.0]),
        },
    )
    native = tmp_path / "native/run_1"
    native.mkdir(parents=True)
    domain.save(native / "domain_run_1.pdmsh")
    prepare_case(
        domain,
        tmp_path / "data/run_1.zarr",
        source=str(native / "domain_run_1.pdmsh"),
        chunk_points=2,
    )
    cfg = OmegaConf.create(
        {
            "seed": 42,
            "model": {
                "surface_channels": 4,
                "volume_channels": 5,
                "parameter_channels": 2,
                "surface_query_feature_channels": 3,
                "volume_query_feature_channels": 4,
            },
            "data": {
                "surface_points": 2,
                "volume_points": 3,
                "geometry_points": None,
                "geometry_source": "cfd_surface",
                "coordinate_frame": "cfd_surface_bbox",
                "geometry_rng_streams": True,
                "reference_scale": [12.0, 4.5, 3.25],
                "reference_density": 1.205,
                "reference_speed": 30.0,
                "normalization_dir": str(tmp_path),
                "io_threads": 1,
                "pin_memory": False,
                "volume_read_multiplier": 1,
                "cfd_feature_mesh_root": str(tmp_path / "native"),
                "field_statistics": {
                    "surface": {"mean": [0.0] * 4, "std": [1.0] * 4},
                    "volume": {"mean": [0.0] * 5, "std": [1.0] * 5},
                },
            },
            "validation": {"points": 2},
        }
    )
    return tmp_path, cfg, surface


@pytest.mark.parametrize("native", [False, True])
def test_reader_alignment_reproducible_samples_and_no_targets(
    prepared_case, native, monkeypatch
):
    """Verify shuffled field alignment, repeatable samples, and target-free inference."""
    root, cfg, surface = prepared_case

    def forbidden_cad(*args, **kwargs):
        """Fail immediately if a CFD-only path attempts to open a CAD reader."""
        raise AssertionError("CFD-only features must not read CAD")

    monkeypatch.setattr("physicsnemo.datapipes.MeshReader", forbidden_cad)
    reader_type = DrivAerMLMeshReader if native else DrivAerMLReader
    path = root / ("native" if native else "data")
    baseline_cfg = copy.deepcopy(cfg)
    baseline_cfg.model.surface_query_feature_channels = 0
    baseline_cfg.model.volume_query_feature_channels = 0
    reader = reader_type(path, cfg, training=True)
    baseline = reader_type(path, baseline_cfg, training=True)
    snapshots = []
    try:
        centers = surface.points[surface.cells].mean(1)
        for epoch in (0, 3, 0):
            reader.set_epoch(epoch)
            baseline.set_epoch(epoch)
            row, _ = reader[0]
            original, _ = baseline[0]
            for key in original.keys():
                torch.testing.assert_close(row[key], original[key], atol=0, rtol=0)
            batch = {key: value.unsqueeze(0) for key, value in row.items()}
            augment_query_features(batch, cfg)
            assert model_inputs(batch)["volume_query_features"].shape == (1, 3, 4)
            queries = (
                batch["surface_queries"][0] * row["cfd_query_scale"]
                + row["cfd_query_center"]
            )
            for index, point in enumerate(queries):
                face = int((centers - point).square().sum(-1).argmin())
                torch.testing.assert_close(
                    batch["surface_query_features"][0, index],
                    surface.cell_normals[face],
                )
            snapshots.append(batch["volume_query_features"].clone())
        torch.testing.assert_close(snapshots[0], snapshots[2], atol=0, rtol=0)
    finally:
        reader.close()
        baseline.close()
    no_targets = reader_type(path, cfg, training=False, targets=False)
    try:
        if not native:
            assert not any(key.endswith("_fields") for key in no_targets._user_fields)
        row, _ = no_targets[0]
        assert not any("targets" in key for key in row.keys())
        batch = {key: value.unsqueeze(0) for key, value in row.items()}
        augment_query_features(batch, cfg)
        assert torch.isfinite(batch["volume_query_features"]).all()
    finally:
        no_targets.close()


def test_triangle_queries_with_float64_default(triangle_geometry, queries):
    """Warp output buffers keep their required dtype under a float64 default."""
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        geometry = CFDQueryFeatures(*triangle_geometry)
        result = geometry.volume(queries)
        assert result.dtype == torch.float32
        assert torch.isfinite(result).all()
    finally:
        torch.set_default_dtype(previous)


def test_triangle_queries_reject_integer_coordinates(triangle_geometry, queries):
    """Do not silently quantize malformed mesh or query coordinates."""
    vertices, faces = triangle_geometry
    with pytest.raises(ValueError, match="floating-point"):
        CFDQueryFeatures(vertices.long(), faces)
    geometry = CFDQueryFeatures(vertices, faces)
    with pytest.raises(ValueError, match="floating-point"):
        geometry.volume(queries.long())
