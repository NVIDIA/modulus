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

"""DrivAerML field, sampling, and normalization acceptance tests."""

from pathlib import Path

import numpy as np
import pytest
import torch
from data import build_dataset, model_inputs
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from physicsnemo.mesh import DomainMesh, Mesh


@pytest.mark.parametrize("prepared", [False, True])
def test_cfd_only_reader_never_opens_cad(native_case, prepared, monkeypatch):
    """Verify native and cached inputs obtain geometry exclusively from CFD."""
    from prepare_drivaerml import prepare_case

    root, cfg, _ = native_case
    cfg.data.volume_read_multiplier = 2
    domain = DomainMesh.load(root / "run_1/domain_run_1.pdmsh")
    surface = domain.boundaries["vehicle"]
    points = surface.points[surface.cells].mean(1)
    center = (points.amin(0) + points.amax(0)) * 0.5
    if prepared:
        output = root / "cfd-only"
        prepare_case(
            domain,
            output / "run_1.zarr",
            source="CFD-only fixture",
            density=2.5,
            speed=4,
            chunk_points=8,
        )
        assert not list(output.rglob("stl_*"))
        root = output
        cfg.data.format = "drivaerml_zarr"
    cfg.data.geometry_source = "cfd_surface"
    cfg.data.geometry_points = None
    OmegaConf.update(cfg, "data.coordinate_frame", "cfd_surface_bbox", force_add=True)

    def forbidden(*args, **kwargs):
        """Fail if a CFD-only code path attempts to read CAD."""
        raise AssertionError("CFD-only reader tried to open CAD")

    monkeypatch.setattr("physicsnemo.datapipes.MeshReader", forbidden)
    for training, targets in [(True, True), (False, True), (False, False)]:
        data = build_dataset(root, cfg, training=training, targets=targets)
        try:
            value, _ = data[0]
            geometry = value["geometry"]
            torch.testing.assert_close(
                geometry[geometry[:, 0].argsort()],
                (points - center)[points[:, 0].argsort()],
                rtol=0,
                atol=0,
            )
            assert len(geometry) == surface.n_cells
            again, _ = data[0]
            assert all(torch.equal(value[k], again[k]) for k in value.keys())
            # Query-target alignment survives translation and full-pool loading.
            if targets:
                recovered = value["surface_queries"] * data.reader.scale + center
                assert torch.isfinite(recovered).all()
                assert value["surface_targets"].shape[0] == recovered.shape[0]
        finally:
            data.close()


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize(
    "training,targets", [(True, True), (False, True), (False, False)]
)
def test_full_cfd_geometry_is_independent_of_query_samples(
    native_case, prepared, training, targets
):
    """Use every CFD surface center without changing field samples or requiring labels."""
    root, cfg, _ = native_case
    cfg.data.geometry_points = None
    cfg.data.volume_read_multiplier = 2
    OmegaConf.update(cfg, "data.cache_validation", True, force_add=True)
    domain = DomainMesh.load(root / "run_1/domain_run_1.pdmsh")
    surface = domain.boundaries["vehicle"]
    centers = surface.points[surface.cells].mean(1)
    expected = centers - (centers.amin(0) + centers.amax(0)) * 0.5
    if prepared:
        from prepare_drivaerml import prepare_case

        output = root / "prepared"
        prepare_case(
            domain,
            output / "run_1.zarr",
            source="fixture",
            density=2.5,
            speed=4,
            chunk_points=8,
        )
        root = output
        cfg.data.format = "drivaerml_zarr"
    cfg.data.geometry_points = 10
    capped = build_dataset(root, cfg, training=training, targets=targets)
    cfd_cfg = OmegaConf.create(OmegaConf.to_container(cfg))
    cfd_cfg.data.geometry_points = None
    cfd = build_dataset(root, cfd_cfg, training=training, targets=targets)
    try:
        for epoch in [0, 1]:
            capped.reader.set_epoch(epoch)
            cfd.reader.set_epoch(epoch)
            a, _ = capped[0]
            b, _ = cfd[0]
            # Preparation shuffles complete, aligned CFD coordinate/field rows.
            actual = b["geometry"]
            assert actual.shape == (20, 3)
            torch.testing.assert_close(
                actual[actual[:, 0].argsort()],
                expected[expected[:, 0].argsort()],
                rtol=0,
                atol=0,
            )
            assert len(actual) > len(b["surface_queries"])
            assert len(a["geometry"]) == 10
            for key in a.keys():
                if key != "geometry":
                    assert torch.equal(a[key], b[key]), key
            again, _ = cfd[0]
            assert all(torch.equal(b[key], again[key]) for key in b.keys())
        # A volume-only invocation still sees the complete CFD boundary, even
        # though it does not request surface query coordinates or target fields.
        cfd_cfg.model.surface_channels = 0
        volume_only = build_dataset(root, cfd_cfg, training=training, targets=targets)
        try:
            c, _ = volume_only[0]
            assert torch.equal(c["geometry"], b["geometry"])
            assert "surface_queries" not in c and "surface_targets" not in c
        finally:
            volume_only.close()
    finally:
        capped.close()
        cfd.close()


def test_cfd_geometry_requires_separate_rng_streams(native_case):
    """Reject coupled RNG streams that would perturb matched field sampling."""
    root, cfg, _ = native_case
    cfg.data.geometry_source = "cfd_surface"
    cfg.data.geometry_rng_streams = False
    with pytest.raises(ValueError, match="CFD geometry requires independent"):
        build_dataset(root, cfg, training=True)


@pytest.mark.parametrize("prepared", [False, True])
def test_full_cfd_pool_preserves_fields_and_sampling_streams(native_case, prepared):
    """Removing the outer pool changes only geometry, with exact coordinate order."""
    root, cfg, center = native_case
    cfg.data.geometry_rng_streams = True
    cfg.data.volume_read_multiplier = 2
    OmegaConf.update(cfg, "data.cache_validation", True, force_add=True)
    if prepared:
        from prepare_drivaerml import prepare_case

        output = root / "prepared"
        prepare_case(
            DomainMesh.load(root / "run_1/domain_run_1.pdmsh"),
            output / "run_1.zarr",
            source="fixture",
            density=2.5,
            speed=4,
            chunk_points=8,
        )
        cfg.data.format = "drivaerml_zarr"
        root = output
    capped = build_dataset(root, cfg, training=True)
    full_cfg = OmegaConf.create(OmegaConf.to_container(cfg))
    full_cfg.data.geometry_points = None
    full = build_dataset(root, full_cfg, training=True)
    surface = DomainMesh.load(native_case[0] / "run_1/domain_run_1.pdmsh").boundaries[
        "vehicle"
    ]
    expected = surface.points[surface.cells].mean(1) - center
    try:
        epoch_seeds = []
        for epoch in [0, 1, 17]:
            capped.reader.set_epoch(epoch)
            full.reader.set_epoch(epoch)
            a, _ = capped[0]
            b, _ = full[0]
            assert a["geometry"].shape == (10, 3)
            torch.testing.assert_close(
                b["geometry"][b["geometry"][:, 0].argsort()],
                expected[expected[:, 0].argsort()],
                rtol=0,
                atol=0,
            )
            for key in a.keys():
                if key != "geometry":
                    assert torch.equal(a[key], b[key]), key
            repeat, _ = full[0]
            assert all(torch.equal(b[k], repeat[k]) for k in b.keys())
            global_rng = torch.random.get_rng_state().clone()
            inputs = model_inputs(b.unsqueeze(0))
            draw = torch.rand(8, generator=inputs["geometry_generator"])
            assert torch.equal(global_rng, torch.random.get_rng_state())
            assert torch.equal(
                draw,
                torch.rand(
                    8, generator=model_inputs(b.unsqueeze(0))["geometry_generator"]
                ),
            )
            epoch_seeds.append(b["geometry_seed"].item())
        assert len(set(epoch_seeds)) == 3
        validation = build_dataset(root, full_cfg, training=False)
        try:
            a, _ = validation[0]
            torch.testing.assert_close(
                a["geometry"][a["geometry"][:, 0].argsort()],
                expected[expected[:, 0].argsort()],
                rtol=0,
                atol=0,
            )
            assert "geometry_seed" not in a
        finally:
            validation.close()
    finally:
        capped.close()
        full.close()


@pytest.fixture
def native_case(tmp_path):
    """Create original native-mesh test fields with a checkable coordinate relation."""
    torch.set_num_threads(2)
    case = tmp_path / "run_1"
    case.mkdir()
    points = torch.arange(180, dtype=torch.float32).reshape(60, 3) / 20
    cells = torch.arange(60).reshape(20, 3)
    surface = Mesh(
        points=points,
        cells=cells,
        cell_data={
            "pMeanTrim": points[cells].mean(1)[:, 0] * 40,
            "wallShearStressMeanTrim": torch.full((20, 3), 80.0),
        },
    )
    volume = Mesh(
        points=points + 1,
        point_data={
            "UMeanTrim": (points + 1) * 4,
            "pMeanTrim": (points[:, 0] + 1) * 40,
            "nutMeanTrim": torch.full((60,), 120.0),
        },
    )
    domain = DomainMesh(
        interior=volume,
        boundaries={"vehicle": surface},
        global_data={
            "rho_inf": torch.tensor(2.5),
            "U_inf": torch.tensor([4.0, 0, 0]),
        },
    )
    domain.save(case / "domain_run_1.pdmsh")
    geometry = Mesh(points=points + 10, cells=cells)
    geometry.save(case / "drivaer_1.stl.pmsh")
    geometry.save(case / "drivaer_1_single_solid.stl.pmsh")
    for field, channels in (("surface", 4), ("volume", 5)):
        np.savez(
            tmp_path / f"{field}_fields_normalization.npz",
            mean=np.ones(channels),
            std=np.full(channels, 2.0),
        )
    with initialize_config_dir(
        config_dir=str(Path(__file__).resolve().parents[1] / "conf"), version_base="1.3"
    ):
        cfg = compose(config_name="drivaerml")
    cfg.model.surface_channels = 4
    cfg.model.volume_channels = 5
    cfg.model.surface_query_feature_channels = 0
    cfg.model.volume_query_feature_channels = 0
    cfg.data.format = "drivaerml_mesh"
    cfg.data.reference_density = None
    cfg.data.reference_speed = None
    cfg.data.normalization_dir = str(tmp_path)
    cfg.data.field_statistics = None
    cfg.data.workers = 0
    cfg.data.to_device = False
    cfg.data.pin_memory = False
    cfg.data.geometry_points = 10
    cfg.data.surface_points = 12
    cfg.data.volume_points = 15
    cfg.data.reference_scale = [1, 1, 1]
    cfg.validation.points = 10
    centers = surface.points[surface.cells].mean(1)
    return tmp_path, cfg, (centers.amin(0) + centers.amax(0)) * 0.5


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("normalize_parameters", [False, True])
def test_native_field_mapping_and_repeatability(
    native_case, prepared, normalize_parameters
):
    """Verify all nine field units, geometry separation, and deterministic samples."""
    root, cfg, center = native_case
    if normalize_parameters:
        cfg.data.parameter_mean = [1.5, 2.0]
        cfg.data.parameter_std = [2.0, 4.0]
    OmegaConf.update(cfg, "data.cache_validation", True, force_add=True)
    if prepared:
        from prepare_drivaerml import prepare_case

        domain = DomainMesh.load(root / "run_1/domain_run_1.pdmsh")
        prepared_root = root / "prepared"
        report = prepare_case(
            domain,
            prepared_root / "run_1.zarr",
            source="test fixture",
            density=2.5,
            speed=4,
            chunk_points=8,
        )
        assert report["point_counts"] == {"surface": 20, "volume": 60}
        assert report["geometry_points"] == 20
        repeated_report = prepare_case(
            domain,
            prepared_root / "run_1.zarr",
            source="test fixture",
            density=2.5,
            speed=4,
            chunk_points=8,
        )
        assert repeated_report == report
        with pytest.raises(ValueError, match="differs"):
            prepare_case(domain, prepared_root / "run_1.zarr", source="different")
        cfg.data.format = "drivaerml_zarr"
        cfg.data.volume_read_multiplier = 2
        root = prepared_root
    dataset = build_dataset(root, cfg, training=False)
    sample, _ = dataset[0]
    repeated, _ = dataset[0]
    assert dataset.reader._sample_cache
    for key in sample.keys():
        torch.testing.assert_close(sample[key], repeated[key], atol=0, rtol=0)
    assert sample["geometry"].shape == (10, 3)
    assert sample["surface_targets"].shape == (10, 4)
    assert sample["volume_targets"].shape == (10, 5)
    expected_parameters = [0.5, 0.5] if normalize_parameters else [2.5, 4.0]
    torch.testing.assert_close(sample["parameters"], torch.tensor(expected_parameters))
    for field in ("surface", "volume"):
        original = sample[f"{field}_targets"] * 2 + 1
        coordinates = sample[f"{field}_queries"] + center
        pressure_index = 0 if field == "surface" else 3
        torch.testing.assert_close(original[:, pressure_index], coordinates[:, 0])
        if field == "surface":
            torch.testing.assert_close(original[:, 1:], torch.full((10, 3), 2.0))
        else:
            torch.testing.assert_close(original[:, :3], coordinates)
            torch.testing.assert_close(
                original[:, 4], torch.full((10,), 120.0 / (4.0 * 8.85))
            )
    dataset.set_epoch(1)
    changed, _ = dataset[0]
    assert not torch.equal(sample["volume_queries"], changed["volume_queries"])


def test_native_target_free_and_bad_counts(native_case):
    """Inference requires no target tensors and oversized sampling fails explicitly."""
    root, cfg, _ = native_case
    cfg.validation.points = None
    sample, _ = build_dataset(root, cfg, training=False, targets=False)[0]
    assert sample["volume_queries"].shape == (60, 3)
    assert not any(key.endswith("targets") for key in sample.keys())
    cfg.data.surface_points = 100
    with pytest.raises(ValueError, match="fewer"):
        build_dataset(root, cfg, training=True)[0]


def test_repacked_resume_validates_source_and_scaling(native_case):
    """Storage migration must retain the original case identities and field units."""
    import json

    from prepare_drivaerml import prepare_case
    from train import validate_resume_data

    root, cfg, _ = native_case
    native = root / "run_1/domain_run_1.pdmsh"
    packed = root / "prepared/run_1.zarr"
    prepare_case(
        DomainMesh.load(native),
        packed,
        source=str(native.resolve()),
        density=2.5,
        speed=4,
    )
    previous = dict(
        format="drivaerml_mesh",
        train_path=str(root),
        val_path=str(root),
        reference_density=2.5,
        reference_speed=4,
        workers=1,
    )
    current = dict(
        previous,
        format="drivaerml_zarr",
        train_path=str(root / "prepared"),
        val_path=str(root / "prepared"),
        workers=4,
        cache_validation=True,
    )
    validate_resume_data(previous, current, repacked=True)
    with pytest.raises(ValueError, match="differs in data"):
        validate_resume_data(previous, current, repacked=False)
    with pytest.raises(ValueError, match="scaling"):
        validate_resume_data(
            previous, dict(current, reference_density=1.0), repacked=True
        )
    metadata = json.loads((packed / ".zattrs").read_text())
    metadata["preparation"]["source"] = "/different/case"
    (packed / ".zattrs").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="source identities"):
        validate_resume_data(previous, current, repacked=True)


@pytest.mark.parametrize("std", [[1.0, 0.0], []])
def test_invalid_parameter_normalization(native_case, std):
    """Reject partial or singular condition scaling instead of ignoring it."""
    root, cfg, _ = native_case
    cfg.data.parameter_mean = [1.0, 2.0]
    cfg.data.parameter_std = std
    with pytest.raises(ValueError, match="Parameter mean/std"):
        build_dataset(root, cfg, training=False)


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("cached", [False, True])
def test_reader_context_releases_sample_cache(native_case, prepared, cached):
    """Both storage formats release validation samples and allow repeated close."""
    from drivaerml import DrivAerMLMeshReader, DrivAerMLReader

    root, cfg, _ = native_case
    OmegaConf.update(cfg, "data.cache_validation", cached, force_add=True)
    reader_type = DrivAerMLMeshReader
    if prepared:
        from prepare_drivaerml import prepare_case

        prepare_case(
            DomainMesh.load(root / "run_1/domain_run_1.pdmsh"),
            root / "prepared/run_1.zarr",
            source="test fixture",
            density=2.5,
            speed=4,
            chunk_points=8,
        )
        root = root / "prepared"
        reader_type = DrivAerMLReader
        cfg.data.volume_read_multiplier = 2
    reader = reader_type(root, cfg, training=False)
    with reader:
        reader._load_sample(0)
        if cached:
            assert len(reader._sample_cache) == 1
    assert not reader._sample_cache
    reader.close()
