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

"""Regression checks for independent SMART fields and geometric query features."""

import pytest
import torch

from physicsnemo import Module
from physicsnemo.experimental.models.smart import SMART


def field_model(field, **extra):
    config = dict(
        surface_channels=4 if field == "surface" else 0,
        volume_channels=5 if field == "volume" else 0,
        surface_query_feature_channels=3 if field == "surface" else 0,
        volume_query_feature_channels=4 if field == "volume" else 0,
        hidden_dim=24,
        num_heads=4,
        num_layers=2,
        num_latents=8,
        geometry_subsample=16,
        qk_normalization=True,
        qk_temperature_max=8.0,
    )
    config.update(extra)
    return SMART(**config)


@pytest.mark.parametrize("field", ["surface", "volume"])
def test_feature_alignment_chunks_gradients_and_checkpoint(field, tmp_path):
    torch.manual_seed(57)
    model = field_model(field).eval()
    geometry = torch.randn(1, 30, 3)
    queries = torch.randn(1, 17, 3)
    features = torch.randn(1, 17, 3 if field == "surface" else 4)
    context = model.encode_geometry(geometry)
    expected = model.decode_queries(
        queries, context, field=field, query_features=features
    )
    chunks = torch.cat(
        [
            model.decode_queries(q, context, field=field, query_features=f)
            for q, f in zip(queries.split(6, 1), features.split(6, 1))
        ],
        dim=1,
    )
    torch.testing.assert_close(chunks, expected, atol=2e-6, rtol=2e-5)
    permutation = torch.randperm(17)
    permuted = model.decode_queries(
        queries[:, permutation],
        context,
        field=field,
        query_features=features[:, permutation],
    )
    torch.testing.assert_close(permuted, expected[:, permutation], atol=2e-6, rtol=2e-5)
    changed = model.decode_queries(
        queries,
        context,
        field=field,
        query_features=-features,
    )
    torch.testing.assert_close(expected, changed, atol=0, rtol=0)
    expected.square().mean().backward()
    assert all(
        p.grad is not None and p.grad.isfinite().all() for p in model.parameters()
    )
    projection = getattr(model, f"{field}_query_projection")
    assert projection.weight.grad.abs().max() > 0
    with torch.no_grad():
        projection.weight.add_(-0.1 * projection.weight.grad)
        learned = model.decode_queries(
            queries, context, field=field, query_features=features
        )
        changed = model.decode_queries(
            queries, context, field=field, query_features=-features
        )
        assert (learned - changed).abs().max() > 1e-6
        projection.weight.zero_()
    assert all(
        "volume" not in name if field == "surface" else "surface" not in name
        for name, _ in model.named_parameters()
    )
    kwargs = {f"{field}_queries": queries, f"{field}_query_features": features}
    result = model(geometry, **kwargs)
    index = 0 if field == "surface" else 1
    torch.testing.assert_close(result[index], expected)
    assert result[1 - index] is None
    path = tmp_path / f"{field}.mdlus"
    model.save(path)
    restored = Module.from_checkpoint(path).eval()
    torch.testing.assert_close(
        restored(geometry, **kwargs)[index], expected, atol=0, rtol=0
    )


def test_legacy_parameter_count_and_no_optional_state():
    model = SMART(
        surface_channels=4,
        volume_channels=5,
        parameter_channels=2,
        num_layers=16,
        qk_normalization=True,
    )
    assert sum(p.numel() for p in model.parameters()) == 25_520_425
    assert not any(
        "temperature" in key or "query_projection" in key for key in model.state_dict()
    )


@pytest.mark.parametrize("field", ["surface", "volume"])
def test_required_feature_validation(field):
    model = field_model(field)
    queries = torch.randn(1, 7, 3)
    contexts = model.encode_geometry(torch.randn(1, 20, 3))
    for features in [None, torch.randn(1, 6, 3), torch.ones(1, 7, 3, dtype=torch.long)]:
        with pytest.raises(ValueError, match="query_features"):
            model.decode_queries(
                queries, contexts, field=field, query_features=features
            )
    with pytest.raises(ValueError, match="requires"):
        model(
            torch.randn(1, 20, 3),
            **{
                f"{'volume' if field == 'surface' else 'surface'}_queries": queries,
                f"{field}_query_features": torch.randn(1, 7, 3),
            },
        )
    with pytest.raises(ValueError, match="disabled"):
        SMART(surface_channels=0, volume_channels=3, surface_query_feature_channels=3)
    plain = SMART(hidden_dim=24, num_heads=4, num_layers=2)
    with pytest.raises(ValueError, match="disabled"):
        plain.decode_queries(queries, contexts, query_features=torch.randn(1, 7, 3))


def test_temperature_initial_equivalence_and_bounded_sharpness():
    torch.manual_seed(22)
    plain = field_model("surface", qk_temperature_max=None)
    learned = field_model("surface")
    missing, unexpected = learned.load_state_dict(plain.state_dict(), strict=False)
    assert missing and all(name.endswith("temperature_logit") for name in missing)
    assert not unexpected
    geometry, queries, features = (
        torch.randn(1, 30, 3),
        torch.randn(1, 7, 3),
        torch.randn(1, 7, 3),
    )
    kwargs = dict(surface_queries=queries, surface_query_features=features)
    plain.eval()
    learned.eval()
    torch.testing.assert_close(
        plain(geometry, **kwargs)[0],
        learned(geometry, **kwargs)[0],
        atol=2e-6,
        rtol=2e-5,
    )
    logits = [
        p
        for name, p in learned.named_parameters()
        if name.endswith("temperature_logit")
    ]
    assert len(logits) == 2 * learned.num_layers
    initial = learned(geometry, **kwargs)[0]
    with torch.no_grad():
        for parameter in logits:
            parameter.fill_(100)
    sharp = learned(geometry, **kwargs)[0]
    assert sharp.isfinite().all()
    assert (sharp - initial).abs().max() > 1e-5
    for module in learned.modules():
        if getattr(module, "temperature_logit", None) is not None:
            assert torch.all(
                module.qk_temperature_max * module.temperature_logit.sigmoid() <= 8
            )
    for maximum in [1, float("inf"), float("nan")]:
        with pytest.raises(ValueError, match="qk_temperature_max"):
            field_model("surface", qk_temperature_max=maximum)
    with pytest.raises(ValueError, match="qk_temperature_max"):
        field_model("surface", qk_normalization=False)


@pytest.mark.parametrize("field", ["surface", "volume"])
def test_shared_initialization_is_identical_between_feature_arms(field):
    torch.manual_seed(189)
    feature_model = field_model(field, qk_temperature_max=4.0)
    torch.manual_seed(189)
    plain_model = field_model(
        field, qk_temperature_max=4.0, **{f"{field}_query_feature_channels": 0}
    )
    plain_state = plain_model.state_dict()
    feature_state = feature_model.state_dict()
    for name, tensor in plain_state.items():
        assert torch.equal(tensor, feature_state[name]), name
    extras = set(feature_state) - set(plain_state)
    assert extras == {f"{field}_query_projection.weight"}
    assert feature_state[extras.pop()].count_nonzero() == 0


@pytest.mark.parametrize("field", ["surface", "volume"])
def test_feature_checkpointed_chunk_gradients_match_direct(field):
    from torch.utils.checkpoint import checkpoint

    torch.manual_seed(29)
    model = field_model(field, qk_temperature_max=4.0).eval()
    reference = field_model(field, qk_temperature_max=4.0).eval()
    reference.load_state_dict(model.state_dict())
    geometry, queries = torch.randn(1, 30, 3), torch.randn(1, 17, 3)
    features = torch.randn(1, 17, 3 if field == "surface" else 4)
    context = model.encode_geometry(geometry)
    direct = model.decode_queries(
        queries, context, field=field, query_features=features
    )
    direct.square().sum().backward()
    context = reference.encode_geometry(geometry)

    def decode(query, feature, *latents):
        return reference.decode_queries(
            query, latents, field=field, query_features=feature
        )

    chunked = torch.cat(
        [
            checkpoint(decode, query, feature, *context, use_reentrant=False)
            for query, feature in zip(queries.split(6, 1), features.split(6, 1))
        ],
        dim=1,
    )
    chunked.square().sum().backward()
    torch.testing.assert_close(chunked, direct, atol=2e-6, rtol=2e-5)
    for (name, p), (_, q) in zip(
        model.named_parameters(), reference.named_parameters()
    ):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-5, rtol=5e-5, msg=name)
