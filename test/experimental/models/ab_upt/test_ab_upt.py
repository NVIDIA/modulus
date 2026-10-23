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

"""Equation, field-invariance, optimization, and serialization checks for AB-UPT."""

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from physicsnemo import Module
from physicsnemo.experimental.models.ab_upt import ABUPT
from physicsnemo.experimental.models.ab_upt.model import _AnchorBlock
from test.common import (
    validate_amp,
    validate_checkpoint,
    validate_combo_optims,
    validate_forward_accuracy,
)

DATA = Path(__file__).parent / "data"
DEFAULTS = dict(
    surface_channels=4,
    volume_channels=7,
    hidden_dim=192,
    num_heads=3,
    num_interaction_blocks=2,
    num_shared_layers=1,
    num_decoder_layers=6,
    mlp_ratio=4,
    pooling_radius=0.25,
    pooling_chunk_size=4096,
    parameter_channels=0,
    radius_search_backend=None,
)
SMALL = dict(
    hidden_dim=24,
    num_heads=3,
    num_interaction_blocks=1,
    num_shared_layers=1,
    num_decoder_layers=1,
    pooling_radius=0.8,
    pooling_chunk_size=17,
    radius_search_backend="torch",
)
CONFIGS = [
    SMALL,
    SMALL
    | dict(
        hidden_dim=48,
        num_heads=4,
        num_interaction_blocks=2,
        num_shared_layers=0,
        num_decoder_layers=2,
        parameter_channels=2,
        surface_channels=2,
        volume_channels=4,
    ),
]


def inputs(device="cpu", dtype=torch.float32, parameters=0):
    g = torch.linspace(-0.7, 0.9, 2 * 19 * 3, device=device, dtype=dtype).reshape(
        2, 19, 3
    )
    s = g[:, ::4].contiguous()
    v = torch.linspace(-0.9, 0.8, 2 * 7 * 3, device=device, dtype=dtype).reshape(
        2, 7, 3
    )
    return g, s, s, v


@pytest.fixture(autouse=True)
def precise_small_tests():
    precision = torch.get_float32_matmul_precision()
    threads = torch.get_num_threads()
    torch.set_float32_matmul_precision("highest")
    torch.set_num_threads(1)
    yield
    torch.set_float32_matmul_precision(precision)
    torch.set_num_threads(threads)


@pytest.mark.parametrize("config", [{}, CONFIGS[1]])
def test_constructor(config):
    model = ABUPT(**config)
    for name, value in (DEFAULTS | config).items():
        assert getattr(model, name) == value
    assert isinstance(model, Module)
    assert (
        len(model.shared_blocks)
        == 2 * model.num_interaction_blocks + model.num_shared_layers
    )
    assert (
        len(model.surface_decoder)
        == len(model.volume_decoder)
        == model.num_decoder_layers
    )
    assert all(a is not b for a, b in zip(model.surface_decoder, model.volume_decoder))
    assert model.meta.amp_cpu and model.meta.amp_gpu and model.meta.auto_grad
    assert not model.meta.jit and not model.meta.cuda_graphs


@pytest.mark.parametrize("index", [0, 1])
def test_frozen_reference_and_checkpoint(index, device):
    reference = torch.load(DATA / f"ab_upt_{index}_methods.pth", weights_only=True)
    model = (
        Module.from_checkpoint(str(DATA / f"ab_upt_{index}.mdlus")).to(device).eval()
    )
    for name, value in (DEFAULTS | CONFIGS[index]).items():
        assert getattr(model, name) == value
    args = tuple(t.to(device) if t is not None else None for t in reference["inputs"])
    assert validate_forward_accuracy(
        model,
        args,
        rtol=2e-5,
        atol=2e-6,
        file_name=f"experimental/models/ab_upt/data/ab_upt_{index}_output.pth",
    )
    geometry = model.encode_geometry(*args[:2])
    context = model.encode(*args[:4], parameters=args[6])
    torch.testing.assert_close(
        geometry, reference["geometry"].to(device), rtol=2e-5, atol=2e-6
    )
    for field in ("surface", "volume"):
        actual = model.decode_queries(
            args[4 if field == "surface" else 5], context, field
        )
        torch.testing.assert_close(
            actual, reference[field].to(device), rtol=2e-5, atol=2e-6
        )
        for (k, v), (rk, rv) in zip(
            getattr(context, field), reference[field + "_context"]
        ):
            torch.testing.assert_close(k, rk.to(device), rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(v, rv.to(device), rtol=2e-5, atol=2e-6)
    vort = model.decode_vorticity(args[5], context, step=0.01)
    torch.testing.assert_close(
        vort, reference["vorticity"].to(device), rtol=1e-3, atol=2e-4
    )


def test_checkpoint_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert validate_checkpoint(ABUPT(**SMALL), ABUPT(**SMALL), inputs())


@pytest.mark.parametrize("channels", [(4, 7), (4, 0), (0, 7)])
def test_gradients_and_disabled_heads(channels, device):
    model = ABUPT(
        **(
            SMALL
            | dict(
                surface_channels=channels[0],
                volume_channels=channels[1],
                parameter_channels=2,
            )
        )
    ).to(device)
    args = inputs(device)
    conditioning = torch.randn(2, 2, device=device, requires_grad=True)
    result = model(*args, parameters=conditioning)
    sum(x.square().mean() for x in result if x is not None).backward()
    assert conditioning.grad is not None and conditioning.grad.abs().sum() > 0
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
    assert (result[0] is None) == (channels[0] == 0)
    assert (result[1] is None) == (channels[1] == 0)


@pytest.mark.parametrize("field", ["surface", "volume"])
def test_query_independence_chunks_permutations_and_gradients(field, device):
    torch.manual_seed(9)
    model = ABUPT(**SMALL).to(device).eval()
    args = inputs(device)
    context = model.encode(*args)
    q = torch.randn(2, 13, 3, device=device, requires_grad=True)
    full = model.decode_queries(q, context, field)
    chunked = torch.cat(
        [model.decode_queries(c, context, field) for c in q.split(4, dim=1)], dim=1
    )
    torch.testing.assert_close(full, chunked, rtol=2e-5, atol=2e-6)
    order = torch.randperm(q.shape[1], device=device)
    torch.testing.assert_close(
        full[:, order],
        model.decode_queries(q[:, order], context, field),
        rtol=2e-5,
        atol=2e-6,
    )
    short = model.decode_queries(q[:, :2], context, field)
    torch.testing.assert_close(full[:, :2], short, rtol=2e-5, atol=2e-6)
    grad = torch.autograd.grad(full[:, 0].sum(), q, retain_graph=True)[0]
    assert grad[:, 0].abs().sum() > 0
    assert torch.count_nonzero(grad[:, 1:]) == 0
    full_grad = torch.autograd.grad(
        full.square().sum(), model.surface_embedding.layers[0].weight, retain_graph=True
    )[0]
    chunk_grad = torch.autograd.grad(
        chunked.square().sum(), model.surface_embedding.layers[0].weight
    )[0]
    torch.testing.assert_close(full_grad, chunk_grad, rtol=1e-4, atol=2e-5)
    anchor_index = 2 if field == "surface" else 3
    torch.testing.assert_close(
        model.decode_queries(args[anchor_index], context, field),
        getattr(context, field + "_predictions"),
        rtol=2e-5,
        atol=2e-6,
    )
    assert model.decode_queries(q[:, :0], context, field).shape[1] == 0


def test_batch_separation_and_anchor_permutation():
    torch.manual_seed(2)
    model = ABUPT(**SMALL).eval()
    args = inputs()
    q = torch.rand(2, 9, 3)
    both = model(*args, q, q)
    for b in range(2):
        single = model(*(x[b : b + 1] for x in args), q[b : b + 1], q[b : b + 1])
        for a, e in zip(single, both):
            torch.testing.assert_close(a, e[b : b + 1], rtol=2e-5, atol=2e-6)
    permuted = model(
        args[0][:, torch.randperm(19)],
        args[1].flip(1),
        args[2].flip(1),
        args[3].flip(1),
        q,
        q,
    )
    for a, e in zip(permuted, both):
        torch.testing.assert_close(a, e, rtol=2e-5, atol=2e-6)


def test_cross_branch_context_couples_predictions():
    model = ABUPT(**SMALL)
    g, sp, s, v = inputs()
    v = v.clone().requires_grad_()
    q = torch.rand(2, 3, 3)
    surface, _ = model(g, sp, s, v, q, q)
    grad = torch.autograd.grad(surface.sum(), v)[0]
    assert torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_anchor_block_against_explicit_softmax_equation():
    torch.manual_seed(11)
    block = _AnchorBlock(24, 3, 2).double()
    anchors = torch.randn(2, 5, 24, dtype=torch.float64)
    queries = torch.randn(2, 8, 24, dtype=torch.float64)
    apos = torch.randn(2, 5, 3, dtype=torch.float64)
    qpos = torch.randn(2, 8, 3, dtype=torch.float64)
    kv = block._key_value(anchors, apos)
    result = block(queries, qpos, kv)
    q = block.query(block.norm(queries)).reshape(2, 8, 3, 8).transpose(1, 2)
    q = block._rotate(q, qpos)
    weights = (q @ kv[0].transpose(-2, -1) / (8**0.5)).softmax(-1)
    expected = queries + block.output((weights @ kv[1]).transpose(1, 2).flatten(-2))
    expected = expected + block.mlp(block.mlp_norm(expected))
    torch.testing.assert_close(result, expected, rtol=1e-12, atol=1e-12)
    # A dense attention mask excludes every query from the key/value set.
    all_values = torch.cat((anchors, queries), dim=1)
    all_pos = torch.cat((apos, qpos), dim=1)
    all_kv = block._key_value(all_values, all_pos)
    all_q = block._rotate(
        block.query(block.norm(all_values)).reshape(2, 13, 3, 8).transpose(1, 2),
        all_pos,
    )
    logits = all_q @ all_kv[0].transpose(-2, -1) / (8**0.5)
    logits[..., 5:] = float("-inf")
    masked = all_values + block.output(
        (logits.softmax(-1) @ all_kv[1]).transpose(1, 2).flatten(-2)
    )
    masked = masked + block.mlp(block.mlp_norm(masked))
    torch.testing.assert_close(result, masked[:, 5:], rtol=1e-12, atol=1e-12)


def test_pooling_chunk_independence_empty_neighbors_and_coordinate_gradients():
    model = ABUPT(**SMALL).double()
    args = inputs(dtype=torch.float64)
    g = args[0].clone().requires_grad_()
    result = model.encode_geometry(g, args[1])
    model.pooling_chunk_size = 4096
    torch.testing.assert_close(
        result, model.encode_geometry(g, args[1]), rtol=1e-12, atol=1e-12
    )
    result.square().mean().backward()
    assert torch.isfinite(g.grad).all() and g.grad.abs().sum() > 0
    distant = model.encode_geometry(g, args[1] + 100)
    assert torch.isfinite(distant).all()


def test_vorticity_scaling_against_analytic_polynomial(monkeypatch):
    model = ABUPT(**SMALL).double()
    q = torch.randn(2, 6, 3, dtype=torch.float64)

    def polynomial(queries, context, field):
        x, y, z = queries.unbind(-1)
        return torch.stack((x * 0, y * y, z * z, x * x, x * 0, x * 0, x * 0), dim=-1)

    monkeypatch.setattr(model, "decode_queries", polynomial)
    result = model.decode_vorticity(
        q, None, step=0.01, coordinate_scale=(2, 3, 4), velocity_scale=(5, 6, 7)
    )
    x, y, z = q.unbind(-1)
    expected = torch.stack((-48 * z, -28 * x, -30 * y), dim=-1)
    torch.testing.assert_close(result, expected, rtol=1e-12, atol=1e-11)


def test_model_curl_agrees_with_autograd_and_has_parameter_gradients():
    model = ABUPT(**SMALL).double()
    args = inputs(dtype=torch.float64)
    ctx = model.encode(*args)
    q = torch.randn(2, 4, 3, dtype=torch.float64, requires_grad=True)
    pred = model.decode_queries(q, ctx, "volume")[..., 1:4]
    jac = [
        torch.autograd.grad(pred[..., i].sum(), q, retain_graph=True)[0]
        for i in range(3)
    ]
    exact = torch.stack(
        (
            jac[2][..., 1] - jac[1][..., 2],
            jac[0][..., 2] - jac[2][..., 0],
            jac[1][..., 0] - jac[0][..., 1],
        ),
        dim=-1,
    )
    approx = model.decode_vorticity(q, ctx, step=1e-4)
    torch.testing.assert_close(approx, exact, rtol=1e-5, atol=1e-7)
    approx.square().mean().backward()
    assert model.volume_head.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("helper", [validate_amp, validate_combo_optims])
def test_supported_optimizations(helper, device):
    assert helper(ABUPT(**SMALL).to(device), inputs(device), iterations=2)


@pytest.mark.parametrize(
    "config",
    [
        dict(hidden_dim=25),
        dict(num_heads=0),
        dict(num_interaction_blocks=-1),
        dict(surface_channels=0, volume_channels=0),
        dict(pooling_radius=0),
        dict(pooling_radius=float("nan")),
        dict(pooling_chunk_size=0),
        dict(radius_search_backend="missing"),
        dict(parameter_channels=-1),
        dict(surface_channels=0, num_interaction_blocks=0),
    ],
)
def test_bad_constructor(config):
    with pytest.raises(ValueError):
        ABUPT(**(SMALL | config))


@pytest.mark.parametrize(
    "index,bad",
    [
        (0, torch.rand(19, 3)),
        (1, torch.rand(2, 0, 3)),
        (2, torch.rand(3, 4, 3)),
        (3, torch.rand(2, 5, 2)),
    ],
)
def test_bad_point_shapes(index, bad):
    args = list(inputs())
    args[index] = bad
    with pytest.raises(ValueError):
        ABUPT(**SMALL)(*args)


def test_bad_context_and_parameters():
    model = ABUPT(**SMALL)
    args = inputs()
    ctx = model.encode(*args)
    q = torch.rand(2, 3, 3)
    with pytest.raises(ValueError, match="parameters"):
        model(*args, parameters=torch.ones(2, 2))
    with pytest.raises(ValueError, match="layer contexts"):
        model.decode_queries(q, replace(ctx, surface=ctx.surface[:-1]))
    bad = ((torch.zeros(2, 1, 3, 8), torch.zeros(2, 1, 3, 8)),) + ctx.surface[1:]
    with pytest.raises(ValueError, match="shape"):
        model.decode_queries(q, replace(ctx, surface=bad))
    with pytest.raises(ValueError, match="field"):
        model.decode_queries(q, ctx, "bad")
    with pytest.raises(ValueError, match="step"):
        model.decode_vorticity(q, ctx, step=0)
    with pytest.raises(ValueError, match="velocity_channels"):
        model.decode_vorticity(q, ctx, velocity_channels=(1, 1, 3))
    with pytest.raises(ValueError, match="coordinate_scale"):
        model.decode_vorticity(q, ctx, coordinate_scale=(1, 0, 1))
    disabled = ABUPT(**(SMALL | dict(surface_channels=0)))
    with pytest.raises(ValueError, match="disabled"):
        disabled(*args, surface_queries=q)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for Warp GPU comparison"
)
def test_cuda_warp_pooling_matches_torch():
    m = ABUPT(**SMALL).cuda().eval()
    args = inputs("cuda")
    expected = m(*args)
    m.radius_search_backend = "warp"
    actual = m(*args)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=1e-4, atol=2e-5)
