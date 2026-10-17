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

"""Tests for ``packed_apply``: the same results as ``TensorDict.apply``, in fewer calls."""

import pytest
import torch
from tensordict import TensorDict

from physicsnemo.mesh.utilities import _packing
from physicsnemo.mesh.utilities._packing import packed_apply
from physicsnemo.mesh.utilities._scatter_ops import scatter_aggregate


def _fields(n: int, device: str) -> TensorDict:
    g = torch.Generator().manual_seed(0)
    return TensorDict(
        {
            "p": torch.randn(n, generator=g),
            "u": torch.randn(n, 3, generator=g),
            "nested": {
                "stress": torch.randn(n, 2, 3, generator=g),
                "k": torch.randn(n, generator=g),
            },
            "p64": torch.randn(n, generator=g, dtype=torch.float64),
            "u64": torch.randn(n, 2, generator=g, dtype=torch.float64),
            "tag": torch.randint(0, 5, (n,), generator=g),
            "flag": torch.rand(n, generator=g) > 0.5,
        },
        batch_size=[n],
    ).to(device)


def _assert_same(actual: TensorDict, expected: TensorDict) -> None:
    keys = list(expected.keys(include_nested=True, leaves_only=True))
    assert list(actual.keys(include_nested=True, leaves_only=True)) == keys
    assert actual.batch_size == expected.batch_size
    for key in keys:
        assert actual[key].dtype == expected[key].dtype, key
        assert actual[key].is_contiguous(), key
        torch.testing.assert_close(actual[key], expected[key], rtol=1e-6, atol=1e-6)


def test_gather_matches_apply(device):
    data = _fields(10, device)
    index = torch.tensor([3, 0, 9, 3, 1], device=device)
    fn = lambda x: x[index]  # noqa: E731
    _assert_same(packed_apply(data, fn, batch_size=[5]), data.apply(fn, batch_size=[5]))


def test_scatter_mean_matches_apply(device):
    """Integer and boolean fields average in float64, packed or not."""
    data = _fields(12, device)
    mapping = torch.tensor([0, 1, 2, 0, 1, 2, 3, 3, 3, 0, 1, 2], device=device)
    fn = lambda x: scatter_aggregate(x, mapping, 4, None, "mean")  # noqa: E731
    actual = packed_apply(data, fn, batch_size=[4])
    _assert_same(actual, data.apply(fn, batch_size=[4]))
    assert actual["tag"].dtype == torch.float64


def test_extra_result_dims_before_the_components(device):
    """A result of shape (N, D, *S), like a gradient, unpacks per leaf."""
    data = _fields(6, device)
    scale = torch.arange(1.0, 4.0, device=device)

    def fn(x):
        shape = (1, 3) + (1,) * (x.ndim - 1)
        return x.unsqueeze(1) * scale.view(shape).to(x.dtype)

    float_data = data.exclude("tag", "flag")
    _assert_same(
        packed_apply(float_data, fn, batch_size=[6]),
        float_data.apply(fn, batch_size=[6]),
    )


def test_calls_once_per_group():
    """Consecutive leaves of one dtype share a call while they fit max(widest, 4)."""
    calls: list[int] = []

    def fn(x):
        calls.append(x.shape[-1] if x.ndim == 2 else 1)
        return x * 2

    data = TensorDict({f"s{i}": torch.ones(5) for i in range(8)}, batch_size=[5])
    packed_apply(data, fn, batch_size=[5])
    assert calls == [4, 4]

    calls.clear()
    data = TensorDict({"p": torch.ones(5), "u": torch.ones(5, 3)}, batch_size=[5])
    packed_apply(data, fn, batch_size=[5])
    assert calls == [4]

    ### Two vectors exceed the 4-component bound, so each runs alone, unpacked.
    calls.clear()
    data = TensorDict({"u": torch.ones(5, 3), "v": torch.ones(5, 3)}, batch_size=[5])
    packed_apply(data, fn, batch_size=[5])
    assert calls == [3, 3]


@pytest.mark.cuda
@pytest.mark.parametrize("pad_rows", [True, False])
def test_cuda_pads_rows_of_16_bytes(monkeypatch, pad_rows):
    """On CUDA before PyTorch 2.16, four float32 components get a fifth, zero column."""
    monkeypatch.setattr(_packing, "_PAD_ROWS_OF_16_BYTES", pad_rows)
    widths: list[int] = []

    def fn(x):
        widths.append(x.shape[-1])
        return x[torch.tensor([1, 0], device=x.device)]

    data = TensorDict(
        {"p": torch.arange(4.0), "u": torch.arange(12.0).view(4, 3)},
        batch_size=[4],
        device="cuda",
    )
    out = packed_apply(data, fn, batch_size=[2])
    assert widths == [5 if pad_rows else 4]
    _assert_same(
        out,
        data.apply(lambda x: x[torch.tensor([1, 0], device="cuda")], batch_size=[2]),
    )


def test_keeps_empty_nested_tensordicts():
    data = TensorDict(
        {"a": torch.ones(4), "sub": TensorDict({}, batch_size=[4]), "b": torch.ones(4)},
        batch_size=[4],
    )
    fn = lambda x: x[:2]  # noqa: E731
    out = packed_apply(data, fn, batch_size=[2])
    assert list(out.keys()) == list(data.apply(fn, batch_size=[2]).keys())
    assert out["sub"].batch_size == torch.Size([2])


def test_a_field_named_like_the_pad_column():
    """A leaf may share the name of the padding column."""
    data = TensorDict(
        {"_packed_apply_pad": torch.arange(4.0), "u": torch.ones(4, 3)}, batch_size=[4]
    )
    fn = lambda x: x[torch.tensor([3, 1])]  # noqa: E731
    _assert_same(packed_apply(data, fn, batch_size=[2]), data.apply(fn, batch_size=[2]))


@pytest.mark.cuda
def test_cuda_pads_a_single_float64_vector(monkeypatch):
    """One float64 (N, 2) field has 16-byte rows, so it is padded even alone."""
    monkeypatch.setattr(_packing, "_PAD_ROWS_OF_16_BYTES", True)
    widths: list[int] = []
    index = torch.tensor([2, 0, 1], device="cuda")

    def fn(x):
        widths.append(x.shape[-1])
        return x[index]

    data = TensorDict(
        {"v": torch.arange(8.0, dtype=torch.float64).view(4, 2)},
        batch_size=[4],
        device="cuda",
    )
    out = packed_apply(data, fn, batch_size=[3])
    assert widths == [3]
    _assert_same(out, data.apply(lambda x: x[index], batch_size=[3]))


def test_empty_tensordict():
    data = TensorDict({}, batch_size=[3])
    out = packed_apply(data, lambda x: x[:2], batch_size=[2])
    assert out.batch_size == torch.Size([2])
    assert len(list(out.keys())) == 0
