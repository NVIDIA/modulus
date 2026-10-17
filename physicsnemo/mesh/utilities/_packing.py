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

"""Run a component-wise operation on the fields of a TensorDict in a few packed calls."""

import math
from typing import Callable

import torch
from tensordict import NestedKey, TensorDict

### Name of the zero column that keeps packed rows off multiples of 16 bytes.
_PAD_KEY = "_packed_apply_pad"


def packed_apply(
    data: TensorDict,
    fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    batch_size: torch.Size | tuple[int, ...],
) -> TensorDict:
    r"""``data.apply(fn, batch_size=batch_size)``, with the leaves packed together.

    ``fn`` must map a field of shape :math:`(N, *S)` to a result of shape
    :math:`(*R, *S)`, treating each of the field's components independently,
    as gathers and scatters of rows, means and other weighted sums of rows,
    and least-squares solves with one right-hand side per component do.  Then
    applying ``fn`` once to several fields packed side by side, in a tensor of
    shape :math:`(N, F)`, gives the results of the fields side by side.

    Leaves of one dtype are packed with :meth:`TensorDict.pack
    <tensordict.TensorDictBase.pack>`, so ``fn`` runs once per group of leaves
    instead of once per leaf.  A group holds consecutive leaves whose
    components add up to at most those of its widest leaf, or 4 if more:
    ``fn``'s temporaries grow with the number of packed components, and this
    bound keeps them near those of a 4-component field.  On CUDA, a group
    whose rows are a multiple of 16 bytes wide gets a column of zeros, because
    gathering rows of that width is 10-20x slower
    (`pytorch/pytorch#200939 <https://github.com/pytorch/pytorch/issues/200939>`_).

    Parameters
    ----------
    data : TensorDict
        Fields with one batch dimension, all on one device.
    fn : Callable[[torch.Tensor], torch.Tensor]
        Component-wise operation, as above.
    batch_size : torch.Size or tuple[int, ...]
        Batch size of the result: a prefix of :math:`R`.

    Returns
    -------
    TensorDict
        ``fn`` of each leaf, contiguous, under the leaf's key and in the order
        of ``data``.

    Examples
    --------
    >>> data = TensorDict(
    ...     {"p": torch.tensor([1.0, 2.0, 3.0]), "u": torch.ones(3, 3)}, batch_size=[3]
    ... )
    >>> out = packed_apply(data, lambda x: x[torch.tensor([2, 0])], batch_size=[2])
    >>> out["p"].tolist(), out["u"].shape
    ([3.0, 1.0], torch.Size([2, 3]))
    """
    leaves = list(data.items(include_nested=True, leaves_only=True))

    ### Consecutive leaves of one dtype, at most ``max(widest, 4)`` components.
    by_dtype: dict[torch.dtype, list[tuple[NestedKey, int]]] = {}
    for key, value in leaves:
        by_dtype.setdefault(value.dtype, []).append((key, math.prod(value.shape[1:])))
    groups: list[tuple[list[NestedKey], int]] = []
    for entries in by_dtype.values():
        cap = max(4, *(width for _, width in entries))
        keys: list[NestedKey] = []
        n_components = 0
        for key, width in entries:
            if keys and n_components + width > cap:
                groups.append((keys, n_components))
                keys, n_components = [], 0
            keys.append(key)
            n_components += width
        groups.append((keys, n_components))

    results: dict[NestedKey, torch.Tensor] = {}
    for keys, n_components in groups:
        first = data.get(keys[0])
        pad = (
            first.device.type == "cuda"
            and n_components * first.element_size() % 16 == 0
        )
        if len(keys) == 1 and not pad:
            results[keys[0]] = fn(first).contiguous()
            continue
        group = data.select(*keys)
        if pad:
            group[_PAD_KEY] = first.new_zeros(data.batch_size)
        flat, layout = group.pack()
        for key, value in layout.unpack(fn(flat)).items(
            include_nested=True, leaves_only=True
        ):
            if key != _PAD_KEY:
                results[key] = value.contiguous()

    out = TensorDict(batch_size=batch_size, device=data.device)
    for key, _ in leaves:
        out.set(key, results[key])
    return out
