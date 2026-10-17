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
from typing import Callable, cast

import torch
from tensordict import NestedKey, TensorDict, TensorDictBase

### Name of the zero column that keeps packed rows off multiples of 16 bytes.
_PAD_KEY = "_packed_apply_pad"


def packed_apply(
    data: TensorDict,
    fn: Callable[[torch.Tensor], torch.Tensor],
    *,
    batch_size: torch.Size | tuple[int, ...],
    group: bool = True,
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
    instead of once per leaf.  A group holds leaves of one dtype, consecutive
    among those of that dtype, whose components add up to at most those of
    the widest such leaf, or 4 if more:
    ``fn``'s temporaries grow with the number of packed components, and this
    bound keeps them near those of a 4-component field.  This pays off when
    each call of ``fn`` has a large fixed cost, such as the batched
    least-squares solves of the LSQ derivatives.  For operations bound by
    memory traffic, such as gathers and scatters of rows, packing saves little
    time and raises peak memory, so ``group=False`` runs each leaf alone.

    On CUDA, a group (or leaf) whose rows are a multiple of 16 bytes wide gets
    a column of zeros, because gathering rows of that width is 10-20x slower
    (`pytorch/pytorch#200939 <https://github.com/pytorch/pytorch/issues/200939>`_).

    Parameters
    ----------
    data : TensorDict
        Fields with one batch dimension, all on one device.
    fn : Callable[[torch.Tensor], torch.Tensor]
        Component-wise operation, as above.
    batch_size : torch.Size or tuple[int, ...]
        Batch size of the result: a prefix of :math:`R`.
    group : bool, optional, default=True
        Whether to pack several leaves per call of ``fn``.  If ``False``,
        each leaf runs alone, and only the padding above applies.

    Returns
    -------
    TensorDict
        ``fn`` of each leaf, contiguous, under the leaf's key and in the order
        of ``data``.  Nested tensordicts without leaves are kept, empty.
        ``data`` must not hold non-tensor data, which :meth:`TensorDict.pack
        <tensordict.TensorDictBase.pack>` refuses.

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

    def needs_pad(n_components: int, value: torch.Tensor) -> bool:
        return (
            value.device.type == "cuda"
            and n_components * value.element_size() % 16 == 0
        )

    ### One leaf at a time, and no leaf to pad: ``apply`` itself.
    if not group and not any(
        needs_pad(math.prod(value.shape[1:]), value) for _, value in leaves
    ):
        return cast(TensorDict, data.apply(fn, batch_size=batch_size))

    ### Consecutive leaves of one dtype, at most ``max(widest, 4)`` components.
    by_dtype: dict[torch.dtype, list[tuple[NestedKey, int]]] = {}
    for key, value in leaves:
        by_dtype.setdefault(value.dtype, []).append((key, math.prod(value.shape[1:])))
    groups: list[tuple[list[NestedKey], int]] = []
    for entries in by_dtype.values():
        cap = max(4, *(width for _, width in entries)) if group else 0
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
        pad = needs_pad(n_components, first)
        if len(keys) == 1 and not pad:
            results[keys[0]] = fn(first).contiguous()
            continue
        selected = data.select(*keys)
        pad_key = _PAD_KEY
        while pad_key in selected.keys():
            pad_key += "_"
        if pad:
            selected[pad_key] = first.new_zeros(data.batch_size)
        flat, layout = selected.pack()
        for key, value in layout.unpack(fn(flat)).items(
            include_nested=True, leaves_only=True
        ):
            if key != pad_key:
                results[key] = value.contiguous()

    ### In the order of ``data``, as ``apply`` returns them.
    out = TensorDict(batch_size=batch_size, device=data.device)
    for key, value in data.items(include_nested=True):
        if key in results:
            out.set(key, results[key])
        elif isinstance(value, TensorDictBase) and value.is_empty():
            out.set(key, TensorDict(batch_size=batch_size, device=data.device))
    return out
