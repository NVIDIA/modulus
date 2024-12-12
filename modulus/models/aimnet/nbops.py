# ignore_header_test
# ruff: noqa: E402,S101

""""""
"""
AIMNet model. This code was modified from,
https://github.com/isayevlab/aimnetcentral

The following license is provided from their source,

MIT License

Copyright (c) 2024, Roman Zubatyuk

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from typing import Dict, Tuple

import torch
from torch import Tensor


def set_nb_mode(data: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """Logic to guess and set the neighbor mode."""
    if "nbmat" in data:
        if data["nbmat"].ndim == 2:
            data["_nb_mode"] = torch.tensor(1)
        elif data["nbmat"].ndim == 3:
            data["_nb_mode"] = torch.tensor(2)
        else:
            raise ValueError(f"Invalid neighbor matrix shape: {data['nbmat'].shape}")
    else:
        data["_nb_mode"] = torch.tensor(0)
    return data


def get_nb_mode(data: Dict[str, Tensor]) -> int:
    """Get the neighbor mode    ."""
    return int(data["_nb_mode"].item())


def calc_masks(data: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """Calculate neighbor masks"""
    nb_mode = get_nb_mode(data)
    if nb_mode == 0:
        data["mask_i"] = data["numbers"] == 0
        data["mask_ij"] = torch.eye(
            data["numbers"].shape[1], device=data["numbers"].device, dtype=torch.bool
        ).unsqueeze(0)
        if data["mask_i"].any():
            data["_input_padded"] = torch.tensor(True)
            data["_natom"] = data["mask_i"].logical_not().sum(-1)
            data["mol_sizes"] = (~data["mask_i"]).sum(-1)
            data["mask_ij"] = data["mask_ij"] | (
                data["mask_i"].unsqueeze(-2) + data["mask_i"].unsqueeze(-1)
            )
        else:
            data["_input_padded"] = torch.tensor(False)
            data["_natom"] = torch.tensor(
                data["numbers"].shape[1], device=data["numbers"].device
            )
            data["mol_sizes"] = torch.tensor(
                data["numbers"].shape[1], device=data["numbers"].device
            )
        data["mask_ij_lr"] = data["mask_ij"]
    elif nb_mode == 1:
        # padding must be the last atom
        data["mask_i"] = torch.zeros(
            data["numbers"].shape[0], device=data["numbers"].device, dtype=torch.bool
        )
        data["mask_i"][-1] = True
        for suffix in ("", "_lr"):
            if f"nbmat{suffix}" in data:
                data[f"mask_ij{suffix}"] = (
                    data[f"nbmat{suffix}"] == data["numbers"].shape[0] - 1
                )
        data["_input_padded"] = torch.tensor(True)
        data["mol_sizes"] = torch.bincount(data["mol_idx"])
        # last atom is padding
        data["mol_sizes"][-1] -= 1
    elif nb_mode == 2:
        data["mask_i"] = data["numbers"] == 0
        w = torch.where(data["mask_i"])
        pad_idx = w[0] * data["numbers"].shape[1] + w[1]
        for suffix in ("", "_lr"):
            if f"nbmat{suffix}" in data:
                data[f"mask_ij{suffix}"] = torch.isin(data[f"nbmat{suffix}"], pad_idx)
        data["_input_padded"] = torch.tensor(True)
        data["mol_sizes"] = (~data["mask_i"]).sum(-1)
    else:
        raise ValueError(f"Invalid neighbor mode: {nb_mode}")

    return data


def mask_ij_(
    x: Tensor,
    data: Dict[str, Tensor],
    mask_value: float = 0.0,
    inplace: bool = True,
    suffix: str = "",
) -> Tensor:
    mask = data[f"mask_ij{suffix}"]
    for _i in range(x.ndim - mask.ndim):
        mask = mask.unsqueeze(-1)
    if inplace:
        x.masked_fill_(mask, mask_value)
    else:
        x = x.masked_fill(mask, mask_value)
    return x


def mask_i_(
    x: Tensor, data: Dict[str, Tensor], mask_value: float = 0.0, inplace: bool = True
) -> Tensor:
    nb_mode = get_nb_mode(data)
    if nb_mode == 0:
        if data["_input_padded"].item():
            mask = data["mask_i"]
            for _i in range(x.ndim - mask.ndim):
                mask = mask.unsqueeze(-1)
            if inplace:
                x.masked_fill_(mask, mask_value)
            else:
                x = x.masked_fill(mask, mask_value)
    elif nb_mode == 1:
        if inplace:
            x[-1] = mask_value
        else:
            x = torch.cat([x[:-1], torch.zeros_like(x[:1])], dim=0)
    elif nb_mode == 2:
        if inplace:
            x[:, -1] = mask_value
        else:
            x = torch.cat([x[:, :-1], torch.zeros_like(x[:, :1])], dim=1)
    else:
        raise ValueError(f"Invalid neighbor mode: {nb_mode}")
    return x


def get_ij(
    x: Tensor, data: Dict[str, Tensor], suffix: str = ""
) -> Tuple[Tensor, Tensor]:
    """Returns neighbors based on the neighbor mode."""
    nb_mode = get_nb_mode(data)
    if nb_mode == 0:
        x_i = x.unsqueeze(2)
        x_j = x.unsqueeze(1)
    elif nb_mode == 1:
        x_i = x.unsqueeze(1)
        idx = data[f"nbmat{suffix}"]
        x_j = torch.index_select(x, 0, idx.flatten()).unflatten(0, idx.shape)
    elif nb_mode == 2:
        x_i = x.unsqueeze(2)
        idx = data[f"nbmat{suffix}"]
        x_j = torch.index_select(x.flatten(0, 1), 0, idx.flatten()).unflatten(
            0, idx.shape
        )
    else:
        raise ValueError(f"Invalid neighbor mode: {nb_mode}")
    return x_i, x_j


def mol_sum(x: Tensor, data: Dict[str, Tensor]) -> Tensor:
    nb_mode = get_nb_mode(data)
    if nb_mode in (0, 2):
        res = x.sum(dim=1)
    elif nb_mode == 1:
        assert x.ndim in (
            1,
            2,
        ), "Invalid tensor shape for mol_sum, ndim should be 1 or 2"
        idx = data["mol_idx"]
        # assuming mol_idx is sorted, replace with max if not
        out_size = int(idx[-1].item()) + 1
        if x.ndim == 1:
            res = torch.zeros(out_size, device=x.device, dtype=x.dtype)
        else:
            idx = idx.unsqueeze(-1).expand(-1, x.shape[1])
            res = torch.zeros(out_size, x.shape[1], device=x.device, dtype=x.dtype)
        res.scatter_add_(0, idx, x)
    else:
        raise ValueError(f"Invalid neighbor mode: {nb_mode}")
    return res
