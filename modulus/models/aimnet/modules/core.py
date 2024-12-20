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

from typing import Any, Callable, Dict, List, Optional

import torch
from torch import Tensor, nn

from modulus.models.aimnet import constants, nbops, ops
from modulus.models.aimnet.config import get_init_module, get_module


def MLP(
    n_in: int,
    n_out: int,
    hidden: Optional[List[int]] = None,
    activation_fn: Callable | str = "torch.nn.GELU",
    activation_kwargs: Optional[Dict[str, Any]] = None,
    weight_init_fn: Callable | str = "torch.nn.init.xavier_normal_",
    bias: bool = True,
    last_linear: bool = True,
):
    """Convenience function to build MLP from config"""
    if hidden is None:
        hidden = []
    if activation_kwargs is None:
        activation_kwargs = {}
    # hp search hack
    hidden = [x for x in hidden if x > 0]
    if isinstance(activation_fn, str):
        activation_fn = get_init_module(activation_fn, kwargs=activation_kwargs)
    if isinstance(weight_init_fn, str):
        weight_init_fn = get_module(weight_init_fn)
    sizes = [n_in, *hidden, n_out]
    layers = []
    for i in range(1, len(sizes)):
        n_in, n_out = sizes[i - 1], sizes[i]
        layer = nn.Linear(n_in, n_out, bias=bias)
        with torch.no_grad():
            weight_init_fn(layer.weight)
            if bias:
                nn.init.zeros_(layer.bias)
        layers.append(layer)
        if not (last_linear and i == len(sizes) - 1):
            layers.append(activation_fn)
    return nn.Sequential(*layers)


class Embedding(nn.Embedding):
    """Atomic embeddings."""

    def __init__(self, init: Optional[Dict[int, Any]] = None, **kwargs):
        super().__init__(**kwargs)
        with torch.no_grad():
            if init is not None:
                for i in range(self.weight.shape[0]):
                    if self.padding_idx is not None and i == self.padding_idx:
                        continue
                    if i in init:
                        self.weight[i] = init[i]
                    else:
                        self.weight[i].fill_(float("nan"))
                for k, v in init.items():
                    self.weight[k] = v

    def reset_parameters(self) -> None:
        nn.init.orthogonal_(self.weight)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)


class DSequential(nn.Module):
    def __init__(self, *modules):
        super().__init__()
        self.module = nn.ModuleList(modules)

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        for m in self.module:
            data = m(data)
        return data


class AtomicShift(nn.Module):
    """Atomic shift."""

    def __init__(
        self,
        key_in: str,
        key_out: str,
        num_types: int = 64,
        dtype: torch.dtype = torch.float,
        requires_grad: bool = True,
        reduce_sum=False,
    ):
        super().__init__()
        shifts = nn.Embedding(num_types, 1, padding_idx=0, dtype=dtype)
        shifts.weight.requires_grad_(requires_grad)
        self.shifts = shifts
        self.key_in = key_in
        self.key_out = key_out
        self.reduce_sum = reduce_sum

    def extra_repr(self) -> str:
        return f"key_in: {self.key_in}, key_out: {self.key_out}"

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        shifts = self.shifts(data["numbers"]).squeeze(-1)
        if self.reduce_sum:
            shifts = nbops.mol_sum(shifts, data)
        data[self.key_out] = data[self.key_in] + shifts
        return data


class AtomicSum(nn.Module):
    """Atomic sum."""

    def __init__(self, key_in: str, key_out: str):
        super().__init__()
        self.key_in = key_in
        self.key_out = key_out

    def extra_repr(self) -> str:
        return f"key_in: {self.key_in}, key_out: {self.key_out}"

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        data[self.key_out] = nbops.mol_sum(data[self.key_in], data)
        return data


class Output(nn.Module):
    """Output."""

    def __init__(
        self, mlp: Dict | nn.Module, n_in: int, n_out: int, key_in: str, key_out: str
    ):
        super().__init__()
        self.key_in = key_in
        self.key_out = key_out
        if not isinstance(mlp, nn.Module):
            mlp = MLP(n_in=n_in, n_out=n_out, **mlp)
        self.mlp = mlp

    def extra_repr(self) -> str:
        return f"key_in: {self.key_in}, key_out: {self.key_out}"

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        v = self.mlp(data[self.key_in]).squeeze(-1)
        if data["_input_padded"].item():
            v = nbops.mask_i_(v, data, mask_value=0.0)
        data[self.key_out] = v
        return data


class Forces(nn.Module):
    """Forces."""

    def __init__(
        self,
        module: nn.Module,
        x: str = "coord",
        y: str = "energy",
        key_out: str = "forces",
    ):
        super().__init__()
        self.module = module
        self.x = x
        self.y = y
        self.key_out = key_out

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        prev = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        data[self.x].requires_grad_(True)
        data = self.module(data)
        y = data[self.y]
        g = torch.autograd.grad([y.sum()], [data[self.x]], create_graph=self.training)[
            0
        ]
        assert g is not None
        data[self.key_out] = -g
        torch.set_grad_enabled(prev)
        return data


class Dipole(nn.Module):
    def __init__(
        self,
        key_in: str = "charges",
        key_out: str = "dipole",
        center_coord: bool = False,
    ):
        super().__init__()
        self.center_coord = center_coord
        self.key_out = key_out
        self.key_in = key_in
        self.register_buffer("mass", constants.get_masses())

    def extra_repr(self) -> str:
        return f"key_in: {self.key_in}, key_out: {self.key_out}, center_coord: {self.center_coord}"

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        q = data[self.key_in]
        r = data["coord"]
        if self.center_coord:
            r = ops.center_coordinates(r, data, self.mass[data["numbers"]])
        data[self.key_out] = nbops.mol_sum(q.unsqueeze(-1) * r, data)
        return data


class Quadrupole(Dipole):
    """Output."""

    def __init__(
        self,
        key_in: str = "charges",
        key_out: str = "quadrupole",
        center_coord: bool = False,
    ):
        super().__init__(key_in=key_in, key_out=key_out, center_coord=center_coord)

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        q = data[self.key_in]
        r = data["coord"]
        if self.center_coord:
            r = ops.center_coordinates(r, data, self.mass[data["numbers"]])
        _x = torch.cat([r.pow(2), r * r.roll(-1, -1)], dim=-1)
        quad = nbops.mol_sum(q.unsqueeze(-1) * _x, data)
        _x1, _x2 = quad.split(3, dim=-1)
        _x1 = _x1 - _x1.mean(dim=-1, keepdim=True)
        quad = torch.cat([_x1, _x2], dim=-1)
        data[self.key_out] = quad
        return data


class SRRep(nn.Module):
    """GFN1-stype short range repulsion function"""

    def __init__(self, key_out="e_rep", cutoff_fn="none", rc=5.2, reduce_sum=True):
        super().__init__()
        from modulus.models.aimnet.constants import get_gfn1_rep

        self.key_out = key_out
        self.cutoff_fn = cutoff_fn
        self.reduce_sum = reduce_sum

        self.register_buffer("rc", torch.tensor(rc))
        gfn1_repa, gfn1_repb = get_gfn1_rep()
        weight = torch.stack([gfn1_repa, gfn1_repb], dim=-1)
        self.params = nn.Embedding(87, 2, padding_idx=0, _weight=weight)
        self.params.weight.requires_grad_(False)

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        p = self.params(data["numbers"])
        p_i, p_j = nbops.get_ij(p, data)
        p_ij = p_i * p_j
        alpha_ij, zeff_ij = p_ij.unbind(-1)
        d_ij = data["d_ij"]
        e = torch.exp(-alpha_ij * d_ij.pow(1.5)) * zeff_ij / d_ij
        e = nbops.mask_ij_(e, data, 0.0)
        if self.cutoff_fn == "exp_cutoff":
            e = e * ops.exp_cutoff(d_ij, self.rc)
        elif self.cutoff_fn == "cosine_cutoff":
            e = e * ops.cosine_cutoff(d_ij, self.rc)
        e = e.sum(-1)
        if self.reduce_sum:
            e = nbops.mol_sum(e, data)
        if self.key_out in data:
            data[self.key_out] = data[self.key_out] + e
        else:
            data[self.key_out] = e
        return data
