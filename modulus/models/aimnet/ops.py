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

import math
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

from . import nbops


def lazy_calc_dij_lr(data: Dict[str, Tensor]) -> Dict[str, Tensor]:
    if "d_ij_lr" not in data:
        nb_mode = nbops.get_nb_mode(data)
        if nb_mode == 0:
            data["d_ij_lr"] = data["d_ij"]
        else:
            data["d_ij_lr"] = calc_distances(data, suffix="_lr")[0]
    return data


def calc_distances(
    data: Dict[str, Tensor], suffix: str = "", pad_value: float = 1.0
) -> Tuple[Tensor, Tensor]:
    coord_i, coord_j = nbops.get_ij(data["coord"], data, suffix)
    if f"shifts{suffix}" in data:
        assert "cell" in data, "cell is required if shifts are provided"
        nb_mode = nbops.get_nb_mode(data)
        if nb_mode == 2:
            shifts = torch.einsum(
                "bnmd,bdh->bnmh", data[f"shifts{suffix}"], data["cell"]
            )
        else:
            shifts = data[f"shifts{suffix}"] @ data["cell"]
        coord_j = coord_j + shifts
    r_ij = coord_j - coord_i
    d_ij = torch.norm(r_ij, p=2, dim=-1)
    d_ij = nbops.mask_ij_(
        d_ij, data, mask_value=pad_value, inplace=False, suffix=suffix
    )
    return d_ij, r_ij


def center_coordinates(
    coord: Tensor, data: Dict[str, Tensor], masses: Optional[Tensor] = None
) -> Tensor:
    if masses is not None:
        masses = masses.unsqueeze(-1)
        center = (
            nbops.mol_sum(coord * masses, data)
            / nbops.mol_sum(masses, data)
            / data["mol_sizes"].unsqueeze(-1)
        )
    else:
        center = nbops.mol_sum(coord, data) / data["mol_sizes"]
    nb_mode = nbops.get_nb_mode(data)
    if nb_mode in (0, 2):
        center = center.unsqueeze(-2)
    coord = coord - center
    return coord


def cosine_cutoff(d_ij: Tensor, rc: float) -> Tensor:
    fc = 0.5 * (torch.cos(d_ij.clamp(min=1e-6, max=rc) * (math.pi / rc)) + 1.0)
    return fc


def exp_cutoff(d: Tensor, rc: Tensor) -> Tensor:
    fc = (
        torch.exp(-1.0 / (1.0 - (d / rc).clamp(0, 1.0 - 1e-6).pow(2)))
        / 0.36787944117144233
    )
    return fc


def exp_expand(d_ij: Tensor, shifts: Tensor, eta: float) -> Tensor:
    # expand on axis -1, e.g. (b, n, m) -> (b, n, m, shifts)
    return torch.exp(-eta * (d_ij.unsqueeze(-1) - shifts) ** 2)


# pylint: disable=invalid-name
def nse(
    Q: Tensor,
    q_u: Tensor,
    f_u: Tensor,
    data: Dict[str, Tensor],
    epsilon: float = 1.0e-6,
) -> Tensor:
    # Q and q_u and f_u must have last dimension size 1 or 2
    F_u = nbops.mol_sum(f_u, data) + epsilon
    Q_u = nbops.mol_sum(q_u, data)
    dQ = Q - Q_u
    # for loss
    data["_dQ"] = dQ

    nb_mode = nbops.get_nb_mode(data)
    if nb_mode in (0, 2):
        F_u = F_u.unsqueeze(-2)
        dQ = dQ.unsqueeze(-2)
    elif nb_mode == 1:
        data["mol_sizes"][-1] += 1
        F_u = torch.repeat_interleave(F_u, data["mol_sizes"], dim=0)
        dQ = torch.repeat_interleave(dQ, data["mol_sizes"], dim=0)
        data["mol_sizes"][-1] -= 1
    else:
        raise ValueError(f"Invalid neighbor mode: {nb_mode}")
    f = f_u / F_u
    q = q_u + f * dQ
    return q


def coulomb_matrix_dsf(
    d_ij: Tensor, Rc: float, alpha: float, data: Dict[str, Tensor]
) -> Tensor:
    _c1 = (alpha * d_ij).erfc() / d_ij
    _c2 = math.erfc(alpha * Rc) / Rc
    _c3 = _c2 / Rc
    _c4 = 2 * alpha * math.exp(-((alpha * Rc) ** 2)) / (Rc * math.pi**0.5)
    J = _c1 - _c2 + (d_ij - Rc) * (_c3 + _c4)
    # mask for d_ij > Rc
    mask = data["mask_ij_lr"] & (d_ij > Rc)
    J.masked_fill_(mask, 0.0)
    return J


def coulomb_matrix_sf(
    q_j: Tensor, d_ij: Tensor, Rc: float, data: Dict[str, Tensor]
) -> Tensor:
    _c1 = 1.0 / d_ij
    _c2 = 1.0 / Rc
    _c3 = _c2 / Rc
    J = _c1 - _c2 + (d_ij - Rc) * _c3
    mask = data["mask_ij_lr"] & (d_ij > Rc)
    J.masked_fill_(mask, 0.0)
    return J


def get_shifts_within_cutoff(cell: Tensor, cutoff: Tensor) -> Tensor:
    """Calculates shifts with cutoffs."""
    assert cell.shape == (3, 3), "Batch cell is not supported"
    cell_inv = torch.inverse(cell).mT
    inv_distances = cell_inv.norm(p=2, dim=-1)
    num_repeats = torch.ceil(cutoff * inv_distances).to(torch.long)
    device = cell.device
    shifts = torch.cartesian_prod(
        torch.arange(-num_repeats[0], num_repeats[0] + 1, device=device),  # type: ignore[attr-defined]
        torch.arange(-num_repeats[1], num_repeats[1] + 1, device=device),  # type: ignore[attr-defined]
        torch.arange(-num_repeats[2], num_repeats[2] + 1, device=device),  # type: ignore[attr-defined]
    ).to(torch.float)
    return shifts


def coulomb_matrix_ewald(coord: Tensor, cell: Tensor) -> Tensor:
    # single molecule implementation. nb_mode == 1
    assert coord.ndim == 2 and cell.ndim == 2, "Only single molecule is supported"
    accuracy = 1e-8
    N = coord.shape[0]
    volume = torch.det(cell)
    eta = ((volume**2 / N) ** (1 / 6)) / math.sqrt(2.0 * math.pi)
    cutoff_real = math.sqrt(-2.0 * math.log(accuracy)) * eta
    cutoff_recip = math.sqrt(-2.0 * math.log(accuracy)) / eta

    # real space
    _grad_mode = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    shifts = get_shifts_within_cutoff(cell, cutoff_real)  # (num_shifts, 3)
    torch.set_grad_enabled(_grad_mode)
    disps_ij = coord[None, :, :] - coord[:, None, :]
    disps = disps_ij[None, :, :, :] + torch.matmul(shifts, cell)[:, None, None, :]
    distances_all = disps.norm(p=2, dim=-1)  # (num_shifts, num_atoms, num_atoms)
    within_cutoff = (distances_all > 0.1) & (distances_all < cutoff_real)
    distances = distances_all[within_cutoff]
    e_real_matrix_aug = torch.zeros_like(distances_all)
    e_real_matrix_aug[within_cutoff] = (
        torch.erfc(distances / (math.sqrt(2) * eta)) / distances
    )
    e_real_matrix = e_real_matrix_aug.sum(dim=0)

    # reciprocal space
    recip = 2 * math.pi * torch.transpose(torch.linalg.inv(cell), 0, 1)
    _grad_mode = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    shifts = get_shifts_within_cutoff(recip, cutoff_recip)
    torch.set_grad_enabled(_grad_mode)
    ks_all = torch.matmul(shifts, recip)
    length_all = ks_all.norm(p=2, dim=-1)
    within_cutoff = (length_all > 0.1) & (length_all < cutoff_recip)
    ks = ks_all[within_cutoff]
    length = length_all[within_cutoff]
    # disps_ij[i, j, :] is displacement vector r_{ij}, (num_atoms, num_atoms, 3)
    # disps_ij = coord[None, :, :] - coord[:, None, :] # computed above
    phases = torch.sum(ks[:, None, None, :] * disps_ij[None, :, :, :], dim=-1)
    e_recip_matrix_aug = (
        torch.cos(phases)
        * torch.exp(-0.5 * torch.square(eta * length[:, None, None]))
        / torch.square(length[:, None, None])
    )
    e_recip_matrix = 4.0 * math.pi / volume * torch.sum(e_recip_matrix_aug, dim=0)
    # self interaction
    device = coord.device
    diag = -math.sqrt(2.0 / math.pi) / eta * torch.ones(N, device=device)
    e_self_matrix = torch.diag(diag)

    J = e_real_matrix + e_recip_matrix + e_self_matrix
    return J


def huber(x: Tensor, delta: float = 1.0) -> Tensor:
    """Huber loss"""
    return torch.where(x.abs() < delta, 0.5 * x**2, delta * (x.abs() - 0.5 * delta))


def bumpfn(x: Tensor, low: float = 0.0, high: float = 1.0) -> Tensor:
    """For x > 0, return smooth transition function which is 0 for x <= low and 1 for x >= high"""
    x = (x - low) / (high - low)
    x = x.clamp(min=1e-6, max=1 - 1e-6)
    a = (-1 / x).exp()
    b = (-1 / (1 - x)).exp()
    return a / (a + b)


def smoothstep(x: Tensor, low: float = 0.0, high: float = 1.0) -> Tensor:
    """For x > 0, return smooth transition function which is 0 for x <= low and 1 for x >= high"""
    x = (x - low) / (high - low)
    x = x.clamp(min=0, max=1)
    return x.pow(3) * (x * (x * 6 - 15) + 10)


def expstep(x: Tensor, low: float = 0.0, high: float = 1.0) -> Tensor:
    """For x > 0, return smooth transition function which is 0 for x <= low and 1 for x >= high"""
    x = (x - low) / (high - low)
    x = x.clamp(min=1e-6, max=1 - 1e-6)
    return (-1 / (1 - x.pow(2))).exp() / 0.36787944117144233
