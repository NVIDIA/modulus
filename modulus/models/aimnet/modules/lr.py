# ignore_header_test
# ruff: noqa: E402

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

from typing import Dict, Optional

import torch
from torch import Tensor, nn

from modulus.models.aimnet import constants, nbops, ops


class LRCoulomb(nn.Module):
    """LRCoulomb."""

    def __init__(
        self,
        key_in: str = "charges",
        key_out: str = "e_h",
        rc: float = 4.6,
        method: str = "simple",
        dsf_alpha: float = 0.2,
        dsf_rc: float = 15.0,
    ):
        super().__init__()
        self.key_in = key_in
        self.key_out = key_out
        self._factor = constants.half_Hartree * constants.Bohr
        self.register_buffer("rc", torch.tensor(rc))
        self.dsf_alpha = dsf_alpha
        self.dsf_rc = dsf_rc
        if method in ("simple", "dsf", "ewald"):
            self.method = method
        else:
            raise ValueError(f"Unknown method {method}")

    def coul_simple(self, data: Dict[str, Tensor]) -> Tensor:
        data = ops.lazy_calc_dij_lr(data)
        d_ij = data["d_ij_lr"]
        q = data[self.key_in]
        q_i, q_j = nbops.get_ij(q, data, suffix="_lr")
        q_ij = q_i * q_j
        fc = 1.0 - ops.exp_cutoff(d_ij, self.rc)
        e_ij = fc * q_ij / d_ij
        e_ij = nbops.mask_ij_(e_ij, data, 0.0, suffix="_lr")
        e_i = e_ij.sum(-1)
        e = self._factor * nbops.mol_sum(e_i, data)
        return e

    def coul_simple_sr(self, data: Dict[str, Tensor]) -> Tensor:
        d_ij = data["d_ij"]
        q = data[self.key_in]
        q_i, q_j = nbops.get_ij(q, data)
        q_ij = q_i * q_j
        fc = ops.exp_cutoff(d_ij, self.rc)
        e_ij = fc * q_ij / d_ij
        e_ij = nbops.mask_ij_(e_ij, data, 0.0)
        e_i = e_ij.sum(-1)
        e = self._factor * nbops.mol_sum(e_i, data)
        return e

    def coul_dsf(self, data: Dict[str, Tensor]) -> Tensor:
        data = ops.lazy_calc_dij_lr(data)
        d_ij = data["d_ij_lr"]
        q = data[self.key_in]
        q_i, q_j = nbops.get_ij(q, data, suffix="_lr")
        J = ops.coulomb_matrix_dsf(d_ij, self.dsf_rc, self.dsf_alpha, data)
        e = (q_i * q_j * J).sum(-1)
        e = self._factor * nbops.mol_sum(e, data)
        e = e - self.coul_simple_sr(data)
        return e

    def coul_ewald(self, data: Dict[str, Tensor]) -> Tensor:
        J = ops.coulomb_matrix_ewald(data["coord"], data["cell"])
        q_i, q_j = data["charges"].unsqueeze(-1), data["charges"].unsqueeze(-2)
        e = self._factor * (q_i * q_j * J).flatten(-2, -1).sum(-1)
        e = e - self.coul_simple_sr(data)
        return e

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if self.method == "simple":
            e = self.coul_simple(data)
        elif self.method == "dsf":
            e = self.coul_dsf(data)
        elif self.method == "ewald":
            e = self.coul_ewald(data)
        else:
            raise ValueError(f"Unknown method {self.method}")
        if self.key_out in data:
            data[self.key_out] = data[self.key_out] + e
        else:
            data[self.key_out] = e
        return data


class DispParam(nn.Module):
    def __init__(
        self,
        ref_c6: Optional[Dict[int, Tensor] | Tensor] = None,
        ref_alpha: Optional[Dict[int, Tensor] | Tensor] = None,
        ptfile: Optional[str] = None,
        key_in: str = "disp_param",
        key_out: str = "disp_param",
    ):
        super().__init__()
        if (
            ptfile is None
            and (ref_c6 is None or ref_alpha is None)
            or ptfile is not None
            and (ref_c6 is not None or ref_alpha is not None)
        ):
            raise ValueError(
                "Either ptfile or ref_c6 and ref_alpha should be supplied."
            )
        # load data
        ref = torch.load(ptfile) if ptfile is not None else torch.zeros(87, 2)
        for i, p in enumerate([ref_c6, ref_alpha]):
            if p is not None:
                if isinstance(p, Tensor):
                    ref[: p.shape[0], i] = p
                else:
                    for k, v in p.items():
                        ref[k, i] = v
        # c6=0 and alpha=1 for dummy atom
        ref[0, 0] = 0.0
        ref[0, 1] = 1.0
        self.register_buffer("disp_param0", ref)
        self.key_in = key_in
        self.key_out = key_out

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        disp_param_mult = data[self.key_in].clamp(min=-4, max=4).exp()
        disp_param = self.disp_param0[data["numbers"]]
        vals = disp_param * disp_param_mult
        data[self.key_out] = vals
        return data


class D3TS(nn.Module):
    """DFT-D3-like pairwise dispersion with TS combination rule"""

    def __init__(
        self,
        a1: float,
        a2: float,
        s8: float,
        s6: float = 1.0,
        key_in="disp_param",
        key_out="energy",
    ):
        super().__init__()
        self.register_buffer("r4r2", constants.get_r4r2())
        self.a1 = a1
        self.a2 = a2
        self.s6 = s6
        self.s8 = s8
        self.key_in = key_in
        self.key_out = key_out

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        disp_param = data[self.key_in]
        disp_param_i, disp_param_j = nbops.get_ij(disp_param, data, suffix="_lr")
        c6_i, alpha_i = disp_param_i.unbind(dim=-1)
        c6_j, alpha_j = disp_param_j.unbind(dim=-1)

        # TS combination rule
        c6ij = (
            2
            * c6_i
            * c6_j
            / (c6_i * alpha_j / alpha_i + c6_j * alpha_i / alpha_j).clamp(min=1e-4)
        )

        rr = self.r4r2[data["numbers"]]
        rr_i, rr_j = nbops.get_ij(rr, data, suffix="_lr")
        rrij = 3 * rr_i * rr_j
        rrij = nbops.mask_ij_(rrij, data, 1.0, suffix="_lr")
        r0ij = self.a1 * rrij.sqrt() + self.a2

        ops.lazy_calc_dij_lr(data)
        d_ij = data["d_ij_lr"] * constants.Bohr_inv
        e_ij = c6ij * (
            self.s6 / (d_ij.pow(6) + r0ij.pow(6))
            + self.s8 * rrij / (d_ij.pow(8) + r0ij.pow(8))
        )
        e = -constants.half_Hartree * nbops.mol_sum(e_ij.sum(-1), data)

        if self.key_out in data:
            data[self.key_out] = data[self.key_out] + e
        else:
            data[self.key_out] = e

        return data


class DFTD3(nn.Module):
    """DFT-D3 implementation.
    BJ dumping, C6 and C8 terms, without 3-body term.
    """

    def __init__(
        self, s8: float, a1: float, a2: float, s6: float = 1.0, key_out="energy"
    ):
        super().__init__()
        self.key_out = key_out
        # BJ damping parameters
        self.s6 = s6
        self.s8 = s8
        self.s9 = 4.0 / 3.0
        self.a1 = a1
        self.a2 = a2
        self.a3 = 16.0
        # CN parameters
        self.k1 = -16.0
        self.k3 = -4.0
        # data
        self.register_buffer("c6ab", torch.zeros(95, 95, 5, 5, 3))
        self.register_buffer("r4r2", torch.zeros(95))
        self.register_buffer("rcov", torch.zeros(95))
        self.register_buffer("cnmax", torch.zeros(95))
        sd = constants.get_dftd3_param()
        self.load_state_dict(sd)

    def _calc_c6ij(self, data: Dict[str, Tensor]) -> Tensor:
        # CN part
        # short range for CN
        # d_ij = data["d_ij"] * constants.Bohr_inv
        data = ops.lazy_calc_dij_lr(data)
        d_ij = data["d_ij_lr"] * constants.Bohr_inv

        numbers = data["numbers"]
        numbers_i, numbers_j = nbops.get_ij(numbers, data, suffix="_lr")
        rcov_i, rcov_j = nbops.get_ij(self.rcov[numbers], data, suffix="_lr")
        rcov_ij = rcov_i + rcov_j
        cn_ij = 1.0 / (1.0 + torch.exp(self.k1 * (rcov_ij / d_ij - 1.0)))
        cn_ij = nbops.mask_ij_(cn_ij, data, 0.0, suffix="_lr")
        cn = cn_ij.sum(-1)
        cn = torch.clamp(cn, max=self.cnmax[numbers]).unsqueeze(-1).unsqueeze(-1)
        cn_i, cn_j = nbops.get_ij(cn, data, suffix="_lr")
        c6ab = self.c6ab[numbers_i, numbers_j]
        c6ref, cnref_i, cnref_j = torch.unbind(c6ab, dim=-1)
        c6ref = nbops.mask_ij_(c6ref, data, 0.0, suffix="_lr")
        l_ij = torch.exp(self.k3 * ((cn_i - cnref_i).pow(2) + (cn_j - cnref_j).pow(2)))
        w = l_ij.flatten(-2, -1).sum(-1)
        z = torch.einsum("...ij,...ij->...", c6ref, l_ij)
        _w = w < 1e-5
        z[_w] = 0.0
        c6_ij = z / w.clamp(min=1e-5)
        return c6_ij

    def forward(self, data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        c6ij = self._calc_c6ij(data)

        rr = self.r4r2[data["numbers"]]
        rr_i, rr_j = nbops.get_ij(rr, data, suffix="_lr")
        rrij = 3 * rr_i * rr_j
        rrij = nbops.mask_ij_(rrij, data, 1.0, suffix="_lr")
        r0ij = self.a1 * rrij.sqrt() + self.a2

        ops.lazy_calc_dij_lr(data)
        d_ij = data["d_ij_lr"] * constants.Bohr_inv
        e_ij = c6ij * (
            self.s6 / (d_ij.pow(6) + r0ij.pow(6))
            + self.s8 * rrij / (d_ij.pow(8) + r0ij.pow(8))
        )
        e = -constants.half_Hartree * nbops.mol_sum(e_ij.sum(-1), data)

        if self.key_out in data:
            data[self.key_out] = data[self.key_out] + e
        else:
            data[self.key_out] = e
        return data
