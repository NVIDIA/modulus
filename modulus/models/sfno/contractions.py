# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# TODO cleanup
import torch

# # Helper routines for FNOs

# @torch.jit.script
# def compl_contract2d_fwd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
#     ac = torch.view_as_complex(a)
#     bc = torch.view_as_complex(b)
#     res = torch.einsum("bixy,kixy->bkxy", ac, bc)
#     return torch.view_as_real(res)

# @torch.jit.script
# def compl_contract_fwd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
#     ac = torch.view_as_complex(a)
#     bc = torch.view_as_complex(b)
#     res = torch.einsum("bin,kin->bkn", ac, bc)
#     return torch.view_as_real(res)

# helper routines for non-linear (S)FNOs


@torch.jit.script
def compl_mul1d_fwd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: # pragma: no cover
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    resc = torch.einsum("bix,io->box", ac, bc)
    res = torch.view_as_real(resc)
    return res


@torch.jit.script
def compl_muladd1d_fwd(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor: # pragma: no cover
    tmpcc = torch.view_as_complex(compl_mul1d_fwd(a, b))
    cc = torch.view_as_complex(c)
    return torch.view_as_real(tmpcc + cc)


@torch.jit.script
def compl_mul2d_fwd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: # pragma: no cover
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    resc = torch.einsum("bixy,io->boxy", ac, bc)
    res = torch.view_as_real(resc)
    return res


@torch.jit.script
def compl_muladd2d_fwd(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor: # pragma: no cover
    tmpcc = torch.view_as_complex(compl_mul2d_fwd(a, b))
    cc = torch.view_as_complex(c)
    return torch.view_as_real(tmpcc + cc)


# # for the real-valued case:
# @torch.jit.script
# def compl_mul1d_fwd_r(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
#     res = torch.einsum("bix,io->box", a, b)
#     return res

# @torch.jit.script
# def compl_muladd1d_fwd_r(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
#     tmp = compl_mul1d_fwd_r(a, b)
#     return tmp + c

# Helper routines for FFT MLPs

# # for the real-valued case:
# @torch.jit.script
# def compl_mul2d_fwd_r(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
#     res = torch.einsum("bixy,io->boxy", a, b)
#     return res

# @torch.jit.script
# def compl_muladd2d_fwd_r(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
#     tmp = compl_mul2d_fwd_c(a, b)
#     return torch.view_as_real(tmp + c)


@torch.jit.script
def _contract_localconv_fwd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: # pragma: no cover
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    resc = torch.einsum("bixy,iox->boxy", ac, bc)
    res = torch.view_as_real(resc)
    return res


# @torch.jit.script
# def _contractadd_localconv_fwd(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
#     tmpcc = torch.view_as_complex(_contract_localconv_fwd(a, b))
#     cc = torch.view_as_complex(c)
#     return torch.view_as_real(tmpcc + cc)


@torch.jit.script
def _contract_blockconv_fwd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: # pragma: no cover
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    resc = torch.einsum("bim,imn->bin", ac, bc)
    res = torch.view_as_real(resc)
    return res


@torch.jit.script
def _contractadd_blockconv_fwd(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor: # pragma: no cover
    tmpcc = torch.view_as_complex(_contract_blockconv_fwd(a, b))
    cc = torch.view_as_complex(c)
    return torch.view_as_real(tmpcc + cc)


# for the experimental layer
@torch.jit.script
def compl_exp_mul2d_fwd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: # pragma: no cover
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    resc = torch.einsum("bixy,xio->boxy", ac, bc)
    res = torch.view_as_real(resc)
    return res


@torch.jit.script
def compl_exp_muladd2d_fwd(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor: # pragma: no cover
    tmpcc = torch.view_as_complex(compl_exp_mul2d_fwd(a, b))
    cc = torch.view_as_complex(c)
    return torch.view_as_real(tmpcc + cc)


@torch.jit.script
def real_mul2d_fwd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: # pragma: no cover
    res = torch.einsum("bixy,io->boxy", a, b)
    return res


@torch.jit.script
def real_muladd2d_fwd(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor
) -> torch.Tensor: # pragma: no cover
    res = real_mul2d_fwd(a, b) + c
    return res


# new contractions set to replace older ones. We use complex


@torch.jit.script
def _contract_diagonal(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: # pragma: no cover
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    resc = torch.einsum("bixy,ioxy->boxy", ac, bc)
    res = torch.view_as_real(resc)
    return res


@torch.jit.script
def _contract_dhconv(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: # pragma: no cover
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    resc = torch.einsum("bixy,iox->boxy", ac, bc)
    res = torch.view_as_real(resc)
    return res


@torch.jit.script
def _contract_sep_diagonal(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: # pragma: no cover
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    resc = torch.einsum("bixy,ixy->boxy", ac, bc)
    res = torch.view_as_real(resc)
    return res


@torch.jit.script
def _contract_sep_dhconv(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: # pragma: no cover
    ac = torch.view_as_complex(a)
    bc = torch.view_as_complex(b)
    resc = torch.einsum("bixy,ix->boxy", ac, bc)
    res = torch.view_as_real(resc)
    return res
