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

import torch

@torch.jit.script
def _contract_rank(xc: torch.Tensor, wc: torch.Tensor, ac: torch.Tensor, bc: torch.Tensor) -> torch.Tensor:
    # return torch.einsum("bixy,ior,xr,yr->boxy", x, w, a, b)
    #xc = torch.view_as_complex(x)
    #wc = w #torch.view_as_complex(w)
    #ac = torch.view_as_complex(a)
    #bc = torch.view_as_complex(b)
    resc = torch.einsum("bixy,ior,xr,yr->boxy", xc, wc, ac, bc)
    #res = torch.view_as_real(resc)
    return resc

# # Helper routines for FNOs
@torch.jit.script
def compl_mul1d_fwd(ac: torch.Tensor, bc: torch.Tensor) -> torch.Tensor:
    #ac = torch.view_as_complex(a)
    #bc = torch.view_as_complex(b)
    resc = torch.einsum("bix,io->box", ac, bc)
    #res = torch.view_as_real(resc)
    return resc

@torch.jit.script
def compl_muladd1d_fwd(ac: torch.Tensor, bc: torch.Tensor, cc: torch.Tensor) -> torch.Tensor:
    tmpcc = compl_mul1d_fwd(ac, bc)
    #cc = torch.view_as_complex(c)
    return tmpcc + cc

@torch.jit.script
def compl_mul2d_fwd(ac: torch.Tensor, bc: torch.Tensor) -> torch.Tensor:
    #ac = torch.view_as_complex(a)
    #bc = torch.view_as_complex(b)
    resc = torch.einsum("bixy,io->boxy", ac, bc)
    #res = torch.view_as_real(resc)
    return resc

@torch.jit.script
def compl_muladd2d_fwd(ac: torch.Tensor, bc: torch.Tensor, cc: torch.Tensor) -> torch.Tensor:
    tmpcc = compl_mul2d_fwd(ac, bc)
    #cc = torch.view_as_complex(c)
    return tmpcc + cc

@torch.jit.script
def _contract_localconv_fwd(ac: torch.Tensor, bc: torch.Tensor) -> torch.Tensor:
    #ac = torch.view_as_complex(a)
    #bc = torch.view_as_complex(b)
    resc = torch.einsum("bixy,iox->boxy", ac, bc)
    #res = torch.view_as_real(resc) 
    return resc

@torch.jit.script
def _contract_blockconv_fwd(ac: torch.Tensor, bc: torch.Tensor) -> torch.Tensor:
    #ac = torch.view_as_complex(a)
    #bc = torch.view_as_complex(b)
    resc = torch.einsum("bim,imn->bin", ac, bc)
    #res = torch.view_as_real(resc)
    return resc

@torch.jit.script
def _contractadd_blockconv_fwd(ac: torch.Tensor, bc: torch.Tensor, cc: torch.Tensor) -> torch.Tensor:
    tmpcc = _contract_blockconv_fwd(ac, bc)
    #cc = torch.view_as_complex(c)
    return tmpcc + cc

# for the experimental layer
@torch.jit.script
def compl_exp_mul2d_fwd(ac: torch.Tensor, bc: torch.Tensor) -> torch.Tensor:
    #ac = torch.view_as_complex(a)
    #bc = torch.view_as_complex(b)
    resc = torch.einsum("bixy,xio->boxy", ac, bc)
    #res = torch.view_as_real(resc)
    return resc

@torch.jit.script
def compl_exp_muladd2d_fwd(ac: torch.Tensor, bc: torch.Tensor, cc: torch.Tensor) -> torch.Tensor:
    tmpcc = compl_exp_mul2d_fwd(ac, bc)
    #cc = torch.view_as_complex(c)
    return tmpcc + cc

@torch.jit.script
def real_mul2d_fwd(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    res = torch.einsum("bixy,io->boxy", a, b)
    return res

@torch.jit.script
def real_muladd2d_fwd(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    res = real_mul2d_fwd(a, b) + c
    return res

# new contractions set to replace older ones. We use complex

@torch.jit.script
def _contract_diagonal(ac: torch.Tensor, bc: torch.Tensor) -> torch.Tensor:
    #ac = torch.view_as_complex(a)
    #bc = torch.view_as_complex(b)
    resc = torch.einsum("bixy,ioxy->boxy", ac, bc)
    #res = torch.view_as_real(resc)
    return resc

@torch.jit.script
def _contract_dhconv(ac: torch.Tensor, bc: torch.Tensor) -> torch.Tensor:
    #ac = torch.view_as_complex(a)
    #bc = torch.view_as_complex(b)
    resc = torch.einsum("bixy,iox->boxy", ac, bc)
    #res = torch.view_as_real(resc)
    return resc

@torch.jit.script
def _contract_sep_diagonal(ac: torch.Tensor, bc: torch.Tensor) -> torch.Tensor:
    #ac = torch.view_as_complex(a)
    #bc = torch.view_as_complex(b)
    resc = torch.einsum("bixy,ixy->boxy", ac, bc)
    #res = torch.view_as_real(resc)
    return resc

@torch.jit.script
def _contract_sep_dhconv(ac: torch.Tensor, bc: torch.Tensor) -> torch.Tensor:
    #ac = torch.view_as_complex(a)
    #bc = torch.view_as_complex(b)
    resc = torch.einsum("bixy,ix->boxy", ac, bc)
    #res = torch.view_as_real(resc)
    return resc

@torch.jit.script
def _contract_diagonal_real(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    res = torch.einsum("bixys,ioxy->boxys", a, b).contiguous()
    return res

@torch.jit.script
def _contract_dhconv_real(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    res = torch.einsum("bixys,iox->boxys", a, b).contiguous()
    return res

@torch.jit.script
def _contract_sep_diagonal_real(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    res = torch.einsum("bixys,ixy->boxys", a, b).contiguous()
    return res

@torch.jit.script
def _contract_sep_dhconv_real(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    res = torch.einsum("bixys,ix->boxys", a, b).contiguous()
    return res
