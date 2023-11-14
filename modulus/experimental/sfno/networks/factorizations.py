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

from functools import partial

import tensorly as tl
tl.set_backend('pytorch')

from modulus.experimental.sfno.networks.contractions import _contract_diagonal, _contract_dhconv, _contract_sep_diagonal, _contract_sep_dhconv
from modulus.experimental.sfno.networks.contractions import _contract_diagonal_real, _contract_dhconv_real, _contract_sep_diagonal_real, _contract_sep_dhconv_real


from tltorch.factorized_tensors.core import FactorizedTensor

einsum_symbols = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'


def _contract_dense(x, weight, separable=False, operator_type='diagonal'):
    order = tl.ndim(x)
    # batch-size, in_channels, x, y...
    x_syms = list(einsum_symbols[:order])

    # in_channels, out_channels, x, y...
    weight_syms = list(x_syms[1:]) # no batch-size

    # batch-size, out_channels, x, y...
    if separable:
        out_syms = [x_syms[0]] + list(weight_syms)
    else:
        weight_syms.insert(1, einsum_symbols[order]) # outputs
        out_syms = list(weight_syms)
        out_syms[0] = x_syms[0] 

    if operator_type == 'diagonal':
        pass
    elif operator_type == 'block-diagonal':
        weight_syms.insert(-1, einsum_symbols[order+1])
        out_syms[-1] = weight_syms[-2]
    elif operator_type == 'dhconv':
        weight_syms.pop()
    else:
        raise ValueError(f"Unkonw operator type {operator_type}")

    eq= ''.join(x_syms) + ',' + ''.join(weight_syms) + '->' + ''.join(out_syms)

    if not torch.is_tensor(weight):
        weight = weight.to_tensor()

    res = tl.einsum(eq, x, weight).contiguous()

    return res

def _contract_cp(x, cp_weight, separable=False, operator_type='diagonal'):
    order = tl.ndim(x)

    x_syms = str(einsum_symbols[:order])
    rank_sym = einsum_symbols[order]
    out_sym = einsum_symbols[order+1]
    out_syms = list(x_syms)

    if separable:
        factor_syms = [einsum_symbols[1]+rank_sym] #in only
    else:
        out_syms[1] = out_sym
        factor_syms = [einsum_symbols[1]+rank_sym, out_sym+rank_sym] #in, out
    
    factor_syms += [xs+rank_sym for xs in x_syms[2:]] #x, y, ...

    if operator_type == 'diagonal':
        pass
    elif operator_type == 'block-diagonal':
        out_syms[-1] = einsum_symbols[order+2]
        factor_syms += [out_syms[-1] + rank_sym]
    elif operator_type == 'dhconv':
        factor_syms.pop()
    else:
        raise ValueError(f"Unkonw operator type {operator_type}")

    eq = x_syms + ',' + rank_sym + ',' + ','.join(factor_syms) + '->' + ''.join(out_syms)

    res = tl.einsum(eq, x, cp_weight.weights, *cp_weight.factors).contiguous()

    return res
 

def _contract_tucker(x, tucker_weight, separable=False, operator_type='diagonal'):
    order = tl.ndim(x)

    x_syms = str(einsum_symbols[:order])
    out_sym = einsum_symbols[order]
    out_syms = list(x_syms)
    if separable:
        core_syms = einsum_symbols[order+1:2*order]
        factor_syms = [xs+rs for (xs, rs) in zip(x_syms[1:], core_syms)] #x, y, ...

    else:
        core_syms = einsum_symbols[order+1:2*order+1]
        out_syms[1] = out_sym
        factor_syms = [einsum_symbols[1]+core_syms[0], out_sym+core_syms[1]] #out, in
        factor_syms += [xs+rs for (xs, rs) in zip(x_syms[2:], core_syms[2:])] #x, y, ...

    if operator_type == 'diagonal':
        pass
    elif operator_type == 'block-diagonal':
        raise NotImplementedError(f"Operator type {operator_type} not implemented for Tucker")
    else:
        raise ValueError(f"Unkonw operator type {operator_type}")

    eq = x_syms + ',' + core_syms + ',' + ','.join(factor_syms) + '->' + ''.join(out_syms)

    res = tl.einsum(eq, x, tucker_weight.core, *tucker_weight.factors).contiguous()

    return res

def _contract_tt(x, tt_weight, separable=False, operator_type='diagonal'):
    order = tl.ndim(x)

    x_syms = list(einsum_symbols[:order])
    weight_syms = list(x_syms[1:]) # no batch-size

    if not separable:
        weight_syms.insert(1, einsum_symbols[order]) # outputs
        out_syms = list(weight_syms)
        out_syms[0] = x_syms[0]
    else:
        out_syms = list(x_syms)
    
    if operator_type == 'diagonal':
        pass
    elif operator_type == 'block-diagonal':
        weight_syms.insert(-1, einsum_symbols[order+1])
        out_syms[-1] = weight_syms[-2]
    elif operator_type == 'dhconv':
        weight_syms.pop()
    else:
        raise ValueError(f"Unkonw operator type {operator_type}")

    rank_syms = list(einsum_symbols[order+2:])
    tt_syms = []
    for i, s in enumerate(weight_syms):
        tt_syms.append([rank_syms[i], s, rank_syms[i+1]])
    eq = ''.join(x_syms) + ',' + ','.join(''.join(f) for f in tt_syms) + '->' + ''.join(out_syms)

    res = tl.einsum(eq, x, *tt_weight.factors).contiguous()

    return res

# jitted PyTorch contractions:
def _contract_dense_pytorch(x, weight, separable=False, operator_type='diagonal', complex=True):

    # make sure input is contig
    x = x.contiguous()

    if separable:
        if operator_type == 'diagonal':
            if complex:
                x = _contract_sep_diagonal(x, weight)
            else:
                x = _contract_sep_diagonal_real(x, weight)
        elif operator_type == 'dhconv':
            if complex:
                x = _contract_sep_dhconv(x, weight)
            else:
                x = _contract_sep_dhconv_real(x, weight)
        else:
            raise ValueError(f"Unkonw operator type {operator_type}")
    else:
        if operator_type == 'diagonal':
            if complex:
                x = _contract_diagonal(x, weight)
            else:
                x = _contract_diagonal_real(x, weight)
        elif operator_type == 'dhconv':
            if complex:
                x = _contract_dhconv(x, weight)
            else:
                x = _contract_dhconv_real(x, weight)
        else:
            raise ValueError(f"Unkonw operator type {operator_type}")

    # make contiguous
    x = x.contiguous()
    return x


def _contract_dense_reconstruct(x, weight, separable=False, operator_type='diagonal', complex=True):
    """Contraction for dense tensors, factorized or not"""
    if not torch.is_tensor(weight):
        weight = weight.to_tensor()
        #weight = torch.view_as_real(weight)

    return _contract_dense_pytorch(x, weight, separable=separable, operator_type=operator_type, complex=complex)


def get_contract_fun(weight, implementation='reconstructed', separable=False, operator_type='diagonal', complex=True):
    """Generic ND implementation of Fourier Spectral Conv contraction

    Parameters
    ----------
    weight : tensorly-torch's FactorizedTensor
    implementation : {'reconstructed', 'factorized'}, default is 'reconstructed'
        whether to reconstruct the weight and do a forward pass (reconstructed)
        or contract directly the factors of the factorized weight with the input (factorized)

    Returns
    -------
    function : (x, weight) -> x * weight in Fourier space
    """
    if implementation == 'reconstructed':
        handle = partial(_contract_dense_reconstruct, separable=separable, complex=complex, operator_type=operator_type)
        return handle
    elif implementation == 'factorized':
        if torch.is_tensor(weight):
            handle = partial(_contract_dense_pytorch, separable=separable, complex=complex, operator_type=operator_type)
            return handle
        elif isinstance(weight, FactorizedTensor):
            if weight.name.lower() == 'complexdense' or weight.name.lower() == 'dense':
                return _contract_dense
            elif weight.name.lower() == 'complextucker':
                return _contract_tucker
            elif weight.name.lower() == 'complextt':
                return _contract_tt
            elif weight.name.lower() == 'complexcp':
                return _contract_cp
            else:
                raise ValueError(f'Got unexpected factorized weight type {weight.name}')
        else:
            raise ValueError(f'Got unexpected weight type of class {weight.__class__.__name__}')
    else:
        raise ValueError(f'Got {implementation=}, expected "reconstructed" or "factorized"')
