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
import torch.nn.functional as F
import torch.distributed as dist

from modulus.experimental.sfno.utils import comm

from torch._utils import _flatten_dense_tensors


def get_memory_format(tensor):
    if tensor.is_contiguous(memory_format=torch.channels_last):
        return torch.channels_last
    else:
        return torch.contiguous_format

def gather_uneven(tensor, dim, comm_name):

    if comm.get_size(comm_name) == 1:
        return tensor
    
    # gather dims
    dim_tensor = torch.tensor([tensor.shape[dim]], dtype=torch.int, device=tensor.device)
    dim_list = [torch.empty_like(dim_tensor) for _ in range(comm.get_size(comm_name))]
    dim_list[comm.get_rank(comm_name)] = dim_tensor
    dist.all_gather(dim_list, dim_tensor, group=comm.get_group(comm_name))

    # gather tensor
    gathered_shape = list(tensor.shape)
    tensor_list = []
    for rshape in dim_list:
        gathered_shape[dim] = rshape.item()
        tensor_list.append(torch.empty(gathered_shape, dtype=tensor.dtype, device=tensor.device))

    tensor_list[comm.get_rank(comm_name)] = tensor
    dist.all_gather(tensor_list, tensor, group=comm.get_group(comm_name))

    # concatenate
    result = torch.cat(tensor_list, dim=dim)
    
    return result

def sync_params(model, mode='broadcast'):
    """Helper routine to ensure shared weights are the same after initialization"""
    
    def _sync_param(param, comm_group, mode):
        if comm.get_size(comm_group) > 1:
            if mode == 'broadcast':
                is_complex = param.is_complex()
                if is_complex:
                    param_real = torch.view_as_real(param).clone()
                else:
                    param_real  = param.clone()
                #tlist = [torch.empty_like(param_real) for x in range(comm.get_size(comm_group))]
                #tlist[comm.get_rank(comm_group)] = param_real
                # gather all weights in the comm group
                dist.broadcast(param_real, src=comm.get_root(comm_group), group=comm.get_group(comm_group))
                # use weight of rank 0
                # important to use copy here otherwise the handle gets detaches from the optimizer
                if is_complex:
                    param.copy_(torch.view_as_complex(param_real))
                else:
                    param.copy_(param_real)
            elif mode == 'mean':
                is_complex = param.is_complex()
                if is_complex:
                    dist.all_reduce(torch.view_as_real(param), op=dist.ReduceOp.AVG, group=comm.get_group(comm_group), async_op=False)
                else:
                    dist.all_reduce(param, op=dist.ReduceOp.AVG, group=comm.get_group(comm_group), async_op=False)
            else:
                raise ValueError(f"Unknown weight synchronization mode {mode}")

        return

    with torch.no_grad():
        # distributed sync step
        for param in model.parameters():

            # share along data dim
            _sync_param(param, "data", mode)
            
            if not hasattr(param, "is_shared_mp"):
                param.is_shared_mp = ["model"]
            
            for comm_group in param.is_shared_mp:
                _sync_param(param, comm_group, mode)

    return

# def gather_params(model):
#     """Helper routine to ensure shared weights are the same after initialization"""

#     non_singleton_group_names = [x for x in comm.get_names() if (comm.get_size(x) > 1) and not (x in ["data", "model", "spatial"])]

#     with torch.no_grad():
#         # distributed sync step
#         for param in model.parameters():
            
#             # weights can only be sharded if they have "is_shared_mp" member
#             if hasattr(param, "is_shared_mp"):
#                 for comm_group in param.is_shared_mp:
#                     if comm.get_size(comm_group) > 1:
#                         tlist = [torch.empty_like(param) for x in range(comm.get_size(comm_group))]
#                         tlist[comm.get_rank(comm_group)] = param
#                         # gather all weights in the comm group
#                         dist.all_gather(tlist, param, group=comm.get_group(comm_group))
#                         # use weight of rank 0
#                         # important to use copy here otherwise the handle gets detaches from the optimizer
#                         param.copy_(tlist[0])

#                         else:
#                             raise ValueError(f"Unknown weight synchronization mode {mode}")


def pad_helper(tensor, dim, new_size, mode="zero"):
    ndim = tensor.ndim
    dim = (dim + ndim) % ndim
    ndim_pad = ndim - dim
    output_shape = [0 for _ in range(2*ndim_pad)]
    orig_size = tensor.shape[dim]
    output_shape[1] = new_size - orig_size
    tensor_pad = F.pad(tensor, output_shape, mode='constant', value=0.)
    
    if mode == "conj":
        lhs_slice = [slice(0,x) if idx != dim else slice(orig_size, new_size) for idx,x in enumerate(tensor.shape)]
        rhs_slice = [slice(0,x) if idx != dim else slice(1, output_shape[1]+1) for idx,x in enumerate(tensor.shape)]
        tensor_pad[lhs_slice] = torch.flip(torch.conj(tensor_pad[rhs_slice]), dims=[dim])
        
    return tensor_pad


def truncate_helper(tensor, dim, new_size):
    input_format = get_memory_format(tensor)
    ndim = tensor.ndim
    dim = (dim + ndim) % ndim
    output_slice = [slice(0,x) if idx != dim else slice(0,new_size) for idx,x in enumerate(tensor.shape)]
    tensor_trunc = tensor[output_slice].contiguous(memory_format=input_format)
    
    return tensor_trunc


def split_tensor_along_dim(tensor, dim, num_chunks):
    assert dim < tensor.dim(), f"Error, tensor dimension is {tensor.dim()} which cannot be split along {dim}"
    assert (tensor.shape[dim] % num_chunks == 0), f"Error, cannot split dim {dim} evenly. Dim size is \
                                                  {tensor.shape[dim]} and requested numnber of splits is {num_chunks}"
    chunk_size = tensor.shape[dim] // num_chunks
    tensor_list = torch.split(tensor, chunk_size, dim=dim)
    
    return tensor_list


# distributed primitives
def _transpose(tensor, dim0, dim1, group=None, async_op=False):
    # get input format
    input_format = get_memory_format(tensor)
    
    # get comm params
    comm_size = dist.get_world_size(group=group)

    # split and local transposition
    split_size = tensor.shape[dim0] // comm_size
    x_send = [y.contiguous(memory_format=input_format) for y in torch.split(tensor, split_size, dim=dim0)]
    x_recv = [torch.empty_like(x_send[0]) for _ in range(comm_size)]
    
    # global transposition
    req = dist.all_to_all(x_recv, x_send, group=group, async_op=async_op)
    
    return x_recv, req 


def _reduce(input_, use_fp32=True, group=None):
    """All-reduce the input tensor across model parallel group."""

    # Bypass the function if we are using only 1 GPU.
    if dist.get_world_size(group=group) == 1:
        return input_
    
    # All-reduce.
    if use_fp32:
        dtype = input_.dtype
        inputf_ = input_.float()
        dist.all_reduce(inputf_, group=group)
        input_ = inputf_.to(dtype)
    else:
        dist.all_reduce(input_, group=group)
        
    return input_


def _split(input_, dim_, group=None):
    """Split the tensor along its last dimension and keep the corresponding slice."""
    # get input format
    input_format = get_memory_format(input_)
    
    # Bypass the function if we are using only 1 GPU.
    comm_size = dist.get_world_size(group=group)
    if comm_size == 1:
        return input_
    
    # Split along last dimension.
    input_list = split_tensor_along_dim(input_, dim_, comm_size)
    
    # Note: torch.split does not create contiguous tensors by default.
    rank = dist.get_rank(group=group)
    output = input_list[rank].contiguous(memory_format=input_format)
    
    return output

def _gather(input_, dim_, group=None):
    """Gather tensors and concatinate along the last dimension."""
    # get input format
    input_format = get_memory_format(input_) 

    comm_size = dist.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if comm_size==1:
        return input_

    # sanity checks
    assert(dim_ < input_.dim()), f"Error, cannot gather along {dim_} for tensor with {input_.dim()} dimensions."

    # Size and dimension.
    comm_rank = dist.get_rank(group=group)
    
    input_ = input_.contiguous(memory_format=input_format)
    tensor_list = [torch.empty_like(input_) for _ in range(comm_size)]
    tensor_list[comm_rank] = input_
    dist.all_gather(tensor_list, input_, group=group)
    
    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=dim_).contiguous(memory_format=input_format)
    
    return output
