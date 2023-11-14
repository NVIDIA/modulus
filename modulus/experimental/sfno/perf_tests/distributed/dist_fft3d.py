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

import os
import sys
import time
import types
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.cuda import amp

sys.path.append(os.path.join("/opt", "ERA5_wind"))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modulus.experimental.sfno.utils import comm
from modulus.experimental.sfno.utils.YParams import ParamsBase

from modulus.experimental.sfno.mpu.mappings import gather_from_parallel_region, scatter_to_parallel_region, reduce_from_parallel_region
from modulus.experimental.sfno.mpu.fft3d import RealFFT3, InverseRealFFT3, DistributedRealFFT3, DistributedInverseRealFFT3


def main(args, verify):
    # parameters
    compute_grads = True
    verbose = True
    fft_type = "fft"
    num_warmup = 10
    num_steps = 10
    h_parallel_size = args.h_parallel_size
    w_parallel_size = args.w_parallel_size

    # common parameters
    B, C, D, H, W = 1, 8, 20, 720, 1440
    Hloc = (H + h_parallel_size - 1) // h_parallel_size
    Wloc = (W + w_parallel_size - 1) // w_parallel_size
    Hpad = h_parallel_size * Hloc - H
    Wpad = w_parallel_size * Wloc - W
        
    # initialize comms
    params = ParamsBase()
    params.update_params(dict(wireup_info="env",
                              wireup_store="tcp",
                              log_to_screen=True,
                              model_parallel_sizes=[h_parallel_size, w_parallel_size],
                              model_parallel_names=["h", "w"]))
    comm.init(params, verbose=True)
    comm_rank = comm.get_world_rank()
    comm_model_parallel_size = comm.get_size("model")
    comm_model_parallel_rank = comm.get_rank("model")
    comm_h_rank = comm.get_rank("h")
    comm_w_rank = comm.get_rank("w")
    comm_local_rank = comm.get_local_rank()
    
    # set device
    device = torch.device(f"cuda:{comm_local_rank}")

    if comm_model_parallel_rank == 0:
        print(f"Running {fft_type} for ({B}, {C}, {D}, {H}, {W}) on {comm_model_parallel_size} ranks")
    
    # tune
    torch.manual_seed(333)
    torch.cuda.manual_seed(333)
    torch.cuda.set_device(device)

    # setup transforms
    forward_transform_local = RealFFT3(nd=D, nh=H, nw=W).to(device)
    forward_transform_dist = DistributedRealFFT3(nd=D, nh=H, nw=W).to(device)
    Lloc = (forward_transform_dist.lhpad + forward_transform_dist.lhmax) // h_parallel_size
    Mloc = (forward_transform_dist.lwpad + forward_transform_dist.lwmax) // w_parallel_size

    # set up inputs
    inp_full = torch.randn((B, C, D, H, W), dtype=torch.float32, device=device)

    # pad
    with torch.no_grad():
        inp_pad = F.pad(inp_full, (0, Wpad, 0, Hpad))
        
        # split in W
        inp_local = scatter_to_parallel_region(inp_pad, -1, "w")

        # split in H
        inp_local = scatter_to_parallel_region(inp_local, -2, "h")

    #############################################################
    # local transform
    #############################################################
    # FWD pass
    inp_full.requires_grad = True
    out_full = forward_transform_local(inp_full).contiguous()

    # create input grad
    with torch.no_grad():
        # create full grad
        ograd_full = torch.randn_like(out_full)

    # BWD pass
    out_full.backward(ograd_full)
    igrad_full = inp_full.grad.clone()
    
    #############################################################
    # distributed transform
    ############################################################# 
    # FWD pass
    inp_local.requires_grad = True
    out_local = forward_transform_dist(inp_local).contiguous()

    # create output grad
    with torch.no_grad():
        # pad
        ograd_pad = F.pad(ograd_full, (0, forward_transform_dist.lwpad, 0, forward_transform_dist.lhpad))

        # split in M
        ograd_local = scatter_to_parallel_region(ograd_pad, -1, "w")

        # split in L
        ograd_local = scatter_to_parallel_region(ograd_local, -2, "h")

    # BWD pass
    out_local.backward(ograd_local)
    igrad_local = inp_local.grad.clone()
    
    # gather the local data
    with torch.no_grad():
        # FWD data
        # gather in W
        if w_parallel_size > 1:
            out_full_gather = gather_from_parallel_region(out_local, -1, "w")
            out_full_gather = out_full_gather[..., :forward_transform_dist.lwmax].contiguous()
        else:
            out_full_gather = out_local

        # gather in h
        if h_parallel_size > 1:
            out_full_gather = gather_from_parallel_region(out_full_gather, -2, "h")
            out_full_gather = out_full_gather[..., :forward_transform_dist.lhmax, :].contiguous()

        # BWD data
        # gather in W
        if w_parallel_size > 1:
            igrad_full_gather = gather_from_parallel_region(igrad_local, -1, "w")
            igrad_full_gather = igrad_full_gather[..., :W].contiguous()
        else:
            igrad_full_gather = igrad_local

        # gather in h
        if h_parallel_size > 1:
            igrad_full_gather = gather_from_parallel_region(igrad_full_gather, -2, "h")
            igrad_full_gather = igrad_full_gather[..., :H, :].contiguous()
            
    #############################################################
    # compare results
    #############################################################
    # FWD pass
    isum = reduce_from_parallel_region(inp_local.abs().sum(), "model").item()
    imax = gather_from_parallel_region(inp_local.max().reshape((1,)), 0, "model").max().item()
    imin = gather_from_parallel_region(inp_local.min().reshape((1,)), 0, "model").min().item()
    if comm_rank == 0:
        print(f"Comparing forward pass results:")
        print(f"Local In : sum={isum}, max={imax}, min={imin}")
        print(f"Distr In : sum={inp_full.abs().sum().item()}, max={inp_full.max().item()}, min={inp_full.min().item()}")
        print(f"Local Out: sum={out_full.abs().sum().item()}, max={out_full.abs().max().item()}, min={out_full.abs().min().item()}")
        print(f"Distr Out: sum={out_full_gather.abs().sum().item()}, max={out_full_gather.abs().max().item()}, min={out_full_gather.abs().min().item()}")
        diff = (out_full-out_full_gather).abs()
        print(f"Out Difference: abs={diff.sum().item()}, rel={diff.sum().item() / (0.5*(out_full.abs().sum() + out_full_gather.abs().sum()))}, max={diff.abs().max().item()}")
        print("")

    # BWD pass
    isum = reduce_from_parallel_region(ograd_local.abs().sum(), "model").item()
    imax = gather_from_parallel_region(ograd_local.abs().max().reshape((1,)), 0, "model").max().item()
    imin = gather_from_parallel_region(ograd_local.abs().min().reshape((1,)), 0, "model").min().item()        
    if comm_rank == 0:
        print(f"Comparing backward pass results:")
        print(f"Local Grad In : sum={ograd_full.abs().sum().item()}, max={ograd_full.abs().max().item()}, min={ograd_full.abs().min().item()}")
        print(f"Distr Grad In : sum={isum}, max={imax}, min={imin}")
        print(f"Local Grad Out: sum={igrad_full.abs().sum().item()}, max={igrad_full.abs().max().item()}, min={igrad_full.abs().min().item()}")
        print(f"Distr Grad Out: sum={igrad_full_gather.abs().sum().item()}, max={igrad_full_gather.abs().max().item()}, min={igrad_full_gather.abs().min().item()}")
        diff = (igrad_full-igrad_full_gather).abs()
        print(f"Grad Out Difference: abs={diff.sum().item()}, rel={diff.sum().item() / (0.5*(igrad_full.abs().sum() + igrad_full_gather.abs().sum()))}, max={diff.abs().max().item()}")

    # wait for everybody to finish
    if dist.is_initialized():
        dist.barrier(device_ids=[device.index])
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h_parallel_size", default=1, type=int, help="Parallelism in H direction")
    parser.add_argument("--w_parallel_size", default=1, type=int, help="Parallelism in W direction")
    parser.add_argument("--nmodes", default=None, type=int, help="Number of modes to use")
    args = parser.parse_args()
    
    main(args, verify = True)
