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
import argparse
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from torch.cuda import amp

sys.path.append(os.path.join("/opt", "makani"))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  

from modulus.experimental.sfno.utils import comm  

from torch_harmonics import RealSHT as RealSphericalHarmonicTransform
from torch_harmonics import InverseRealSHT as InverseRealSphericalHarmonicTransform

# profile stuff
from ctypes import cdll
libcudart = cdll.LoadLibrary('libcudart.so')

def cudaProfilerStart(enabled=True):
    if enabled:
        libcudart.cudaProfilerStart()

def cudaProfilerStop(enabled=True):
    if enabled:
        libcudart.cudaProfilerStop()      


def main(args, verify):
    # parameters
    num_layers = args.num_layers
    enable_jit = args.enable_jit
    num_warmup = 10
    num_steps = 10
    batch_size = args.batch_size
    C = args.embed_dim
    H = args.height
    W = args.width
        
    # set device
    device = torch.device(f"cuda:0")
    
    # tune
    torch.cuda.manual_seed(333)
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True

    # set autograd
    #torch.autograd.set_detect_anomaly(True)
    
    # blocks
    model_loc = nn.Sequential(*[RealSphericalHarmonicTransform(H, W, grid="equiangular"),
                                InverseRealSphericalHarmonicTransform(H, W, grid="equiangular")]).to(device)

    if enable_jit:
        model_loc = torch.jit.script(model_loc)
    
    #input
    inp_loc = torch.empty((batch_size, C, H, W), dtype=torch.float32, device=device, requires_grad=True)
    
    # check FW pass
    for _ in range(num_warmup):
        with torch.no_grad():
            inp_loc.normal_()
        model_loc.zero_grad(set_to_none=True) 
        out_loc = model_loc(inp_loc)
        l_loc = torch.mean(out_loc)
        l_loc.backward()

    # clean up the cuda stuff:
    max_mem_gb = torch.cuda.max_memory_allocated(device=device) / (1024. * 1024. * 1024.)
    print(f"Memory high watermark during scaffolding: {max_mem_gb} GB")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device=device) 

    cudaProfilerStart(enabled=args.enable_profiling)
    with torch.autograd.profiler.emit_nvtx(enabled=args.enable_profiling, record_shapes=False):
        start = time.perf_counter_ns()
        for step in range(num_steps):
            torch.cuda.nvtx.range_push(f"step {step}") 
            model_loc.zero_grad(set_to_none=True) 
            out_loc = model_loc(inp_loc)
            l_loc = torch.mean(out_loc)
            l_loc.backward()
            torch.cuda.nvtx.range_pop() 
        if dist.is_initialized():
            dist.barrier(device_ids=[device.index], group=comm.get_model_parallel_group())
        end = time.perf_counter_ns()
    cudaProfilerStop(enabled=args.enable_profiling) 
    
    # print results
    print(f"Time per step local: {(end-start)*10**(-6)/float(num_steps)} ms")
    max_mem_gb = torch.cuda.max_memory_allocated(device=device) / (1024. * 1024. * 1024.)
    print(f"Memory high watermark: {max_mem_gb} GB") 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1, type=int, help="Batch Size")
    parser.add_argument("--num_layers", default=1, type=int, help="Number of Layers")  
    parser.add_argument("--embed_dim", default=64, type=int, help="Embedding dim")
    parser.add_argument("--height", default=721, type=int, help="Input Height")
    parser.add_argument("--width", default=1440, type=int, help="Input Width")
    parser.add_argument("--enable_jit", action="store_true")
    parser.add_argument("--enable_profiling", action="store_true")
    args = parser.parse_args()  
    
    main(args, verify = True)
