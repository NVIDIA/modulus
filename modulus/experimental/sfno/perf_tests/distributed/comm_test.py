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
    enable_amp = True
    orthogonal_gather = False
    num_warmup = 10
    num_steps = 100
    model_parallel_size = args.model_parallel_size
    tensor_size = 84873728
    profile_ranks = [0, 1, 2, 3]
    
    # initialize comms
    params = types.SimpleNamespace(wireup_info="env",
                                   wireup_store="tcp",
                                   log_to_screen=True,
                                   model_parallel_sizes=[model_parallel_size],
                                   model_parallel_names=["mp"])
    comm.init(params, verbose=True)
    comm_model_parallel_size = comm.get_size("mp")
    comm_model_parallel_rank = comm.get_rank("mp")
    comm_local_rank = comm.get_local_rank()
    
    # set device
    device = torch.device(f"cuda:{comm_local_rank}")
    
    # tune
    torch.cuda.manual_seed(333)
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = True
    profile_rank = comm.get_world_rank() in profile_ranks

    # tensors
    B, C, H, W = 1, 1024, 180, 360
    tensor1 = torch.zeros((B, C, H, W), dtype=torch.float32, device=device)
    tensor1list = [torch.empty_like(tensor1) for _ in range(comm_model_parallel_size)]
    tensor1list[comm_model_parallel_rank] = tensor1
    tensor2 = torch.zeros((tensor_size), dtype=torch.float32, device=device)

    if comm.get_world_rank() == 0:
        print("Warming up")
    for _ in range(num_warmup):
        if orthogonal_gather:
            dist.all_gather(tensor1list, tensor1, group=comm.get_group("mp"))
        dist.all_reduce(tensor2, group=comm.get_group("data"))

    if comm.get_world_rank() == 0:
        print("Running")
    dist.barrier(device_ids=[device.index])
    cudaProfilerStart(enabled=profile_rank)
    start = time.perf_counter_ns()
    for _ in range(num_steps):
        if orthogonal_gather:
            dist.all_gather(tensor1list, tensor1, group=comm.get_group("mp"))
        dist.all_reduce(tensor2, group=comm.get_group("data"))
    dist.barrier(device_ids=[device.index])
    end = time.perf_counter_ns()
    cudaProfilerStop(enabled=profile_rank)
    if comm.get_world_rank() == 0:
        print("done")

    if comm.get_world_rank() == 0:
        print(f"Runtime: {(end-start)*10**(-9):.2f} s ({(end-start)/num_steps*10**(-6):.2f} ms per step)")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_parallel_size", default=1, type=int, help="Model parallelism dimension")
    args = parser.parse_args()  
    
    main(args, verify = True)
