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

import gc
import os
import sys
import torch
from torch import nn
from torch.cuda import amp
import time
import apex
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

sys.path.append(os.path.join("/opt", "ERA5_wind"))

from modulus.experimental.sfno.mpu.layers import compl_mul_add_fwd, compl_mul_add_fwd_c


class ComplexMult(nn.Module):
    def __init__(self, num_blocks, block_size, hidden_size_factor, use_complex_kernels=True):
        super(ComplexMult, self).__init__()
        self.weight = nn.Parameter(torch.ones((num_blocks, block_size, block_size * hidden_size_factor, 2), dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros((num_blocks, block_size * hidden_size_factor, 1, 1, 2), dtype=torch.float32))
        self.mult_handle = compl_mul_add_fwd_c if use_complex_kernels else compl_mul_add_fwd

    def forward(self, inp):
        result = self.mult_handle(inp, self.weight, self.bias)
        return result

def main():
    torch.backends.cudnn.benchmark = True

    # set up process groups
    os.environ["LOCAL_RANK"] = str(int(os.getenv("WORLD_RANK", "0")) % torch.cuda.device_count())
    dist.init_process_group(backend = 'nccl',
                            rank = int(os.getenv("WORLD_RANK", "0")),
                            world_size = int(os.getenv("WORLD_SIZE", "1")))

    # input
    device=torch.device(f'cuda:{os.getenv("LOCAL_RANK", "0")}')
    enable_amp=True
    enable_graph=True
    use_complex_kernels=False

    # nn params
    batch_size=1
    num_blocks=8
    H=180
    W=360
    hidden_size=1024
    hidden_size_factor=1
    block_size=hidden_size // num_blocks

    # we need those
    inp = torch.ones((batch_size, hidden_size, H, W // 2 + 1, 2), dtype=torch.float32, device=device)
    model = ComplexMult(num_blocks, block_size, hidden_size_factor, use_complex_kernels=use_complex_kernels).to(device)
    optimizer = apex.optimizers.FusedAdam(model.parameters(), lr=1e-4)
    gscaler = amp.GradScaler(enabled=enable_amp)  

    # capture stuff
    capture_stream = torch.cuda.Stream()
    capture_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(capture_stream):
        model = DistributedDataParallel(model,
                                        device_ids = [device.index],
                                        output_device = device.index,
                                        broadcast_buffers = False,
                                        find_unused_parameters = False,
                                        process_group = None)
        capture_stream.synchronize()

        for _ in range(30):
            optimizer.zero_grad()
            with amp.autocast(enabled=enable_amp):
                x = inp.view(batch_size, num_blocks, block_size, H, W // 2 + 1, 2)
                res = model(x)
                loss = torch.sum(res)
            gscaler.scale(loss).backward()
            gscaler.step(optimizer)
            gscaler.update()

        # sync here and clean up
        capture_stream.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        
        if enable_graph:
            graph = torch.cuda.CUDAGraph()
            optimizer.zero_grad()
            # start capture
            graph.capture_begin()
            with amp.autocast(enabled=enable_amp):
                x = inp.view(batch_size, num_blocks, block_size, H, W // 2 + 1, 2)
                res = model(x)
                loss = torch.sum(res)
            gscaler.scale(loss).backward()
            # end capture
            graph.capture_end()

    torch.cuda.current_stream().wait_stream(capture_stream)

    
    # multiplication
    num_steps = 500
    start = time.perf_counter_ns()
    for n in range(num_steps):
        if enable_graph:
            graph.replay()
        else:
            with amp.autocast(enabled=enable_amp):
                x = inp.view(batch_size, num_blocks, block_size, H, W // 2 + 1, 2)
                res = model(x)
                loss = torch.sum(res)
            gscaler.scale(loss).backward()

        # optimizer step
        gscaler.step(optimizer)
        gscaler.update() 
    
    stop = time.perf_counter_ns()

    print(f"Time per step: {(stop-start)*10**(-6)/float(num_steps):.2f} ms")

if __name__ == "__main__":
    main()

