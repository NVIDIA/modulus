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
import pytest
import torch

from modulus.distributed import DistributedManager
from modulus.distributed.autograd import (
    all_gather_v, gather_v,
    scatter_v,
    indexed_all_gather
)


def run_test_scatter_v(rank, world_size):
    os.environ["RANK"] = f"{rank}"
    os.environ["WORLD_SIZE"] = f"{world_size}"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(12355)

    DistributedManager.initialize()

    manager = DistributedManager()
    assert manager.is_initialized()

    tensor_dim = 4
    sizes = [r + 2 for r in range(world_size)]

    tensor = torch.arange(world_size, device=f"cuda:{rank}", dtype=torch.float32) + 1
    tensor = tensor.view(-1, 1).expand(-1, tensor_dim).contiguous()
    tensor = tensor.repeat_interleave(repeats=torch.tensor(sizes, device=f"cuda:{rank}"), dim=0)
    tensor.requires_grad_(True)

    scattered_tensor = scatter_v(tensor, sizes, dim=0, src=0, group=None)  
    
    expected_tensor = torch.ones(
        (sizes[rank], tensor_dim), device=f"cuda:{rank}", dtype=torch.float32
    ) * (rank + 1)

    assert torch.allclose(expected_tensor, scattered_tensor)

    grad_out = torch.ones_like(scattered_tensor) * (-1)
    scattered_tensor.backward(gradient=grad_out)
    
    if rank == 0:
        expected_grad = torch.ones_like(tensor) * (-1)
        assert torch.allclose(tensor.grad, expected_grad)

    del os.environ["RANK"]
    del os.environ["WORLD_SIZE"]
    del os.environ["MASTER_ADDR"]
    del os.environ["MASTER_PORT"]

    DistributedManager.cleanup()


def run_test_gather_v(rank, world_size):
    os.environ["RANK"] = f"{rank}"
    os.environ["WORLD_SIZE"] = f"{world_size}"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(12355)

    DistributedManager.initialize()

    manager = DistributedManager()
    assert manager.is_initialized()

    tensor_dim = 4
    tensor = (rank + 1) * torch.ones(
        (rank + 2, tensor_dim), device=f"cuda:{rank}", dtype=torch.float32
    )
    tensor.requires_grad_(True)
    sizes = [r + 2 for r in range(world_size)]

    gathered_tensor = gather_v(tensor, sizes, dim=0, dst=0, group=None)  
    
    if rank == 0:
        expected_tensor = torch.arange(world_size, device="cuda:0", dtype=torch.float32) + 1
        expected_tensor = expected_tensor.view(-1, 1).expand(-1, tensor_dim).contiguous()
        expected_tensor = expected_tensor.repeat_interleave(repeats=torch.tensor(sizes, device="cuda:0"), dim=0)

        assert torch.allclose(expected_tensor, gathered_tensor)

    grad_out = torch.ones_like(gathered_tensor) * (-1)
    gathered_tensor.backward(gradient=grad_out)
   
    expected_grad = torch.ones_like(tensor) * (-1)
    assert torch.allclose(tensor.grad, expected_grad)
  
    del os.environ["RANK"]
    del os.environ["WORLD_SIZE"]
    del os.environ["MASTER_ADDR"]
    del os.environ["MASTER_PORT"]

    DistributedManager.cleanup()


def run_test_all_gather_v(rank, world_size):
    os.environ["RANK"] = f"{rank}"
    os.environ["WORLD_SIZE"] = f"{world_size}"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(12355)

    DistributedManager.initialize()

    manager = DistributedManager()
    assert manager.is_initialized()

    tensor_dim = 4
    tensor = (rank + 1) * torch.ones(
        (rank + 2, tensor_dim), device=f"cuda:{rank}", dtype=torch.float32
    )
    tensor.requires_grad_(True)
    sizes = [r + 2 for r in range(world_size)]

    gathered_tensor = all_gather_v(tensor, sizes, dim=0, group=None)  
    
    expected_tensor = torch.arange(world_size, device=f"cuda:{rank}", dtype=torch.float32) + 1
    expected_tensor = expected_tensor.view(-1, 1).expand(-1, tensor_dim).contiguous()
    expected_tensor = expected_tensor.repeat_interleave(repeats=torch.tensor(sizes, device=f"cuda:{rank}"), dim=0)

    assert torch.allclose(expected_tensor, gathered_tensor)

    grad_out = torch.ones_like(gathered_tensor) * (-1)
    gathered_tensor.backward(gradient=grad_out)
   
    expected_grad = torch.ones_like(tensor) * (-1) * world_size
    assert torch.allclose(tensor.grad, expected_grad)
  
    del os.environ["RANK"]
    del os.environ["WORLD_SIZE"]
    del os.environ["MASTER_ADDR"]
    del os.environ["MASTER_PORT"]

    DistributedManager.cleanup()


def run_test_indexed_all_gather_v(rank, world_size):
    os.environ["RANK"] = f"{rank}"
    os.environ["WORLD_SIZE"] = f"{world_size}"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(12355)

    DistributedManager.initialize()

    manager = DistributedManager()
    assert manager.is_initialized()

    tensor_dim = 4
    tensor = torch.arange(1, world_size + 1, device=f"cuda:{rank}", dtype=torch.float32)
    tensor = tensor.view(-1, 1).expand(-1, tensor_dim).contiguous()
    tensor = tensor.repeat_interleave(repeats=rank+1, dim=0)
    tensor.requires_grad_(True)

    send_sizes = [rank + 1 for _ in range(world_size)]
    recv_sizes = [r + 1 for r in range(world_size)]

    scatter_indices = [
        torch.nonzero(tensor[:, 0] == (r + 1)).view(-1) for r in range(world_size)
    ]

    gathered_tensor = indexed_all_gather(tensor, scatter_indices, recv_sizes, send_sizes, group=None)  

    expected_tensor = torch.ones(
        (sum(recv_sizes), tensor_dim), device=f"cuda:{rank}", dtype=torch.float32
    ) * (rank + 1)

    assert torch.allclose(expected_tensor, gathered_tensor)

    grad_out = torch.ones_like(gathered_tensor) * (-1)
    gathered_tensor.backward(gradient=grad_out)
   
    expected_grad = torch.ones_like(tensor) * (-1)
    assert torch.allclose(tensor.grad, expected_grad)
  
    del os.environ["RANK"]
    del os.environ["WORLD_SIZE"]
    del os.environ["MASTER_ADDR"]
    del os.environ["MASTER_PORT"]

    DistributedManager.cleanup()


def run_test_autograd_prim(func):
    num_gpus = torch.cuda.device_count()
    assert num_gpus >= 2, "Not enough GPUs available for test"
    world_size = 2

    torch.multiprocessing.spawn(
        func,
        args=(world_size,),
        nprocs=world_size,
        start_method="spawn",
    )


def test_scatter_v():
    run_test_autograd_prim(run_test_scatter_v)


def test_gather_v():
    run_test_autograd_prim(run_test_gather_v)


def test_all_gather_v():
    run_test_autograd_prim(run_test_all_gather_v)


def test_indexed_all_gather_v():
    run_test_autograd_prim(run_test_indexed_all_gather_v)


# for debugging
if __name__ == "__main__":
    test_indexed_all_gather_v()
    test_scatter_v()
    test_gather_v()
    test_all_gather_v()
