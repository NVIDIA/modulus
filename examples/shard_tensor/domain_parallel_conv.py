"""
In this example, extend our domain-parallel convolution to be differentiated across GPUs
"""
import torch
from time import perf_counter
import argparse


from typing import Tuple, Optional, List

from torch import nn
from torch import Tensor

torch.manual_seed(1234)

import torch.distributed as dist
from torch.distributed import tensor as dist_tensor
# from torch.distributed.tensor import DTensor

from modulus.distributed import DistributedManager

from layers.strategies import ParallelStrategy

from layers.utils import parallelize_module

def generate_input(N: Tuple[int], dtype : torch.dtype = torch.float32, device : torch.device = None, seed : int = None,) -> Tensor:
    """
    Generate a sequence of inputs on a single device


    Parameters
    ----------
    N : int
        The total length of the inputs
    dtype : torch.dtype, optional
        Target datatype, by default torch.float32
    device : torch.device, optional
        Target device, by default None
    seed : int, optional
        If set, override the torch random seed for reproducability, by default None

    Returns
    -------
    Tensor
        The generated inputs
    """
    

    if seed is not None:
        torch.manual_seed(seed)
        
        
    if device is None:
        device = torch.get_default_device()
        
    return torch.rand(size=N, device=device, dtype=dtype)


def validate_conv_1d(batch_size, channels_in, channels_out, kernel_size, stride, dilation, vector_length, padding):
    
    dm = DistributedManager()
    

    # The Distributed manager lets you access the mesh for all processes:
    mesh = dm.global_mesh
    size = dm.world_size
    rank = dm.rank
    
    # Decide the default device for this process as the local rank:
    default_device = torch.device(f"cuda:{local_rank}")
    

    # Generate the input in [batch, channels, length] format
    input = generate_input((batch_size, channels_in, vector_length,),device=default_device)
    input.requires_grad_(True)
    # Now, apply a convolution to the data on rank 0:
    # Define the convolution here:
    conv = torch.nn.Conv1d(channels_in, channels_out, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation).to(default_device)
    
    if rank == 0:
    
        print(f"Input: ", input)
        # Forward:
        output_baseline = conv(input)
        # Loss is very simple MSE compared to 0.
        loss_baseline = torch.pow(output_baseline, 2).sum()
        # Compute the gradients:
        loss_baseline.backward(retain_graph=True)
        print(f"conv baseline: {output_baseline} with shape {output_baseline.shape}")
        print(f"loss_baseline: {loss_baseline} with shape {loss_baseline.shape}")
        print(f"conv.weight.grad: {conv.weight.grad} with shape {conv.weight.grad.shape}")
        print(f"conv.bias.grad: {conv.bias.grad} with shape {conv.bias.grad.shape}")
        
        # Gradient with respect to input:
        print(f"input.grad: {input.grad}")

    # Set the grads to None:
    conv.weight.grad = None
    conv.bias.grad = None

    if rank == 0:
        print("Mesh is: ", mesh)
    
    # Create the mesh, per usual:
    domain_mesh = mesh["domain"]

    
    # First, slice the local input based on domain mesh rank:
    domain_group = domain_mesh.get_group()
    domain_rank = dist.get_group_rank(domain_group, rank)
    domain_size = len(dist.get_process_group_ranks(domain_group))

    # Pytorch's DTensor makes sharding the input easy:
    dtensor = dist_tensor.distribute_tensor(
        input, 
        device_mesh = domain_mesh, 
        placements = (dist_tensor.Shard(2),)
    )
    
    if rank == 0: print(f"dtensor: {dtensor}")
    
    
    # The above operation will broadcast from rank 0 of the domain_mesh. 
    # But in practice, for extra large data, this may not be optimal.
    
    # Now, pass the local tensor and the corresponding mesh to the distributed convolution
    # Distributing the convolution can be tricky.  We have to pass information about the mesh to parallize over
    
    # (Note this is similar to the tensor parallel organization here
    # https://pytorch.org/docs/main/distributed.tensor.parallel.html)
    # We're reimplementing some to demonstrate how Modulus does this and show how to extend to new layers.

    # We create a dictionary for weight parallelization that matches the device mesh axes:
    parallelization_strategy = {
        "ddp"    : ParallelStrategy.REPLICATE,
        "domain" : ParallelStrategy.REPLICATE,
    }

    dist_conv1d = parallelize_module(conv, parallelization_strategy, mesh).to(default_device)
    
    # Send the dtensor through the convolution:
    local_output = dist_conv1d(dtensor)

    print(f"local output shape: {local_output.shape}")
    print(f"local output spec: {local_output._spec}")

    summed_output = local_output.sum()
    print(f"summed_output: {summed_output}")
    
    partial_sum = local_output.sum(axis=(1,-1))
    print(f"partial_sum: {partial_sum}")
    

    # Consolidate the tensor into just one rank:
    global_output = local_output.full_tensor()
    
    print(f"Global output on Rank {dm.rank}: {global_output} (shape: {global_output.shape})")
    # Compute the loss over the dtensor:
    loss = torch.pow(local_output, 2).sum()
    
    print(f"loss type: {type(loss)}")
    print(f"loss typ_spec: {loss._spec}")
    
    return
    
    if dm.rank == 0:
        # Numerical accuracy check!
        print(f"Vec Difference: ", output_baseline - global_output)
        print(f"Max Difference: ", torch.max(torch.abs(output_baseline - global_output)))
        assert torch.allclose(output_baseline, global_output)
        

    print(f"Rank {dm.rank} loss: {loss}")
    
    global_loss = loss.full_tensor()
    if dm.rank == 0:
        print("global_loss: ", global_loss)
        assert torch.allclose(global_loss, loss_baseline)

    # Go backwards through the distributed graph!
    global_loss.backward()
    
    if dm.rank == 0:
        print(f"Dist w grad: {dist_conv1d.unsharded_convolution.weight.grad}")
        print(f"Dist b grad: {dist_conv1d.unsharded_convolution.bias.grad}")
    # Time the distributed co
        
    return

def get_args():
    
    parser = argparse.ArgumentParser(description="Set convolution parameters.")

    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for the input tensor. Default: 1",
    )
    parser.add_argument(
        "--channels_in",
        type=int,
        default=1,
        help="Number of input channels. Default: 1",
    )
    parser.add_argument(
        "--channels_out",
        type=int,
        default=1,
        help="Number of output channels. Default: 1",
    )
    parser.add_argument(
        "--kernel_size",
        type=int,
        default=3,
        help="Size of the convolutional kernel. Default: 3",
    )
    parser.add_argument(
        "--dilation",
        type=int,
        default=1,
        help="Dilation rate of the convolution. Default: 1",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Stride of the convolution. Default: 2",
    )
    parser.add_argument(
        "--vector_length",
        type=int,
        default=20,
        help="Length of the input vector. Default: 20",
    )
    
    parser.add_argument(
        "--padding",
        type=lambda x : x if x in ["same", "valid"] else int(x),
        default="valid",
        help="Padding applied to the input tensor. Can be an integer, 'same', or 'valid'. Default: 'valid'",
    )

    return parser.parse_args()

if __name__ == "__main__":
    
    # Set up the local ranks:
    DistributedManager.initialize()
    dm = DistributedManager()
    # Access the rank easily through the manager:
    rank = dm.rank
    local_rank = dm.local_rank
    
    # Use the distributed manager to generate a mesh
    # Passing -1 for an argument will infer the size based on the world_size
    # (similar to reshaping a tensor with one axis as -1)
    dm.initialize_mesh((1, -1), ("ddp", "domain"))
    
    # These are settings to configure the convolution operation
    
    if rank == 0: print("B,N,nin,nout,K,L,T")


    args = get_args()
    validate_conv_1d(
        args.batch_size,
        args.channels_in,
        args.channels_out,
        args.kernel_size,
        args.stride,
        args.dilation,
        args.vector_length,
        args.padding
    )

    
    DistributedManager.cleanup()