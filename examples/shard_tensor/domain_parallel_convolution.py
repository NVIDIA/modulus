"""
In this example, we'll use modulus's distributed manager to implement a domain-parallel convolution operation.

This will be a 1D convolution spread over data placed on multiple GPUs.
"""
import torch
from time import perf_counter


from typing import Tuple

from torch import nn
from torch import Tensor

import torch.distributed as dist
from torch.distributed import tensor as dist_tensor
from torch.distributed.device_mesh import init_device_mesh
# from torch.distributed.tensor import DTensor

from modulus.distributed import DistributedManager
from modulus.distributed import ProcessGroupNode, ProcessGroupConfig

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



from enum import Enum
class ParallelStrategy(Enum):
    """Docstring for MyEnum."""
    REPLICATE = 0
    WEIGHT_SHARD = 1
    


class DistributedConv1d(nn.Module):
    
    def __init__(self, 
                unsharded_convolution, 
                parallelization_strategy: dict,
                *args, **kwargs):
        super().__init__()
        
        # First, save the convolution:
        self.unsharded_convolution = unsharded_convolution
        
        # Next, find the world mesh:
        world_mesh = DistributedManager().global_mesh
        
        # For a convolution, we have to shard or replicated the weights and biases.
        
        # Work from the outside in, assuming that's the simplest parallelism scheme.
        # No gradient needed here:
        with torch.no_grad():
            for mesh_name, strategy in parallelization_strategy.items():
                
                # Get the appropriate group:
                group = world_mesh[mesh_name].get_group()
                # We need to get rank 0 _in this group_ as the source
                # But, the broadcast needs to know the global rank:
                local_rank_0_global_rank = dist.get_global_rank(group, 0)
                
                # local_rank = world_mesh[mesh_name].get_local_rank()
                # global_rank = dm.rank
                
                if strategy == ParallelStrategy.REPLICATE:
                    # Broadcast from rank 0 outward:
                    w = self.unsharded_convolution.weight
                    dist.broadcast(w, src = local_rank_0_global_rank, group = group)
                    # Set the parameter:
                    self.unsharded_convolution.weight = w
                    
                    b = self.unsharded_convolution.bias
                    if b is not None:
                        dist.broadcast(b, src = local_rank_0_global_rank, group = group)
                        self.unsharded_convolution.bias = b
        
        
                # Place a distributed barrier here for the ops to sync:
                dist.barrier(group = group)
                
                
        # For sharded convolutions, always use padding to compute a little more than planned.
        # This extra is the halo computation.
        # Corner cases will throw parts of it away, but those points will need to wait
        # for the other ranks to complete anyways.  So it is not a penalty.
        
        # For domain-parallel convolutions, only padding="same" is allowed.
        
        # To calculate the padding needed, we need to first determine the receptive field.
        # Requires kernel size and stride (these are stored as tuples)
        stride = self.unsharded_convolution.stride
        kernel_size = self.unsharded_convolution.kernel_size
        
        if "dilation" in kwargs and kwargs["dilation"] != 1:
            raise NotImplementedError("Dilated convolution not supported yet")
            
        
        # The receptive field is how far in the input a pixel in the output can see:
        receptive_field = tuple(
            k + (k - 1) * (s - 1) - 1
            for k, s in zip(kernel_size, stride)
        )
        # It's used to calculate how large the halo computation has to be:
        # print(f"Receptive field: {receptive_field}")
        
        # The number of halo pixels is the casting `int(receptive field/2)`
        # Why?  Assuming a filter in the output image is centered in the input image,
        # we have only half of it's filter to the left.
        self.halo = tuple( int(r / 2) for r in receptive_field )
        # print(f"Halo size: {self.halo}")
        
        # Then, add padding equal to the halo nodes on each side of the image:
        # We add the receptive field to ensure we get the right output shape:
        kwargs["padding"] = receptive_field 
        kwargs["padding_mode"] = "zeros"
        
        # Set the parent conv padding:
        self.unsharded_convolution.padding = receptive_field
        self.unsharded_convolution.padding_mode = "zeros"
    
        
        
    def forward(self, sharded_tensor: dist_tensor.DTensor) -> dist_tensor.DTensor:
        
        # Use the modulus distributed manager to simplify all of this:
        dm = DistributedManager()
        
        # These are useful pieces of information we need for this operation:
        mesh = sharded_tensor.device_mesh
        
        # The local group can come right from the mesh, which the tensor knows already.
        # Here's its implicit mesh is 1D so we don't pass a mesh_dim
        local_group = mesh.get_group()
        local_rank  = mesh.get_local_rank()
        local_size  = dist.get_world_size(group=local_group)

        

        # First, compute the convolution on the local tensor without halo:
        local_output = self.unsharded_convolution(sharded_tensor.to_local())

        # The halo computation depends on the local rank in the mesh.  In a 1D mesh it's simple (only 1 group),
        # but written here as a loop over each halo direction (which is a loop over 1 in this example!):
        
        # A hack, here, to make sure we have the central output defined outside the loop:
        central_output = local_output[:, :,self.halo[0]:-self.halo[0]]


        for halo in self.halo:

            # Initialize peer2peer buffers to be empty
            # They will be left empty unless there is a message to exchange
            all_to_all_source = [ torch.empty(0, device=local_output.device, dtype=local_output.dtype) for _ in range(local_size)]
            all_to_all_dest   = [ torch.empty(0, device=local_output.device, dtype=local_output.dtype) for _ in range(local_size)]
            
            
        
        
            # print(f"Central Value shape: {central_output.shape}")
        
            # Outgoing Halo, send from this rank but index is destination rank:

            if local_rank != 0:
                # Send one left (don't include the bias - it's already accounted for!)
                all_to_all_source[local_rank - 1] = local_output[:,:,:halo].contiguous()
                # If there was bias applied, remove it from the halo term:
                if self.unsharded_convolution.bias is not None:
                    all_to_all_source[local_rank - 1] -= self.unsharded_convolution.bias.reshape((1,-1,1))
                # Receive one left (need to initialize an empty buffer of the right size):
                all_to_all_dest[local_rank - 1] = torch.zeros_like(all_to_all_source[local_rank - 1]).contiguous() 
            if local_rank != local_size - 1: 
                # Send one right:
                all_to_all_source[local_rank + 1] = local_output[:,:,-halo:].contiguous()
                # If there was bias applied, remove it from the halo term:
                if self.unsharded_convolution.bias is not None:
                    all_to_all_source[local_rank + 1] -= self.unsharded_convolution.bias.reshape((1,-1,1))
                # Receive one right:
                all_to_all_dest[local_rank + 1] = torch.zeros_like(all_to_all_source[local_rank + 1]).contiguous()
            

            # Scatter and gather the halo objects:
            dist.all_to_all(all_to_all_dest, all_to_all_source, group=local_group)
            
            
            # Apply the halo correction:
            if local_rank != 0:
                # From one left
                # print(f"halo from left: {all_to_all_dest[local_rank - 1]}")
                central_output[:,:,:halo] += all_to_all_dest[local_rank - 1]
                
            if local_rank != local_size - 1: 
                # From one right:
                # print(f"halo from right: {all_to_all_dest[local_rank + 1]}")
                central_output[:,:,-halo:] += all_to_all_dest[local_rank + 1]
        
        # Create a fresh distributed tensor from the output:
        return dist_tensor.DTensor.from_local(central_output, sharded_tensor.device_mesh, sharded_tensor.placements)
    

def benchmark_conv_1d(batch_size, channels_in, channels_out, kernel_size, vector_length):
    
    dm = DistributedManager()
    

    # The Distributed manager lets you access the mesh for all processes:
    mesh = dm.global_mesh
    size = dm.world_size
    
    # Decide the default device for this process as the local rank:
    default_device = torch.device(f"cuda:{local_rank}")
    

    # Generate the input in [batch, channels, length] format
    input = generate_input((batch_size, channels_in, vector_length,),device=default_device)
    
    # Now, apply a convolution to the data on rank 0:
    # Define the convolution here:
    conv = torch.nn.Conv1d(channels_in, channels_out, kernel_size=kernel_size, padding="same").to(default_device)
    
    if rank == 0 and size == 1:
    
        output_baseline = conv(input)
    
        
        torch.cuda.synchronize()
        # Time the single-gpu convolution:
        times = []
        for _ in range(10):
            start = perf_counter()
            null = conv(input)
            torch.cuda.synchronize()
            end = perf_counter()
            times.append(end-start)
        
        # Best time:
        best_single_gpu_time = min(times)



    # Now, let's do the same thing but parallelize the convolution over multiple GPU processes
    
    
    # First things first, set up the data.  Here we can use a torch.DTensor to make sure it's sharded appropriately.

    # This is the mesh for the domain-parallel operations (we named it domain, above, as opposed to "ddp")
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
    
    # The above operation will broadcast from rank 0 of the domain_mesh.  But in practice, for extra large data, this may not be optimal.
    
    # Now, pass the local tensor and the corresponding mesh to the distributed convolution
    # Distributing the convolution can be tricky.  We have to pass information about the mesh to parallize over
    
    # (Note this is similar to the tensor parallel organization here
    # https://pytorch.org/docs/main/distributed.tensor.parallel.html)
    # We're reimplementing some to demonstrate how Modulus does this and show how to extend to new layers.

    # We create a dictionary for parallelization that matches the device mesh axes:
    parallelization_strategy = {
        "ddp"    : ParallelStrategy.REPLICATE,
        "domain" : ParallelStrategy.REPLICATE,
    }

    dist_conv1d = DistributedConv1d(conv, parallelization_strategy).to(default_device)
    
    # Send the dtensor through the convolution:
    local_output = dist_conv1d(dtensor)

    # Consolidate the tensor into just one rank:
    global_output = local_output.full_tensor()

    # if dm.rank == 0:
    #     # Numerical accuracy check!
    #     assert torch.allclose(output_baseline, global_output)
    #     # print(f"Max Difference: ", torch.max(torch.abs(output_baseline - global_output)))
        
        
    # Time the distributed convolution:
    times = []
    for _ in range(10):
        start = perf_counter()
        local_output = dist_conv1d(dtensor)
        torch.cuda.synchronize()
        end = perf_counter()
        times.append(end-start)
        
    best_distributed_time = min(times)
    if dm.rank == 0: 

        if size == 1: # Only print single gpu when doing a single gpu run:
            print(f"{batch_size},{1},{channels_in},{channels_out},{kernel_size},{vector_length},{best_single_gpu_time}")
        else:
            print(f"{batch_size},{size},{channels_in},{channels_out},{kernel_size},{vector_length},{best_distributed_time}")
    
    return

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

    batch_size=1

    for channels_in in [2, 4, 8, 16, 32, 64, 128, 256]:
        channels_out = channels_in
        for kernel_size in [3, 5, 7]:
            for exp in range(13):
                vector_length = 800 * 2**exp
                benchmark_conv_1d(batch_size, channels_in, channels_out, kernel_size, vector_length)
    
    
    DistributedManager.cleanup()