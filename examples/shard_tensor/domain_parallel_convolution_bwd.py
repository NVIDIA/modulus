"""
In this example, extend our domain-parallel convolution to be differentiated across GPUs
"""
import torch
from time import perf_counter


from typing import Tuple, Optional, List

from torch import nn
from torch import Tensor

torch.manual_seed(1234)

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

from modulus.distributed.autograd import indexed_all_to_all_v

from enum import Enum
class ParallelStrategy(Enum):
    """Docstring for MyEnum."""
    REPLICATE = 0
    WEIGHT_SHARD = 1
    
    
# def halo_1D_wrapper_slice_and_add
    
def halo_ND_wrapper(local_tensor: dist_tensor, halo_size : Tuple[int]):
    """
    Compute the N-Dimensional halo exchange terms for a regular (image-like)
    tensor according to halo_size

    Parameters
    ----------
    local_tensor : dist_tensor
        _description_
    halo_size : Tuple[int]
        _description_

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    NotImplementedError
        _description_
    """
    
    # The local group can come right from the mesh, which the tensor knows already.
    # Here's its implicit mesh is 1D so we don't pass a mesh_dim
    local_group = mesh.get_group()
    local_rank  = mesh.get_local_rank()
    local_size  = dist.get_world_size(group=local_group)



class HaloReduction1D(torch.autograd.Function):
    """
    
    """

    @staticmethod
    def forward(
        ctx,
        local_tensor: torch.Tensor,
        mesh,
        halo_t,
    ) -> torch.Tensor:  # pragma: no cover
        """forward pass of the Distributed HaloReduction in 1D"""
        
        # The local group can come right from the mesh, which the tensor knows already.
        # Here's its implicit mesh is 1D so we don't pass a mesh_dim
        local_group = mesh.get_group()
        local_rank  = mesh.get_local_rank()
        local_size  = dist.get_world_size(group=local_group)


        
        # The halo reduction is now a dedicated layer to allow differentiation:
        central_output = local_tensor[:, :,halo_t[0]:-halo_t[0]]

        for halo in halo_t:

            # Initialize peer2peer buffers to be empty
            # They will be left empty unless there is a message to exchange
            all_to_all_source = [ torch.empty(0, device=local_tensor.device, dtype=local_tensor.dtype) for _ in range(local_size)]
            all_to_all_dest   = [ torch.empty(0, device=local_tensor.device, dtype=local_tensor.dtype) for _ in range(local_size)]
        
        
        
            # Outgoing Halo, send from this rank but index is destination rank:

            if local_rank != 0:
                # Send one left (don't include the bias - it's already accounted for!)
                all_to_all_source[local_rank - 1] = local_tensor[:,:,:halo].contiguous()
                # Receive one left (need to initialize an empty buffer of the right size):
                all_to_all_dest[local_rank - 1] = torch.zeros_like(all_to_all_source[local_rank - 1]).contiguous() 
            if local_rank != local_size - 1: 
                # Send one right:
                all_to_all_source[local_rank + 1] = local_tensor[:,:,-halo:].contiguous()
                # Receive one right:
                all_to_all_dest[local_rank + 1] = torch.zeros_like(all_to_all_source[local_rank + 1]).contiguous()
            

            # Scatter and gather the halo objects:
            dist.all_to_all(all_to_all_dest, all_to_all_source, group=local_group)
            
            
            # Apply the halo correction:
            if local_rank != 0:
                # From one left
                central_output[:,:,:halo] += all_to_all_dest[local_rank - 1]
                
            if local_rank != local_size - 1: 
                # From one right:
                central_output[:,:,-halo:] += all_to_all_dest[local_rank + 1]
        
        ctx.input_shape = local_tensor.shape
        ctx.a2a_recv = all_to_all_source
        ctx.tensor = local_tensor
        ctx.halo_t = halo_t
        ctx.mesh   = mesh

        return central_output



    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        """backward pass of the HaloReduction1D primitive"""

        print(f"Bwd grad_output ({grad_output.shape}) {grad_output}")
        local_group = ctx.mesh.get_group()
        local_rank  = ctx.mesh.get_local_rank()
        local_size  = dist.get_world_size(group=local_group)

        
        # For locations unaffected by the halo exchange, the grad_input is the grad output.
        # Start there:
        grad_input = torch.zeros(ctx.input_shape, device=grad_output.device, dtype=grad_output.dtype)
        grad_input[:,:, ctx.halo_t[0]:-ctx.halo_t[0]] = grad_output
   
   
   
        # In the backward pass, the gradient is largely unaffected.  But, we have 
        # to send the gradient from neighboring regions to pad parts of the vector.
        # Messages are moving opposite to the forward pass!

        for halo in ctx.halo_t:

            # Initialize peer2peer buffers to be empty
            # They will be left empty unless there is a message to exchange
            all_to_all_source = [ torch.empty(0, device=grad_output.device, dtype=grad_output.dtype) for _ in range(local_size)]
            all_to_all_dest   = [ torch.empty(0, device=grad_output.device, dtype=grad_output.dtype) for _ in range(local_size)]
        
        
            # Outgoing Halo, send from this rank but index is destination rank:

            if local_rank != 0:
                # Send one left (don't include the bias - it's already accounted for!)
                all_to_all_source[local_rank - 1] = grad_output[:,:,:halo].contiguous()
                # Receive one left (need to initialize an empty buffer of the right size):
                all_to_all_dest[local_rank - 1] = torch.zeros_like(all_to_all_source[local_rank - 1]).contiguous() 
            if local_rank != local_size - 1: 
                # Send one right:
                all_to_all_source[local_rank + 1] = grad_output[:,:,-halo:].contiguous()
                # Receive one right:
                all_to_all_dest[local_rank + 1] = torch.zeros_like(all_to_all_source[local_rank + 1]).contiguous()
            

            # Scatter and gather the halo objects:
            dist.all_to_all(all_to_all_dest, all_to_all_source, group=local_group)
            
            
            # Apply the halo correction:
            if local_rank != 0:
                # From one left
                grad_input[:,:,:halo] += all_to_all_dest[local_rank - 1]
                
            if local_rank != local_size - 1: 
                # From one right:
                grad_input[:,:,-halo:] += all_to_all_dest[local_rank + 1]
   
   
        print(f"Grad Input: {grad_input}")
        
        if not ctx.needs_input_grad[0]:
            grad_input = None

        return grad_input, None, None, None, None

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
        halo_size = tuple( int(r / 2) for r in receptive_field )
        # print(f"Halo size: {self.halo}")
        
        # Then, add padding equal to the halo nodes on each side of the image:
        # We add the receptive field to ensure we get the right output shape:
        kwargs["padding"] = receptive_field 
        kwargs["padding_mode"] = "zeros"
        
        # Set the parent conv padding:
        self.unsharded_convolution.padding = receptive_field
        self.unsharded_convolution.padding_mode = "zeros"
    
        self.halo = halo_size
        
        # This hook is to collect gradients across the local image shard:
        self.unsharded_convolution.weight.register_hook(self.sync_local_weights)
        self.unsharded_convolution.bias.register_hook(self.correct_bias_grad)
        self.unsharded_convolution.bias.register_hook(self.sync_local_weights)

        
    def forward(self, sharded_tensor: dist_tensor.DTensor) -> dist_tensor.DTensor:
        
        # Use the modulus distributed manager to simplify all of this:
        dm = DistributedManager()
        
        # These are useful pieces of information we need for this operation:
        mesh = sharded_tensor.device_mesh
        

        

        # First, compute the convolution on the local tensor without halo:
        local_output = self.unsharded_convolution(sharded_tensor.to_local())

        # if the bias is not none, remove it from the halo terms:
        # This is not necessarily optimal for performance, since the bias is fused with the convolution
        if self.unsharded_convolution.bias is not None:
            halo_tensor = local_output - self.unsharded_convolution.bias

        # The halo computation depends on the local rank in the mesh.  In a 1D mesh it's simple (only 1 group),
        # but written here as a loop over each halo direction (which is a loop over 1 in this example!):
        central_output = HaloReduction1D.apply(halo_tensor, mesh, self.halo, bias)

        # Add the bias back in, if necessary:
        if self.unsharded_convolution is not None:
            central_output += self.unsharded_convolution.bias

        
        # Cache the mesh to use in the backward hook:
        self.mesh = mesh
        
        # Create a fresh distributed tensor from the output:
        return dist_tensor.DTensor.from_local(central_output, sharded_tensor.device_mesh, sharded_tensor.placements)
    
    def correct_bias_grad(self, grad):
        # Use the captured  grad halo to fix the bias grad.
        raise Exception
    
    def sync_local_weights(self, grad):
        
        # Get the cached mesh:
        mesh = self.mesh
        
        local_group = mesh.get_group()
        local_rank  = mesh.get_local_rank()
        local_size  = dist.get_world_size(group=local_group)
        
        dist.all_reduce(grad, group=local_group)
        
        return grad


def validate_conv_1d(batch_size, channels_in, channels_out, kernel_size, vector_length):
    
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
    
    if rank == 0:
    
        # Forward:
        output_baseline = conv(input)
        # Loss is very simple MSE compared to 0.
        loss_baseline = torch.pow(output_baseline, 2).sum()
        # Compute the gradients:
        loss_baseline.backward()

        print("loss_baseline: ", loss_baseline)
        print("conv.weight.grad: ", conv.weight.grad)
        print("conv.bias.grad: ", conv.bias.grad)

    # Set the grads to None:
    conv.weight.grad = None
    conv.bias.grad = None

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
    
    if dm.rank == 0:
        # Numerical accuracy check!
        # print(f"Vec Difference: ", output_baseline - global_output)
        # print(f"Max Difference: ", torch.max(torch.abs(output_baseline - global_output)))
        assert torch.allclose(output_baseline, global_output)
        
    # Compute the loss over the dtensor:
    loss = torch.pow(local_output, 2).sum()
    global_loss = loss.full_tensor()
    print("global_loss: ", global_loss)
    if dm.rank == 0:
        assert torch.allclose(global_loss, loss_baseline)

    # Go backwards through the distributed graph!
    global_loss.backward()
    
    # if dm.rank == 0:
    print(f"Dist w grad: {dist_conv1d.unsharded_convolution.weight.grad}")
    print(f"Dist b grad: {dist_conv1d.unsharded_convolution.bias.grad}")
    # Time the distributed co
        
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
    channels_in = 1
    channels_out = 1
    kernel_size = 3
    vector_length = 10
    validate_conv_1d(batch_size, channels_in, channels_out, kernel_size, vector_length)

    
    DistributedManager.cleanup()