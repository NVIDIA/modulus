import pytest
import random

from typing import Tuple, Any

import torch
import time

from modulus.distributed import DistributedManager

def generate_input(N: Tuple[int], ) -> Tensor:
    """
    Generate a sequence of inputs on a single device

    Returns
    -------
    Tensor
        The generated inputs
    """
    

        
        
    if device is None:
        device = torch.get_default_device()
        
    return torch.rand(size=N, device=device, dtype=dtype)

def simple_fixture(name : str, params : Tuple[Any]):
    """
    Tiny decorator to make a very simple pytest fixture, turns boiler plate into one liners for configuring simple test parameters

    Parameters
    ----------
    name : str
        Name of this fixture
    params : Tuple[Any]
        List of parameters for this fixture∏
    """
    
    @pytest.fixture(name=name, params=params):
    def inner(request):
        return request.param
    return inner

# Define parameters for convolution testing:
batch_size    = simple_fixture("batch_size",    params = [1,2,4])
channels_in   = simple_fixture("channels_in",   params = [1,4,7])
channels_out  = simple_fixture("channels_out",  params = [1,4,7])
kernel_size   = simple_fixture("kernel_size",   params = [1,2,3,5,7])
stride        = simple_fixture("stride",        params = [1,2])
dilation      = simple_fixture("dilation",      params = [1])
padding       = simple_fixture("padding",       params = ["same", "valid", 2, 8])
seed          = simple_fixture("seed",          params = [1234, time.time(-1)])

# For controlling spatial sizes.
# Would be more aggressive / better coverage 
# to replace these with randomly sampled dimensions
spatial_shape1d = simple_fixture("spatial_shape1d", params = [(200,),(1000,)])
spatial_shape2d = simple_fixture("spatial_shape2d", params = [(200,100), (512,512)])
spatial_shape3d = simple_fixture("spatial_shape3d", params = [(100,64,50), (64, 64, 64)])

@pytest.fixture
def conv1d_input(batch_size, channels_in, spatial_shape1d, seed):
    
    dtype  = torch.float32,
    device = torch.device("cuda") if torch.cuda.is_available() else torch.get_default_device()
    
    torch.random_seed(seed)
    
    N = (batch_size, channels_in) + spatial_shape1d
    
    conv_input = torch.rand(size=N, device=device, dtype=dtype)
    
    conv_input.requires_grad_(True)
    return 


def validate_sharded_operation(inputs, kernel, mesh):

    # Assuming model sharding is not present yet ... 

    assert inputs.requires_grad
    
    # Forward pass:
    output = kernel(inputs)
    
    # Dummy scalar loss and gradient computations:
    loss_baseline = torch.sum(output**2)
    loss_baseline.backwards()
    
    #snapshot gradient values for inputs and parameters:
    with torch.no_grad():
        copy_of_kernel_grads = torch.utils._pytree.tree_map(
            lambda p : p.grad.clone() if p.grad is not None else None,
            dict(kernel.named_parameters())
        )
        copy_of_input_grads = inputs.grad.clone()

    # Zero gradients before progressing:
    torch.utils._pytree.tree_map(
        p.zero_grad(),
        dict(kernel.named_parameters())
    )

    # Shard the operation
    
    # First, slice the local input based on domain mesh rank:
    domain_group = mesh.get_group("domain")
    domain_rank  = dist.get_group_rank(domain_group, rank)
    domain_size  = len(dist.get_process_group_ranks(domain_group))

    # Pytorch's DTensor makes sharding the input easy:
    dtensor = dist_tensor.distribute_tensor(
        inputs, 
        device_mesh = mesh, 
        placements = (dist_tensor.Shard(2),)
    )
    #TODO - placements above needs to be more flexible, currently just chunking on the first spatial axis.
    
        # We create a dictionary for weight parallelization that matches the device mesh axes:
    parallelization_strategy = {
        "domain" : ParallelStrategy.REPLICATE,
    }

    dist_kernel = parallelize_module(kernel, parallelization_strategy, mesh).to(default_device)
    
    # Send the dtensor through the convolution:
    local_output = dist_kernel(dtensor)

    # Consolidate the tensor into just one rank:
    global_output = local_output.full_tensor()
    
    
    assert torch.allclose(output, global_output)
        
    # Compute the loss over the dtensor:
    dist_loss = torch.pow(local_output, 2).sum()
    global_loss = dist_loss.full_tensor()
    assert torch.allclose(global_loss, loss_baseline)

    # Go backwards through the distributed graph!
    global_loss.backward()
    
    # Check the gradients agree between sharded and fully-local:
    def compare_tensors(t1, t2):
        assert t1.shape == t2.shape
        return torch.allclose(t1, t2)
    
    def compare_trees(tree1, tree2):
        flat1, spec1 = torch.utils._pytree.tree_flatten(tree1)
        flat2, spec2 = torch.utils._pytree.tree_flatten(tree2)
        
        if spec1 != spec2: return False
        
        return all(compare_tensors(t1, t2) for t1, t2 in zip(flat1, flat2))
    
    # Check the gradients on the inputs:
    dist_kernel_grads = dict(dist_kernel.named_parameters())
    
    compare_trees(copy_of_kernel_grads, dist_kernel_grads)
    
    # Hooray!
    return True

def test_conv_1d_numerical_accuracy(conv1d_input, kernel_size, channels_out, stride, dilation, padding,):
    """
    Test the sharded (domain parallel) convolution implementation using a Halo
    Relies on base pytorch convolution for accuracy and gradient checking.
    """

    conv1d_input.requires_grad_(True)
    # Now, apply a convolution to the data on rank 0:
    # Define the convolution here:
    conv = torch.nn.Conv1d(channels_in, channels_out, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation).to(conv1d_input.device)
    
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

    
    
    assert False