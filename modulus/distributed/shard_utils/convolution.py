import torch
from torch import nn

from modulus.distributed import DistributedManager

import torch.distributed as dist

from torch.distributed import tensor as dist_tensor
from torch.distributed.device_mesh import DeviceMesh

from . strategies import ParallelStrategy
from . halo       import compute_halo_size, HaloPaddingND



class DistributedConv1d(nn.Module):
    
    def __init__(self, 
                unsharded_convolution : torch.nn.Conv1d, 
                parallelization_strategy: dict,
                world_mesh: DeviceMesh,
                *args, **kwargs):
        super().__init__()
        
        # First, save the convolution:
        self.unsharded_convolution = unsharded_convolution
        
        
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
                
                
        # Requires kernel size and stride (these are stored as tuples)
        stride = self.unsharded_convolution.stride
        kernel_size = self.unsharded_convolution.kernel_size
        
        halo_size = [ compute_halo_size(stride=s, kernel=k, dilation=1) for s, k in zip(stride, kernel_size) ]
        
        
        # Padding is forced to "valid" for each patch because we're manually padding
        # to exactly replicate behavior of  "same" or other paddings, 
        # we would need to include some edge-zeros in the halo step.
        # This is controlled with "edge_padding"

        # Track the unsharded padding to get it right with uneven output shards.
        
        if self.unsharded_convolution.padding == "valid":
            self.edge_padding = "none"
            self.unsharded_padding = (0,)
        elif self.unsharded_convolution.padding == "same":
            self.edge_padding = self.unsharded_convolution.padding_mode
            self.unsharded_padding = None # We don't need to calculate the output size in this case
        else:
            self.unsharded_padding = self.unsharded_convolution.padding
            self.edge_padding = self.unsharded_convolution.padding_mode
            
        self.unsharded_convolution.padding = "valid"
    
        self.halo = halo_size
        
        # # This hook is to collect gradients across the local image shard:
        self.unsharded_convolution.weight.register_hook(self.sync_local_weights)
        # self.unsharded_convolution.bias.register_hook(self.correct_bias_grad)
        self.unsharded_convolution.bias.register_hook(self.sync_local_weights)

        
    def forward(self, sharded_tensor: dist_tensor.DTensor) -> dist_tensor.DTensor:
        
        # These are useful pieces of information we need for this operation:
        self.mesh = sharded_tensor.device_mesh
        
        print(f"Unpadded input: {sharded_tensor.to_local()}")

        # For some convolutions, the output is not evenly distributed across ranks.
        # In this case, we focus here on getting this chunk's output correct
        # And ensure we pass the proper shape to the final DTensor output.
        
        global_shape = sharded_tensor.shape
        
        
        # Compute the expected global output size:
        if self.unsharded_padding is not None:
            out_len = conv_output_shape(
                global_shape[-1], 
                self.unsharded_padding[0], 
                self.unsharded_convolution.stride[0], 
                self.unsharded_convolution.kernel_size[0], 
                self.unsharded_convolution.dilation[0])
            out_shape = global_shape[:-1] + (out_len,)
        else:
            out_shape = global_shape

        out_strides = sharded_tensor.stride()


        # Compute the Halo pass in each dimension:
        halo_padded_tensor = HaloPaddingND.apply(sharded_tensor, self.halo, self.edge_padding, self.unsharded_padding)


        print(f"Expected output shape: {out_shape}")
        
        # For convolutions with stride != 2, things get tricky.
        
        # Then, compute the convolution on the local tensor with halo:
        local_output = self.unsharded_convolution(halo_padded_tensor)
        
        # Some logic here to account for when there are rounding cutoffs in the unsharded conv.
        # Have to do this _here_ because the length of the output depends on the length of the input.
        
        # dm = DistributedManager()
        # if dm.rank == 0:
        #     local_output = local_output[:,:,:-1]

        print(f"local_output is {local_output} with shape {local_output.shape}")

        # Create a fresh distributed tensor from the output:
        return dist_tensor.DTensor.from_local(local_output, 
                                              sharded_tensor.device_mesh, 
                                              sharded_tensor.placements,
                                              shape  = out_shape,
                                              stride = out_strides)
    
    def correct_bias_grad(self, grad):
        # Use the captured  grad halo to fix the bias grad.
        raise Exception
    
    def sync_local_weights(self, grad):
        
        # Get the cached mesh:
        mesh = self.mesh
        
        local_group = mesh.get_group()
        # The `mesh` variable in the code refers to the device mesh associated with the
        # input sharded tensor. The device mesh represents the topology of the
        # distributed devices and how they are interconnected for communication in a
        # distributed setting.
        mesh.get_group()
        local_rank  = mesh.get_local_rank()
        local_size  = dist.get_world_size(group=local_group)
        
        dist.all_reduce(grad, group=local_group)
        
        return grad
