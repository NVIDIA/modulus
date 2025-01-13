import torch
from torch import nn

from torch.distributed.device_mesh import DeviceMesh


from . strategies import ParallelStrategy


def parallelize_module(module : nn.Module, strategy_per_axis : ParallelStrategy, mesh : DeviceMesh):
    """
    Recursively loop over a module and it's children to parallelize it, assuming a sharded input tensor,
    with respect to the device mesh.

    Parameters
    ----------
    module : nn.Module
        Torch module or class inheriting from module
    strategy_per_axis : ParallelStrategy
        Defined strategy for weight parallelization per axis in the mesh
    mesh : DeviceMesh
        Global device mesh for parallelization
    """

    # This function is essentially a lookup table for operations.
    if isinstance(module, nn.Conv1d):
        from . convolution import DistributedConv1d
        return DistributedConv1d(module, strategy_per_axis, mesh)