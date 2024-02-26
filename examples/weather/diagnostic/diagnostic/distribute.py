from typing import Tuple

import torch

from modulus import Module
from modulus.distributed import DistributedManager


def distribute_model(model: Module) -> Tuple[Module, DistributedManager]:
    """Distribute model using DDP.

    Parameters
    ----------
    model: modulus.Module
        The model to be distributed

    Returns
    -------
    (model: modulus.Module, dist: modulus.distributed.DistributedManager)
        A tuple of the local copy of the distributed model and the
        DistributedManager object.
    """

    DistributedManager.initialize()
    dist = DistributedManager()
    model = model.to(dist.device)

    if dist.world_size > 1:
        ddps = torch.cuda.Stream()
        with torch.cuda.stream(ddps):
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[dist.local_rank],
                output_device=dist.device,
                broadcast_buffers=dist.broadcast_buffers,
                find_unused_parameters=dist.find_unused_parameters,
            )
        torch.cuda.current_stream().wait_stream(ddps)

    return (model, dist)
