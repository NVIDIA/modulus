Modulus Distributed
===================

.. automodule:: modulus.distributed
.. currentmodule:: modulus.distributed

Basics
-------

The distributed utilites in Modulus are aimed to to make writing parallel training and
inference scripts easier by providing a unified way to specify the distributed training
environment. The utilites in ``modulus.distributed`` build on top of the utilites from
``torch.distributed`` further abstracting some of the complexities of setting up a
distributed training. 

Below example shows how to setup a simple distributed data parallel training using the
distributed utilites in Modulus. 
The `DistributedDataParallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`_ 
utility from PyTorch provides the framework to achieve data parallelism by synchronizing
across multiple workers. The ``DistributedManager`` provides utilites that can be used
to create this container in a unified fashion. 

Below code shows how to provide the inputs to the ``device_ids``, ``output_device``,
``broadcast_buffers`` and ``find_unused_parameters`` arguments of the
``DistributedDataParallel`` class using the ``DistributedManager`` from Modulus. 

.. code:: python

    import torch
    import modulus
    from torch.nn.parallel import DistributedDataParallel
    from modulus.distributed import DistributedManager
    from modulus.models.mlp.fully_connected import FullyConnected

    def main():
        DistributedManager.initialize()
        dist = DistributedManager()

        arch = FullyConnected(in_features=32, out_features=64).to(dist.device)
        input = torch.randn(128, 32).to(dist.device)
        output = torch.randn(128, 64).to(dist.device)

        # Distributed learning
        if dist.world_size > 1:
            ddps = torch.cuda.Stream()
            with torch.cuda.stream(ddps):
                arch = DistributedDataParallel(
                    arch,
                    device_ids=[dist.local_rank],
                    output_device=dist.device,
                    broadcast_buffers=dist.broadcast_buffers,
                    find_unused_parameters=dist.find_unused_parameters,
                )
            torch.cuda.current_stream().wait_stream(ddps)

        optimizer = torch.optim.Adam(
            arch.parameters(),
            lr=0.001,
        )

        def training_step(invar, outvar):
            predvar = arch(invar)
            loss = torch.sum(torch.pow(predvar - outvar, 2))
            return loss

        # Sample training loop
        for i in range(20):
            print(i)
            loss = training_step(input, output)
            input.copy_(torch.randn(128, 32).to(dist.device))

    if __name__ == "__main__":
        main()

Once the scirpt is defined as above, the simple training can be run on a single GPU
using ``python train.py`` or on multiple GPUs using 
``mpirun -np <num_gpus> python train.py``.

.. autosummary::
   :toctree: generated

modulus.distributed.manager
----------------------------

.. automodule:: modulus.distributed.manager
    :members:
    :show-inheritance:

modulus.distributed.utils
----------------------------

.. automodule:: modulus.distributed.utils
    :members:
    :show-inheritance:

modulus.distributed.autograd
----------------------------

.. automodule:: modulus.distributed.autograd
    :members:
    :show-inheritance: