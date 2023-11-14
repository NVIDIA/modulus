# [imports]
import torch
import numpy as np
import modulus
import matplotlib.pyplot as plt
from modulus.datapipes.benchmarks.darcy import Darcy2D
from modulus.models.fno.fno import FNO
from torch.nn.parallel import DistributedDataParallel
from modulus.distributed import DistributedManager
from modulus.utils import StaticCaptureTraining
from modulus.metrics.general.mse import mse
# [imports]

# [code]

def main():

    DistributedManager.initialize()
    dist = DistributedManager()

    normaliser = {
        "permeability": (1.25, 0.75),
        "darcy": (4.52e-2, 2.79e-2),
    }
    dataloader = modulus.datapipes.benchmarks.darcy.Darcy2D(
        resolution=256, batch_size=64, nr_permeability_freq=5, normaliser=normaliser
    )
    model = FNO(
        in_channels=1,
        out_channels=1,
        decoder_layers=1,
        decoder_layer_size=32,
        dimension=2,
        latent_channels=32,
        num_fno_layers=4,
        num_fno_modes=12,
        padding=5,
    ).to("cuda")
    
    if dist.distributed:
        ddps = torch.cuda.Stream()
        with torch.cuda.stream(ddps):
            model = DistributedDataParallel(
                model,
                device_ids=[dist.local_rank],  # Set the device_id to be
                                               # the local rank of this process on
                                               # this node
                output_device=dist.device,
                broadcast_buffers=dist.broadcast_buffers,
                find_unused_parameters=dist.find_unused_parameters,
            )
        torch.cuda.current_stream().wait_stream(ddps)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: 0.85**step
    )

    # Create training step function with optimization wrapper
    @StaticCaptureTraining(
        model=model,
        optim=optimizer,
        cuda_graph_warmup=11,
    )
    def training_step(invar, outvar):
        predvar = model(invar)
        loss = mse(predvar, outvar)
        return loss

    # run for 20 iterations
    for i in range(21):
        batch = next(iter(dataloader))
        true = batch["darcy"]
        input = batch["permeability"]

if __name__ == "__main__":
    main()

# [code]
