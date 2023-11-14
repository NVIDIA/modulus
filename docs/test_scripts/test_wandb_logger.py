# [imports]
import torch
import numpy as np
import modulus
import matplotlib.pyplot as plt
from modulus.datapipes.benchmarks.darcy import Darcy2D
from modulus.models.fno.fno import FNO
from modulus.metrics.general.mse import mse
from modulus.launch.logging import PythonLogger, LaunchLogger, initialize_wandb
# [imports]

# [code]
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

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lr_lambda=lambda step: 0.85**step
)

# Initialize the console logger
logger = PythonLogger("main")  # General python logger

# Initialize the MLFlow logger
initialize_wandb(
    project="Modulus Tutorials",
    name="Simple Modulus Tutorials",
    entity="Modulus MLFLow Tutorial",
    mode="offline",
)
LaunchLogger.initialize(use_wandb=True)

# Use logger methods to track various information during training
logger.info("Starting Training!")
# run for 20 iterations
for i in range(21):
    # wrap the epoch in launch logger to control frequency of output for console logs
    with LaunchLogger("train", epoch=i) as log:    
        batch = next(iter(dataloader))
        true = batch["darcy"]
        pred = model(batch["permeability"])
        loss = mse(pred, true)
        loss.backward()
        optimizer.step()
        scheduler.step()
        log.log_epoch({"Loss": loss.detach().cpu().numpy()})

        log.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})
        

logger.info("Finished Training!")
# [code]
