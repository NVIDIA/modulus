import modulus
import modulus.sym
from modulus.sym.hydra import to_absolute_path
from modulus.sym.distributed.manager import DistributedManager

import torch
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from typing import Union
import h5py
import numpy as np
import hydra
from omegaconf import DictConfig
from modulus.models.fno import FNO
from torch.utils.data import Dataset, DataLoader
from modulus.launch.logging import PythonLogger, LaunchLogger
from torch.nn import MSELoss
from torch.optim import Adam, lr_scheduler
from modulus.launch.utils import load_checkpoint, save_checkpoint
import torch.nn.functional as F

class HDF5MapStyleDataset(Dataset):
    """Simple map-stype HDF5 dataset"""

    def __init__(
        self,
        file_path,
        device: Union[str, torch.device] = "cuda",
    ):
        self.file_path = file_path
        with h5py.File(file_path, "r") as f:
            self.keys = list(f.keys())
        # Set up device, needed for pipeline
        if isinstance(device, str):
            device = torch.device(device)
        # Need a index id if cuda
        if device.type == "cuda" and device.index == None:
            device = torch.device("cuda:0")
        self.device = device
    def __len__(self):
        with h5py.File(self.file_path, "r") as f:
            return len(f[self.keys[0]])
    def __getitem__(self, idx):
        data = {}
        with h5py.File(self.file_path, "r") as f:
            for key in self.keys:
                data[key] = np.array(f[key][idx])
        invar = torch.from_numpy(data["wavefield_in"])
        outvar = torch.from_numpy(data["wavefield_sol"])
        if self.device.type == "cuda":
            # Move tensors to GPU
            invar = invar.cuda()
            outvar = outvar.cuda()

        return invar, outvar


@hydra.main(version_base="1.3", config_path="./conf/", config_name="config_FNO_launch")
def main(cfg: DictConfig) -> None:
    DistributedManager.initialize()  # Only call this once in the entire script!
    dist = DistributedManager()  # call if required elsewhere

    # initialize monitoring
    logger = PythonLogger(name="brain_fno")

    # initialize monitoring
    LaunchLogger.initialize()

    ### get the data
    train_path = to_absolute_path(
        "./train_sets/data_scale_train.hdf5"
    )
    train_dataset = HDF5MapStyleDataset(train_path, device="cuda")
    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg.training.batch_size, shuffle=True
    )

    test_path = to_absolute_path(
        "./train_sets/data_scale_test.hdf5")
    # rewrite this into class
    test_dataset = HDF5MapStyleDataset(test_path, device="cuda")
    test_dataloader = DataLoader(
    test_dataset, batch_size=cfg.test.batch_size, shuffle=False
    )
    # set device as GPU
    device = "cuda"

    model = FNO(
        in_channels=cfg.arch.fno.in_channels,
        out_channels=cfg.arch.decoder.out_features ,
        decoder_layers=cfg.arch.decoder.layers,
        decoder_layer_size=cfg.arch.decoder.layer_size,
        dimension=cfg.arch.fno.dimension,
        latent_channels=cfg.arch.fno.latent_channels,
        num_fno_layers=cfg.arch.fno.fno_layers,
        num_fno_modes=cfg.arch.fno.fno_modes,
        padding=cfg.arch.fno.padding,
    ).to(dist.device)

    loss_fun = MSELoss(reduction="sum")

    optimizer = Adam(model.parameters())

    scheduler = lr_scheduler.ExponentialLR(
        optimizer, gamma=(cfg.scheduler.decay_rate)**(1.0/cfg.scheduler.decay_steps) )


    loaded_epoch = load_checkpoint(
        "./checkpoints",
        models=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device="cuda",
    )

    # Training loop
    for epoch in range(max(1, loaded_epoch + 1), cfg.max_epochs + 1):
        # wrap epoch in launch logger for console logs
        with LaunchLogger(
            "train",
            epoch=epoch,
            num_mini_batch=len(train_dataloader),
            epoch_alert_freq=1,
        ) as log:
            # go through the full dataset
            for i, data in enumerate(train_dataloader):
                invar, outvar = data
                invar = invar.float()
                outvar = outvar.float()
                optimizer.zero_grad()
                outpred = model(invar)
                loss = loss_fun(outvar, outpred)
                loss.backward()
                optimizer.step()
                scheduler.step()
                log.log_minibatch({"loss": loss.detach()})

            log.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})

        if epoch % cfg.checkpoint_save_freq == 0:
            save_checkpoint(
                "./checkpoints",
                models=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
            )
    logger.info("Finished Training")

if __name__ == "__main__":
    main()

