# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hydra
from omegaconf import DictConfig
import h5py
import torch
import numpy as np
from sympy import Symbol, Function
import matplotlib.pyplot as plt
from hydra.utils import to_absolute_path
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from typing import Union
from modulus.models.mlp import FullyConnected
from modulus.models.fno import FNO
from modulus.launch.logging import PythonLogger, LaunchLogger
from modulus.launch.utils.checkpoint import save_checkpoint

from modulus.sym.eq.pde import PDE


class Darcy(PDE):
    """Darcy PDE using Modulus Sym"""

    name = "Darcy"

    def __init__(self):

        # time
        x, y = Symbol("x"), Symbol("y")

        # make input variables
        input_variables = {"x": x, "y": y}

        # make sol function
        u = Function("sol")(*input_variables)
        k = Function("K")(*input_variables)
        f = 1.0

        # set equation
        self.equations = {}
        self.equations["darcy"] = (
            f
            + k.diff(x) * u.diff(x)
            + k * u.diff(x).diff(x)
            + k.diff(y) * u.diff(y)
            + k * u.diff(y).diff(y)
        )


class HDF5MapStyleDataset(Dataset):
    """Simple map-style HDF5 dataset"""

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

        invar = torch.cat(
            [
                torch.from_numpy(
                    (data["Kcoeff"][:, :240, :240] - 7.48360e00) / 4.49996e00
                ),
                torch.from_numpy(data["Kcoeff_x"][:, :240, :240]),
                torch.from_numpy(data["Kcoeff_y"][:, :240, :240]),
            ]
        )
        outvar = torch.from_numpy(
            (data["sol"][:, :240, :240] - 5.74634e-03) / 3.88433e-03
        )

        x = np.linspace(0, 1, 240)
        y = np.linspace(0, 1, 240)

        xx, yy = np.meshgrid(x, y)
        x_invar = torch.from_numpy(xx.astype(np.float32)).reshape(-1, 1)
        y_invar = torch.from_numpy(yy.astype(np.float32)).reshape(-1, 1)

        if self.device.type == "cuda":
            # Move tensors to GPU
            invar = invar.cuda()
            outvar = outvar.cuda()
            x_invar = x_invar.cuda()
            y_invar = y_invar.cuda()

        return invar, outvar, x_invar, y_invar


def validation_step(model_branch, model_trunk, dataloader, epoch):
    """Validation Step"""
    model_branch.eval()
    model_trunk.eval()

    with torch.no_grad():
        loss_epoch = 0
        for data in dataloader:
            invar, outvar, x_invar, y_invar = data
            coords = torch.cat(
                (x_invar.squeeze(dim=2), y_invar.squeeze(dim=2)), dim=0
            ).reshape(-1, 2)

            branch_out = model_branch(invar)
            trunk_out = model_trunk(coords)
            branch_out = branch_out.reshape(-1, 240 * 240)
            trunk_out = trunk_out.reshape(-1, 240 * 240)
            deepo_out = trunk_out * branch_out
            deepo_out = deepo_out.reshape(-1, 1, 240, 240)
            loss_epoch += F.mse_loss(outvar, deepo_out)

        # convert data to numpy
        outvar = outvar.detach().cpu().numpy()
        predvar = deepo_out.detach().cpu().numpy()
        x_invar = x_invar.detach().cpu().numpy()
        y_invar = y_invar.detach().cpu().numpy()
        invar = invar.detach().cpu().numpy()

        # plotting
        fig, ax = plt.subplots(1, 3, figsize=(25, 5))

        d_min = np.min(outvar[0, 0, ...])
        d_max = np.max(outvar[0, 0, ...])

        im = ax[0].imshow(outvar[0, 0, ...], vmin=d_min, vmax=d_max)
        plt.colorbar(im, ax=ax[0])
        im = ax[1].imshow(predvar[0, 0, ...], vmin=d_min, vmax=d_max)
        plt.colorbar(im, ax=ax[1])
        im = ax[2].imshow(np.abs(predvar[0, 0, ...] - outvar[0, 0, ...]))
        plt.colorbar(im, ax=ax[2])

        ax[0].set_title("True")
        ax[1].set_title("Pred")
        ax[2].set_title("Difference")

        fig.savefig(f"results_{epoch}.png")
        plt.close()

        return loss_epoch / len(dataloader)


@hydra.main(version_base="1.3", config_path="conf", config_name="config.yaml")
def main(cfg: DictConfig):

    logger = PythonLogger("main")  # General python logger
    LaunchLogger.initialize()

    darcy = Darcy()
    darcy_node = darcy.make_nodes()

    dataset = HDF5MapStyleDataset(
        to_absolute_path("./datasets/Darcy_241/train.hdf5"), device="cuda"
    )
    validation_dataset = HDF5MapStyleDataset(
        to_absolute_path("./datasets/Darcy_241/validation.hdf5"), device="cuda"
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

    model_branch = FNO(
        in_channels=3,
        out_channels=1,
        decoder_layers=1,
        decoder_layer_size=32,
        dimension=2,
        latent_channels=32,
        num_fno_layers=4,
        num_fno_modes=12,
        padding=9,
    ).to("cuda")

    model_trunk = FullyConnected(
        in_features=2,
        out_features=1,
        layer_size=128,
        num_layers=3,
    ).to("cuda")

    from itertools import chain

    optimizer = torch.optim.Adam(
        chain(model_branch.parameters(), model_trunk.parameters()),
        betas=(0.9, 0.999),
        lr=0.001,
        weight_decay=0.0,
    )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999948708)

    for epoch in range(20):
        # wrap epoch in launch logger for console logs
        with LaunchLogger(
            "train",
            epoch=epoch,
            num_mini_batch=len(dataloader),
            epoch_alert_freq=10,
        ) as log:
            for data in dataloader:
                optimizer.zero_grad()
                invar = data[0]
                outvar = data[1]
                x_invar = data[2].squeeze(dim=2).requires_grad_(True)
                y_invar = data[3].squeeze(dim=2).requires_grad_(True)
                coords = torch.cat((x_invar, y_invar), dim=0).reshape(-1, 2)

                # compute forward pass
                branch_out = model_branch(invar)
                trunk_out = model_trunk(coords)
                branch_out = branch_out.reshape(-1, 240 * 240)
                trunk_out = trunk_out.reshape(-1, 240 * 240)
                deepo_out = trunk_out * branch_out

                # Compute physics loss
                # note: the derivative computation can be done using Modulus-Sym
                # utilities. However, for the purposes of this example, we show it using
                # torch.autograd. This example will soon be updated to use the graph and
                # autograd computation from Modulus Sym.
                grad_sol = torch.autograd.grad(
                    deepo_out.sum(), [x_invar, y_invar], create_graph=True
                )
                sol_x = grad_sol[0]
                sol_y = grad_sol[1]

                sol_x_x = torch.autograd.grad(
                    sol_x.sum(), [x_invar], create_graph=True
                )[0]
                sol_y_y = torch.autograd.grad(
                    sol_y.sum(), [y_invar], create_graph=True
                )[0]

                k, k_x, k_y = (
                    invar[:, 0].reshape(-1, 240 * 240),
                    invar[:, 1].reshape(-1, 240 * 240),
                    invar[:, 2].reshape(-1, 240 * 240),
                )

                pde_out = darcy_node[0].evaluate(
                    {
                        "sol__x": sol_x,
                        "sol__y": sol_y,
                        "sol__x__x": sol_x_x,
                        "sol__y__y": sol_y_y,
                        "K": k,
                        "K__x": k_x,
                        "K__y": k_y,
                    }
                )

                pde_out_arr = pde_out["darcy"]
                pde_out_arr = pde_out_arr.reshape(-1, 240, 240)
                (
                    pde_out_arr[:, :2, :],
                    pde_out_arr[:, -2:, :],
                    pde_out_arr[:, :, :2],
                    pde_out_arr[:, :, -2:],
                ) = (0, 0, 0, 0)
                loss_pde = pde_out_arr.pow(2).mean()

                # Compute data loss
                deepo_out = deepo_out.reshape(-1, 1, 240, 240)
                loss_data = F.mse_loss(outvar, deepo_out)

                # Compute total loss
                loss = loss_data + 0.01 * loss_pde

                # Backward pass and optimizer and learning rate update
                loss.backward()
                optimizer.step()
                scheduler.step()
                log.log_minibatch(
                    {"loss_data": loss_data.detach(), "loss_pde": loss_pde.detach()}
                )

            log.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})

        with LaunchLogger("valid", epoch=epoch) as log:
            error = validation_step(
                model_branch, model_trunk, validation_dataloader, epoch
            )
            log.log_epoch({"Validation error": error})

        save_checkpoint(
            "./checkpoints",
            models=[model_branch, model_trunk],
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
        )


if __name__ == "__main__":
    main()
