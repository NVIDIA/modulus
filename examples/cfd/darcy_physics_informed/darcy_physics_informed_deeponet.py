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
import torch
import numpy as np
import matplotlib.pyplot as plt
from hydra.utils import to_absolute_path
import torch.nn.functional as F
from torch.utils.data import DataLoader
from itertools import chain

from modulus.models.mlp import FullyConnected
from modulus.models.fno import FNO
from modulus.launch.logging import LaunchLogger
from modulus.launch.utils.checkpoint import save_checkpoint

from utils import HDF5MapStyleDataset
from darcy_pde import Darcy


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

            branch_out = model_branch(invar[:, 0].unsqueeze(dim=1))
            trunk_out = model_trunk(coords)
            branch_out = branch_out.reshape(-1, 240 * 240)
            trunk_out = trunk_out.reshape(-1, 240 * 240)
            deepo_out = trunk_out * branch_out
            deepo_out = deepo_out.reshape(-1, 1, 240, 240)
            loss_epoch += F.mse_loss(outvar, deepo_out)

        # convert data to numpy
        outvar = outvar.detach().cpu().numpy()
        predvar = deepo_out.detach().cpu().numpy()

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
        in_channels=cfg.model.fno.in_channels,
        out_channels=cfg.model.fno.out_channels,
        decoder_layers=cfg.model.fno.decoder_layers,
        decoder_layer_size=cfg.model.fno.decoder_layer_size,
        dimension=cfg.model.fno.dimension,
        latent_channels=cfg.model.fno.latent_channels,
        num_fno_layers=cfg.model.fno.num_fno_layers,
        num_fno_modes=cfg.model.fno.num_fno_modes,
        padding=cfg.model.fno.padding,
    ).to("cuda")

    model_trunk = FullyConnected(
        in_features=cfg.model.fc.in_features,
        out_features=cfg.model.fc.out_features,
        layer_size=cfg.model.fc.layer_size,
        num_layers=cfg.model.fc.num_layers,
    ).to("cuda")

    optimizer = torch.optim.Adam(
        chain(model_branch.parameters(), model_trunk.parameters()),
        betas=(0.9, 0.999),
        lr=cfg.start_lr,
        weight_decay=0.0,
    )

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.gamma)

    for epoch in range(cfg.max_epochs):
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
                x_invar = data[2].squeeze(dim=2).reshape(-1, 1).requires_grad_(True)
                y_invar = data[3].squeeze(dim=2).reshape(-1, 1).requires_grad_(True)
                coords = torch.cat((x_invar, y_invar), dim=1)

                # compute forward pass
                branch_out = model_branch(invar[:, 0].unsqueeze(dim=1))
                trunk_out = model_trunk(coords)
                branch_out = branch_out.reshape(-1, 1)
                trunk_out = trunk_out.reshape(-1, 1)
                deepo_out = trunk_out * branch_out

                # Compute physics loss
                # note: the derivative computation can be done using Modulus-Sym
                # utilities. However, for the purposes of this example, we show it using
                # torch.autograd.
                grad_sol = torch.autograd.grad(
                    deepo_out.sum(),
                    [x_invar, y_invar],
                    create_graph=True,  # grad_outputs=torch.ones_like(deepo_out)
                )
                sol_x = grad_sol[0]
                sol_y = grad_sol[1]

                sol_x_x = torch.autograd.grad(
                    sol_x.sum(),
                    [x_invar],
                    create_graph=True,  # grad_outputs=torch.ones_like(sol_x)
                )[0]
                sol_y_y = torch.autograd.grad(
                    sol_y.sum(),
                    [y_invar],
                    create_graph=True,  # grad_outputs=torch.ones_like(sol_y)
                )[0]

                k, k_x, k_y = (
                    invar[:, 0].reshape(-1, 1),
                    invar[:, 1].reshape(-1, 1),
                    invar[:, 2].reshape(-1, 1),
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
                pde_out_arr = F.pad(
                    pde_out_arr[:, 2:-2, 2:-2], [2, 2, 2, 2], "constant", 0
                )
                loss_pde = F.l1_loss(pde_out_arr, torch.zeros_like(pde_out_arr))

                # Compute data loss
                deepo_out = deepo_out.reshape(-1, 1, 240, 240)
                loss_data = F.mse_loss(outvar, deepo_out)

                # Compute total loss
                loss = loss_data + cfg.phy_wt * loss_pde

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
