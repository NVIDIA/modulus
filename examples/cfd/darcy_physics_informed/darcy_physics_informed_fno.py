# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
import torch.nn.functional as F

from modulus.models.fno import FNO
from modulus.launch.logging import LaunchLogger
from modulus.launch.utils.checkpoint import save_checkpoint
from modulus.sym.eq.pdes.diffusion import Diffusion

from utils import HDF5MapStyleDataset
from ops import dx, ddx


def validation_step(model, dataloader, epoch):
    """Validation Step"""
    model.eval()

    with torch.no_grad():
        loss_epoch = 0
        for data in dataloader:
            invar, outvar, _, _ = data
            out = model(invar[:, 0].unsqueeze(dim=1))

            loss_epoch += F.mse_loss(outvar, out)

        # convert data to numpy
        outvar = outvar.detach().cpu().numpy()
        predvar = out.detach().cpu().numpy()

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


@hydra.main(version_base="1.3", config_path="conf", config_name="config_pino.yaml")
def main(cfg: DictConfig):

    # CUDA support
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    LaunchLogger.initialize()

    # Use Diffusion equation for the Darcy PDE
    darcy = Diffusion(T="u", time=False, dim=2, D="k", Q=1.0 * 4.49996e00 * 3.88433e-03)
    darcy_node = darcy.make_nodes()

    dataset = HDF5MapStyleDataset(
        to_absolute_path("./datasets/Darcy_241/train.hdf5"), device=device
    )
    validation_dataset = HDF5MapStyleDataset(
        to_absolute_path("./datasets/Darcy_241/validation.hdf5"), device=device
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    validation_dataloader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

    model = FNO(
        in_channels=cfg.model.fno.in_channels,
        out_channels=cfg.model.fno.out_channels,
        decoder_layers=cfg.model.fno.decoder_layers,
        decoder_layer_size=cfg.model.fno.decoder_layer_size,
        dimension=cfg.model.fno.dimension,
        latent_channels=cfg.model.fno.latent_channels,
        num_fno_layers=cfg.model.fno.num_fno_layers,
        num_fno_modes=cfg.model.fno.num_fno_modes,
        padding=cfg.model.fno.padding,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
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

                # Compute forward pass
                out = model(invar[:, 0].unsqueeze(dim=1))

                dxf = 1.0 / out.shape[-2]
                dyf = 1.0 / out.shape[-1]

                # Compute gradients using finite difference
                sol_x = dx(out, dx=dxf, channel=0, dim=1, order=1, padding="zeros")
                sol_y = dx(out, dx=dyf, channel=0, dim=0, order=1, padding="zeros")
                sol_x_x = ddx(out, dx=dxf, channel=0, dim=1, order=1, padding="zeros")
                sol_y_y = ddx(out, dx=dyf, channel=0, dim=0, order=1, padding="zeros")

                k_x = dx(invar, dx=dxf, channel=0, dim=1, order=1, padding="zeros")
                k_y = dx(invar, dx=dxf, channel=0, dim=0, order=1, padding="zeros")

                k, _, _ = (
                    invar[:, 0],
                    invar[:, 1],
                    invar[:, 2],
                )

                pde_out = darcy_node[0].evaluate(
                    {
                        "u__x": sol_x,
                        "u__y": sol_y,
                        "u__x__x": sol_x_x,
                        "u__y__y": sol_y_y,
                        "k": k,
                        "k__x": k_x,
                        "k__y": k_y,
                    }
                )

                pde_out_arr = pde_out["diffusion_u"]
                pde_out_arr = F.pad(
                    pde_out_arr[:, :, 2:-2, 2:-2], [2, 2, 2, 2], "constant", 0
                )
                loss_pde = F.l1_loss(pde_out_arr, torch.zeros_like(pde_out_arr))

                # Compute data loss
                loss_data = F.mse_loss(outvar, out)

                # Compute total loss
                loss = loss_data + 1 / 240 * cfg.phy_wt * loss_pde

                # Backward pass and optimizer and learning rate update
                loss.backward()
                optimizer.step()
                scheduler.step()
                log.log_minibatch(
                    {"loss_data": loss_data.detach(), "loss_pde": loss_pde.detach()}
                )

            log.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})

        with LaunchLogger("valid", epoch=epoch) as log:
            error = validation_step(model, validation_dataloader, epoch)
            log.log_epoch({"Validation error": error})

        save_checkpoint(
            "./checkpoints",
            models=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
        )


if __name__ == "__main__":
    main()
