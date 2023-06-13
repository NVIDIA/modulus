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

import numpy as np
import torch
import hydra
import matplotlib.pyplot as plt
import xarray
import datetime

from torch.nn.parallel import DistributedDataParallel
import torch.nn.functional as F
from omegaconf import DictConfig

from era5_hdf5 import ERA5HDF5Datapipe
from modulus.distributed import DistributedManager
from modulus.utils import StaticCaptureTraining, StaticCaptureEvaluateNoGrad

from modulus.models.dlwp import DLWP

from cube_sphere_plotter_no_subplots import cube_sphere_plotter
from modulus.launch.logging import LaunchLogger, PythonLogger, initialize_mlflow
from modulus.launch.utils import load_checkpoint, save_checkpoint
import modulus.utils.sfno.zenith_angle as zenith_angle
from torch.optim.lr_scheduler import ReduceLROnPlateau
from hydra.utils import to_absolute_path

Tensor = torch.Tensor


def loss_func(x, y, p=2.0):
    yv = y.reshape(x.size()[0], -1)
    xv = x.reshape(x.size()[0], -1)
    diff_norms = torch.linalg.norm(xv - yv, ord=p, dim=1)
    y_norms = torch.linalg.norm(yv, ord=p, dim=1)

    return torch.mean(diff_norms / y_norms)


def prepare_input(
    input_list,
    datapipe_start_year,
    idx_list,
    year_idx,
    lsm,
    longrid,
    latgrid,
    topographic_height,
    device,
    batchsize,
):
    # TODO: Add an assertion check here to ensure the idx_list has same number of elements as the input_list!
    for i in range(len(input_list)):
        tisr = []
        sub_idx_list = idx_list[:, i]
        for j, id in enumerate(sub_idx_list):
            year = datapipe_start_year + year_idx[j]
            start_date = datetime.datetime(year.item(), 1, 1, 0, 0)
            time_delta = datetime.timedelta(hours=id.item() * 6)
            result_time = start_date + time_delta
            # print(result_time, year.item(), id.item())
            tisr.append(
                np.maximum(
                    zenith_angle.cos_zenith_angle(result_time, longrid, latgrid), 0
                )
                - (1 / np.pi)
            )  # subtract mean value
        tisr = np.stack(tisr, axis=0)
        tisr = torch.tensor(tisr, dtype=input_list[0].dtype).to(device).unsqueeze(dim=1)
        input_list[i] = torch.cat((input_list[i], tisr), dim=1)

    input_model = torch.cat(
        input_list, dim=1
    )  # concat the time dimension into channels

    repeat_vals = (batchsize, -1, -1, -1, -1)  # repeat along batch dimension
    lsm_tensor = (
        torch.tensor(lsm, dtype=input_list[0].dtype).to(device).unsqueeze(dim=0)
    )
    lsm_tensor = lsm_tensor.expand(*repeat_vals)
    # normalize topographic height
    topographic_height = (topographic_height - 3.724e03) / 8.349e03
    topographic_height_tensor = (
        torch.tensor(topographic_height, dtype=input_list[0].dtype)
        .to(device)
        .unsqueeze(dim=0)
    )
    topographic_height_tensor = topographic_height_tensor.expand(*repeat_vals)

    input_model = torch.cat((input_model, lsm_tensor, topographic_height_tensor), dim=1)
    return input_model


@torch.no_grad()
def validation_step(
    eval_step,
    arch,
    datapipe,
    nr_output_channels=14,
    num_input_steps=2,
    lsm=None,
    longrid=None,
    latgrid=None,
    topographic_height=None,
    epoch=0,
):
    loss_epoch = 0
    num_examples = 0
    # Dealing with DDP wrapper
    if hasattr(arch, "module"):
        arch = arch.module
    arch.eval()
    for i, data in enumerate(datapipe):
        invar = data[0]["invar"]
        outvar = data[0]["outvar"]
        invar_list = torch.split(invar, 1, dim=1)  # split along the time dimension
        invar_list = [tensor.squeeze(dim=1) for tensor in invar_list]
        invar_model = prepare_input(
            invar_list,
            2016,
            data[0]["invar_idx"],
            data[0]["year_idx"],
            lsm,
            longrid,
            latgrid,
            topographic_height,
            invar.device,
            invar.size(0),
        )

        # multi step loss.
        for t in range(outvar.shape[1] // num_input_steps):
            output = eval_step(arch, invar_model)
            invar_model = output
            invar_list = list(
                torch.split(invar_model, (nr_output_channels // num_input_steps), dim=1)
            )
            invar_model = prepare_input(
                invar_list,
                2016,
                data[0]["outvar_idx"][
                    :, t * num_input_steps : (t + 1) * num_input_steps
                ],
                data[0]["year_idx"],
                lsm,
                longrid,
                latgrid,
                topographic_height,
                invar.device,
                invar.size(0),
            )
            output_list = torch.split(
                output, nr_output_channels // num_input_steps, dim=1
            )
            output_list = [tensor.unsqueeze(dim=1) for tensor in output_list]
            output = torch.cat(output_list, dim=1)
            loss_epoch += F.mse_loss(
                outvar[:, t * num_input_steps : t * num_input_steps + num_input_steps],
                output,
            ).detach()
        num_examples += invar.shape[0]

    arch.train()
    return loss_epoch.detach() / num_examples


@torch.no_grad()
def plotting_step(
    arch,
    datapipe,
    datapipe_start_year,
    channels=[0, 1],
    epoch=0,
    nr_output_channels=14,
    num_input_steps=2,
    lsm=None,
    longrid=None,
    latgrid=None,
    topographic_height=None,
):
    arch.eval()
    for i, data in enumerate(datapipe):
        invar = data[0]["invar"].detach()
        outvar = data[0]["outvar"].cpu().detach()
        invar_list = torch.split(invar, 1, dim=1)  # split along the time dimension
        invar_list = [tensor.squeeze(dim=1) for tensor in invar_list]
        invar_model = prepare_input(
            invar_list,
            datapipe_start_year,
            data[0]["invar_idx"],
            data[0]["year_idx"],
            lsm,
            longrid,
            latgrid,
            topographic_height,
            invar.device,
            invar.size(0),
        )

        pred_outvar = torch.zeros_like(outvar)

        # non over-lapping rollout
        for t in range(outvar.shape[1] // num_input_steps):
            # print(t)
            output = arch(invar_model)
            invar_model = output
            invar_list = list(
                torch.split(invar_model, (nr_output_channels // num_input_steps), dim=1)
            )
            # print(data[0]["outvar_idx"][:,t*num_input_steps:(t+1)*num_input_steps], data[0]["year_idx"])
            invar_model = prepare_input(
                invar_list,
                datapipe_start_year,
                data[0]["outvar_idx"][
                    :, t * num_input_steps : (t + 1) * num_input_steps
                ],
                data[0]["year_idx"],
                lsm,
                longrid,
                latgrid,
                topographic_height,
                invar.device,
                invar.size(0),
            )

            output_list = torch.split(
                output, nr_output_channels // num_input_steps, dim=1
            )  # split along the channel dimension
            output_list = [tensor.unsqueeze(dim=1) for tensor in output_list]
            output = torch.cat(output_list, dim=1).cpu().detach()
            pred_outvar[:, t * 2] = output[:, 0]
            pred_outvar[:, t * 2 + 1] = output[:, 1]

        # Plotting
        if i == 0:
            pred_outvar = pred_outvar.numpy()
            outvar = outvar.numpy()
            for chan in channels:
                plt.close("all")
                fig, ax = plt.subplots(
                    3, pred_outvar.shape[1], figsize=(4 * outvar.shape[1], 8)
                )
                for t in range(outvar.shape[1]):
                    vmin, vmax = np.min(pred_outvar[0, t, chan]), np.max(
                        pred_outvar[0, t, chan]
                    )
                    im = ax[0, t].imshow(
                        cube_sphere_plotter(pred_outvar[0, t, chan]),
                        vmin=vmin,
                        vmax=vmax,
                        origin="lower",
                    )
                    fig.colorbar(im, ax=ax[0, t])
                    im = ax[1, t].imshow(
                        cube_sphere_plotter(outvar[0, t, chan]),
                        vmin=vmin,
                        vmax=vmax,
                        origin="lower",
                    )
                    fig.colorbar(im, ax=ax[1, t])
                    im = ax[2, t].imshow(
                        cube_sphere_plotter(
                            pred_outvar[0, t, chan] - outvar[0, t, chan]
                        ),
                        origin="lower",
                    )
                    fig.colorbar(im, ax=ax[2, t])
                    ax[0, t].set_xticks([])
                    ax[0, t].set_yticks([])
                    ax[1, t].set_xticks([])
                    ax[1, t].set_yticks([])
                    ax[2, t].set_xticks([])
                    ax[2, t].set_yticks([])
                    ax[0, t].set_title(f"Pred: {t}")
                    ax[1, t].set_title(f"True: {t}")
                    ax[2, t].set_title(f"Diff: {t}")

                fig.savefig(f"era5_validation_channel{chan}_epoch{epoch}.png", dpi=300)


@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist = DistributedManager()

    initialize_mlflow(
        experiment_name="Modulus-Launch-Dev",
        experiment_desc="Modulus launch development",
        run_name="DLWP-Training",
        run_desc="DLWP ERA5 Training",
        user_name="Modulus User",
        mode="offline",
    )
    LaunchLogger.initialize(use_mlflow=True)
    logger = PythonLogger("main")  # General python logger

    nr_input_channels = cfg.nr_input_channels
    nr_output_channels = cfg.nr_output_channels
    num_input_steps = 2
    num_output_steps = 4

    arch = DLWP(
        nr_input_channels=nr_input_channels, nr_output_channels=nr_output_channels
    ).to(dist.device)

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

    # load static datasets
    lsm = xarray.open_dataset(
        to_absolute_path("./static_datasets/land_sea_mask_rs_cs.nc")
    )["lsm"].values
    topographic_height = xarray.open_dataset(
        to_absolute_path("./static_datasets/geopotential_rs_cs.nc")
    )["z"].values
    latlon_grids = xarray.open_dataset(
        to_absolute_path("./static_datasets/latlon_grid_field_rs_cs.nc")
    )
    latgrid, longrid = latlon_grids["latgrid"].values, latlon_grids["longrid"].values

    optimizer = torch.optim.Adam(
        arch.parameters(),
        betas=(0.9, 0.999),
        lr=0.001,
        weight_decay=0.0,
    )

    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.2, patience=20, min_lr=1e-6, verbose=True
    )

    datapipe = ERA5HDF5Datapipe(
        data_dir="/data/train/",
        stats_dir="/data/stats/",
        channels=None,
        num_samples_per_year=1460
        - num_input_steps
        - num_output_steps,  # Need better shard fix
        # num_samples_per_year=1408,  # Need better shard fix
        num_input_steps=num_input_steps,
        num_output_steps=num_output_steps,
        batch_size=cfg.batch_size.train,
        grid_type="cubesphere",
        patch_size=None,
        device=dist.device,
        num_workers=1,
        shuffle=True,
        process_rank=dist.rank,
        world_size=dist.world_size,
    )

    # if dist.rank == 0:
    val_datapipe = ERA5HDF5Datapipe(
        data_dir="/data/test/",
        stats_dir="/data/stats/",
        channels=None,
        num_samples_per_year=1460
        - num_input_steps
        - num_output_steps,  # Need better shard fix
        # num_samples_per_year=1408,  # Need better shard fix
        num_input_steps=num_input_steps,
        num_output_steps=num_output_steps,
        batch_size=cfg.batch_size.validate,
        grid_type="cubesphere",
        patch_size=None,
        device=dist.device,
        num_workers=1,
        shuffle=False,
        process_rank=dist.rank,
        world_size=dist.world_size,
    )

    if dist.rank == 0:
        out_of_sample_datapipe = ERA5HDF5Datapipe(
            data_dir="/data/out_of_sample/",
            stats_dir="/data/stats/",
            channels=None,
            num_samples_per_year=4,  # Need better shard fix
            num_input_steps=num_input_steps,
            num_output_steps=16,
            batch_size=cfg.batch_size.out_of_sample,
            grid_type="cubesphere",
            patch_size=None,
            device=dist.device,
            num_workers=1,
            shuffle=False,
        )

    loaded_epoch = load_checkpoint(
        "./checkpoints",
        models=arch,
        optimizer=optimizer,
        scheduler=scheduler,
        device=dist.device,
    )

    @StaticCaptureEvaluateNoGrad(
        model=arch, logger=logger, use_graphs=False, use_amp=False
    )
    def eval_step_forward(arch, invar):
        return arch(invar)

    @StaticCaptureTraining(
        model=arch, optim=optimizer, logger=logger, use_graphs=False, use_amp=False
    )
    def train_step_forward(arch, invar, outvar):
        invar_list = torch.split(invar, 1, dim=1)  # split along the time dimension
        invar_list = [tensor.squeeze(dim=1) for tensor in invar_list]
        invar_model = prepare_input(
            invar_list,
            1980,
            data[0]["invar_idx"],
            data[0]["year_idx"],
            lsm,
            longrid,
            latgrid,
            topographic_height,
            dist.device,
            invar.size(0),
        )

        # multi step loss.
        loss = 0.0
        for t in range(outvar.shape[1] // num_input_steps):
            output = arch(invar_model)
            invar_model = output
            invar_list = list(
                torch.split(invar_model, (nr_output_channels // num_input_steps), dim=1)
            )
            invar_model = prepare_input(
                invar_list,
                1980,
                data[0]["outvar_idx"][
                    :, t * num_input_steps : (t + 1) * num_input_steps
                ],
                data[0]["year_idx"],
                lsm,
                longrid,
                latgrid,
                topographic_height,
                dist.device,
                invar.size(0),
            )
            output_list = torch.split(
                output, nr_output_channels // num_input_steps, dim=1
            )
            output_list = [tensor.unsqueeze(dim=1) for tensor in output_list]
            output = torch.cat(output_list, dim=1)
            loss += F.mse_loss(
                outvar[:, t * num_input_steps : t * num_input_steps + num_input_steps],
                output,
            )

        return loss

    # Main training loop
    max_epoch = cfg.max_epoch
    for epoch in range(max(1, loaded_epoch + 1), max_epoch + 1):
        with LaunchLogger(
            "train", epoch=epoch, num_mini_batch=len(datapipe), epoch_alert_freq=1
        ) as log:
            for data in datapipe:
                invar = data[0]["invar"]
                outvar = data[0]["outvar"]

                loss = train_step_forward(arch, invar, outvar)
                log.log_minibatch({"Mini-batch loss": loss.detach()})
            log.log_epoch({"Learning Rate": optimizer.param_groups[0]["lr"]})

        if dist.rank == 0:
            with LaunchLogger("valid", epoch=epoch) as log:
                val_loss = validation_step(
                    eval_step_forward,
                    arch,
                    val_datapipe,
                    nr_output_channels,
                    num_input_steps,
                    lsm,
                    longrid,
                    latgrid,
                    topographic_height,
                    epoch=epoch,
                )
                log.log_epoch({"Val loss": val_loss})

                # plot the data on out of sample dataset
                plotting_step(
                    arch,
                    out_of_sample_datapipe,
                    2018,
                    [0, 1, 2, 3, 4, 5, 6],
                    epoch,
                    nr_output_channels,
                    num_input_steps,
                    lsm,
                    longrid,
                    latgrid,
                    topographic_height,
                )

        if dist.world_size > 1:
            torch.distributed.barrier()

        # scheduler step
        scheduler.step(val_loss)

        if epoch % 2 == 0 and dist.rank == 0:
            save_checkpoint(
                "./checkpoints",
                models=arch,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
            )


if __name__ == "__main__":
    main()
