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

import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from datetime import datetime
import xarray as xr
import zarr
import pandas as pd
import hydra
from physicsnemo.distributed import DistributedManager
from omegaconf import DictConfig
from physicsnemo.models import Module

from utils.nn import regression_model_forward, diffusion_model_forward
from utils.data_loader_hrrr_era5 import HrrrEra5Dataset


@hydra.main(version_base=None, config_path="config", config_name="stormcast_inference")
def main(cfg: DictConfig):

    # Initialize
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device

    initial_time = datetime.fromisoformat(cfg.inference.initial_time)
    n_steps = cfg.inference.n_steps

    # Dataset prep
    dataset = HrrrEra5Dataset(cfg.dataset, train=False)

    base_hrrr_channels, hrrr_channels = dataset._get_hrrr_channel_names()
    diffusion_channels = (
        hrrr_channels
        if cfg.dataset.diffusion_channels == "all"
        else cfg.dataset.diffusion_channels
    )
    input_channels = (
        hrrr_channels
        if cfg.dataset.input_channels == "all"
        else cfg.dataset.input_channels
    )

    diffusion_channel_indices = [
        hrrr_channels.index(channel) for channel in diffusion_channels
    ]

    input_channel_indices = [
        list(hrrr_channels).index(channel) for channel in input_channels
    ]

    hrrr_data = xr.open_zarr(
        os.path.join(
            cfg.dataset.location, cfg.dataset.conus_dataset_name, "valid", "2021.zarr"
        )
    )

    invariant_array = dataset._get_invariants()
    invariant_tensor = torch.from_numpy(invariant_array).to(device).repeat(1, 1, 1, 1)

    if len(cfg.inference.output_hrrr_channels) == 0:
        output_hrrr_channels = diffusion_channels.copy()

    vardict: dict[str, int] = {
        hrrr_channel: i for i, hrrr_channel in enumerate(hrrr_channels)
    }

    vardict_era5 = {
        era5_channel: i for i, era5_channel in enumerate(dataset.era5_channels.values)
    }

    color_limits = {
        "u10m": (-5, 5),
        "v10": (-5, 5),
        "t2m": (260, 310),
        "tcwv": (0, 60),
        "msl": (0.1, 0.3),
        "refc": (-10, 30),
    }

    hours_since_jan_01 = int(
        (initial_time - datetime(initial_time.year, 1, 1, 0, 0)).total_seconds() / 3600
    )

    hrrr_channel_indices = [
        list(base_hrrr_channels).index(channel) for channel in hrrr_channels
    ]
    means_hrrr = dataset.means_hrrr[hrrr_channel_indices]
    stds_hrrr = dataset.stds_hrrr[hrrr_channel_indices]
    means_era5 = dataset.means_era5
    stds_era5 = dataset.stds_era5

    # Load pretrained models
    net = Module.from_checkpoint(cfg.inference.regression_checkpoint)
    regression_model = net.to(device)
    net = Module.from_checkpoint(cfg.inference.diffusion_checkpoint)
    diffusion_model = net.to(device)

    # initialize zarr
    zarr_output_path = os.path.join(cfg.inference.rundir, "data.zarr")
    group = zarr.open_group(zarr_output_path, mode="w")
    group.array("latitude", data=hrrr_data["latitude"].values)
    group.array("longitude", data=hrrr_data["longitude"].values)

    edm_prediction_group = group.create_group("edm_prediction")
    noedm_prediction_group = group.create_group("noedm_prediction")
    target_group = group.create_group("target")
    hrrr, _ = dataset[0]["hrrr"]
    assert hrrr.ndim == 3

    grid_size = hrrr.shape[1:]

    for name in output_hrrr_channels:
        target_group.empty(
            name, shape=(n_steps,) + grid_size, chunks=[1, *grid_size], compressor=None
        )
        edm_prediction_group.empty(
            name, shape=(n_steps,) + grid_size, chunks=[1, *grid_size], compressor=None
        )
        noedm_prediction_group.empty(
            name, shape=(n_steps,) + grid_size, chunks=[1, *grid_size], compressor=None
        )

    with torch.no_grad():

        for i in range(n_steps):
            data = dataset[i + hours_since_jan_01]
            print(i)

            if i == 0:
                inp = data["hrrr"][0].cuda().float().unsqueeze(0)
                boundary = data["era5"][0].cuda().float().unsqueeze(0)
                out = inp
                out_edm = out.clone()
                out_noedm = out.clone()

            assert out_edm.shape == (1, len(hrrr_channels)) + grid_size
            assert out_noedm.shape == (1, len(hrrr_channels)) + grid_size
            # write hrrr
            denorm_out_edm = out_edm.cpu().numpy() * stds_hrrr + means_hrrr
            denorm_out_noedm = out_noedm.cpu().numpy() * stds_hrrr + means_hrrr

            for name in output_hrrr_channels:
                k = vardict[name]
                edm_prediction_group[name][i] = denorm_out_edm[0, k]
                noedm_prediction_group[name][i] = denorm_out_noedm[0, k]
                target_data = (
                    data["hrrr"][0][k].cpu().numpy() * stds_hrrr[k] + means_hrrr[k]
                )
                target_group[name][i] = target_data

            if i > n_steps:
                break

            hrrr_0 = out
            out = regression_model_forward(
                regression_model, hrrr_0, boundary, invariant_tensor
            )
            out_noedm = out.clone()
            hrrr_0 = torch.cat(
                (
                    hrrr_0[:, input_channel_indices, :, :],
                    out[:, input_channel_indices, :, :],
                ),
                dim=1,
            )
            edm_corrected_outputs = diffusion_model_forward(
                diffusion_model,
                hrrr_0,
                diffusion_channel_indices,
                invariant_tensor,
                sampler_args=dict(cfg.sampler.args),
            )
            out[0, diffusion_channel_indices] += edm_corrected_outputs[0].float()
            out_edm = out.clone()
            boundary = data["era5"][0].cuda().float().unsqueeze(0)

            varidx = vardict[cfg.inference.plot_var_hrrr]

            fig, ax = plt.subplots(1, 4, figsize=(20, 5))

            pred = out.cpu().numpy()
            tar = data["hrrr"][1].unsqueeze(0).cpu().numpy()
            era5 = data["era5"][0].unsqueeze(0).cpu().numpy()
            pred = pred * stds_hrrr + means_hrrr
            tar = tar * stds_hrrr + means_hrrr
            era5 = era5 * stds_era5 + means_era5

            error = pred - tar

            if cfg.inference.plot_var_hrrr in color_limits:
                im = ax[0].imshow(
                    pred[0, varidx],
                    origin="lower",
                    cmap="magma",
                    clim=color_limits[cfg.inference.plot_var_hrrr],
                )
            else:
                im = ax[0].imshow(pred[0, varidx], origin="lower", cmap="magma")

            fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
            ax[0].set_title(
                "Predicted, {}, \n initial time {} \n lead_time {} hours".format(
                    cfg.inference.plot_var_hrrr, initial_time, i
                )
            )
            if cfg.inference.plot_var_hrrr in color_limits:
                im = ax[1].imshow(
                    tar[0, varidx],
                    origin="lower",
                    cmap="magma",
                    clim=color_limits[cfg.inference.plot_var_hrrr],
                )
            else:
                im = ax[1].imshow(tar[0, varidx], origin="lower", cmap="magma")
            fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
            ax[1].set_title("Actual, {}".format(cfg.inference.plot_var_hrrr))
            if cfg.inference.plot_var_era5 in color_limits:
                im = ax[2].imshow(
                    era5[0, vardict_era5[cfg.inference.plot_var_era5]],
                    origin="lower",
                    cmap="magma",
                    clim=color_limits[cfg.inference.plot_var_era5],
                )
            else:
                im = ax[2].imshow(
                    era5[0, vardict_era5[cfg.inference.plot_var_era5]],
                    origin="lower",
                    cmap="magma",
                )
            fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)
            ax[2].set_title("ERA5, {}".format(cfg.inference.plot_var_era5))
            maxerror = np.max(np.abs(error[0, varidx]))
            im = ax[3].imshow(
                error[0, varidx],
                origin="lower",
                cmap="RdBu_r",
                vmax=maxerror,
                vmin=-maxerror,
            )
            fig.colorbar(im, ax=ax[3], fraction=0.046, pad=0.04)
            ax[3].set_title("Error, {}".format(cfg.inference.plot_var_hrrr))

            plt.savefig(f"{cfg.inference.rundir}/out_{i}.png")

    level_names = [
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "13",
        "15",
        "20",
        "25",
        "30",
        "35",
        "40",
    ]
    vertical_vars = ["u", "v", "t", "q", "z", "p", "w"]
    horizontal_vars = ["msl", "refc", "u10m", "v10m"]

    zarr_group = group
    lons = zarr_group["longitude"][:, :]
    lats = zarr_group["latitude"][:, :]
    initial_time_pd = pd.to_datetime(initial_time)
    val_times = []
    for i in range(n_steps):
        val_times.append(initial_time_pd + pd.Timedelta(seconds=i * hours_since_jan_01))

    def convert_strings_to_ints(string_list):
        return [int(i) for i in string_list]

    model_levels = convert_strings_to_ints(level_names)

    ds_pred_edm = xr.Dataset()
    ds_pred_noedm = xr.Dataset()
    ds_targ = xr.Dataset()

    for var in vertical_vars:
        dsp_edm = xr.Dataset()
        dsp_noedm = xr.Dataset()
        dst = xr.Dataset()

        for i, level in enumerate(level_names):
            key = f"{var}{level}"

            if key in zarr_group["edm_prediction"]:
                # Extract the data from zarr_group
                data_pred_edm = zarr_group["edm_prediction"][key][:, :]
            else:
                data_pred_edm = np.full(
                    (len(val_times), lats.shape[0], lats.shape[1]), np.nan
                )
                print(f"Key {key} not found in Zarr group. Filling with NaNs.")
            if key in zarr_group["noedm_prediction"]:
                # Extract the data from zarr_group
                data_pred_noedm = zarr_group["noedm_prediction"][key][:, :]
            else:
                data_pred_noedm = np.full(
                    (len(val_times), lats.shape[0], lats.shape[1]), np.nan
                )
                print(f"Key {key} not found in Zarr group. Filling with NaNs.")

            if key in zarr_group["target"]:
                data_targ = zarr_group["target"][key][:, :]
            else:
                data_targ = np.full(
                    (len(val_times), lats.shape[0], lats.shape[1]), np.nan
                )
                print(f"Key {key} not found in Zarr group. Filling with NaNs.")

            dap_edm = xr.DataArray(
                data_pred_edm,
                dims=("time", "y", "x"),
                coords={
                    "time": val_times,
                    "y": np.arange(lats.shape[0]),
                    "x": np.arange(lats.shape[1]),
                    "levels": model_levels[i],
                },
            )
            dap_noedm = xr.DataArray(
                data_pred_noedm,
                dims=("time", "y", "x"),
                coords={
                    "time": val_times,
                    "y": np.arange(lats.shape[0]),
                    "x": np.arange(lats.shape[1]),
                    "levels": model_levels[i],
                },
            )
            dat = xr.DataArray(
                data_targ,
                dims=("time", "y", "x"),
                coords={
                    "time": val_times,
                    "y": np.arange(lats.shape[0]),
                    "x": np.arange(lats.shape[1]),
                    "levels": model_levels[i],
                },
            )

            dsp_edm[f"var_{i}"] = dap_edm
            dsp_noedm[f"var_{i}"] = dap_noedm
            dst[f"var_{i}"] = dat

        combined_pred_edm = xr.concat(
            [dsp_edm[var] for var in dsp_edm.data_vars], dim="levels"
        )
        combined_pred_noedm = xr.concat(
            [dsp_noedm[var] for var in dsp_noedm.data_vars], dim="levels"
        )
        combined_targ = xr.concat([dst[var] for var in dst.data_vars], dim="levels")

        reshaped_pred_edm = combined_pred_edm.transpose("time", "levels", "y", "x")
        reshaped_pred_noedm = combined_pred_noedm.transpose("time", "levels", "y", "x")
        reshaped_targ = combined_targ.transpose("time", "levels", "y", "x")

        ds_pred_edm[f"{var}_comb"] = reshaped_pred_edm
        ds_pred_noedm[f"{var}_comb"] = reshaped_pred_noedm
        ds_targ[f"{var}_comb"] = reshaped_targ

    for var in horizontal_vars:
        data_pred_edm = zarr_group["edm_prediction"][var][:, :]
        data_pred_noedm = zarr_group["noedm_prediction"][var][:, :]
        data_targ = zarr_group["target"][var][:, :]
        dap_edm = xr.DataArray(
            data_pred_edm,
            dims=("time", "y", "x"),
            coords={
                "time": val_times,
                "y": np.arange(lats.shape[0]),
                "x": np.arange(lats.shape[1]),
            },
        )
        dap_noedm = xr.DataArray(
            data_pred_noedm,
            dims=("time", "y", "x"),
            coords={
                "time": val_times,
                "y": np.arange(lats.shape[0]),
                "x": np.arange(lats.shape[1]),
            },
        )
        dat = xr.DataArray(
            data_targ,
            dims=("time", "y", "x"),
            coords={
                "time": val_times,
                "y": np.arange(lats.shape[0]),
                "x": np.arange(lats.shape[1]),
            },
        )
        ds_pred_edm.update({var: dap_edm})
        ds_pred_noedm.update({var: dap_noedm})
        ds_targ.update({var: dat})

    ds_pred_edm["longitude"] = xr.DataArray(lons, dims=("y", "x"))
    ds_pred_edm["latitude"] = xr.DataArray(lats, dims=("y", "x"))
    ds_pred_edm = ds_pred_edm.assign_coords(levels=model_levels)

    ds_pred_noedm["longitude"] = xr.DataArray(lons, dims=("y", "x"))
    ds_pred_noedm["latitude"] = xr.DataArray(lats, dims=("y", "x"))
    ds_pred_noedm = ds_pred_noedm.assign_coords(levels=model_levels)

    ds_targ["longitude"] = xr.DataArray(lons, dims=("y", "x"))
    ds_targ["latitude"] = xr.DataArray(lats, dims=("y", "x"))
    ds_targ = ds_targ.assign_coords(levels=model_levels)

    ds_out_path = cfg.inference.rundir

    ds_pred_edm.to_netcdf(f"{ds_out_path}/ds_pred_edm.nc", format="NETCDF4")
    ds_pred_noedm.to_netcdf(f"{ds_out_path}/ds_pred_noedm.nc", format="NETCDF4")
    ds_targ.to_netcdf(f"{ds_out_path}/ds_targ.nc", format="NETCDF4")


if __name__ == "__main__":
    main()
