from utils.YParams import YParams
from importlib import reload
import utils
from utils.data_loader_hrrr_era5 import get_inference_data_loader, get_dataset
import matplotlib.pyplot as plt
import numpy as np
import os
from networks.swinv2_hrrr import swinv2net
import torch
from datetime import datetime
import xarray as xr
import zarr
import pandas as pd
from score_inference import make_movie
import json
import utils.diffusions.networks
import pickle

import typer

reload(utils.data_loader_hrrr_era5)

def main(
    model_shortname: str = "diffusion_regression_a2a_v3_1_exclude_w_pstep_pos_embed_v2_noema_16M",
    output_directory: str = "./diffusion_regression_a2a_v3_1_exclude_w_pstep_pos_embed_v2_noema_16M",
    device: str = "cuda:0",
    initial_time: datetime = datetime(2022, 11, 4, 21, 0),
    plot_var_hrrr: str = "refc",
    plot_var_era5: str = "t2m",
    n_steps: int = 12,
    output_hrrr_channels: list[str] = [],
):

    #load model registry:
    with open("./config/registry.json", "r") as f:

        registry = json.load(f)
        model_info = registry["models"][model_shortname]
        use_regression_model = True if model_info["diffusion_only"] == 'False' else False

    use_diffusion = True

    if use_regression_model:
        params = YParams(model_info["regression_config_file"], model_info["regression_config_name"])
        params.local_batch_size = 1
        params.valid_years = [2022]
        residual = params.residual

    if use_diffusion:
        from utils.diffusions.run_edm import EDMRunner
        edm_config = model_info["edm_config_name"]
        edm_params = YParams(model_info["edm_config_file"], edm_config)
        diffusion_channels = edm_params.diffusion_channels
        input_channels = edm_params.input_channels
        resume_pkl = model_info["edm_checkpoint_path"] 
        posthoc_ema_sigma = model_info["posthoc_ema_sigma"]
        edm_runner = EDMRunner(edm_params, resume_pkl=resume_pkl, posthoc_ema_sigma=posthoc_ema_sigma, ema=False)
        assert use_regression_model == edm_params.use_regression_net, \
               "Model registry and diffusion model config must agree on use_regression_model"
        residual = edm_params.residual
        log_scale_residual = edm_params.log_scale_residual
        if use_diffusion and not use_regression_model:
            params = edm_params
            params.local_batch_size = 1
            params.valid_years = [2022]
    
    os.makedirs(output_directory, exist_ok=True)

    if params.boundary_padding_pixels > 0:
        params.era5_img_size = (
            params.hrrr_img_size[0] + 2 * params.boundary_padding_pixels,
            params.hrrr_img_size[1] + 2 * params.boundary_padding_pixels,
        )
    else:
        params.era5_img_size = params.hrrr_img_size

    loader, dataset = get_inference_data_loader(params)

    if use_regression_model:
        if edm_params.regression_net_type == "swin":
            model = swinv2net(params).cuda()
            model.eval()

            checkpoint_path = model_info["regression_checkpoint_path"] 
            checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))

            # remove the module. prefix from the keys
            new_state_dict = {}
            for k, v in checkpoint["model_state"].items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v

                model.load_state_dict(new_state_dict, strict=False)

        elif edm_params.regression_net_type == "unet":

            from utils.diffusions.networks import EasyRegression

            net_name = "song-unet-regression"
            resolution = 512
            target_channels = len(['u10m', 'v10m', 't2m', 'msl', 'u1', 'u2', 'u3', 'u4', 'u5', 'u6', 'u7', 'u8', 'u9', 'u10', 'u11', 'u13', 'u15', 'u20', 'u25', 'u30', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'v13', 'v15', 'v20', 'v25', 'v30', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't13', 't15', 't20', 't25', 't30', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10', 'q11', 'q13', 'q15', 'q20', 'q25', 'q30', 'z1', 'z2', 'z3', 'z4', 'z5', 'z6', 'z7', 'z8', 'z9', 'z10', 'z11', 'z13', 'z15', 'z20', 'z25', 'z30', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p13', 'p15', 'p20', 'refc']) 
            conditional_channels = target_channels + len(params.invariants) + 26

            net = utils.diffusions.networks.get_preconditioned_architecture(
                name=net_name,
                resolution=resolution,
                target_channels=target_channels,
                conditional_channels=conditional_channels,
                label_dim=0,
                spatial_embedding=params.spatial_pos_embed,
                attn_resolutions=params.attn_resolutions,
                )

            resume_pkl = model_info["regression_checkpoint_path"]

            with open(resume_pkl, 'rb') as f:
                data = pickle.load(f)
            net.load_state_dict(data['net'].state_dict(), strict=True)

            latent_shape = [target_channels, 512, 640]

            model = EasyRegression(net, latent_shape).to(device)
        
        elif edm_params.regression_net_type == "unet2":

            print("using unet2 for regression")

            from utils.diffusions.networks import EasyRegressionV2

            net_name = "song-unet-regression-v2"
            resolution = 512
            target_channels = len(['u10m', 'v10m', 't2m', 'msl', 'u1', 'u2', 'u3', 'u4', 'u5', 'u6', 'u7', 'u8', 'u9', 'u10', 'u11', 'u13', 'u15', 'u20', 'u25', 'u30', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'v13', 'v15', 'v20', 'v25', 'v30', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't13', 't15', 't20', 't25', 't30', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10', 'q11', 'q13', 'q15', 'q20', 'q25', 'q30', 'z1', 'z2', 'z3', 'z4', 'z5', 'z6', 'z7', 'z8', 'z9', 'z10', 'z11', 'z13', 'z15', 'z20', 'z25', 'z30', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p13', 'p15', 'p20', 'refc']) 
            conditional_channels = target_channels + len(params.invariants) + 26

            net = utils.diffusions.networks.get_preconditioned_architecture(
                name=net_name,
                resolution=resolution,
                target_channels=target_channels,
                conditional_channels=conditional_channels,
                label_dim=0,
                spatial_embedding=params.spatial_pos_embed,
                attn_resolutions=params.attn_resolutions,
                )

            resume_pkl = model_info["regression_checkpoint_path"]

            with open(resume_pkl, 'rb') as f:
                data = pickle.load(f)
            net.load_state_dict(data['net'].state_dict(), strict=True)

            latent_shape = [target_channels, 512, 640]

            model = EasyRegressionV2(net).to(device)
    hrrr_data = xr.open_zarr(os.path.join(params.location, params.conus_dataset_name, "valid", "2021.zarr"))

    dataset_obj = get_dataset(params, train=False)

    if use_regression_model:
        if edm_params.regression_net_type in ["unet", "unet2"]:
            invariant_array = dataset._get_invariants()
            invariant_tensor = torch.from_numpy(invariant_array).to(device).repeat(1, 1, 1, 1)
            model.set_invariant(invariant_tensor)

    dataset_location = dataset_obj.location

    base_hrrr_channels, hrrr_channels = dataset_obj._get_hrrr_channel_names()
    hrrr_channel_indices = [ list(base_hrrr_channels).index(channel) for channel in hrrr_channels]
    if len(output_hrrr_channels) == 0:
        output_hrrr_channels = hrrr_channels.copy()

    if use_diffusion:
        if diffusion_channels == "all":
            diffusion_channels = hrrr_channels
        diffusion_channel_indices = [ list(hrrr_channels).index(channel) for channel in diffusion_channels ]
        if input_channels == "all":
            input_channels = hrrr_channels
        input_channel_indices = [ list(hrrr_channels).index(channel) for channel in input_channels ]

    vardict: dict[str, int] = {
        hrrr_channel: i for i, hrrr_channel in enumerate(hrrr_channels)
    }

    era5_data_path = os.path.join(params.location, "era5", "valid", "2021.zarr")

    era5_data = xr.open_zarr(era5_data_path)

    era5_channels = era5_data.channel.values

    vardict_era5 = {era5_channel: i for i, era5_channel in enumerate(era5_channels)}

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

    means_hrrr = dataset.means_hrrr[hrrr_channel_indices]
    stds_hrrr = dataset.stds_hrrr[hrrr_channel_indices]

    means_era5 = dataset.means_era5
    stds_era5 = dataset.stds_era5

    # initialize zarr
    zarr_output_path = os.path.join(
        output_directory, initial_time.strftime("%Y-%m-%dT%H_%M_%S"), "data.zarr"
    )
    group = zarr.open_group(zarr_output_path, mode="w")
    group.array("latitude", data=hrrr_data["latitude"].values)
    group.array("longitude", data=hrrr_data["longitude"].values)

    edm_prediction_group = group.create_group("edm_prediction")
    if use_regression_model: noedm_prediction_group = group.create_group("noedm_prediction")
    target_group = group.create_group("target")
    hrrr, _ = dataset[0]["hrrr"]
    assert hrrr.ndim == 3
     
    grid_size = hrrr.shape[1:]

    for name in output_hrrr_channels:
        target_group.empty(
            name, shape=(n_steps,) + grid_size, chunks=[1, *grid_size], compressor=None
        )

    for name in output_hrrr_channels:
        edm_prediction_group.empty(
            name, shape=(n_steps,) + grid_size, chunks=[1, *grid_size], compressor=None
        )
    if use_regression_model:
        for name in output_hrrr_channels:
            noedm_prediction_group.empty(
                name, shape=(n_steps,) + grid_size, chunks=[1, *grid_size], compressor=None
            )

    with torch.no_grad():
        # for i, data in enumerate(loader):
        for i in range(n_steps):
            data = dataset[i + hours_since_jan_01]
            print(i)

            if i == 0:
                inp = data["hrrr"][0].cuda().float().unsqueeze(0)
                boundary = data["era5"][0].cuda().float().unsqueeze(0)
                out = inp
                out_edm = out.clone()
                if use_regression_model: out_noedm = out.clone()

            assert out_edm.shape == (1, len(hrrr_channels)) + grid_size
            if use_regression_model: assert out_noedm.shape == (1, len(hrrr_channels)) + grid_size
            # write hrrr
            denorm_out_edm = out_edm.cpu().numpy() * stds_hrrr + means_hrrr
            if use_regression_model: denorm_out_noedm = out_noedm.cpu().numpy() * stds_hrrr + means_hrrr
            
            for name in output_hrrr_channels:
                k = vardict[name]
                edm_prediction_group[name][i] = denorm_out_edm[0, k]
                if use_regression_model: noedm_prediction_group[name][i] = denorm_out_noedm[0, k]
                target_data = data["hrrr"][0][k].cpu().numpy() * stds_hrrr[k] + means_hrrr[k]
                target_group[name][i] = target_data

            if i > n_steps:
                break

            if not use_regression_model: 
                if edm_params.pure_diffusion:
                    hrrr_0 = out
                    hrrr_0 = torch.cat((hrrr_0[:, input_channel_indices], boundary), dim=1)
                    edm_corrected_outputs, edm_channels = edm_runner.run(hrrr_0)
                    if residual:
                        out[0, diffusion_channel_indices] += edm_corrected_outputs[0].float()
                    else:
                        out[0, diffusion_channel_indices] = edm_corrected_outputs[0].float()
                    out_edm = out.clone()
            else:
                if edm_params.previous_step_conditioning:
                    assert use_regression_model, 'previous step conditioning not yet supported without swin regression'
                    hrrr_0 = out
                    out = model(hrrr_0, boundary, mask=None)
                    if use_regression_model: out_noedm = out.clone()
                    hrrr_0 = torch.cat((hrrr_0[:, input_channel_indices, :, :], out[:, input_channel_indices, :, :]), dim=1)
                    if use_diffusion:
                        edm_corrected_outputs, edm_channels = edm_runner.run(hrrr_0)
                        if residual:
                            if log_scale_residual:
                                edm_corrected_outputs = torch.sign(edm_corrected_outputs) * (torch.expm1(torch.abs(edm_corrected_outputs)))
                            out[0, diffusion_channel_indices] += edm_corrected_outputs[0].float()
                        else:
                            out[0, diffusion_channel_indices] = edm_corrected_outputs[0].float()
                        out_edm = out.clone()
                else:
                    if use_regression_model:
                        out = model(out, boundary, mask = None)
                        if use_regression_model: out_noedm = out.clone()
                    if use_diffusion:
                        edm_corrected_outputs, edm_channels = edm_runner.run(out[:, input_channel_indices, :, :])
                        if residual:
                            if log_scale_residual:
                                print("unscaling log scale residual")
                                edm_corrected_outputs = torch.sign(edm_corrected_outputs) * (torch.expm1(torch.abs(edm_corrected_outputs)))
                            out[0, diffusion_channel_indices] += edm_corrected_outputs[0].float()
                        else:
                            out[0, diffusion_channel_indices] = edm_corrected_outputs[0].float()
                        out_edm = out.clone()

            boundary = data["era5"][0].cuda().float().unsqueeze(0)

            varidx = vardict[plot_var_hrrr]

            fig, ax = plt.subplots(1, 4, figsize=(20, 5))

            pred = out.cpu().numpy()
            tar = data["hrrr"][1].unsqueeze(0).cpu().numpy()
            era5 = data["era5"][0].unsqueeze(0).cpu().numpy()
            pred = pred * stds_hrrr + means_hrrr
            tar = tar * stds_hrrr + means_hrrr
            era5 = era5 * stds_era5 + means_era5

            error = pred - tar

            if plot_var_hrrr in color_limits:
                im = ax[0].imshow(
                    pred[0, varidx],
                    origin="lower",
                    cmap="magma",
                    clim=color_limits[plot_var_hrrr],
                )
            else:
                im = ax[0].imshow(pred[0, varidx], origin="lower", cmap="magma")
            # colorbar
            fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
            ax[0].set_title(
                "Predicted, {}, \n initial time {} \n lead_time {} hours".format(
                    plot_var_hrrr, initial_time, i
                )
            )
            if plot_var_hrrr in color_limits:
                im = ax[1].imshow(
                    tar[0, varidx],
                    origin="lower",
                    cmap="magma",
                    clim=color_limits[plot_var_hrrr],
                )
            else:
                im = ax[1].imshow(tar[0, varidx], origin="lower", cmap="magma")
            fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
            ax[1].set_title("Actual, {}".format(plot_var_hrrr))
            if plot_var_era5 in color_limits:
                im = ax[2].imshow(
                    era5[0, vardict_era5[plot_var_era5]],
                    origin="lower",
                    cmap="magma",
                    clim=color_limits[plot_var_era5],
                )
            else:
                im = ax[2].imshow(
                    era5[0, vardict_era5[plot_var_era5]], origin="lower", cmap="magma"
                )
            fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)
            ax[2].set_title("ERA5, {}".format(plot_var_era5))
            maxerror = np.max(np.abs(error[0, varidx]))
            im = ax[3].imshow(
                error[0, varidx],
                origin="lower",
                cmap="RdBu_r",
                vmax=maxerror,
                vmin=-maxerror,
            )
            fig.colorbar(im, ax=ax[3], fraction=0.046, pad=0.04)
            ax[3].set_title("Error, {}".format(plot_var_hrrr))

            plt.savefig(f"{output_directory}/out_{i}.png")
    
    #create output name with config, edm_config, initial_time
    edm_config = model_info["edm_config_name"] if use_diffusion else "no_edm" 
    reg_config = model_info["regression_config_name"] if use_regression_model else "no_reg"
    output_gif_name = "{}_{}_{}".format(reg_config, edm_config, initial_time.isoformat()) 
    make_movie(zarr_output_path, os.path.join(output_directory, initial_time.isoformat()), output_name = output_gif_name) 
    
    level_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 
               '11', '13', '15', '20', '25', '30', '35', '40']
    vertical_vars = ['u', 'v', 't', 'q', 'z', 'p', 'w']
    horizontal_vars = ['msl','refc','u10m','v10m']
    
    zarr_group = group
    lons = zarr_group['longitude'][:,:]
    lats = zarr_group['latitude'][:,:]
    initial_time_pd = pd.to_datetime(initial_time)
    val_times = []
    for i in range(n_steps):
        val_times.append(initial_time_pd+pd.Timedelta(seconds=i*hours_since_jan_01))
    
    def convert_strings_to_ints(string_list):
        return [int(i) for i in string_list]
    
    model_levels = convert_strings_to_ints(level_names)
    
    ds_pred_edm = xr.Dataset()
    if use_regression_model: ds_pred_noedm = xr.Dataset()
    ds_targ = xr.Dataset()

    for var in vertical_vars:
        dsp_edm = xr.Dataset()
        if use_regression_model: dsp_noedm = xr.Dataset()
        dst = xr.Dataset()

        for i, level in enumerate(level_names):
            key = f'{var}{level}' 

            if key in zarr_group['edm_prediction']:
                # Extract the data from zarr_group
                data_pred_edm = zarr_group['edm_prediction'][key][:, :]
            else:
                data_pred_edm = np.full((len(val_times), lats.shape[0], lats.shape[1]), np.nan)
                print(f"Key {key} not found in Zarr group. Filling with NaNs.")
            if use_regression_model:
                if key in zarr_group['noedm_prediction']:
                    # Extract the data from zarr_group
                    data_pred_noedm = zarr_group['noedm_prediction'][key][:, :]
                else:
                    data_pred_noedm = np.full((len(val_times), lats.shape[0], lats.shape[1]), np.nan)
                    print(f"Key {key} not found in Zarr group. Filling with NaNs.")
                
            if key in zarr_group['target']:
                data_targ = zarr_group['target'][key][:, :]
            else:
                data_targ = np.full((len(val_times), lats.shape[0], lats.shape[1]), np.nan)
                print(f"Key {key} not found in Zarr group. Filling with NaNs.")


            dap_edm = xr.DataArray(data_pred_edm, dims=('time', 'y', 'x'), coords={'time': val_times, 'y': np.arange(lats.shape[0]), 'x': np.arange(lats.shape[1]), 'levels': model_levels[i]})
            if use_regression_model:  dap_noedm = xr.DataArray(data_pred_noedm, dims=('time', 'y', 'x'), coords={'time': val_times, 'y':
                                                                                                                 np.arange(lats.shape[0]),
                                                                                                                 'x':
                                                                                                                 np.arange(lats.shape[1]),
                                                                                                                 'levels': model_levels[i]})
            dat = xr.DataArray(data_targ, dims=('time', 'y', 'x'), coords={'time': val_times, 'y': np.arange(lats.shape[0]), 'x': np.arange(lats.shape[1]), 'levels': model_levels[i]})

            dsp_edm[f'var_{i}'] = dap_edm
            if use_regression_model: dsp_noedm[f'var_{i}'] = dap_noedm
            dst[f'var_{i}'] = dat

        combined_pred_edm = xr.concat([dsp_edm[var] for var in dsp_edm.data_vars], dim='levels')
        if use_regression_model: combined_pred_noedm = xr.concat([dsp_noedm[var] for var in dsp_noedm.data_vars], dim='levels')
        combined_targ = xr.concat([dst[var] for var in dst.data_vars], dim='levels')

        reshaped_pred_edm = combined_pred_edm.transpose('time', 'levels', 'y', 'x')
        if use_regression_model: reshaped_pred_noedm = combined_pred_noedm.transpose('time', 'levels', 'y', 'x')
        reshaped_targ = combined_targ.transpose('time', 'levels', 'y', 'x')

        ds_pred_edm[f'{var}_comb'] = reshaped_pred_edm
        if use_regression_model: ds_pred_noedm[f'{var}_comb'] = reshaped_pred_noedm
        ds_targ[f'{var}_comb'] = reshaped_targ
    
    for var in horizontal_vars:
        data_pred_edm = zarr_group['edm_prediction'][var][:, :]
        if use_regression_model: data_pred_noedm = zarr_group['noedm_prediction'][var][:, :]
        data_targ = zarr_group['target'][var][:, :]
        dap_edm = xr.DataArray(data_pred_edm,dims=('time','y','x'),
                         coords={'time':val_times,
                                'y': np.arange(lats.shape[0]), 'x': np.arange(lats.shape[1])})
        if use_regression_model: 
            dap_noedm = xr.DataArray(data_pred_noedm,dims=('time','y','x'),
                         coords={'time':val_times,
                                'y': np.arange(lats.shape[0]), 'x': np.arange(lats.shape[1])})
        dat = xr.DataArray(data_targ,dims=('time','y','x'),
                         coords={'time':val_times,
                                'y': np.arange(lats.shape[0]), 'x': np.arange(lats.shape[1])})
        ds_pred_edm.update({var:dap_edm})
        if use_regression_model: ds_pred_noedm.update({var:dap_noedm})
        ds_targ.update({var:dat})
    
    ds_pred_edm['longitude'] = xr.DataArray(lons,dims=('y','x'))
    ds_pred_edm['latitude'] = xr.DataArray(lats,dims=('y','x'))
    ds_pred_edm = ds_pred_edm.assign_coords(levels=model_levels)
    
    if use_regression_model:
        ds_pred_noedm['longitude'] = xr.DataArray(lons,dims=('y','x'))
        ds_pred_noedm['latitude'] = xr.DataArray(lats,dims=('y','x'))
        ds_pred_noedm = ds_pred_noedm.assign_coords(levels=model_levels)

    ds_targ['longitude'] = xr.DataArray(lons,dims=('y','x'))
    ds_targ['latitude'] = xr.DataArray(lats,dims=('y','x'))
    ds_targ = ds_targ.assign_coords(levels=model_levels)
    
    ds_out_path = os.path.join(output_directory, initial_time.strftime("%Y-%m-%dT%H_%M_%S"))
    
    ds_pred_edm.to_netcdf(f"{ds_out_path}/ds_pred_edm.nc",format='NETCDF4')
    if use_regression_model : ds_pred_noedm.to_netcdf(f"{ds_out_path}/ds_pred_noedm.nc",format='NETCDF4')
    ds_targ.to_netcdf(f"{ds_out_path}/ds_targ.nc",format='NETCDF4')

    
    

if __name__ == "__main__":
    typer.run(main)
