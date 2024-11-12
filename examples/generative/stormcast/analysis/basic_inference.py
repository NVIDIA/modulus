import os
import sys 
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.getcwd())

from utils.YParams import YParams
from importlib import reload
import utils
from utils.data_loader_hrrr_era5 import get_inference_data_loader, get_dataset
import matplotlib.pyplot as plt
import numpy as np
from networks.swinv2_hrrr import swinv2net
import torch
from datetime import datetime
import xarray as xr
import pandas as pd
import json
import typer
import shutil
import pickle


reload(utils.data_loader_hrrr_era5)


def main(config_file, registry_file="./config/registry.json"):
    with open(config_file, 'r') as file:
        config = json.load(file)

    model_shortname = config["model_shortname"]
    output_directory = config["output_directory"]
    device = config["device"]
    initial_time_str = config["initial_time"]
    initial_time = datetime.strptime(initial_time_str, "%Y-%m-%d %H:%M")
    n_steps = config["n_steps"]
    output_hrrr_channels = config["output_hrrr_channels"]
    regression_only = eval(config["regression_only"]) #this is set by run_analysis
    
    level_names = ['1','2','3','4','5','6','7','8','9','10',
                   '11','13','15','20','25','30']
    vertical_vars = ['u', 'v', 't', 'q', 'z', 'p']
    horizontal_vars = ['msl','refc','u10m','v10m']

    print("Using registry file: ", registry_file)
    print("Registry file has the following models: ", json.load(open(registry_file))["models"].keys())

    with open(registry_file, "r") as f:
        registry = json.load(f)
        model_info = registry["models"][model_shortname]
        if model_info["diffusion_only"] == 'False':
            use_regression_model = True 
        elif regression_only:
            use_regression_model = True
        else:
            use_regression_model = False

    diffusion_only = True if model_info["diffusion_only"] == 'True' else False

    if regression_only:
        use_diffusion = False
    else:
        use_diffusion = True

    identity_regression = False
    if diffusion_only and use_regression_model:
        #this is the case when regression_only = eval(config["regression_only"]) is set by run_analysis.py
        identity_regression = True #use a dummy regression model that returns the input

    #if diffusion_only: 
    #   use_diffusion = True
    #   use_regression_model = False
       
    #print(use_diffusion)

    if use_regression_model:
        if identity_regression:
            #from types import SimpleNamespace
            #params = SimpleNamespace()
            edm_config = model_info["edm_config_name"]
            edm_params = YParams(model_info["edm_config_file"], edm_config)
            params = edm_params
            params.local_batch_size = 1
            params.valid_years = [2022]
            residual = False
            model_info["regression_net_type"] = "identity"

        else:
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
        if "force_no_ema" in model_info:
            force_no_ema = model_info["force_no_ema"]
            print("setting ema to {}".format(not force_no_ema))
        else:
            force_no_ema = False
        edm_runner = EDMRunner(edm_params, resume_pkl=resume_pkl, posthoc_ema_sigma=posthoc_ema_sigma, ema=not force_no_ema)
        assert use_regression_model == edm_params.use_regression_net, \
               "Model registry and diffusion model config must agree on use_regression_model"
        residual = edm_params.residual
        log_scale_residual = edm_params.log_scale_residual
        if use_diffusion and not use_regression_model:
            params = edm_params
            params.local_batch_size = 1
            params.valid_years = [2022]
    
    if config['event_name'] is None:
        event_name = initial_time.strftime("%Y-%m-%dT%H_%M_%S")
    else:
        event_name = config['event_name']
        
    os.makedirs(output_directory, exist_ok=True)
    ds_out_path = os.path.join(output_directory, event_name)
    os.makedirs(ds_out_path, exist_ok=True)
    shutil.copy(config_file, os.path.join(ds_out_path,"config.json"))

    if params.boundary_padding_pixels > 0:
        params.era5_img_size = (
            params.hrrr_img_size[0] + 2 * params.boundary_padding_pixels,
            params.hrrr_img_size[1] + 2 * params.boundary_padding_pixels,
        )
    else:
        params.era5_img_size = params.hrrr_img_size

    loader, dataset = get_inference_data_loader(params)


    if use_regression_model:
        if model_info["regression_net_type"] == "swin":
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

        elif model_info["regression_net_type"] == "unet":

            print("using unet for regression")

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

            model = EasyRegression(net).to(device)

        elif model_info["regression_net_type"] == "unet2":

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

        elif identity_regression:

            from torch import nn

            class IdentityRegression(nn.Module):
                def __init__(self):
                    super(IdentityRegression, self).__init__()

                def forward(self, x, boundary, mask=None):
                    return x
            
            model = IdentityRegression().to(device)




    hrrr_data = xr.open_zarr(os.path.join(params.location, params.conus_dataset_name, "valid", "2021.zarr"))

    dataset_obj = get_dataset(params, train=False)

    if use_regression_model:
        if model_info["regression_net_type"] in ["unet", "unet2"]:
             invariant_array = dataset_obj._get_invariants()
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

    hrrr, _ = dataset[0]["hrrr"]
    assert hrrr.ndim == 3
     
    grid_size = hrrr.shape[1:]
    if use_diffusion: 
        prediction_data = {name: np.zeros((n_steps,) + grid_size) for name in output_hrrr_channels}
    if regression_only:
        prediction_data_noedm = {name: np.zeros((n_steps,) + grid_size) for name in output_hrrr_channels}
        
    target_data = {name: np.zeros((n_steps,) + grid_size) for name in output_hrrr_channels}

    with torch.no_grad():
        data = dataset[hours_since_jan_01]["hrrr"][0].cuda().float().unsqueeze(0)
        out = data
        assert data.shape == (1, len(hrrr_channels)) + grid_size
        if regression_only: out_noedm = data.clone()
        for name in output_hrrr_channels:
            k = vardict[name]
            target_data[name][0] = data[0][k].cpu().numpy() * stds_hrrr[k] + means_hrrr[k]
            if use_diffusion: prediction_data[name][0] = target_data[name][0]
            if regression_only: prediction_data_noedm[name][0] = target_data[name][0]


        for i in range(n_steps):
            data = dataset[i + hours_since_jan_01]
            boundary = data["era5"][0].cuda().float().unsqueeze(0)
            if regression_only:
                hrrr_0 = out
                out = model(hrrr_0, boundary, mask=None)
                out_noedm = out.clone()
                #hrrr_0 = torch.cat((hrrr_0[:, input_channel_indices, :, :], out[:, input_channel_indices, :, :]), dim=1)
            elif diffusion_only:
                assert input_channel_indices == diffusion_channel_indices
                hrrr_0 = out
                hrrr_0 = torch.cat((hrrr_0[:, input_channel_indices], boundary), dim=1)
                edm_corrected_outputs, edm_channels = edm_runner.run(hrrr_0)
                if residual:
                    raise Exception("not implemented yet")
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
                else:
                    if use_regression_model:
                        out = model(out, boundary, mask = None)
                        out_noedm = out.clone()
                    if use_diffusion:
                        edm_corrected_outputs, edm_channels = edm_runner.run(out[:, input_channel_indices, :, :])
                        if residual:
                            if log_scale_residual:
                                print("using log scale residual")
                                edm_corrected_outputs = torch.sign(edm_corrected_outputs) * (torch.expm1(torch.abs(edm_corrected_outputs)))
                            out[0, diffusion_channel_indices] += edm_corrected_outputs[0].float()
                        else:
                            out[0, diffusion_channel_indices] = edm_corrected_outputs[0].float()

            if use_diffusion: denorm_out = out.cpu().numpy() * stds_hrrr + means_hrrr
            if regression_only: denorm_out_noedm = out_noedm.cpu().numpy() * stds_hrrr + means_hrrr
            for name in output_hrrr_channels:
                k = vardict[name]
                if use_diffusion: prediction_data[name][i] = denorm_out[0, k]
                if regression_only: prediction_data_noedm[name][i] = denorm_out_noedm[0, k]
                target_data[name][i] = data["hrrr"][1][k].cpu().numpy() * stds_hrrr[k] + means_hrrr[k]

    vertical_vars = ['u', 'v', 't', 'q', 'z', 'p', 'w']
    horizontal_vars = ['msl','refc','u10m','v10m']
    
    lons = dataset.hrrr_lon.values
    lats = dataset.hrrr_lat.values
    initial_time_pd = pd.to_datetime(initial_time)
    val_times = []
    for i in range(n_steps):
        val_times.append(initial_time_pd+pd.Timedelta(seconds=i*hours_since_jan_01))
    
    def convert_strings_to_ints(string_list):
        return [int(i) for i in string_list]
    
    model_levels = convert_strings_to_ints(level_names)
    
    if use_diffusion: ds_pred = xr.Dataset()
    if regression_only: ds_pred_noedm = xr.Dataset()
    ds_targ = xr.Dataset()

    for var in vertical_vars:
        if use_diffusion: dsp = xr.Dataset()
        if regression_only: dsp_noedm = xr.Dataset()
        dst = xr.Dataset()

        for i, level in enumerate(level_names):
            key = f'{var}{level}' 
            if regression_only:
                if key in prediction_data_noedm:
                    data_pred_noedm = prediction_data_noedm[key][:, :]
                else:
                    data_pred_noedm = np.full((len(val_times), lats.shape[0], lats.shape[1]), np.nan)
                    print(f"Key {key} not found in Zarr group. Filling with NaNs.")
            else:
                if key in prediction_data:
                    data_pred = prediction_data[key][:, :]
                else:
                    data_pred = np.full((len(val_times), lats.shape[0], lats.shape[1]), np.nan)
                    print(f"Key {key} not found in Zarr group. Filling with NaNs.")
            if key in target_data:
                data_targ = target_data[key][:, :]
            else:
                data_targ = np.full((len(val_times), lats.shape[0], lats.shape[1]), np.nan)
                print(f"Key {key} not found in Zarr group. Filling with NaNs.")


            if use_diffusion: dap = xr.DataArray(data_pred, dims=('time', 'y', 'x'), 
                               coords={'time': val_times, 'y': np.arange(lats.shape[0]), 
                                       'x': np.arange(lats.shape[1]), 'levels': model_levels[i]})
            if regression_only: dap_noedm = xr.DataArray(data_pred_noedm, dims=('time', 'y', 'x'), 
                               coords={'time': val_times, 'y': np.arange(lats.shape[0]), 
                                       'x': np.arange(lats.shape[1]), 'levels': model_levels[i]})
            dat = xr.DataArray(data_targ, dims=('time', 'y', 'x'), 
                               coords={'time': val_times, 'y': np.arange(lats.shape[0]), 
                                       'x': np.arange(lats.shape[1]), 'levels': model_levels[i]})

            if use_diffusion: dsp[f'var_{i}'] = dap
            if regression_only: dsp_noedm[f'var_{i}'] = dap_noedm
            dst[f'var_{i}'] = dat

        if use_diffusion: combined_pred = xr.concat([dsp[var] for var in dsp.data_vars], dim='levels')
        if regression_only: combined_pred_noedm = xr.concat([dsp_noedm[var] for var in dsp_noedm.data_vars], dim='levels')
        combined_targ = xr.concat([dst[var] for var in dst.data_vars], dim='levels')

        if use_diffusion: reshaped_pred = combined_pred.transpose('time', 'levels', 'y', 'x')
        if regression_only: reshaped_pred_noedm = combined_pred_noedm.transpose('time', 'levels', 'y', 'x')
        reshaped_targ = combined_targ.transpose('time', 'levels', 'y', 'x')

        if use_diffusion: ds_pred[f'{var}_comb'] = reshaped_pred
        if regression_only: ds_pred_noedm[f'{var}_comb'] = reshaped_pred_noedm
        ds_targ[f'{var}_comb'] = reshaped_targ
    
    for var in horizontal_vars:
        if use_diffusion: data_pred = prediction_data[var][:, :]
        if regression_only: data_pred_noedm = prediction_data_noedm[var][:, :]
        data_targ = target_data[var][:, :]
        if use_diffusion: dap = xr.DataArray(data_pred,dims=('time','y','x'),
                         coords={'time':val_times,
                                'y': np.arange(lats.shape[0]), 'x': np.arange(lats.shape[1])})
        if regression_only: dap_noedm = xr.DataArray(data_pred_noedm,dims=('time','y','x'),
                         coords={'time':val_times,
                                'y': np.arange(lats.shape[0]), 'x': np.arange(lats.shape[1])})
        dat = xr.DataArray(data_targ,dims=('time','y','x'),
                         coords={'time':val_times,
                                'y': np.arange(lats.shape[0]), 'x': np.arange(lats.shape[1])})
        if use_diffusion: ds_pred.update({var:dap})
        if regression_only: ds_pred_noedm.update({var:dap_noedm})
        ds_targ.update({var:dat})
    
    if use_diffusion:
        ds_pred['longitude'] = xr.DataArray(lons,dims=('y','x'))
        ds_pred['latitude'] = xr.DataArray(lats,dims=('y','x'))
        ds_pred = ds_pred.assign_coords(levels=model_levels)
        ds_pred.attrs['config'] = json.dumps(config)
    
    if regression_only:
        ds_pred_noedm['longitude'] = xr.DataArray(lons,dims=('y','x'))
        ds_pred_noedm['latitude'] = xr.DataArray(lats,dims=('y','x'))
        ds_pred_noedm = ds_pred_noedm.assign_coords(levels=model_levels)
        ds_pred_noedm.attrs['config'] = json.dumps(config)

    ds_targ['longitude'] = xr.DataArray(lons,dims=('y','x'))
    ds_targ['latitude'] = xr.DataArray(lats,dims=('y','x'))
    ds_targ = ds_targ.assign_coords(levels=model_levels)
    ds_targ.attrs['config'] = json.dumps(config)
    
    if use_diffusion: ds_pred.to_netcdf(os.path.join(ds_out_path,"ds_pred_edm.nc"),format='NETCDF4')
    if regression_only: ds_pred_noedm.to_netcdf(os.path.join(ds_out_path,"ds_pred_noedm.nc"),format='NETCDF4')
    ds_targ.to_netcdf(os.path.join(ds_out_path,"ds_targ.nc"),format='NETCDF4')
    

if __name__ == "__main__":

    #if len(sys.argv) < 2:
    #    print("Usage: python run_inference.py <path_to_config_file>")
    #    sys.exit(1)
    #config_file = sys.argv[1]
    #main(config_file)

    #set up argparse
    import argparse
    parser = argparse.ArgumentParser(description='Run inference on a dataset using a trained model.')
    parser.add_argument('--config_file', type=str, help='Path to the config file.', default=None)
    parser.add_argument('--registry_file', type=str, help='Path to the registry file.', default='config/registry.json')

    args = parser.parse_args()
    main(args.config_file, args.registry_file)
    
