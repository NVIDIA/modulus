from utils.YParams import YParams
from importlib import reload
import utils
from utils.data_loader_hrrr_era5 import get_inference_data_loader, get_dataset
from utils.gfsdataset import GFSDataSet
import matplotlib.pyplot as plt
import numpy as np
import os
from networks.swinv2_hrrr import swinv2net
import torch
from datetime import datetime, timedelta
import xarray as xr
import zarr
import pandas as pd
from score_inference import make_movie
import json
import utils.diffusions.networks
import pickle
from realtime import machine_info
import typer
import pathlib
import cartopy.crs
import cartopy.feature
from metpy.plots import ctables
from pathlib import Path
from utils.metrics import probability_matched_mean

reload(utils.data_loader_hrrr_era5)

def make_movie(path: str, dest: str, initial_time: datetime, output_format: str = "gif", output_name: str = "out"):    

    hrrr_forecast_dir = machine_info.savepath_hrrr_forecast
    year= initial_time.year
    month = initial_time.month
    day = initial_time.day
    hour = initial_time.hour
    hrrr_forecast_str = f"hrrr_{year}{month:02d}{day:02d}_{hour:02d}z_forecast.zarr"
    hrrr_forecast_path = os.path.join(hrrr_forecast_dir, hrrr_forecast_str) 

    ml_forecast = xr.open_zarr(path)
    lat = ml_forecast["latitude"].values
    lon = ml_forecast["longitude"].values
    print(ml_forecast)

    
    fcst = xr.open_dataset(hrrr_forecast_path)
    refc_fcst = fcst["REFC"]


    projection = cartopy.crs.LambertConformal(
        central_longitude=-95, central_latitude=35
    )
    src_crs = cartopy.crs.PlateCarree()


    def plot_latlon(ax, z):
        z = np.where(z > 0, z, np.nan)
        norm, cmap = ctables.registry.get_with_steps('NWSReflectivity', -0, 5)
        im = ax.pcolormesh(lon, lat, z, transform=src_crs, vmax=60, cmap=cmap)
        ax.coastlines()
        ax.add_feature(cartopy.feature.STATES, edgecolor="0.5")
        return im

    #image_root = os.path.join(dest, "images")
    image_root = "./current_run/ensemble_images/"

    os.makedirs(image_root, exist_ok=True)
    n = len(ml_forecast.time)
    ensemble_size = len(ml_forecast.ensemble.values)
    for i in range(n):


        image_path = os.path.join(image_root, f"{i:06}.png")

        #compute PMM
        pmm = probability_matched_mean(ml_forecast['data'].isel(time=i).sel(channel='refc').values)

        #grid of ensembe_size +1 subplots to accomodate ensemble members and hrrr forecast
        n_plots = ensemble_size + 1
        n_cols = int(np.ceil(np.sqrt(n_plots)))
        n_rows = int(np.ceil(n_plots / n_cols))
        fig_width = 7.5 * n_cols
        fig_height = 7.5 * n_rows

        fig, axs = plt.subplots( n_rows, n_cols, figsize=(fig_width, fig_height), subplot_kw=dict(projection=projection))
        axs = axs.flatten()

        for ens in range(ensemble_size):
            im = plot_latlon(axs[ens], ml_forecast['data'].isel(time=i).sel(channel="refc", ensemble=ens).values)
            axs[ens].set_title(f"ML Ensemble Member {ens}")
        time = initial_time + i * timedelta(hours=1)
        hrrr_refc = refc_fcst.isel(time=i).values
        #TODO: check if this is the correct time
        fig.suptitle(f"Valid Time: {time.isoformat()} \n Tag: {output_name} \n . ")
        im = plot_latlon(axs[-1], hrrr_refc)
        axs[-1].set_title("HRRR Forecast (unverified/not ground truth)")
        im = plot_latlon(axs[-2], pmm)
        axs[-2].set_title("ML ensemble PMM")
        plt.tight_layout()
        plt.savefig(image_path)
        plt.close(fig)

    from PIL import Image

    images = []
    for i in range(n):
        image_path = os.path.join(image_root, f"{i:06}.png")
        images.append(Image.open(image_path))
    images[0].save(
        os.path.join(dest, "current_forecast_ensemble.gif"),
        save_all=True,
        append_images=images[1:],
        duration=1000,
        loop=0,
    )

    image_root = "./current_run/images/"

    for i in range(n):


        image_path = os.path.join(image_root, f"{i:06}.png")

        fig, axs = plt.subplots( 1, 2, figsize=(12, 6), subplot_kw=dict(projection=projection))

        im = plot_latlon(axs[0], ml_forecast['data'].isel(time=i).sel(channel="refc", ensemble=0).values)
        axs[0].set_title(f"ML Ensemble Member")
        time = initial_time + i * timedelta(hours=1)
        hrrr_refc = refc_fcst.isel(time=i).values
        #TODO: check if this is the correct time
        fig.suptitle(f"Valid Time: {time.isoformat()} \n Tag: {output_name} \n . ")
        im = plot_latlon(axs[-1], hrrr_refc)
        cb = plt.colorbar(im, ax=axs[:], orientation="horizontal", shrink=0.2)
        cb.set_label("Composite Reflectivity dBZ")
        #axs[0].set_title("ML Forecast")
        axs[-1].set_title("HRRR Forecast (unverified/not ground truth)")
        plt.savefig(image_path)
        plt.close(fig)

    from PIL import Image

    images = []
    for i in range(n):
        image_path = os.path.join(image_root, f"{i:06}.png")
        images.append(Image.open(image_path))
    images[0].save(
        os.path.join(dest, "current_forecast.gif"),
        save_all=True,
        append_images=images[1:],
        duration=1000,
        loop=0,
    )
    """

    #create gif of previous day 12 hour forecast with hrrr analysis
    prev_initialization_date = initial_time - timedelta(hours=24)
    year = prev_initialization_date.year
    month = prev_initialization_date.month
    day = prev_initialization_date.day
    hour = prev_initialization_date.hour
    hrrr_forecast_str = f"hrrr_{year}{month:02d}{day:02d}_{hour:02d}z_forecast.zarr"
    hrrr_forecast_path = os.path.join(hrrr_forecast_dir, hrrr_forecast_str)
    fcst = xr.open_dataset(hrrr_forecast_path)
    refc_fcst = fcst["REFC"]

    datestr = prev_initialization_date.strftime("%Y%m%d")
    init_z = prev_initialization_date.hour
    hrrr_analysis_str = f"hrrr_{datestr}_{init_z:02d}z_anl.zarr"
    hrrr_analysis = os.path.join(hrrr_forecast_dir, hrrr_analysis_str)
    anl = xr.open_dataset(hrrr_analysis)
    refc_anl = anl["REFC"]

    #image_root = os.path.join(dest, "images_prev")
    image_root = "./"
    os.makedirs(image_root, exist_ok=True)
    #get parent dir of "path"
    model_forecast_dir = Path(os.path.dirname(path)).parent
    hindcast_path = os.path.join(model_forecast_dir, f"{year}-{month:02d}-{day:02d}T{hour:02d}_00_00")
    #g = zarr.open(hindcast_path + "/data.zarr", mode="r")
    #prediction = g["edm_prediction"]["refc"]
    hindcast = xr.open_zarr(hindcast_path)

    n = prediction.shape[0]
    for i in range(n):

        image_path = os.path.join(image_root, f"{i:06}.png")

        fig, axs = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(projection=projection))
        im = plot_latlon(axs[0], hindcast['data'].isel(time=i).sel(channel="refc").values)
        time = prev_initialization_date + i * timedelta(hours=1)
        hrrr_refc = refc_fcst.isel(time=i).values
        hrrr_anl = refc_anl.isel(time=i).values
        fig.suptitle(f"Valid Time: {time.isoformat()} \n Tag: {output_name} \n . ")
        cb = plt.colorbar(im, ax=axs[0:3], orientation="horizontal", shrink=0.8)
        cb.set_label("Composite Reflectivity dBZ")
        im = plot_latlon(axs[1], hrrr_refc)
        axs[0].set_title("ML Forecast")
        axs[1].set_title("HRRR Forecast (unverified/not ground truth)")
        im = plot_latlon(axs[2], hrrr_anl)
        axs[2].set_title("HRRR Analysis")
        plt.savefig(image_path)
        plt.close(fig)

    images = []
    for i in range(n):
        image_path = os.path.join(image_root, f"{i:06}.png")
        images.append(Image.open(image_path))
    images[0].save(
        os.path.join(dest, "previous_forecast.gif"),
        save_all=True,
        append_images=images[1:],
        duration=1000,
        loop=0,
    )
"""



def get_invariants(invariant_location, invariant_channels):

    invariants = xr.open_zarr(invariant_location)

    invariant_channels_in_dataset = list(invariants.channel.values)

    for invariant in invariant_channels:
        assert invariant in invariant_channels_in_dataset, f"Requested invariant {invariant} not in dataset"
    
    invariant_array = invariants["HRRR_invariants"].sel(channel=invariant_channels).values

    return invariant_array

def run_forecast(
    model_shortname: str = "diffusion_regression_a2a_v3_1_exclude_w_pstep_pos_embed_v2_noema_16M",
    device: str = "cuda:0",
    n_steps: int = 18,
    output_hrrr_channels: list[str] = [],
    initial_time: datetime = datetime(2024, 6, 9, 0, 0),
    use_swiftstack: bool = False,
    swiftstack_staging_path: str = "",
    ensemble_size: int = 1,
    movie: bool = True
):
        
    if use_swiftstack:
        path_hrrr = swiftstack_staging_path + "/hrrr_{}_{}z_f01.zarr".format(initial_time.strftime("%Y%m%d"), initial_time.strftime("%H"))
        path_gfs = swiftstack_staging_path + "/gfs_{}_{}z.zarr".format(initial_time.strftime("%Y%m%d"), initial_time.strftime("%H"))
    else:
        path_hrrr = machine_info.savepath_hrrr_zarr + "/hrrr_{}_{}z_f01.zarr".format(initial_time.strftime("%Y%m%d"), initial_time.strftime("%H"))
        path_gfs = machine_info.savepath_gfs + "/gfs_{}_{}z.zarr".format(initial_time.strftime("%Y%m%d"), initial_time.strftime("%H"))

    #load model registry:
    with open("./config/registry.json", "r") as f:

        registry = json.load(f)
        model_info = registry["models"][model_shortname]
        use_regression_model = True if model_info["diffusion_only"] == 'False' else False

    use_diffusion = True

    if use_regression_model:
        params = YParams(model_info["regression_config_file"], model_info["regression_config_name"])
        params.local_batch_size = 1
        residual = params.residual
        if machine_info.environment_name == "local":
            params.location = "/data/hrrr_v3_subset/"

    if use_diffusion:
        from utils.diffusions.run_edm import EDMRunner
        edm_config = model_info["edm_config_name"]
        edm_params = YParams(model_info["edm_config_file"], edm_config)
        diffusion_channels = edm_params.diffusion_channels
        input_channels = edm_params.input_channels
        resume_pkl = model_info["edm_checkpoint_path"] 
        posthoc_ema_sigma = model_info["posthoc_ema_sigma"]
        if machine_info.environment_name == "local":
            resume_pkl = resume_pkl.replace("/pscratch/sd/j/jpathak/hrrr_experiments_eos/", "/data/checkpoints/")
            edm_params["location"] = "/data/hrrr_v3_subset/" 
        edm_runner = EDMRunner(edm_params, resume_pkl=resume_pkl, posthoc_ema_sigma=posthoc_ema_sigma, ema=False)
        assert use_regression_model == edm_params.use_regression_net, \
               "Model registry and diffusion model config must agree on use_regression_model"
        residual = edm_params.residual
        log_scale_residual = edm_params.log_scale_residual
        if use_diffusion and not use_regression_model:
            params = edm_params
            params.local_batch_size = 1
    
    #os.makedirs(output_directory, exist_ok=True)

    dataset = GFSDataSet(
        location=params.location,
        conus_dataset_name=params.conus_dataset_name,
        hrrr_stats=params.hrrr_stats,
        exclude_channels=params.exclude_channels,
        path_gfs=path_gfs,
        path_hrrr=path_hrrr
    )

    if use_regression_model:
        if edm_params.regression_net_type == "swin":
            model = swinv2net(params).cuda()
            model.eval()

            checkpoint_path = model_info["regression_checkpoint_path"] 
            if machine_info.environment_name == "local":
                checkpoint_path = checkpoint_path.replace("/pscratch/sd/j/jpathak/hrrr_experiments_eos/", "/data/checkpoints/")

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
            if machine_info.environment_name == "local":
                resume_pkl = resume_pkl.replace("/pscratch/sd/j/jpathak/hrrr_experiments_eos/", "/data/checkpoints/")

            with open(resume_pkl, 'rb') as f:
                data = pickle.load(f)
            net.load_state_dict(data['net'].state_dict(), strict=True)

            latent_shape = [target_channels, 512, 640]

            model = EasyRegression(net, latent_shape).to(device)

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
            if machine_info.environment_name == "local":
                resume_pkl = resume_pkl.replace("/pscratch/sd/j/jpathak/hrrr_experiments_eos/", "/data/checkpoints/")

            with open(resume_pkl, 'rb') as f:
                data = pickle.load(f)
            net.load_state_dict(data['net'].state_dict(), strict=True)

            latent_shape = [target_channels, 512, 640]

            model = EasyRegressionV2(net).to(device)

    if use_regression_model:
        if edm_params.regression_net_type in ["unet", "unet2"]:
            invariant_array = get_invariants(invariant_location="/data/hrrr_v3_subset/hrrr_v3/invariants.zarr", invariant_channels=params.invariants)
            invariant_tensor = torch.from_numpy(invariant_array).to(device).repeat(1, 1, 1, 1)
            model.set_invariant(invariant_tensor)

    base_hrrr_channels, hrrr_channels = dataset.get_hrrr_channel_names()
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

    means_hrrr = dataset.means_hrrr[hrrr_channel_indices]
    stds_hrrr = dataset.stds_hrrr[hrrr_channel_indices]

    means_era5 = dataset.means_era5
    stds_era5 = dataset.stds_era5
    if ensemble_size > 1:
        means_hrrr = np.repeat(means_hrrr[np.newaxis, :, :], ensemble_size, axis=0)
        stds_hrrr = np.repeat(stds_hrrr[np.newaxis, :, :], ensemble_size, axis=0)
        means_era5 = np.repeat(means_era5[np.newaxis, :, :], ensemble_size, axis=0)
        stds_era5 = np.repeat(stds_era5[np.newaxis, :, :], ensemble_size, axis=0)

    gfs_data, hrrr_data = dataset[0]
    assert hrrr_data.ndim == 3

    grid_size = hrrr_data.shape[1:]

    save_channels = ["refc", "u10m", "v10m", "t2m"]


    times = [initial_time + timedelta(hours=i + 1) for i in range(n_steps)] #the model is initialized at using f01, hence the +1

    prediction_array = np.zeros((n_steps, ensemble_size, len(save_channels), *grid_size)) 

    latitudes = dataset.hrrr_latitudes
    longitudes = dataset.hrrr_longitudes

    with torch.no_grad():
        # for i, data in enumerate(loader):
        for i in range(n_steps):

            gfs_data, hrrr_data = dataset[i]
            gfs_data = torch.from_numpy(gfs_data)
            hrrr_data = torch.from_numpy(hrrr_data) if hrrr_data is not None else None

            if i == 0:
                inp = hrrr_data.cuda().float().unsqueeze(0)
                boundary = gfs_data.cuda().float().unsqueeze(0)
                out = inp
                if ensemble_size > 1:
                    out = out.repeat(ensemble_size, 1, 1, 1)
                    boundary = boundary.repeat(ensemble_size, 1, 1, 1)
                out_edm = out.clone()
                if use_regression_model: out_noedm = out.clone()

            assert out_edm.shape == (ensemble_size, len(hrrr_channels)) + grid_size
            denorm_out_edm = out_edm.cpu().numpy() * stds_hrrr + means_hrrr
            
            for cid, name in enumerate(save_channels):
                k = vardict[name]
                prediction_array[i, :, cid] = denorm_out_edm[:, k]

            if i > n_steps:
                break

            if not use_regression_model: 
                if edm_params.pure_diffusion:
                    hrrr_0 = out
                    hrrr_0 = torch.cat((hrrr_0[:, input_channel_indices], boundary), dim=1)
                    edm_corrected_outputs, edm_channels = edm_runner.run(hrrr_0)
                    if residual:
                        out[:, diffusion_channel_indices] += edm_corrected_outputs.float()
                    else:
                        out[:, diffusion_channel_indices] = edm_corrected_outputs.float()
                    out_edm = out.clone()
            else:
                if edm_params.previous_step_conditioning:
                    assert use_regression_model, 'previous step conditioning not yet supported without swin regression'
                    hrrr_0 = out
                    out = model(hrrr_0, boundary, mask=None)
                    hrrr_0 = torch.cat((hrrr_0[:, input_channel_indices, :, :], out[:, input_channel_indices, :, :]), dim=1)
                    if use_diffusion:
                        edm_corrected_outputs, edm_channels = edm_runner.run(hrrr_0)
                        if residual:
                            if log_scale_residual:
                                edm_corrected_outputs = torch.sign(edm_corrected_outputs) * (torch.expm1(torch.abs(edm_corrected_outputs)))
                            out[:, diffusion_channel_indices] += edm_corrected_outputs.float()
                        else:
                            out[:, diffusion_channel_indices] = edm_corrected_outputs.float()
                        out_edm = out.clone()
                else:
                    if use_regression_model:
                        out = model(out, boundary, mask = None)
                    if use_diffusion:
                        edm_corrected_outputs, edm_channels = edm_runner.run(out[:, input_channel_indices, :, :])
                        if residual:
                            if log_scale_residual:
                                print("unscaling log scale residual")
                                edm_corrected_outputs = torch.sign(edm_corrected_outputs) * (torch.expm1(torch.abs(edm_corrected_outputs)))
                            out[:, diffusion_channel_indices] += edm_corrected_outputs.float()
                        else:
                            out[:, diffusion_channel_indices] = edm_corrected_outputs.float()
                        out_edm = out.clone()

            boundary = gfs_data.cuda().float().unsqueeze(0)
            if ensemble_size > 1:
                boundary = boundary.repeat(ensemble_size, 1, 1, 1)
    
    edm_config = model_info["edm_config_name"] if use_diffusion else "no_edm" 
    reg_config = model_info["regression_config_name"] if use_regression_model else "no_reg"
    output_gif_name = "{}_{}_{}".format(reg_config, edm_config, initial_time.isoformat()) 
    gif_output_directory = "./current_run/"


    prediction_dataset = xr.Dataset(
        data_vars=dict(
            data=(["time", "ensemble", "channel", "y", "x"], prediction_array.astype(np.float32)),
        ),
        coords=dict(
            time=times,
            ensemble=np.arange(ensemble_size),
            channel=save_channels,
            longitude=(["y", "x"], longitudes),
            latitude=(["y", "x"], latitudes),
        ), 
    )

    #ml_20240501_12z.zarr
    year, month, day, hour = initial_time.year, initial_time.month, initial_time.day, initial_time.hour
    zarr_output_filename = f"ml_{year}{month:02d}{day:02d}_{hour:02d}z.zarr" 
    zarr_output_path = machine_info.savepath_ml_forecast + "/" + f"{model_shortname}" +"/" + zarr_output_filename
    if not os.path.exists(machine_info.savepath_ml_forecast + "/" + f"{model_shortname}"):
        os.makedirs(machine_info.savepath_ml_forecast + "/" + f"{model_shortname}")
    prediction_dataset.to_zarr(zarr_output_path, mode='w')

    if movie:
        os.makedirs(gif_output_directory, exist_ok=True)
        make_movie(zarr_output_path, dest=gif_output_directory, initial_time = initial_time, output_name = output_gif_name)


    """
    
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
        val_times.append(initial_time_pd+pd.Timedelta(hours=i))
    
    def convert_strings_to_ints(string_list):
        return [int(i) for i in string_list]
    
    model_levels = convert_strings_to_ints(level_names)
    
    ds_pred_edm = xr.Dataset()
    ds_targ = xr.Dataset()

    for var in vertical_vars:
        dsp_edm = xr.Dataset()
        dst = xr.Dataset()

        for i, level in enumerate(level_names):
            key = f'{var}{level}' 

            if key in zarr_group['edm_prediction']:
                # Extract the data from zarr_group
                data_pred_edm = zarr_group['edm_prediction'][key][:, :]
            else:
                data_pred_edm = np.full((len(val_times), lats.shape[0], lats.shape[1]), np.nan)
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

        reshaped_pred_edm = combined_pred_edm.transpose('time', 'levels', 'y', 'x')

        ds_pred_edm[f'{var}_comb'] = reshaped_pred_edm
    
    for var in horizontal_vars:
        data_pred_edm = zarr_group['edm_prediction'][var][:, :]
        dap_edm = xr.DataArray(data_pred_edm,dims=('time','y','x'),
                         coords={'time':val_times,
                                'y': np.arange(lats.shape[0]), 'x': np.arange(lats.shape[1])})
        
        dat = xr.DataArray(data_targ,dims=('time','y','x'),
                         coords={'time':val_times,
                                'y': np.arange(lats.shape[0]), 'x': np.arange(lats.shape[1])})
        ds_pred_edm.update({var:dap_edm})
    
    ds_pred_edm['longitude'] = xr.DataArray(lons,dims=('y','x'))
    ds_pred_edm['latitude'] = xr.DataArray(lats,dims=('y','x'))
    ds_pred_edm = ds_pred_edm.assign_coords(levels=model_levels)
    
    ds_out_path = os.path.join(output_directory, initial_time.strftime("%Y-%m-%dT%H_%M_%S"))
    
    ds_pred_edm.to_netcdf(f"{ds_out_path}/ds_pred_edm.nc",format='NETCDF4')

    """
    
    

if __name__ == "__main__":
    typer.run(run_forecast)
