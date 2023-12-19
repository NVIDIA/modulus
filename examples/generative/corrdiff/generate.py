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

"""Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import cftime
import datetime
import json
import hydra
import netCDF4 as nc
import numpy as np
import torch
import tqdm
import training.dataset
import training.time
from einops import rearrange
from omegaconf import DictConfig
from training.dataset import (
    denormalize,
)

from torch.distributed import gather
from modulus.distributed import DistributedManager
from modulus.launch.logging import PythonLogger, RankZeroLoggingWrapper
from modulus.utils.generative import (
    ablation_sampler,
    parse_int_list,
    StackedRandomGenerator,
    construct_class_by_name,
)


@hydra.main(version_base="1.2", config_path="conf", config_name="config_generate")
def main(cfg: DictConfig) -> None:
    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".
    """

    # Parse options
    res_ckpt_filename = getattr(cfg, "res_ckpt_filename")
    reg_ckpt_filename = getattr(cfg, "reg_ckpt_filename")
    image_outdir = getattr(cfg, "image_outdir")
    seeds = parse_int_list(getattr(cfg, "seeds", "0-63"))
    class_idx = getattr(cfg, "class_idx", None)  # TODO: is this needed?
    num_steps = getattr(cfg, "num_steps", 18)
    sample_res = getattr(cfg, "sample_res", "full")
    res_edm = getattr(cfg, "res_edm", True)
    sampling_method = getattr(cfg, "sampling_method", "stochastic")
    seed_batch_size = getattr(cfg, "seed_batch_size", 1)

    # Parse deterministic sampler options
    sigma_min = getattr(cfg, "sigma_min", None)
    sigma_max = getattr(cfg, "sigma_max", None)
    rho = getattr(cfg, "rho", 7)
    solver = getattr(cfg, "solver", "heun")
    discretization = getattr(cfg, "discretization", "edm")
    schedule = getattr(cfg, "schedule", "linear")
    scaling = getattr(cfg, "scaling", None)
    S_churn = getattr(cfg, "S_churn", 0)
    S_min = getattr(cfg, "S_min", 0)
    S_max = getattr(cfg, "S_max", float("inf"))
    S_noise = getattr(cfg, "S_noise", 1)

    # Parse data options
    train_data_path = getattr(cfg, "train_data_path")
    patch_size = getattr(cfg, "patch_size", 448)
    crop_size_x = getattr(cfg, "crop_size_x", 448)
    crop_size_y = getattr(cfg, "crop_size_y", 448)
    n_history = getattr(cfg, "n_history", 0)
    in_channels = getattr(
        cfg, "in_channels", [0, 1, 2, 3, 4, 9, 10, 11, 12, 17, 18, 19]
    )
    out_channels = getattr(cfg, "out_channels", [0, 17, 18, 19])
    img_shape_x = getattr(cfg, "img_shape_x", 448)
    img_shape_y = getattr(cfg, "img_shape_y", 448)
    roll = getattr(cfg, "roll", False)
    add_grid = getattr(cfg, "add_grid", True)
    ds_factor = getattr(cfg, "ds_factor", 1)
    min_path = getattr(cfg, "min_path", None)
    max_path = getattr(cfg, "max_path", None)
    global_means_path = getattr(cfg, "global_means_path", None)
    global_stds_path = getattr(cfg, "global_stds_path", None)
    gridtype = getattr(cfg, "gridtype", "sinusoidal")
    N_grid_channels = getattr(cfg, "N_grid_channels", 4)
    normalization = getattr(cfg, "normalization", "v2")
    times = getattr(cfg, "times", ["2021-02-02T00:00:00"])

    # Sampler kwargs
    if sampling_method == "stochastic":
        sampler_kwargs = {}
    elif sampling_method == "deterministic":
        sampler_kwargs = {
            "sigma_min": sigma_min,
            "sigma_max": sigma_max,
            "rho": rho,
            "solver": solver,
            "discretization": discretization,
            "schedule": schedule,
            "scaling": scaling,
            "S_churn": S_churn,
            "S_min": S_min,
            "S_max": S_max,
            "S_noise": S_noise,
        }
    else:
        raise ValueError(f"Unknown sampling method {sampling_method}")

    # Initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    # Initialize logger
    logger = PythonLogger("generate")  # General python logger
    logger0 = RankZeroLoggingWrapper(logger, dist)
    logger.file_logging("generate.log")

    logger0.info(f"Train data path: {train_data_path}")
    dataset, sampler = get_dataset_and_sampler(
        path=train_data_path,  # TODO check if this should be train data path
        n_history=n_history,
        in_channels=in_channels,
        out_channels=out_channels,
        img_shape_x=img_shape_x,
        img_shape_y=img_shape_y,
        crop_size_x=crop_size_x,
        crop_size_y=crop_size_y,
        roll=roll,
        add_grid=add_grid,
        train=None,
        ds_factor=ds_factor,
        min_path=min_path,
        max_path=max_path,
        global_means_path=global_means_path,
        global_stds_path=global_stds_path,
        gridtype=gridtype,
        N_grid_channels=N_grid_channels,
        times=times,
        normalization=normalization,
        all_times=True,
    )

    logger0.info(f"torch.__version__: {torch.__version__}")

    # Load diffusion network
    logger0.info(f'Loading residual network from "{res_ckpt_filename}"...')
    net_res_state_dict = torch.load(res_ckpt_filename)
    with open("checkpoints/args_diffusion.json", "r") as json_file:
        args_diffusion = json.load(json_file)
    net_res = construct_class_by_name(**args_diffusion)
    net_res.load_state_dict(net_res_state_dict, strict=False)

    # load regression network
    logger0.info(f'Loading network from "{reg_ckpt_filename}"...')
    net_reg_state_dict = torch.load(reg_ckpt_filename) if res_edm else None
    if res_edm:
        with open("checkpoints/args_regression.json", "r") as json_file:
            args_regression = json.load(json_file)
        net_reg = construct_class_by_name(**args_regression)
        net_reg.load_state_dict(net_reg_state_dict, strict=False)
    else:
        net_reg = None

    # move to device
    torch.cuda.set_device(dist.rank)  # TODO is this needed?
    device = dist.device
    net_res = net_res.eval()
    net_reg = net_reg.eval() if net_reg else None

    def generate_fn(image_lr):
        """Function to generate an image

        Args:
            image_lr: low resolution input. shape: (b, c, h, w)

        Return
            image_hr: high resolution output: shape (b, c, h, w)
        """
        if sample_res == "full":
            image_lr_patch = image_lr
        else:
            image_lr_patch = rearrange(
                image_lr,
                "b c (h1 h) (w1 w) -> (b h1 w1) c h w",
                h1=crop_size_x // patch_size,
                w1=crop_size_y // patch_size,
            )

        sample_seeds = seeds

        logger0.info(f"seeds: {sample_seeds}")
        if net_reg:
            net_reg.to(device)
            image_mean = generate(
                net=net_reg,
                img_lr=image_lr_patch,
                seed_batch_size=1,
                seeds=sample_seeds,
                pretext="reg",
                class_idx=class_idx,
            )
            net_reg.cpu()

            net_res.to(device)
            image_out = image_mean + generate(
                net=net_res,
                img_lr=image_lr_patch.expand(seed_batch_size, -1, -1, -1),
                seed_batch_size=seed_batch_size,
                sampling_method=sampling_method,
                seeds=sample_seeds,
                pretext="gen",
                class_idx=class_idx,
                num_steps=num_steps,
                **sampler_kwargs,
            )
            net_res.cpu()

        else:
            net_reg.to(device)
            image_out = generate(
                net=net_res,
                img_lr=image_lr_patch.expand(seed_batch_size, -1, -1, -1),
                seed_batch_size=seed_batch_size,
                sampling_method=sampling_method,
                seeds=sample_seeds,
                pretext="gen",
                class_idx=class_idx,
                num_steps=num_steps,
                **sampler_kwargs,
            )

        # reshape: (1*9*9)x3x50x50  --> 1x3x450x450
        if sample_res != "full":
            image_out = rearrange(
                image_out,
                "(b h1 w1) c h w -> b c (h1 h) (w1 w)",
                h1=crop_size_x // patch_size,
                w1=crop_size_y // patch_size,
            )

        # Gather tensors on rank 0
        if dist.world_size > 1:
            if dist.rank == 0:
                gathered_tensors = [
                    torch.zeros_like(
                        image_out, dtype=image_out.dtype, device=image_out.device
                    )
                    for _ in range(dist.world_size)
                ]
            else:
                gathered_tensors = None

            torch.distributed.barrier()
            gather(
                image_out,
                gather_list=gathered_tensors if dist.rank == 0 else None,
                dst=0,
            )

            if dist.rank == 0:
                return torch.cat(gathered_tensors)
            else:
                return None
        else:
            return image_out

    # generate images
    logger0.info("Generating images...")
    with nc.Dataset(f"{image_outdir}_{dist.rank}.nc", "w") as f:
        # add attributes
        f.cfg = str(cfg)
        generate_and_save(dataset, sampler, f, generate_fn, device, logger0)

    logger0.info("Done.")


def _get_name(channel_info):
    plev = (
        ""
        if np.isnan(channel_info["pressure"])
        else "{:d}".format(int(channel_info["pressure"]))
    )
    return channel_info["variable"] + plev


def get_dataset_and_sampler(
    path,
    n_history,
    in_channels,
    out_channels,
    img_shape_x,
    img_shape_y,
    crop_size_x,
    crop_size_y,
    roll,
    add_grid,
    train,
    ds_factor,
    min_path,
    max_path,
    global_means_path,
    global_stds_path,
    gridtype,
    N_grid_channels,
    times,
    normalization="v1",
    all_times=False,
):
    """
    Get a dataset and sampler for generation.
    """
    dataset = training.dataset.get_zarr_dataset(
        path=path,
        n_history=n_history,
        in_channels=in_channels,
        out_channels=out_channels,
        img_shape_x=img_shape_x,
        img_shape_y=img_shape_y,
        crop_size_x=crop_size_x,
        crop_size_y=crop_size_y,
        roll=roll,
        add_grid=add_grid,
        train=train,
        ds_factor=ds_factor,
        min_path=min_path,
        max_path=max_path,
        global_means_path=global_means_path,
        global_stds_path=global_stds_path,
        gridtype=gridtype,
        N_grid_channels=N_grid_channels,
        normalization=normalization,
        all_times=all_times,
    )
    plot_times = [
        training.time.convert_datetime_to_cftime(
            datetime.datetime.strptime(time, "%Y-%m-%dT%H:%M:%S")
        )
        for time in times
    ]
    all_times = dataset.time()
    time_indices = [all_times.index(t) for t in plot_times]
    sampler = time_indices

    return dataset, sampler


def generate_and_save(dataset, sampler, f: nc.Dataset, generate_fn, device, logger):
    """
    This function generates model predictions from the input data using the specified
    `generate_fn`, and saves the predictions to the provided NetCDF file. It iterates
    through the dataset using a data loader, computes predictions, and saves them along
    with associated metadata.

    Parameters:
        dataset: Input dataset.
        sampler: Sampler for selecting data samples.
        f: NetCDF file for saving predictions.
        generate_fn: Function for generating model predictions.
        device: Device (e.g., GPU) for computation.
        logger: Logger for logging information.
    """
    # Instantiate distributed manager.
    dist = DistributedManager()
    device = dist.device

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset, sampler=sampler, batch_size=1, pin_memory=True
    )
    time_index = -1
    writer = writer_from_input_dataset(f, dataset)

    for image_tar, image_lr, index in iter(data_loader):
        time_index += 1
        if dist.rank == 0:
            logger.info(f"starting index: {time_index}")  # TODO print on rank zero
        # continue
        input_data = image_lr = image_lr.to(device=device).to(torch.float32)
        image_tar = image_tar.to(device=device).to(torch.float32)
        image_out = generate_fn(image_lr)

        # for validation - make 3x450x450 to an ordered sequence of 50x50 patches
        # input; 1x3x450x450 --> (1*9*9)x3x50x50

        if dist.rank == 0:
            # weather sub-plot
            mx, sx = dataset.info()["input_normalization"]
            mx = mx[dataset.in_channels]
            image_lr2 = image_lr[0].unsqueeze(0)

            # add zeros for grid embeddings
            padding = image_lr2.shape[1] - len(mx)
            if not padding >= 0:
                raise ValueError("padding must be non-negative")

            mx = np.concatenate([mx, np.zeros(padding)])
            # add zeros for grid embeddings
            sx = sx[dataset.in_channels]
            sx = np.concatenate([sx, np.ones(padding)])
            image_lr2 = image_lr2.cpu().numpy()
            image_lr2 = denormalize(image_lr2, mx, sx)

            my, sy = dataset.info()["target_normalization"]
            my = my[dataset.out_channels]
            sy = sy[dataset.out_channels]
            image_tar2 = image_tar[0].unsqueeze(0)
            image_tar2 = image_tar2.cpu().numpy()
            image_tar2 = denormalize(image_tar2, my, sy)

            # some runtime assertions
            if image_tar2.ndim != 4:
                raise ValueError("image_tar2 must be 4-dimensional")

            for idx in range(image_out.shape[0]):
                image_out2 = image_out[idx].unsqueeze(0)
                if image_out2.ndim != 4:
                    raise ValueError("image_out2 must be 4-dimensional")

                # Denormalize the input and outputs
                image_out2 = image_out2.cpu().numpy()
                image_out2 = denormalize(image_out2, my, sy)

                t_index = index[0]
                if len(index) != 1:
                    raise ValueError(
                        "len(index) must be 1"
                    )  # TODO allow len(index) > 1
                time = dataset.time()[t_index]
                writer.write_time(time_index, time)
                for channel_idx in range(image_out2.shape[1]):
                    output_channels = dataset.output_channels()
                    info = output_channels[channel_idx]
                    channel_name = _get_name(info)
                    truth = image_tar2[0, channel_idx]

                    writer.write_truth(channel_name, time_index, truth)
                    writer.write_prediction(
                        channel_name, time_index, idx, image_out2[0, channel_idx]
                    )

                input_channels = dataset.input_channels()
                for channel_idx in range(len(input_channels)):
                    info = input_channels[channel_idx]
                    channel_name = _get_name(info)
                    writer.write_input(
                        channel_name, time_index, image_lr2[0, channel_idx]
                    )


def generate(
    net,
    seeds,
    class_idx,
    seed_batch_size,
    sampling_method=None,
    img_lr=None,
    pretext=None,
    **sampler_kwargs,
):
    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".
    """

    if sampling_method == "stochastic":
        # import stochastic sampler
        try:
            from edmss import edm_sampler
        except ImportError:
            raise ImportError(
                "Please get the edm_sampler by running: pip install git+https://github.com/mnabian/edmss.git"
            )

    # Instantiate distributed manager.
    dist = DistributedManager()
    device = dist.device

    num_batches = (
        (len(seeds) - 1) // (seed_batch_size * dist.world_size) + 1
    ) * dist.world_size
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.rank :: dist.world_size]

    # Synchronize
    if dist.world_size > 1:
        torch.distributed.barrier()

    # Loop over batches.
    all_images = []
    for batch_seeds in tqdm.tqdm(rank_batches, unit="batch", disable=(dist.rank != 0)):
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        # Pick latents and labels.
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn(
            [
                seed_batch_size,
                net.img_out_channels,
                net.img_resolution,
                net.img_resolution,
            ],
            device=device,
        )

        class_labels = None
        if net.label_dim:
            class_labels = torch.eye(net.label_dim, device=device)[
                rnd.randint(net.label_dim, size=[seed_batch_size], device=device)
            ]
        if class_idx is not None:
            class_labels[:, :] = 0
            class_labels[:, class_idx] = 1

        # Generate images.
        sampler_kwargs = {
            key: value for key, value in sampler_kwargs.items() if value is not None
        }

        if pretext == "gen":
            if sampling_method == "deterministic":
                sampler_fn = ablation_sampler
            elif sampling_method == "stochastic":
                sampler_fn = edm_sampler
            else:
                raise ValueError(
                    f"Unknown sampling method {sampling_method}. Should be either 'stochastic' or 'deterministic'."
                )
        elif pretext == "reg":
            latents = torch.zeros_like(latents)
            sampler_fn = unet_regression
        else:
            raise ValueError(
                f"Unknown pretext {pretext}. Should be either 'gen' or 'reg'."
            )

        with torch.inference_mode():
            images = sampler_fn(
                net.to(device),
                latents,
                img_lr,
                class_labels,
                randn_like=rnd.randn_like,
                **sampler_kwargs,
            )
        all_images.append(images)
    return torch.cat(all_images)


def unet_regression(  # TODO a lot of redundancy, need to clean up
    net,
    latents,
    img_lr,
    class_labels=None,
    randn_like=torch.randn_like,
    num_steps=2,
    sigma_min=0.0,
    sigma_max=0.0,
    rho=7,
    S_churn=0,
    S_min=0,
    S_max=float("inf"),
    S_noise=0.0,
):
    """
    Perform U-Net regression with temporal sampling.

    Parameters:
        net (torch.nn.Module): U-Net model for regression.
        latents (torch.Tensor): Latent representation.
        img_lr (torch.Tensor): Low-resolution input image.
        class_labels (torch.Tensor, optional): Class labels for conditional generation.
        randn_like (function, optional): Function for generating random noise.
        num_steps (int, optional): Number of time steps for temporal sampling.
        sigma_min (float, optional): Minimum noise level.
        sigma_max (float, optional): Maximum noise level.
        rho (int, optional): Exponent for noise level interpolation.
        S_churn (float, optional): Churning parameter.
        S_min (float, optional): Minimum churning value.
        S_max (float, optional): Maximum churning value.
        S_noise (float, optional): Noise level for churning.

    Returns:
        torch.Tensor: Predicted output at the next time step.
    """

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (
        sigma_max ** (1 / rho)
        + step_indices
        / (num_steps - 1)
        * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat(
        [net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]
    )  # t_N = 0

    # conditioning
    x_lr = img_lr

    # Main sampling loop.
    x_hat = latents.to(torch.float64) * t_steps[0]
    t_hat = torch.tensor(1.0).to(torch.float64).cuda()

    x_next = net(x_hat, x_lr, t_hat, class_labels).to(torch.float64)

    return x_next


class NetCDFWriter:
    """NetCDF Writer"""

    def __init__(self, f, lat, lon, input_channels, output_channels):
        self._f = f

        # create unlimited dimensions
        f.createDimension("time")
        f.createDimension("ensemble")

        if lat.shape != lon.shape:
            raise ValueError("lat and lon must have the same shape")
        ny, nx = lat.shape

        # create lat/lon grid
        f.createDimension("x", nx - 2)
        f.createDimension("y", ny - 2)

        v = f.createVariable("lat", "f", dimensions=("y", "x"))
        v[:] = lat[1:-1, 1:-1]
        v.standard_name = "latitude"
        v.units = "degrees_north"

        v = f.createVariable("lon", "f", dimensions=("y", "x"))
        v[:] = lon[1:-1, 1:-1]
        v.standard_name = "longitude"
        v.units = "degrees_east"

        # create time dimension
        v = f.createVariable("time", "i8", ("time"))
        v.calendar = "standard"
        v.units = "hours since 1990-01-01 0:0:0"

        self.truth_group = f.createGroup("truth")
        self.prediction_group = f.createGroup("prediction")
        self.input_group = f.createGroup("input")

        for variable in output_channels:
            name = _get_name(variable)
            self.truth_group.createVariable(name, "f", dimensions=("time", "y", "x"))
            self.prediction_group.createVariable(
                name, "f", dimensions=("ensemble", "time", "y", "x")
            )

        # setup input data in netCDF

        n_grid_inputs = 4  # TODO get this from the model object
        for i in range(n_grid_inputs):
            input_channels.append({"variable": "grid", "pressure": i})

        for variable in input_channels:
            name = _get_name(variable)
            self.input_group.createVariable(name, "f", dimensions=("time", "y", "x"))

    def write_input(self, channel_name, time_index, val):
        """Write input data to NetCDF file."""
        self.input_group[channel_name][time_index] = val

    def write_truth(self, channel_name, time_index, val):
        """Write ground truth data to NetCDF file."""
        self.truth_group[channel_name][time_index] = val

    def write_prediction(self, channel_name, time_index, ensemble_index, val):
        """Write prediction data to NetCDF file."""
        self.prediction_group[channel_name][ensemble_index, time_index] = val

    def write_time(self, time_index, time):
        """Write time information to NetCDF file."""
        time_v = self._f["time"]
        self._f["time"][time_index] = cftime.date2num(
            time, time_v.units, time_v.calendar
        )


def writer_from_input_dataset(f, dataset):
    """Create a NetCDFWriter object from an input dataset."""
    return NetCDFWriter(
        f,
        lat=dataset.latitude(),
        lon=dataset.longitude(),
        input_channels=dataset.input_channels(),
        output_channels=dataset.output_channels(),
    )


# ----------------------------------------------------------------------------


if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
