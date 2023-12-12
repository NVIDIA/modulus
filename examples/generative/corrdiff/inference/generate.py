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

from modulus.distributed import DistributedManager
from modulus.launch.logging import PythonLogger, RankZeroLoggingWrapper
from modulus.utils.generative import parse_int_list


def unet_regression(
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


def ablation_sampler(
    net,
    latents,
    img_lr,
    class_labels=None,
    randn_like=torch.randn_like,
    num_steps=18,
    sigma_min=None,
    sigma_max=None,
    rho=7,
    solver="heun",
    discretization="edm",
    schedule="linear",
    scaling="none",
    epsilon_s=1e-3,
    C_1=0.001,
    C_2=0.008,
    M=1000,
    alpha=1,
    S_churn=0,
    S_min=0,
    S_max=float("inf"),
    S_noise=1,
):
    """
    Generalized sampler, representing the superset of all sampling methods discussed
    in the paper "Elucidating the Design Space of Diffusion-Based Generative Models"
    """

    # conditioning
    x_lr = img_lr

    if solver not in ["euler", "heun"]:
        raise ValueError(f"Unknown solver {solver}")
    if discretization not in ["vp", "ve", "iddpm", "edm"]:
        raise ValueError(f"Unknown discretization {discretization}")
    if schedule not in ["vp", "ve", "linear"]:
        raise ValueError(f"Unknown schedule {schedule}")
    if scaling not in ["vp", "none"]:
        raise ValueError(f"Unknown scaling {scaling}")

    # Helper functions for VP & VE noise level schedules.
    vp_sigma = (
        lambda beta_d, beta_min: lambda t: (
            np.e ** (0.5 * beta_d * (t**2) + beta_min * t) - 1
        )
        ** 0.5
    )
    vp_sigma_deriv = (
        lambda beta_d, beta_min: lambda t: 0.5
        * (beta_min + beta_d * t)
        * (sigma(t) + 1 / sigma(t))
    )
    vp_sigma_inv = (
        lambda beta_d, beta_min: lambda sigma: (
            (beta_min**2 + 2 * beta_d * (sigma**2 + 1).log()).sqrt() - beta_min
        )
        / beta_d
    )
    ve_sigma = lambda t: t.sqrt()
    ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
    ve_sigma_inv = lambda sigma: sigma**2

    # Select default noise level range based on the specified time step discretization.
    if sigma_min is None:
        vp_def = vp_sigma(beta_d=19.1, beta_min=0.1)(t=epsilon_s)
        sigma_min = {"vp": vp_def, "ve": 0.02, "iddpm": 0.002, "edm": 0.002}[
            discretization
        ]
    if sigma_max is None:
        vp_def = vp_sigma(beta_d=19.1, beta_min=0.1)(t=1)
        sigma_max = {"vp": vp_def, "ve": 100, "iddpm": 81, "edm": 80}[discretization]

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Compute corresponding betas for VP.
    vp_beta_d = (
        2
        * (np.log(sigma_min**2 + 1) / epsilon_s - np.log(sigma_max**2 + 1))
        / (epsilon_s - 1)
    )
    vp_beta_min = np.log(sigma_max**2 + 1) - 0.5 * vp_beta_d

    # Define time steps in terms of noise level.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    if discretization == "vp":
        orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
        sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
    elif discretization == "ve":
        orig_t_steps = (sigma_max**2) * (
            (sigma_min**2 / sigma_max**2) ** (step_indices / (num_steps - 1))
        )
        sigma_steps = ve_sigma(orig_t_steps)
    elif discretization == "iddpm":
        u = torch.zeros(M + 1, dtype=torch.float64, device=latents.device)
        alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
        for j in torch.arange(M, 0, -1, device=latents.device):  # M, ..., 1
            u[j - 1] = (
                (u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1
            ).sqrt()
        u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
        sigma_steps = u_filtered[
            ((len(u_filtered) - 1) / (num_steps - 1) * step_indices)
            .round()
            .to(torch.int64)
        ]
    else:
        sigma_steps = (
            sigma_max ** (1 / rho)
            + step_indices
            / (num_steps - 1)
            * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
        ) ** rho

    # Define noise level schedule.
    if schedule == "vp":
        sigma = vp_sigma(vp_beta_d, vp_beta_min)
        sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
        sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
    elif schedule == "ve":
        sigma = ve_sigma
        sigma_deriv = ve_sigma_deriv
        sigma_inv = ve_sigma_inv
    else:
        sigma = lambda t: t
        sigma_deriv = lambda t: 1
        sigma_inv = lambda sigma: sigma

    # Define scaling schedule.
    if scaling == "vp":
        s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
        s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
    else:
        s = lambda t: 1
        s_deriv = lambda t: 0

    # Compute final time steps based on the corresponding noise levels.
    t_steps = sigma_inv(net.round_sigma(sigma_steps))
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0

    # Main sampling loop.
    t_next = t_steps[0]
    x_next = latents.to(torch.float64) * (sigma(t_next) * s(t_next))
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = (
            min(S_churn / num_steps, np.sqrt(2) - 1)
            if S_min <= sigma(t_cur) <= S_max
            else 0
        )
        t_hat = sigma_inv(net.round_sigma(sigma(t_cur) + gamma * sigma(t_cur)))
        x_hat = s(t_hat) / s(t_cur) * x_cur + (
            sigma(t_hat) ** 2 - sigma(t_cur) ** 2
        ).clip(min=0).sqrt() * s(t_hat) * S_noise * randn_like(x_cur)

        # Euler step.
        h = t_next - t_hat
        denoised = net(
            x_hat / s(t_hat), x_lr / s(t_hat), sigma(t_hat), class_labels
        ).to(torch.float64)
        d_cur = (
            sigma_deriv(t_hat) / sigma(t_hat) + s_deriv(t_hat) / s(t_hat)
        ) * x_hat - sigma_deriv(t_hat) * s(t_hat) / sigma(t_hat) * denoised
        x_prime = x_hat + alpha * h * d_cur
        t_prime = t_hat + alpha * h

        # Apply 2nd order correction.
        if solver == "euler" or i == num_steps - 1:
            x_next = x_hat + h * d_cur
        else:
            denoised = net(
                x_prime / s(t_prime), x_lr / s(t_prime), sigma(t_prime), class_labels
            ).to(torch.float64)
            d_prime = (
                sigma_deriv(t_prime) / sigma(t_prime) + s_deriv(t_prime) / s(t_prime)
            ) * x_prime - sigma_deriv(t_prime) * s(t_prime) / sigma(t_prime) * denoised
            x_next = x_hat + h * (
                (1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_prime
            )

    return x_next


class StackedRandomGenerator:
    """
    Wrapper for torch.Generator that allows specifying a different random seed
    for each sample in a minibatch.
    """

    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [
            torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds
        ]

    def randn(self, size, **kwargs):
        if size[0] != len(self.generators):
            raise ValueError(f"Expected first dimension of size {len(self.generators)}")
        return torch.stack(
            [torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators]
        )

    def randn_like(self, input):
        return self.randn(
            input.shape, dtype=input.dtype, layout=input.layout, device=input.device
        )

    def randint(self, *args, size, **kwargs):
        if size[0] != len(self.generators):
            raise ValueError(f"Expected first dimension of size {len(self.generators)}")
        return torch.stack(
            [
                torch.randint(*args, size=size[1:], generator=gen, **kwargs)
                for gen in self.generators
            ]
        )


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


# ----------------------------------------------------------------------------


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
    pretext = getattr(cfg, "pretext", None)
    res_edm = getattr(cfg, "res_edm", True)
    sampling_method = getattr(cfg, "sampling_method", "stochastic")

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

    det_batch = None
    gen_batch = None  # NOTE: This will always set the batch size to 1. Should be fixed.

    if gen_batch is None:
        gen_batch = 1  # max(4096 // net.img_resolution, 1)
    if det_batch is None:
        det_batch = 1  # max(gen_batch, 64)
    if det_batch % gen_batch != 0:
        raise ValueError(
            f"det_batch ({det_batch}) must be divisible by gen_batch ({gen_batch})"
        )

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

    with nc.Dataset(f"{image_outdir}_{dist.rank}.nc", "w") as f:
        # add attributes
        f.cfg = str(cfg)

        # Load network
        logger.info(f"torch.__version__: {torch.__version__}")
        logger0.info(f'Loading residual network from "{res_ckpt_filename}"...')
        net_res = torch.load(res_ckpt_filename)
        logger0.info(f'Loading network from "{reg_ckpt_filename}"...')
        net_reg = torch.load(reg_ckpt_filename) if res_edm else None

        # move to device
        torch.cuda.set_device(dist.rank)
        device = dist.device
        net_res = net_res.to(device)
        net_reg = net_reg.to(device) if net_reg else None

        batch_size = min(
            len(sampler), det_batch
        )  # TODO: check if batch size can be more than 1

        def generate_fn(image_lr):
            """Function to generate an image with

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
                image_mean = generate(
                    net=net_reg,
                    img_lr=image_lr_patch,
                    max_batch_size=image_lr_patch.shape[0],
                    seeds=sample_seeds,
                    pretext="reg",
                    class_idx=class_idx,
                )
                image_out = image_mean + generate(
                    net=net_res,
                    img_lr=image_lr_patch,
                    max_batch_size=image_lr_patch.shape[0],
                    sampling_method=sampling_method,
                    seeds=sample_seeds,
                    pretext="gen",
                    class_idx=class_idx,
                    num_steps=num_steps,
                    **sampler_kwargs,
                )
            else:
                image_out = generate(
                    net=net_res,
                    img_lr=image_lr_patch,
                    max_batch_size=image_lr_patch.shape[0],
                    sampling_method=sampling_method,
                    seeds=sample_seeds,
                    pretext=pretext,
                    class_idx=class_idx,
                    num_steps=num_steps,
                    **sampler_kwargs,
                )

            # reshape: (1*9*9)x3x50x50  --> 1x3x450x450
            if sample_res == "full":
                image_lr_patch = image_lr
            else:
                image_out = rearrange(
                    image_out,
                    "(b h1 w1) c h w -> b c (h1 h) (w1 w)",
                    h1=crop_size_x // patch_size,
                    w1=crop_size_y // patch_size,
                )

            return image_out

        # generate images
        logger0.info("Generating images...")
        generate_and_save(dataset, sampler, f, generate_fn, device, batch_size, logger0)

    # Done.
    if dist.world_size > 1:
        torch.distributed.barrier()
    logger0.info("Done.")


def _get_name(channel_info):
    plev = (
        ""
        if np.isnan(channel_info["pressure"])
        else "{:d}".format(int(channel_info["pressure"]))
    )
    return channel_info["variable"] + plev


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


def generate_and_save(
    dataset, sampler, f: nc.Dataset, generate_fn, device, batch_size, logger
):
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
        batch_size: Batch size for data loading.
        logger: Logger for logging information.
    """
    # Instantiate distributed manager.
    dist = DistributedManager()
    device = dist.device

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset, sampler=sampler, batch_size=batch_size, pin_memory=True
    )
    time_index = -1
    writer = writer_from_input_dataset(f, dataset)

    for image_tar, image_lr, index in iter(data_loader):
        time_index += 1
        if dist.rank == 0:
            logger.info(f"starting index: {time_index}")  # TODO print on rank zero
        input_data = image_lr = image_lr.to(device=device).to(torch.float32)
        image_tar = image_tar.to(device=device).to(torch.float32)
        image_out = generate_fn(image_lr)

        # for validation - make 3x450x450 to an ordered sequence of 50x50 patches
        # input; 1x3x450x450 --> (1*9*9)x3x50x50

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
                raise ValueError("len(index) must be 1")
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
                writer.write_input(channel_name, time_index, image_lr2[0, channel_idx])


def generate(
    net,
    seeds,
    class_idx,
    max_batch_size,
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
        (len(seeds) - 1) // (max_batch_size * dist.world_size) + 1
    ) * dist.world_size
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.rank :: dist.world_size]

    # Rank 0 goes first.
    if dist.world_size > 1 and dist.rank != 0:
        torch.distributed.barrier()

        # Other ranks follow.
        if dist.world_size > 1 and dist.rank == 0:
            torch.distributed.barrier()

    # Loop over batches.
    all_images = []
    for batch_seeds in tqdm.tqdm(rank_batches, unit="batch", disable=(dist.rank != 0)):
        if dist.world_size > 1:
            torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        # Pick latents and labels.
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn(
            [
                max_batch_size,
                net.img_out_channels,
                net.img_resolution,
                net.img_resolution,
            ],
            device=device,
        )

        class_labels = None
        if net.label_dim:
            class_labels = torch.eye(net.label_dim, device=device)[
                rnd.randint(net.label_dim, size=[batch_size], device=device)
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

        images = sampler_fn(
            net,
            latents,
            img_lr,
            class_labels,
            randn_like=rnd.randn_like,
            **sampler_kwargs,
        )
        all_images.append(images)

    return torch.cat(all_images, dim=0)


# ----------------------------------------------------------------------------


if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
