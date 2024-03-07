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

"""Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

from concurrent.futures import ThreadPoolExecutor
import datetime

import cftime
from einops import rearrange
import hydra
import netCDF4 as nc
import nvtx
from omegaconf import OmegaConf, DictConfig
import torch
from torch.distributed import gather
import torch._dynamo
import tqdm


from modulus.distributed import DistributedManager
from modulus.launch.logging import PythonLogger, RankZeroLoggingWrapper
from modulus.utils.generative import (
    ablation_sampler,
    parse_int_list,
    StackedRandomGenerator,
)
from module import Module  # TODO import from Core once the kwargs issue is fixed

from datasets.base import DownscalingDataset
from datasets.dataset import init_dataset_from_config
from training.time import convert_datetime_to_cftime, time_range


time_format = "%Y-%m-%dT%H:%M:%S"


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
    force_fp16 = getattr(cfg, "force_fp16", False)
    use_torch_compile = getattr(cfg, "use_torch_compile", True)

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
    times_range = getattr(cfg, "time_range", None)
    patch_size = getattr(cfg, "patch_size", 448)
    patch_shape_x = getattr(cfg, "patch_shape_x", 448)
    patch_shape_y = getattr(cfg, "patch_shape_y", 448)
    overlap_pix = getattr(cfg, "overlap_pix", 0)
    boundary_pix = getattr(cfg, "boundary_pix", 2)

    if times_range is not None:
        times = []
        t_start = datetime.datetime.strptime(times_range[0], time_format)
        t_end = datetime.datetime.strptime(times_range[1], time_format)
        dt = datetime.timedelta(hours=(times_range[2] if len(times_range) > 2 else 1))
        times = [
            t.strftime(time_format)
            for t in time_range(t_start, t_end, dt, inclusive=True)
        ]
    else:
        times = getattr(cfg, "times", ["2021-02-02T00:00:00"])

    # writer options
    num_writer_workers = getattr(cfg, "num_writer_workers", 1)

    # Create dataset object
    dataset_cfg = OmegaConf.to_container(cfg.dataset)
    dataset, sampler = get_dataset_and_sampler(dataset_cfg=dataset_cfg, times=times)
    (img_shape_y, img_shape_x) = dataset.image_shape()

    # Sampler kwargs
    if sampling_method == "stochastic":
        sampler_kwargs = {
            "img_shape": img_shape_x,
            "patch_shape": patch_shape_x,
            "overlap_pix": overlap_pix,
            "boundary_pix": boundary_pix,
        }
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

    if patch_shape_x != patch_shape_y:
        raise NotImplementedError("Rectangular patch not supported yet")
    if patch_shape_x % 32 != 0 or patch_shape_y % 32 != 0:
        raise ValueError("Patch shape needs to be a factor of 32")
    if patch_shape_x > img_shape_x:
        patch_shape_x = img_shape_x
    if patch_shape_y > img_shape_y:
        patch_shape_y = img_shape_y
    if patch_shape_x != img_shape_x or patch_shape_y != img_shape_y:
        logger0.info("Patch-based generation enabled")
    else:
        logger0.info("Patch-based generation disabled")

    logger0.info(f"torch.__version__: {torch.__version__}")

    # Load diffusion network
    logger0.info(f'Loading residual network from "{res_ckpt_filename}"...')
    net_res = Module.from_checkpoint(res_ckpt_filename)

    # load regression network
    if res_edm:
        logger0.info(f'Loading network from "{reg_ckpt_filename}"...')
        net_reg = Module.from_checkpoint(reg_ckpt_filename)
    else:
        net_reg = None

    # move to device
    device = dist.device
    net_res = net_res.eval().to(device).to(memory_format=torch.channels_last)
    net_reg = (
        net_reg.eval().to(device).to(memory_format=torch.channels_last)
        if net_reg
        else None
    )

    # change precision if needed
    if force_fp16:
        net_reg.use_fp16 = True
        net_res.use_fp16 = True

    # Reset since we are using a different mode.
    if use_torch_compile:
        torch._dynamo.reset()
        compile_mode = "reduce-overhead"
        # Only compile residual network
        # Overhead of compiling regression network outweights any benefits
        net_res = torch.compile(net_res, mode=compile_mode)

    def generate_fn(image_lr):
        """Function to generate an image

        Args:
            image_lr: low resolution input. shape: (b, c, h, w)

        Return
            image_hr: high resolution output: shape (b, c, h, w)
        """
        with nvtx.annotate("generate_fn", color="green"):
            if sample_res == "full":
                image_lr_patch = image_lr
            else:
                torch.cuda.nvtx.range_push("rearrange")
                image_lr_patch = rearrange(
                    image_lr,
                    "b c (h1 h) (w1 w) -> (b h1 w1) c h w",
                    h1=img_shape_y // patch_size,
                    w1=img_shape_x // patch_size,
                )
                torch.cuda.nvtx.range_pop()
            image_lr_patch = image_lr_patch.to(memory_format=torch.channels_last)

            sample_seeds = seeds

            logger0.info(f"seeds: {sample_seeds}")
            if net_reg:
                with nvtx.annotate("regression_model", color="yellow"):
                    image_mean = generate(
                        net=net_reg,
                        img_lr=image_lr_patch,
                        seed_batch_size=1,
                        seeds=[
                            0,
                        ],  # Only run regression model once
                        pretext="reg",
                        class_idx=class_idx,
                    )

                with nvtx.annotate("diffusion model", color="purple"):
                    image_out = image_mean + generate(
                        net=net_res,
                        img_lr=image_lr_patch.expand(seed_batch_size, -1, -1, -1).to(
                            memory_format=torch.channels_last
                        ),
                        seed_batch_size=seed_batch_size,
                        sampling_method=sampling_method,
                        seeds=sample_seeds,
                        pretext="gen",
                        class_idx=class_idx,
                        num_steps=num_steps,
                        **sampler_kwargs,
                    )

            else:
                with nvtx.annotate("diffusion model", color="purple"):
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
                    h1=img_shape_y // patch_size,
                    w1=img_shape_x // patch_size,
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
        with torch.cuda.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():
                generate_and_save(
                    dataset,
                    sampler,
                    f,
                    generate_fn,
                    device,
                    num_writer_workers,
                    logger0,
                )

    logger0.info("Done.")


def _get_name(channel_info):
    return channel_info.name + channel_info.level


def get_dataset_and_sampler(dataset_cfg, times):
    """
    Get a dataset and sampler for generation.
    """
    (dataset, _) = init_dataset_from_config(dataset_cfg, batch_size=1)
    plot_times = [
        convert_datetime_to_cftime(datetime.datetime.strptime(time, time_format))
        for time in times
    ]
    all_times = dataset.time()
    time_indices = [all_times.index(t) for t in plot_times]
    sampler = time_indices

    return dataset, sampler


def generate_and_save(
    dataset, sampler, f: nc.Dataset, generate_fn, device, num_writer_workers, logger
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
    warmup_steps = 2

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    batch_size = 1

    # Initialize threadpool for writers
    writer_executor = ThreadPoolExecutor(max_workers=num_writer_workers)
    writer_threads = []

    times = dataset.time()

    for image_tar, image_lr, index in iter(data_loader):
        time_index += 1
        if dist.rank == 0:
            logger.info(f"starting index: {time_index}")  # TODO print on rank zero

        if time_index == warmup_steps:
            start.record()

        # continue
        image_lr = (
            image_lr.to(device=device)
            .to(torch.float32)
            .to(memory_format=torch.channels_last)
        )
        image_tar = image_tar.to(device=device).to(torch.float32)
        image_out = generate_fn(image_lr)

        batch_size = image_out.shape[0]

        # for validation - make 3x450x450 to an ordered sequence of 50x50 patches
        # input; 1x3x450x450 --> (1*9*9)x3x50x50

        if dist.rank == 0:
            # write out data in a seperate thread so we don't hold up inferencing
            writer_threads.append(
                writer_executor.submit(
                    save_images,
                    writer,
                    dataset,
                    list(times),
                    image_out.cpu(),
                    image_tar.cpu(),
                    image_lr.cpu(),
                    time_index,
                    index[0],
                )
            )
    end.record()
    end.synchronize()
    elapsed_time = start.elapsed_time(end) / 1000.0  # Convert ms to s
    timed_steps = time_index + 1 - warmup_steps
    average_time_per_batch_element = elapsed_time / timed_steps / batch_size
    if dist.rank == 0:
        logger.info(
            f"Total time to run {timed_steps} and {batch_size} ensembles = {elapsed_time} s"
        )
        logger.info(
            f"Average time per batch element = {average_time_per_batch_element} s"
        )

    # make sure all the workers are done writing
    for thread in list(writer_threads):
        thread.result()
        writer_threads.remove(thread)
    writer_executor.shutdown()


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

    img_lr = img_lr.to(memory_format=torch.channels_last)

    # Loop over batches.
    all_images = []
    for batch_seeds in tqdm.tqdm(rank_batches, unit="batch", disable=(dist.rank != 0)):
        with nvtx.annotate(f"generate {len(all_images)}", color="rapids"):
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
            ).to(memory_format=torch.channels_last)

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
                latents = torch.zeros_like(latents, memory_format=torch.channels_last)
                sampler_fn = unet_regression
            else:
                raise ValueError(
                    f"Unknown pretext {pretext}. Should be either 'gen' or 'reg'."
                )

            with torch.inference_mode():
                images = sampler_fn(
                    net,
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
    num_steps=2,
    sigma_min=0.0,
    sigma_max=0.0,
    rho=7,
    **kwargs,
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

    # Run regression on just a single batch element and then repeat
    x_next = net(x_hat[0:1], x_lr, t_hat, class_labels).to(torch.float64)
    if x_hat.shape[0] > 1:
        x_next = x_next.repeat([d if i == 0 else 1 for i, d in enumerate(x_hat.shape)])

    return x_next


def save_images(
    writer,
    dataset: DownscalingDataset,
    times,
    image_out,
    image_tar,
    image_lr,
    time_index,
    t_index,
):
    """
    Saves inferencing result along with the baseline

    Parameters
    ----------

    writer (NetCDFWriter): Where the data is being written
    in_channels (List): List of the input channels being used
    input_channel_info (Dict): Description of the input channels
    out_channels (List): List of the output channels being used
    output_channel_info (Dict): Description of the output channels
    input_norm (Tuple): Normalization data for input
    target_norm (Tuple): Normalization data for the target
    image_out (torch.Tensor): Generated output data
    image_tar (torch.Tensor): Ground truth data
    image_lr (torch.Tensor): Low resolution input data
    time_index (int): Epoch number
    t_index (int): index where times are located
    """
    # weather sub-plot
    image_lr2 = image_lr[0].unsqueeze(0)
    image_lr2 = image_lr2.cpu().numpy()
    image_lr2 = dataset.denormalize_input(image_lr2)

    image_tar2 = image_tar[0].unsqueeze(0)
    image_tar2 = image_tar2.cpu().numpy()
    image_tar2 = dataset.denormalize_output(image_tar2)

    # some runtime assertions
    if image_tar2.ndim != 4:
        raise ValueError("image_tar2 must be 4-dimensional")

    for idx in range(image_out.shape[0]):
        image_out2 = image_out[idx].unsqueeze(0)
        if image_out2.ndim != 4:
            raise ValueError("image_out2 must be 4-dimensional")

        # Denormalize the input and outputs
        image_out2 = image_out2.cpu().numpy()
        image_out2 = dataset.denormalize_output(image_out2)

        time = times[t_index]
        writer.write_time(time_index, time)
        for channel_idx in range(image_out2.shape[1]):
            info = dataset.output_channels()[channel_idx]
            channel_name = _get_name(info)
            truth = image_tar2[0, channel_idx]

            writer.write_truth(channel_name, time_index, truth)
            writer.write_prediction(
                channel_name, time_index, idx, image_out2[0, channel_idx]
            )

        input_channel_info = dataset.input_channels()
        for channel_idx in range(len(input_channel_info)):
            info = input_channel_info[channel_idx]
            channel_name = _get_name(info)
            writer.write_input(channel_name, time_index, image_lr2[0, channel_idx])


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
        f.createDimension("x", nx)
        f.createDimension("y", ny)

        v = f.createVariable("lat", "f", dimensions=("y", "x"))
        v[:] = lat
        v.standard_name = "latitude"
        v.units = "degrees_north"

        v = f.createVariable("lon", "f", dimensions=("y", "x"))
        v[:] = lon
        v.standard_name = "longitude"
        v.units = "degrees_east"

        # create time dimension
        v = f.createVariable("time", "i8", ("time"))
        v.calendar = "standard"
        v.units = "hours since 1990-01-01 00:00:00"

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
