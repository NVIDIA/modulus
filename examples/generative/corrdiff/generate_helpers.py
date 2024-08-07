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

import cftime
import datetime
import torch
import tqdm
import nvtx
import inspect

from modulus.utils.generative import StackedRandomGenerator
from einops import rearrange
from torch.distributed import gather


from datasets.base import DownscalingDataset
from datasets.dataset import init_dataset_from_config
from modulus.utils.generative import convert_datetime_to_cftime, time_range


############################################################################
#                           GENERATION HELPERS                             #
############################################################################

def _generate(
    net,
    sampler_fn,
    seed_batch_size,
    img_shape,
    img_out_channels,
    rank_batches,
    img_lr,
    rank,
    device,
    mean_hr=None,
):
    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".
    """

    img_lr = img_lr.to(memory_format=torch.channels_last)
    if sampler_fn:  # TODO better handling
        sig = inspect.signature(sampler_fn)
        if "mean_hr" in sig.parameters:
            additional_args = {"mean_hr": mean_hr}
        else:
            additional_args = {}
    else:
        additional_args = {}
    # Loop over batches.
    all_images = []
    for batch_seeds in tqdm.tqdm(rank_batches, unit="batch", disable=(rank != 0)):
        with nvtx.annotate(f"generate {len(all_images)}", color="rapids"):
            batch_size = len(batch_seeds)
            if batch_size == 0:
                continue

            # Pick latents and labels.
            rnd = StackedRandomGenerator(device, batch_seeds)

            if sampler_fn:
                latents = rnd.randn(
                    [
                        seed_batch_size,
                        img_out_channels,
                        img_shape[1],
                        img_shape[0],
                    ],
                    device=device,
                ).to(memory_format=torch.channels_last)
                with torch.inference_mode():
                    images = sampler_fn(
                        net, latents, img_lr, randn_like=rnd.randn_like, **additional_args
                    )
            else:
                latents = torch.zeros((1, img_out_channels, img_shape[1], img_shape[0])).to(device, memory_format=torch.channels_last)
                with torch.inference_mode():
                    images = unet_regression(
                        net, latents, img_lr
                    )
            all_images.append(images)
    return torch.cat(all_images)


def generate_fn(
    sampler_fn,
    image_lr,
    sample_res,
    img_shape,
    img_out_channels,
    patch_shape,
    seeds,
    net_reg,
    net_res,
    seed_batch_size,
    rank_batches,
    inference_mode,
    use_mean_hr,
    rank,
    world_size,
    device,
):
    """Function to generate an image

    Args:
        image_lr: low resolution input. shape: (b, c, h, w)

    Return
        image_hr: high resolution output: shape (b, c, h, w)
    """
    img_shape_y, img_shape_x = img_shape
    with nvtx.annotate("generate_fn", color="green"):
        if sample_res == "full":
            image_lr_patch = image_lr
        else:
            torch.cuda.nvtx.range_push("rearrange")
            image_lr_patch = rearrange(
                image_lr,
                "b c (h1 h) (w1 w) -> (b h1 w1) c h w",
                h1=img_shape_y // patch_shape[0],
                w1=img_shape_x // patch_shape[1],
            )
            torch.cuda.nvtx.range_pop()
        image_lr_patch = image_lr_patch.to(memory_format=torch.channels_last)

        sample_seeds = seeds

        if net_reg:
            with nvtx.annotate("regression_model", color="yellow"):
                image_reg = _generate(
                    net=net_reg,
                    sampler_fn=None,
                    seed_batch_size=1,
                    img_shape=img_shape,
                    img_out_channels=img_out_channels,
                    rank_batches=[torch.tensor([rank])],
                    img_lr=image_lr_patch,
                    rank=rank,
                    device=device,
                )
        if net_res:
            if use_mean_hr:
                mean_hr = image_reg[0:1]
            else:
                mean_hr = None
            with nvtx.annotate("diffusion model", color="purple"):
                image_res = _generate(
                    net=net_res,
                    sampler_fn=sampler_fn,
                    seed_batch_size=seed_batch_size,
                    img_shape=img_shape,
                    img_out_channels=img_out_channels,
                    rank_batches=rank_batches,
                    img_lr=image_lr_patch.expand(seed_batch_size, -1, -1, -1).to(
                        memory_format=torch.channels_last
                    ),
                    rank=rank,
                    device=device,
                    mean_hr=mean_hr,
                )
        if inference_mode == "regression":
            image_out = image_reg
        elif inference_mode == "diffusion":
            image_out = image_res
        else:
            image_out = image_reg + image_res

        if sample_res != "full":
            image_out = rearrange(
                image_out,
                "(b h1 w1) c h w -> b c (h1 h) (w1 w)",
                h1=img_shape_y // patch_size,
                w1=img_shape_x // patch_size,
            )

        # Gather tensors on rank 0
        if world_size > 1:
            if rank == 0:
                gathered_tensors = [
                    torch.zeros_like(
                        image_out, dtype=image_out.dtype, device=image_out.device
                    )
                    for _ in range(world_size)
                ]
            else:
                gathered_tensors = None

            torch.distributed.barrier()
            gather(
                image_out,
                gather_list=gathered_tensors if rank == 0 else None,
                dst=0,
            )

            if rank == 0:
                return torch.cat(gathered_tensors)
            else:
                return None
        else:
            return image_out


def unet_regression(
    net,
    latents,
    img_lr,
    class_labels=None,
):
    """
    Perform U-Net regression with temporal sampling.

    Parameters:
        net (torch.nn.Module): U-Net model for regression.
        latents (torch.Tensor): Latent representation.
        img_lr (torch.Tensor): Low-resolution input image.
        class_labels (torch.Tensor, optional): Class labels for conditional generation.

    Returns:
        torch.Tensor: Predicted output at the next time step.
    """


    x_hat = torch.zeros_like(latents).to(torch.float64)
    t_hat = torch.tensor(1.0).to(torch.float64).cuda()

    # Run regression on just a single batch element and then repeat
    x_next = net(x_hat[0:1], img_lr, t_hat, class_labels).to(torch.float64)
    if x_hat.shape[0] > 1:
        x_next = x_next.repeat([d if i == 0 else 1 for i, d in enumerate(x_hat.shape)])
    return x_next


############################################################################
#                              DATA HELPERS                                #
############################################################################


def get_dataset_and_sampler(dataset_cfg, times):
    """
    Get a dataset and sampler for generation.
    """
    (dataset, _) = init_dataset_from_config(dataset_cfg, batch_size=1)
    plot_times = [
        convert_datetime_to_cftime(
            datetime.datetime.strptime(time, "%Y-%m-%dT%H:%M:%S")
        )
        for time in times
    ]
    all_times = dataset.time()
    time_indices = [all_times.index(t) for t in plot_times]
    sampler = time_indices

    return dataset, sampler


def get_time_from_range(times_range, time_format="%Y-%m-%dT%H:%M:%S"):
    """Generates a list of times within a given range.

    Args:
        times_range: A list containing start time, end time, and optional interval (hours).
        time_format: The format of the input times (default: "%Y-%m-%dT%H:%M:%S").

    Returns:
        A list of times within the specified range.
    """

    start_time = datetime.datetime.strptime(times_range[0], time_format)
    end_time = datetime.datetime.strptime(times_range[1], time_format)
    interval = (
        datetime.timedelta(hours=times_range[2])
        if len(times_range) > 2
        else datetime.timedelta(hours=1)
    )

    times = [
        t.strftime(time_format)
        for t in time_range(start_time, end_time, interval, inclusive=True)
    ]
    return times


def _get_name(channel_info):
    return channel_info.name + channel_info.level


############################################################################
#                              WRITER HELPERS                              #
############################################################################


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
            if channel_idx == image_lr2.shape[1] - 1:
                break
