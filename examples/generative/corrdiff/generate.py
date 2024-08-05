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
from omegaconf import OmegaConf, DictConfig
import torch
import torch._dynamo
import numpy as np
import netCDF4 as nc
from modulus.distributed import DistributedManager
from modulus.launch.logging import PythonLogger, RankZeroLoggingWrapper
from modulus import Module
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from modulus.utils.generative import ablation_sampler

from generate_helpers import get_dataset_and_sampler, get_time_from_range,writer_from_input_dataset, save_images, generate_fn
from training_helpers import set_patch_shape


@hydra.main(version_base="1.2", config_path="conf", config_name="config_generate")
def main(cfg: DictConfig) -> None:
    """Generate random images using the techniques described in the paper
    "Elucidating the Design Space of Diffusion-Based Generative Models".
    """

    # Initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device

    # Initialize logger
    logger = PythonLogger("generate")  # General python logger
    logger0 = RankZeroLoggingWrapper(logger, dist)
    logger.file_logging("generate.log")

    # Handle the batch size
    seeds = list(np.arrange(cfg.sampler.seeds))
    num_batches = (
        (len(seeds) - 1) // (cfg.generation.seed_batch_size * dist.world_size) + 1
    ) * dist.world_size
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.rank :: dist.world_size]

    # Synchronize
    if dist.world_size > 1:
        torch.distributed.barrier()

    # Parse the inference input times
    if cfg.generate.times_range and times:
        raise ValueError("Either times_range or times must be provided, but not both")
    if cfg.generation.times_range:
        times = get_time_from_range(cfg.generation.times_range)
    else:
        times = cfg.generation.times

    # Create dataset object
    dataset_cfg = OmegaConf.to_container(cfg.dataset)
    dataset, sampler = get_dataset_and_sampler(dataset_cfg=dataset_cfg, times=times)
    img_shape = dataset.image_shape()
    use_mean_hr = False

    # Parse the patch shape
    if hasattr(cfg, "training.hp.patch_shape_x"):  # TODO better config handling
        patch_shape_x = cfg.training.hp.patch_shape_x
    else:
        patch_shape_x = None
    if hasattr(cfg, "training.hp.patch_shape_y"):
        patch_shape_y = cfg.training.hp.patch_shape_y
    else:
        patch_shape_y = None
    patch_shape = (patch_shape_y, patch_shape_x)
    patch_shape, img_shape = set_patch_shape(img_shape, patch_shape)
    if patch_shape != img_shape:
        logger0.info("Patch-based training enabled")
    else:
        logger0.info("Patch-based training disabled")

    # Parse the inference mode
    if cfg.generation.inference_mode == "regression":
        load_net_reg, load_net_res = True, False
    elif cfg.generation.inference_mode == "diffusion":
        load_net_reg, load_net_res = False, True
    elif cfg.generation.inference_mode == "regression_and_diffusion":
        load_net_reg, load_net_res = True, True
    else:
        raise ValueError(f"Invalid inference mode {cfg.generation.inference_mode}")
    
    # Load diffusion network, move to device, change precision
    if load_net_res:
        res_ckpt_filename = cfg.generation.res_ckpt_filename
        logger0.info(f'Loading residual network from "{res_ckpt_filename}"...')
        net_res = Module.from_checkpoint(res_ckpt_filename)
        net_res = net_res.eval().to(device).to(memory_format=torch.channels_last)
        if cfg.generation.force_fp16:
            net_res.use_fp16 = True
    else:
        net_res = None

    # load regression network, move to device, change precision
    if load_net_reg:
        reg_ckpt_filename = cfg.generation.reg_ckpt_filename
        logger0.info(f'Loading network from "{reg_ckpt_filename}"...')
        net_reg = Module.from_checkpoint(reg_ckpt_filename)
        net_reg = net_reg.eval().to(device).to(memory_format=torch.channels_last)
        if cfg.generation.force_fp16:
            net_reg.use_fp16 = True
    else:
        net_reg = None

    # Reset since we are using a different mode.
    if cfg.generation.use_torch_compile:
        torch._dynamo.reset()
        # Only compile residual network
        # Overhead of compiling regression network outweights any benefits
        if net_res:
            net_res = torch.compile(net_res, mode="reduce-overhead")
    
    # Partially instantiate the sampler based on the configs
    if hr_mean_conditioning and sampling_method == "stochastic":
    if cfg.sampler.type == "deterministic":
        sampler_fn = partial(ablation_sampler,
                             num_steps=cfg.sampler.num_steps,
                             num_ensembles=cfg.sampler.num_ensembles,
                             solver=cfg.sampler.solver
                             )
    elif cfg.sampler.type == "stochastic":
        try:
            from edmss import edm_sampler
        except ImportError:
            raise ImportError(
                "Please get the edm_sampler by running: pip install git+https://github.com/mnabian/edmss.git"
            )
        if cfg.generation.hr_mean_conditioning:
            use_mean_hr = True
        sampler_fn = partial(edm_sampler,
                             img_shape=img_shape[1],
                             patch_shape=patch_shape[1],
                             boundary_pix=cfg.sampler.boundary_pix,
                             overlap_pix=cfg.sampler.overlap_pix,
        )
    else:
        raise ValueError(f"Unknown sampling method {cfg.sampling.type}")

    # Initialize threadpool for writers
    writer = writer_from_input_dataset(f, dataset)  # TODO what is f?
    writer_executor = ThreadPoolExecutor(max_workers=cfg.generation.perf.num_writer_workers)
    writer_threads = []

    # generate images
    logger0.info("Generating images...")
    warmup_steps = 2
    batch_size = 1
    time_index = -1
    data_loader = torch.utils.data.DataLoader(
                    dataset=dataset, sampler=sampler, batch_size=batch_size, pin_memory=True
                )

    # Generates model predictions from the input data using the specified
    # `generate_fn`, and save the predictions to the provided NetCDF file. It iterates
    # through the dataset using a data loader, computes predictions, and saves them along
    # with associated metadata.
    with nc.Dataset(f"output_{dist.rank}.nc", "w") as f:
        # add attributes
        f.cfg = str(cfg)
        with torch.cuda.profiler.profile():
            with torch.autograd.profiler.emit_nvtx():            
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                times = dataset.time()
                for image_tar, image_lr, index in iter(data_loader):
                    time_index += 1
                    if dist.rank == 0:
                        logger0.info(f"starting index: {time_index}")

                    if time_index == warmup_steps:
                        start.record()

                    # continue
                    image_lr = (
                        image_lr.to(device=device)
                        .to(torch.float32)
                        .to(memory_format=torch.channels_last)
                    )
                    image_tar = image_tar.to(device=device).to(torch.float32)
                    image_out = generate_fn(sampler_fn, image_lr, cfg.generation.sample_res, img_shape, patch_size, seeds, net_reg, net_res, cfg.generation.seed_batch_size, rank_batches, cfg.generationinference_mode, use_mean_hr, dist.rank, dist.world_size, device)

                    if dist.rank == 0:
                        batch_size = image_out.shape[0]
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
                if dist.rank == 0:
                    average_time_per_batch_element = elapsed_time / timed_steps / batch_size
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

    logger0.info("Generation Completed.")

if __name__ == "__main__":
    main()
