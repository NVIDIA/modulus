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

"""Main training loop."""

import os
import time
import copy
import json
from utils.YParams import YParams
import psutil
import numpy as np
import torch
from modulus.distributed import DistributedManager
from utils import training_stats
from utils import misc
from utils.misc import print0
from utils.diffusions.generate import edm_sampler
from utils.diffusions.networks import get_preconditioned_architecture, EasyRegressionV2
from utils.diffusions.losses import EDMLoss, RegressionLossV2
from utils.data_loader_hrrr_era5 import get_dataset, worker_init
import matplotlib.pyplot as plt
import wandb
from utils.spectrum import compute_ps1d
from torch.nn.utils import clip_grad_norm_

# ----------------------------------------------------------------------------


def get_pretrained_regression_net(
    checkpoint_path, config_file, regression_config, target_channels, device
):
    """
    Load a pretrained regression network as specified by a given config
    """

    hyperparams = YParams(config_file, regression_config)
    resolution = hyperparams.hrrr_img_size[0]

    conditional_channels = (
        target_channels + len(hyperparams.invariants) + hyperparams.n_era5_channels
    )

    net = get_preconditioned_architecture(
        name="regression",
        resolution=resolution,
        target_channels=target_channels,
        conditional_channels=conditional_channels,
        label_dim=0,
        spatial_embedding=hyperparams.spatial_pos_embed,
        attn_resolutions=hyperparams.attn_resolutions,
    )

    chkpt = torch.load(checkpoint_path, weights_only=True)
    net.load_state_dict(chkpt["net"], strict=True)
    net = EasyRegressionV2(net)

    return net.to(device)


def training_loop(
    run_dir=".",  # Output directory.
    optimizer_kwargs={},  # Options for optimizer.
    seed=0,  # Global random seed.
    lr_rampup_kimg=2000,  # Learning rate ramp-up duration.
    state_dump_ticks=50,  # How often to dump training state, None = disable.
    resume_state_dump=None,  # Start from the given training state, None = reset training state.
    resume_kimg=0,  # Start from the given training progress.
    cudnn_benchmark=True,  # Enable torch.backends.cudnn.benchmark?
    device=torch.device("cuda"),
    config_file=None,
    config_name=None,
    log_to_wandb=False,
):
    dist = DistributedManager()
    params = YParams(config_file, config_name)
    batch_size = params.batch_size
    local_batch_size = batch_size // dist.world_size
    optimizer_kwargs["lr"] = params.lr
    img_per_tick = params.img_per_tick
    use_regression_net = params.use_regression_net
    previous_step_conditioning = params.previous_step_conditioning
    loss_type = params.loss
    if loss_type == "regression_v2":
        train_regression_unet = True
        net_name = "regression"
        print0("Using regression_v2")
    elif loss_type == "edm":
        train_regression_unet = False
        net_name = "diffusion"

    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.world_size + dist.rank) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    total_kimg = params.total_kimg

    if resume_state_dump is not None:
        print0("Resuming from state dump:", resume_state_dump)
        print0("Resuming from kimg:", resume_kimg)

    # Load dataset.
    print0("Loading dataset...")

    dataset_train = get_dataset(params, train=True)
    dataset_valid = get_dataset(params, train=False)

    _, hrrr_channels = dataset_train._get_hrrr_channel_names()
    diffusion_channels = (
        hrrr_channels
        if params.diffusion_channels == "all"
        else params.diffusion_channels
    )
    input_channels = (
        hrrr_channels if params.input_channels == "all" else params.input_channels
    )
    input_channel_indices = [hrrr_channels.index(channel) for channel in input_channels]
    diffusion_channel_indices = [
        hrrr_channels.index(channel) for channel in diffusion_channels
    ]

    sampler = misc.InfiniteSampler(
        dataset=dataset_train,
        rank=dist.rank,
        num_replicas=dist.world_size,
        seed=seed,
    )
    valid_sampler = misc.InfiniteSampler(
        dataset=dataset_valid,
        rank=dist.rank,
        num_replicas=dist.world_size,
        seed=seed,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=local_batch_size,
        num_workers=params.num_data_workers,
        sampler=sampler,
        worker_init_fn=worker_init,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )

    valid_data_loader = torch.utils.data.DataLoader(
        dataset=dataset_valid,
        batch_size=local_batch_size,
        num_workers=params.num_data_workers,
        sampler=valid_sampler,
        drop_last=True,
        pin_memory=torch.cuda.is_available(),
    )

    dataset_iterator = iter(data_loader)
    valid_dataset_iterator = iter(valid_data_loader)

    # load pretrained regression net if training diffusion
    if use_regression_net:
        regression_net = get_pretrained_regression_net(
            checkpoint_path=params.regression_weights,
            config_file=config_file,
            regression_config=params.regression_config,
            target_channels=len(diffusion_channels),
            device=device,
        )

    # Construct network
    print0("Constructing network...")
    resolution = 512
    target_channels = len(diffusion_channels)
    if train_regression_unet:
        conditional_channels = (
            len(input_channels) + 26
        )  # 26 is the number of era5 channels
    else:
        conditional_channels = (
            len(input_channels)
            if not previous_step_conditioning
            else 2 * len(input_channels)
        )

    conditional_channels += len(params.invariants)
    invariant_array = dataset_train._get_invariants()
    invariant_tensor = torch.from_numpy(invariant_array).to(device)

    if not train_regression_unet:
        regression_net.set_invariant(invariant_tensor)

    print0("hrrr_channels", hrrr_channels)
    print0("target_channels for diffusion", target_channels)
    print0("conditional_channels for diffusion", conditional_channels)

    net = get_preconditioned_architecture(
        name=net_name,
        resolution=resolution,
        target_channels=target_channels,
        conditional_channels=conditional_channels,
        label_dim=0,
        spatial_embedding=params.spatial_pos_embed,
        attn_resolutions=params.attn_resolutions,
    )

    if not params.loss in ["regression", "regression_v2"]:
        assert net.sigma_min < net.sigma_max
    net.train().requires_grad_(True).to(device)

    # Setup optimizer.
    print0("Setting up optimizer...")
    if params.loss == "regression_v2":
        loss_fn = RegressionLossV2()
    elif params.loss == "edm":
        loss_fn = EDMLoss(P_mean=params.P_mean)
    optimizer = torch.optim.Adam(net.parameters(), **optimizer_kwargs)
    augment_pipe = None
    ddp = torch.nn.parallel.DistributedDataParallel(
        net, device_ids=[device], broadcast_buffers=False
    )

    total_steps = 0

    # Resume training from previous snapshot.
    if resume_state_dump:
        print0(f'Loading training state from "{resume_state_dump}"...')
        data = torch.load(
            resume_state_dump, map_location=torch.device("cpu"), weights_only=True
        )
        net.load_state_dict(data["net"])
        total_steps = data["total_steps"]
        optimizer.load_state_dict(data["optimizer_state"])
        del data  # conserve memory

    # Train.
    print0(f"Training for {total_kimg} kimg...")
    print0()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    stats_jsonl = None
    wandb_logs = {}

    while True:
        # Accumulate gradients.
        optimizer.zero_grad(set_to_none=True)
        batch = next(dataset_iterator)
        hrrr_0 = batch["hrrr"][0].to(device).to(torch.float32)
        hrrr_1 = batch["hrrr"][1].to(device).to(torch.float32)

        if use_regression_net:
            era5 = batch["era5"][0].to(device).to(torch.float32)

            with torch.no_grad():
                reg_out = regression_net(hrrr_0, era5, mask=None)
                hrrr_0 = torch.cat(
                    (
                        hrrr_0[:, input_channel_indices, :, :],
                        reg_out[:, input_channel_indices, :, :],
                    ),
                    dim=1,
                )
                hrrr_1 = hrrr_1 - reg_out
                del reg_out

        elif train_regression_unet:
            assert diffusion_channel_indices == input_channel_indices

            era5 = batch["era5"][0].to(device).to(torch.float32)

            hrrr_0 = torch.cat((hrrr_0[:, input_channel_indices, :, :], era5), dim=1)

        hrrr_1 = hrrr_1[
            :, diffusion_channel_indices, :, :
        ]  # targets of the diffusion model

        invariant_tensor_ = invariant_tensor.unsqueeze(0)
        invariant_tensor_ = invariant_tensor.repeat(hrrr_0.shape[0], 1, 1, 1)
        hrrr_0 = torch.cat((hrrr_0, invariant_tensor_), dim=1)

        loss = loss_fn(net=ddp, x=hrrr_1, condition=hrrr_0, augment_pipe=augment_pipe)
        channelwise_loss = loss.mean(dim=(0, 2, 3))
        channelwise_loss_dict = {
            f"ChLoss/{diffusion_channels[i]}": channelwise_loss[i].item()
            for i in range(target_channels)
        }
        training_stats.report("Loss/loss", loss.mean())
        loss_value = loss.sum() / target_channels
        if log_to_wandb:
            wandb_logs["channelwise_loss"] = channelwise_loss_dict

        loss_value.backward()

        if params.clip_grad_norm is not None:
            clip_grad_norm_(net.parameters(), params.clip_grad_norm)

        # Update weights.
        for g in optimizer.param_groups:
            g["lr"] = optimizer_kwargs["lr"] * min(
                cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1
            )
            if log_to_wandb:
                wandb_logs["lr"] = g["lr"]
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(
                    param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad
                )

        optimizer.step()

        # Perform maintenance tasks once per tick.
        effective_batch_size = (batch_size // local_batch_size) * hrrr_0.shape[0]
        total_steps += 1
        cur_nimg += effective_batch_size
        # done = (cur_nimg >= total_kimg * 1000)
        done = cur_nimg >= 10  # TODO remove this line (testing only)

        if (
            (not done)
            and (cur_tick != 0)
            and (cur_nimg < tick_start_nimg + img_per_tick)
        ):
            continue

        # make inference
        if cur_tick % params.validate_every == 0:
            batch = next(valid_dataset_iterator)

            with torch.no_grad():
                # n = 1
                hrrr_0, hrrr_1 = batch["hrrr"]
                hrrr_0 = hrrr_0.to(torch.float32).to(device)
                hrrr_1 = hrrr_1.to(torch.float32).to(device)

                invariant_tensor_ = invariant_tensor.unsqueeze(0)
                invariant_tensor_ = invariant_tensor.repeat(hrrr_0.shape[0], 1, 1, 1)

                if use_regression_net:
                    with torch.no_grad():
                        era5 = batch["era5"][0].to(device).to(torch.float32)
                        reg_out = regression_net(hrrr_0, era5, mask=None)
                        hrrr_0 = torch.cat(
                            (
                                hrrr_0[:, input_channel_indices, :, :],
                                reg_out[:, input_channel_indices, :, :],
                            ),
                            dim=1,
                        )
                        latents = torch.randn_like(
                            hrrr_1[:, diffusion_channel_indices, :, :]
                        )
                        loss_target = hrrr_1 - reg_out
                        output_images = edm_sampler(
                            net,
                            latents=latents,
                            condition=torch.cat((hrrr_0, invariant_tensor_), dim=1),
                        )
                        valid_loss = loss_fn(
                            net=ddp,
                            x=loss_target[:, diffusion_channel_indices],
                            condition=torch.cat((hrrr_0, invariant_tensor_), dim=1),
                            augment_pipe=augment_pipe,
                        )
                        output_images += reg_out[:, diffusion_channel_indices, :, :]
                        del reg_out

                elif train_regression_unet:
                    assert (
                        use_regression_net == False
                    ), "use_regression_net must be False when training regression unet"
                    assert (
                        input_channel_indices == diffusion_channel_indices
                    ), "input_channel_indices must be equal to diffusion_channel_indices when training regression unet"
                    condition = torch.cat(
                        (
                            hrrr_0[:, input_channel_indices, :, :],
                            era5[:],
                            invariant_tensor_,
                        ),
                        dim=1,
                    )
                    latents = torch.zeros_like(
                        hrrr_1[:, diffusion_channel_indices, :, :],
                        device=hrrr_1.device,
                    )
                    rnd_normal = torch.randn(
                        [latents.shape[0], 1, 1, 1], device=latents.device
                    )
                    sigma = (
                        rnd_normal * 1.2 - 1.2
                    ).exp()  # this isn't used by the code
                    output_images = net(sigma=sigma, condition=condition)
                    valid_loss = loss_fn(
                        net=ddp,
                        x=hrrr_1[:, diffusion_channel_indices, :, :],
                        condition=condition,
                        augment_pipe=augment_pipe,
                    )
                    channelwise_valid_loss = valid_loss.mean(dim=[0, 2, 3])
                    channelwise_valid_loss_dict = {
                        f"ChLoss_valid/{diffusion_channels[i]}": channelwise_valid_loss[
                            i
                        ].item()
                        for i in range(target_channels)
                    }
                    if log_to_wandb:
                        wandb_logs[
                            "channelwise_valid_loss"
                        ] = channelwise_valid_loss_dict

                hrrr_1 = hrrr_1[:, diffusion_channel_indices, :, :]

                training_stats.report("Loss/valid_loss", valid_loss.mean())

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [
            f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"
        ]
        fields += [
            f"time {misc.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"
        ]
        fields += [
            f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"
        ]
        fields += [
            f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"
        ]
        fields += [
            f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"
        ]
        fields += [
            f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"
        ]
        fields += [
            f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"
        ]
        fields += [
            f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"
        ]
        torch.cuda.reset_peak_memory_stats()
        print0(" ".join(fields))

        if cur_tick % params.validate_every == 0:
            if log_to_wandb:
                if dist.rank == 0:
                    print0("logging to wandb")
                    wandb.log(wandb_logs, step=cur_nimg)

            if dist.rank == 0:
                # TODO: improve the image saving and run_dir setup for thread safe image saving from all ranks

                for i in range(output_images.shape[0]):
                    image = output_images[i].cpu().numpy()
                    # hrrr_channels = dataset_train.hrrr_channels
                    fields = ["u10m", "v10m", "t2m", "refc", "q1", "q5", "q10"]

                    # Compute spectral metrics
                    figs, spec_ratios = compute_ps1d(
                        output_images[i], hrrr_1[i], fields, diffusion_channels
                    )
                    if log_to_wandb:
                        wandb.log(spec_ratios, step=cur_nimg)
                        for figname, fig in figs.items():
                            wandb.log({figname: wandb.Image(fig)}, step=cur_nimg)

                    for f_ in fields:
                        f_index = diffusion_channels.index(f_)
                        image_dir = os.path.join(run_dir, "images", f_)
                        generated = image[f_index]
                        truth = hrrr_1[i, f_index].cpu().numpy()

                        fig, (a, b) = plt.subplots(1, 2)
                        im = a.imshow(generated)
                        a.set_title("generated, {}.png".format(f_))
                        plt.colorbar(im, fraction=0.046, pad=0.04)
                        im = b.imshow(truth)
                        b.set_title("truth")
                        plt.colorbar(im, fraction=0.046, pad=0.04)
                        os.makedirs(image_dir, exist_ok=True)
                        plt.savefig(os.path.join(image_dir, f"{cur_tick}_{i}_{f_}.png"))
                        plt.close("all")

                        specfig = "PS1D_" + f_
                        figs[specfig].savefig(
                            os.path.join(image_dir, f"{cur_tick}{i}{f_}_spec.png")
                        )
                        plt.close(figs[specfig])

                        # log the images to wandb
                        if log_to_wandb:
                            # log fig to wandb
                            wandb.log({f"generated_{f_}": fig}, step=cur_nimg)

        # Save full dump of the training state.
        if (
            (state_dump_ticks is not None)
            and (done or cur_tick % state_dump_ticks == 0)
            and cur_tick != 0
            and dist.rank == 0
        ):
            torch.save(
                dict(
                    net=net.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    total_steps=total_steps,
                ),
                os.path.join(run_dir, f"training-state-{cur_nimg//1000:06d}.pt"),
            )

        # Update logs.
        training_stats.default_collector.update()
        if dist.rank == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, "stats.jsonl"), "at")

            stats_dict = dict(
                training_stats.default_collector.as_dict(), timestamp=time.time()
            )
            if True:
                wandb_logs["loss"] = stats_dict["Loss/loss"]["mean"]
                wandb_logs["valid_loss"] = stats_dict["Loss/valid_loss"]["mean"]
                print0("loss: ", wandb_logs["loss"])
                print0("valid_loss: ", wandb_logs["valid_loss"])
            stats_jsonl.write(json.dumps(stats_dict) + "\n")
            stats_jsonl.flush()

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    torch.distributed.barrier()
    print0()
    print0("Exiting...")
