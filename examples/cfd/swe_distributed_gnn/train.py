# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2022 The torch-harmonics Authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

import argparse
import json
import os
import time
import uuid
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torch.cuda import amp

import numpy as np
import matplotlib.pyplot as plt

from pde_dataset import PdeDataset

from modulus.distributed import DistributedManager, mark_module_as_shared
from modulus.models.graphcast.graph_cast_net import GraphCastNet


def l2loss_sphere(prd, tar, solver, relative=False, squared=True):
    loss = solver.integrate_grid((prd - tar) ** 2, dimensionless=True).sum(dim=-1)
    if relative:
        loss = loss / solver.integrate_grid(tar**2, dimensionless=True).sum(dim=-1)

    if not squared:
        loss = torch.sqrt(loss)
    loss = loss.mean()

    return loss


def l2loss(prd, tar, *args, **kwargs):
    loss = ((prd - tar) ** 2).mean()
    return loss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# rolls out the model and compares to the classical solver
def autoregressive_inference(
    model,
    dataset,
    path_root,
    nsteps,
    dist_manager,
    loss_fn: str = "l2",
    autoreg_steps=10,
    nskip=1,
    plot_channel=1,
    nics=20,
):
    if loss_fn == "l2":
        loss_fn = l2loss
    elif loss_fn == "l2sphere":
        loss_fn = l2loss_sphere
    else:
        raise ValueError(
            f"loss_fn={loss_fn} not supported, expected 'l2' or 'l2sphere'."
        )

    losses = np.zeros(nics)
    fno_times = np.zeros(nics)
    nwp_times = np.zeros(nics)

    for iic in range(nics):
        if dist_manager.rank == 0:
            ic = dataset.solver.random_initial_condition(mach=0.1)
            inp_mean = dataset.inp_mean
            inp_var = dataset.inp_var

            prd = (dataset.solver.spec2grid(ic) - inp_mean) / torch.sqrt(inp_var)
            prd = prd.unsqueeze(0)
            uspec = ic.clone()

        else:
            prd = torch.empty((1, 3, dataset.nlat, dataset.nlon), device=model.device)

        # ML model
        start_time = time.time()
        for i in range(1, autoreg_steps + 1):
            # evaluate the ML model
            prd = model(prd)

            if (
                (dist_manager.rank == 0)
                and (iic == nics - 1)
                and (nskip > 0)
                and (i % nskip == 0)
            ):

                # do plotting
                fig = plt.figure(figsize=(7.5, 6))
                dataset.solver.plot_griddata(prd[0, plot_channel], fig, vmax=4, vmin=-4)
                plt.savefig(os.path.join(path_root, f"pred_{i//nskip}.png"))
                plt.clf()

        fno_times[iic] = time.time() - start_time

        # classical model, not parallel so only on rank 0
        if dist_manager.rank == 0:
            start_time = time.time()
            for i in range(1, autoreg_steps + 1):

                # advance classical model
                uspec = dataset.solver.timestep(uspec, nsteps)

                if iic == nics - 1 and i % nskip == 0 and nskip > 0:
                    ref = (dataset.solver.spec2grid(uspec) - inp_mean) / torch.sqrt(
                        inp_var
                    )

                    fig = plt.figure(figsize=(7.5, 6))
                    dataset.solver.plot_griddata(
                        ref[plot_channel], fig, vmax=4, vmin=-4
                    )
                    plt.savefig(os.path.join(path_root, f"truth_{i//nskip}.png"))
                    plt.clf()

            nwp_times[iic] = time.time() - start_time

            ref = dataset.solver.spec2grid(uspec)
            prd = prd * torch.sqrt(inp_var) + inp_mean

            losses[iic] = loss_fn(prd, ref, solver=dataset.solver, relative=True).item()

    return losses, fno_times, nwp_times


# training function
def train_model(
    model,
    dataloader,
    optimizer,
    scheduler,
    gscaler,
    dist_manager,
    metrics,
    num_epochs=10,
    num_future=0,
    num_examples=512,
    num_valid=8,
    loss_fn="l2",
    enable_amp=False,
):
    if loss_fn == "l2":
        loss_fn = l2loss
    elif loss_fn == "l2sphere":
        loss_fn = l2loss_sphere
    else:
        raise ValueError(
            f"loss_fn={loss_fn} not supported, expected 'l2' or 'l2sphere'."
        )
    train_start = time.time()

    # count iterations
    iters = 0

    metrics["current_lr"] = []
    metrics["epoch_time"] = []
    metrics["acc_loss"] = []
    metrics["val_loss"] = []
    metrics["epochs"] = []

    for epoch in range(num_epochs):
        # time each epoch
        epoch_start = time.time()

        dataloader.dataset.set_initial_condition("random")
        dataloader.dataset.set_num_examples(num_examples)

        # get the solver for its convenience functions
        solver = dataloader.dataset.solver

        # do the training
        acc_loss = 0
        model.train()

        for inp, tar in dataloader:
            with amp.autocast(enabled=enable_amp):
                prd = model(inp)
                for _ in range(num_future):
                    prd = model(prd)
                loss = loss_fn(prd, tar, solver=solver, relative=False)

            acc_loss += loss.item() * inp.size(0)
            optimizer.zero_grad(set_to_none=True)
            if dist_manager.rank == 0:
                gscaler.scale(loss).backward()
            gscaler.step(optimizer)
            gscaler.update()

            iters += 1

        acc_loss = acc_loss / len(dataloader.dataset)

        dataloader.dataset.set_initial_condition("random")
        dataloader.dataset.set_num_examples(num_valid)

        # perform validation
        valid_loss = 0.0
        model.eval()
        with torch.no_grad():
            for inp, tar in dataloader:
                prd = model(inp)
                for _ in range(num_future):
                    prd = model(prd)
                loss = loss_fn(prd, tar, solver=solver, relative=True)
                valid_loss += loss.item() * inp.size(0)

        valid_loss = valid_loss / len(dataloader.dataset)

        epoch_time = time.time() - epoch_start

        if metrics is not None:
            current_lr = optimizer.param_groups[0]["lr"]
            metrics["epochs"].append(epoch)
            metrics["current_lr"].append(current_lr)
            metrics["epoch_time"].append(epoch_time)
            metrics["acc_loss"].append(acc_loss)
            metrics["val_loss"].append(valid_loss)

        scheduler.step()

    train_time = time.time() - train_start

    if dist_manager.rank == 0:
        print(
            f"--------------------------------------------------------------------------------"
        )
        print(f"done. Training took {train_time}.")


def get_random_data(dataset, device):
    inp = torch.randn((1, 3, dataset.nlat, dataset.nlon), device=device)
    tar = torch.randn((1, 3, dataset.nlat, dataset.nlon), device=device)
    return inp, tar


def train_model_on_dummy_data(
    model,
    dataloader,
    optimizer,
    scheduler,
    gscaler,
    dist_manager,
    metrics,
    num_epochs=10,
    num_future=0,
    num_examples=512,
    num_valid=8,
    loss_fn="l2",
    enable_amp=False,
):
    train_start = time.time()

    # count iterations
    iters = 0

    metrics["current_lr"] = []
    metrics["epoch_time"] = []
    metrics["acc_loss"] = []
    metrics["val_loss"] = []
    metrics["epochs"] = []

    for epoch in range(num_epochs):
        # time each epoch
        epoch_start = time.time()

        dataloader.dataset.set_num_examples(num_examples)

        # do the training
        acc_loss = 0
        model.train()

        for train_batch in range(num_examples):
            inp, tar = get_random_data(dataloader.dataset, model.device)
            with amp.autocast(enabled=enable_amp):
                prd = model(inp)
                for _ in range(num_future):
                    prd = model(prd)
                loss = l2loss(prd, tar)

            acc_loss += loss.item() * inp.size(0)
            optimizer.zero_grad(set_to_none=True)
            if dist_manager.rank == 0:
                gscaler.scale(loss).backward()
            gscaler.step(optimizer)
            gscaler.update()

            iters += 1

        acc_loss = acc_loss / len(dataloader.dataset)

        dataloader.dataset.set_num_examples(num_valid)

        # perform validation
        valid_loss = 0.0
        model.eval()
        with torch.no_grad():
            for val_batch in range(num_valid):
                inp, tar = get_random_data(dataloader.dataset, model.device)
                prd = model(inp)
                for _ in range(num_future):
                    prd = model(prd)
                loss = l2loss(prd, tar)
                valid_loss += loss.item() * inp.size(0)

        valid_loss = valid_loss / len(dataloader.dataset)

        epoch_time = time.time() - epoch_start

        if metrics is not None:
            current_lr = optimizer.param_groups[0]["lr"]
            metrics["epochs"].append(epoch)
            metrics["current_lr"].append(current_lr)
            metrics["epoch_time"].append(epoch_time)
            metrics["acc_loss"].append(acc_loss)
            metrics["val_loss"].append(valid_loss)

        scheduler.step()

    train_time = time.time() - train_start

    if dist_manager.rank == 0:
        print(
            f"--------------------------------------------------------------------------------"
        )
        print(f"done. Training took {train_time}.")


def main(
    mesh_size: int = 6,
    dummy_data: bool = False,
    seed: int = 1234,
    output_dir: Optional[str] = None,
    loss_fn: str = "l2",
    enable_amp: bool = False,
    hidden_dim: int = 128,
    processor_layers: int = 16,
    num_examples: int = 512,
    num_epochs: int = 10,
    angular_resolution: float = 0.703125,
    use_lat_lon_partitioning: bool = False,
):
    # realistic FP32 / TF32 settings
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    train = True
    load_checkpoint = False
    save_checkpoint = False
    enable_amp = False

    num_future = 0  # for now, no multi-step-rollout training
    # default of 0.703125 gives (256, 512)
    input_res = (
        int(180.0 / angular_resolution),
        int(360.0 / angular_resolution),
    )

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # initialize DistributedManager and define process group
    # across which graph is partitioned and model is parallel across
    # for now, based
    DistributedManager.initialize()
    if DistributedManager().distributed:
        graph_partition_pg_name = "graph_partition"
        world_size = torch.distributed.get_world_size()
        DistributedManager.create_process_subgroup(
            name=graph_partition_pg_name,
            size=world_size,
            verbose=True,
        )
    else:
        world_size = 1
        graph_partition_pg_name = None

    dist_manager = DistributedManager()

    hyperparameters = {
        "loss_fn": loss_fn,
        "processor_layers": processor_layers,
        "hidden_dim": hidden_dim,
        "num_examples": num_examples,
        "num_epochs": num_epochs,
        "angular_resolution": angular_resolution,
        "input_resolution": input_res,
        "mesh_size": mesh_size,
        "use_lat_lon_partitioning": use_lat_lon_partitioning,
        "enable_amp": enable_amp,
        "seed": seed,
        "world_size": world_size,
    }

    run_suffix = str(uuid.uuid4())

    if dist_manager.rank == 0:
        print(f"starting to run ...")

    mesh_base_path = os.path.dirname(os.path.realpath(__file__))
    meshgraph_path = os.path.join(mesh_base_path, f"./icospheres_{mesh_size}.json")
    model = GraphCastNet(
        meshgraph_path=meshgraph_path,
        static_dataset_path=None,
        input_res=input_res,
        input_dim_grid_nodes=3,
        input_dim_mesh_nodes=3,
        input_dim_edges=4,
        output_dim_grid_nodes=3,
        processor_layers=processor_layers,
        hidden_dim=hidden_dim,
        partition_size=dist_manager.group_size(graph_partition_pg_name),
        partition_group_name=graph_partition_pg_name,
        expect_partitioned_input=False,
        global_features_on_rank_0=True,
        produce_aggregated_output=True,
        produce_aggregated_output_on_all_ranks=False,
        use_lat_lon_partitioning=use_lat_lon_partitioning,
    ).to(device=dist_manager.device)

    # since model is "tensor-parallel" in graph-partition
    # mark model as "shared" which sets gradient hooks and
    # aggregates gradients in the backward pass accordingly
    if (
        dist_manager.is_initialized()
        and dist_manager.group_size(graph_partition_pg_name) > 1
    ):
        mark_module_as_shared(model, graph_partition_pg_name)

    # iterate over models and train each model
    if output_dir is None:
        output_dir = os.path.dirname(__file__)

    num_params = count_parameters(model)
    # prepare dicts containing models and corresponding metrics
    metrics = {**hyperparameters}

    if dist_manager.rank == 0:
        print(f"number of trainable params: {num_params}")
    metrics["num_params"] = num_params

    if load_checkpoint:
        model.load_state_dict(
            torch.load(os.path.join(output_dir, run_suffix, "checkpoints", "model.pt"))
        )

    # 1 hour prediction steps
    dt = 1 * 3600
    dt_solver = 150
    nsteps = dt // dt_solver
    dataset = PdeDataset(
        dt=dt,
        nsteps=nsteps,
        dims=input_res,
        device=dist_manager.device,
        normalize=True,
        rank=dist_manager.rank,
    )
    # There is still an issue with parallel dataloading. Do NOT use it at the moment
    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, persistent_workers=True)
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=0, persistent_workers=False
    )

    # run the training
    if train:
        # optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[10, 20], gamma=0.1
        )
        gscaler = amp.GradScaler(enabled=enable_amp)

        start_time = time.time()

        if dist_manager.rank == 0:
            print(f"Training, single step")

        if dummy_data:
            train_fn = train_model_on_dummy_data
        else:
            train_fn = train_model

        train_fn(
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            gscaler=gscaler,
            dist_manager=dist_manager,
            metrics=metrics,
            num_examples=num_examples,
            num_epochs=num_epochs,
            loss_fn=loss_fn,
            num_future=num_future,
            enable_amp=enable_amp,
        )

        training_time = time.time() - start_time

        if dist_manager.rank == 0:
            os.makedirs(os.path.join(output_dir, run_suffix, "figures"), exist_ok=True)
            os.makedirs(
                os.path.join(output_dir, run_suffix, "output_data"), exist_ok=True
            )
            if save_checkpoint:
                os.makedirs(
                    os.path.join(output_dir, run_suffix, "checkpoints"), exist_ok=True
                )
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, run_suffix, "checkpoints", "model.pt"),
                )

    # skip inference for dummy_data
    if not dummy_data:
        # set seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        with torch.inference_mode():
            losses, fno_times, nwp_times = autoregressive_inference(
                model,
                dataset,
                os.path.join(output_dir, run_suffix, "figures"),
                dist_manager=dist_manager,
                loss_fn=loss_fn,
                nsteps=nsteps,
                autoreg_steps=10,
            )
            metrics["loss_mean"] = np.mean(losses)
            metrics["loss_std"] = np.std(losses)
            metrics["fno_time_mean"] = np.mean(fno_times)
            metrics["fno_time_std"] = np.std(fno_times)
            metrics["nwp_time_mean"] = np.mean(nwp_times)
            metrics["nwp_time_std"] = np.std(nwp_times)

    max_mem = torch.tensor(
        [
            torch.cuda.max_memory_allocated() * 1.0 / (1024**3),
            torch.cuda.max_memory_reserved() * 1.0 / (1024**3),
        ],
        dtype=torch.float32,
        device=dist_manager.device,
    ).view(1, 2)
    if dist_manager.rank == 0:
        gather_list = [
            torch.empty_like(max_mem) for r in range(dist_manager.world_size)
        ]
    else:
        gather_list = []

    if DistributedManager().distributed:
        torch.distributed.gather(max_mem, gather_list, dst=0)
    else:
        max_mem = gather_list[0]

    if dist_manager.rank == 0:
        if train:
            max_mem = torch.vstack(gather_list)
            metrics["training_time"] = training_time
            metrics[f"max_memory_allocated_gib"] = max_mem[:, 0].tolist()
            metrics[f"max_memory_reserved_gib"] = max_mem[:, 1].tolist()

        with open(
            os.path.join(output_dir, run_suffix, "output_data", "metrics.json"), "w"
        ) as f:
            json.dump(metrics, f)

    if DistributedManager().distributed:
        DistributedManager.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_size", type=int, default=6)
    parser.add_argument("--dummy_data", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--loss", type=str, default="l2")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--processor_layers", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--num_examples", type=int, default=512)
    parser.add_argument("--angular_resolution", type=float, default=0.703125)
    parser.add_argument("--enable_amp", action="store_true")
    parser.add_argument("--lat_lon_part", action="store_true")
    args = parser.parse_args()

    main(
        mesh_size=args.mesh_size,
        dummy_data=args.dummy_data,
        seed=args.seed,
        output_dir=args.output_dir,
        loss_fn=args.loss,
        hidden_dim=args.hidden_dim,
        processor_layers=args.processor_layers,
        num_epochs=args.num_epochs,
        num_examples=args.num_examples,
        angular_resolution=args.angular_resolution,
        enable_amp=args.enable_amp,
        use_lat_lon_partitioning=args.lat_lon_part,
    )
