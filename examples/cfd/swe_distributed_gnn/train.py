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
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torch.cuda import amp

import numpy as np
import matplotlib.pyplot as plt

from pde_dataset import PdeDataset

from modulus.distributed import DistributedManager, mark_module_as_shared
from modulus.models.graphcast.graph_cast_net import GraphCastNet

import wandb


# control precision to a certain degree to
# make sure that the loss pattern is easier
# to sompare in slightly different accumulation
USE_TF32 = False
torch.backends.cuda.matmul.allow_tf32 = USE_TF32
torch.backends.cudnn.allow_tf32 = USE_TF32
torch.set_float32_matmul_precision("highest")
torch.use_deterministic_algorithms(True, warn_only=True)


def l2loss_sphere(solver, prd, tar, relative=False, squared=True):
    loss = solver.integrate_grid((prd - tar) ** 2, dimensionless=True).sum(dim=-1)
    if relative:
        loss = loss / solver.integrate_grid(tar**2, dimensionless=True).sum(dim=-1)

    if not squared:
        loss = torch.sqrt(loss)
    loss = loss.mean()

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
    autoreg_steps=10,
    nskip=1,
    plot_channel=1,
    nics=20,
):
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
            losses[iic] = l2loss_sphere(dataset.solver, prd, ref, relative=True).item()

    return losses, fno_times, nwp_times


# training function
def train_model(
    model,
    dataloader,
    optimizer,
    gscaler,
    dist_manager,
    scheduler=None,
    nepochs=10,
    nfuture=0,
    num_examples=512,
    num_valid=8,
    loss_fn="l2",
    enable_amp=False,
    log_grads=0,
):
    train_start = time.time()

    # count iterations
    iters = 0

    for epoch in range(nepochs):

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
                for _ in range(nfuture):
                    prd = model(prd)
                loss = l2loss_sphere(solver, prd, tar, relative=False)

            acc_loss += loss.item() * inp.size(0)
            optimizer.zero_grad(set_to_none=False)
            gscaler.scale(loss).backward()
            gscaler.step(optimizer)
            gscaler.update()

            iters += 1

        acc_loss = acc_loss / len(dataloader.dataset)

        dataloader.dataset.set_initial_condition("random")
        dataloader.dataset.set_num_examples(num_valid)

        # perform validation
        valid_loss = 0
        model.eval()
        with torch.no_grad():
            for inp, tar in dataloader:
                prd = model(inp)
                for _ in range(nfuture):
                    prd = model(prd)
                loss = l2loss_sphere(solver, prd, tar, relative=True)
                valid_loss += loss.item() * inp.size(0)

        valid_loss = valid_loss / len(dataloader.dataset)

        if scheduler is not None:
            scheduler.step(valid_loss)

        epoch_time = time.time() - epoch_start

        if dist_manager.rank == 0:
            print(
                f"--------------------------------------------------------------------------------"
            )
            print(f"Epoch {epoch} summary:")
            print(f"time taken: {epoch_time}")
            print(f"accumulated training loss: {acc_loss}")
            print(f"relative validation loss: {valid_loss}")

            if wandb.run is not None:
                current_lr = optimizer.param_groups[0]["lr"]
                wandb.log(
                    {
                        "loss": acc_loss,
                        "validation loss": valid_loss,
                        "learning rate": current_lr,
                    }
                )

    train_time = time.time() - train_start

    if dist_manager.rank == 0:
        print(
            f"--------------------------------------------------------------------------------"
        )
        print(f"done. Training took {train_time}.")

    return valid_loss


def main(
    mesh_size: int = 6,
    dummy_dataset: bool = False,
    short_run: bool = False,
    seed: int = 1234,
    root_path: Optional[str] = None,
):
    train = True
    load_checkpoint = False
    save_checkpoint = False
    enable_amp = False
    log_grads = 0

    nepochs = 10 if not short_run else 64
    num_examples = 512 if not short_run else 1
    input_res = (256, 512)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # initialize DistributedManager and define process group
    # across which graph is partitioned and model is parallel across
    # for now, based
    DistributedManager.initialize()
    if DistributedManager().distributed:
        graph_partition_pg_name = "graph_partition"
        world_size = torch.distributed.get_world_size()
        run_suffix = f"mg_{world_size}_mesh{mesh_size}_dummy{dummy_dataset}_seed{seed}"
        DistributedManager.create_process_subgroup(
            name=graph_partition_pg_name,
            size=world_size,
            verbose=True,
        )
    else:
        run_suffix = f"sg_mesh{mesh_size}_dummy{dummy_dataset}_seed{seed}"
        graph_partition_pg_name = None

    dist_manager = DistributedManager()

    if dist_manager.rank == 0:
        wandb.login()

    model = GraphCastNet(
        meshgraph_path=f"./icospheres_{mesh_size}.json",
        static_dataset_path=None,
        input_res=input_res,
        input_dim_grid_nodes=3,
        input_dim_mesh_nodes=3,
        input_dim_edges=4,
        output_dim_grid_nodes=3,
        processor_layers=16,
        hidden_dim=128,
        partition_size=dist_manager.group_size(graph_partition_pg_name),
        partition_group_name=graph_partition_pg_name,
        expect_partitioned_input=False,
        global_features_on_rank_0=True,
        produce_aggregated_output=True,
        produce_aggregated_output_on_all_ranks=False,
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
    if root_path is None:
        root_path = os.path.dirname(__file__)

    num_params = count_parameters(model)
    # prepare dicts containing models and corresponding metrics
    metrics = {}

    if dist_manager.rank == 0:
        print(f"number of trainable params: {num_params}")
    metrics["num_params"] = num_params

    if load_checkpoint:
        model.load_state_dict(
            torch.load(os.path.join(root_path, "checkpoints", run_suffix, "model.pt"))
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
        dummy_dataset=dummy_dataset,
    )
    # There is still an issue with parallel dataloading. Do NOT use it at the moment
    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, persistent_workers=True)
    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=0, persistent_workers=False
    )

    # run the training
    if train:
        if dist_manager.rank == 0:
            run = wandb.init(
                project="swe", group="SWE_GC", name="SWE_GC" + "_" + str(time.time())
            )

        # optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[4, 8], gamma=0.5
        )
        gscaler = amp.GradScaler(enabled=enable_amp)

        start_time = time.time()

        if dist_manager.rank == 0:
            print(f"Training, single step")

        train_model(
            model,
            dataloader,
            optimizer,
            gscaler,
            dist_manager,
            scheduler=scheduler,
            num_examples=num_examples,
            nepochs=nepochs,
            loss_fn="l2",
            enable_amp=enable_amp,
            log_grads=log_grads,
        )

        training_time = time.time() - start_time

        if dist_manager.rank == 0:
            os.makedirs(os.path.join(root_path, "figures", run_suffix), exist_ok=True)
            os.makedirs(
                os.path.join(root_path, "output_data", run_suffix), exist_ok=True
            )
            if save_checkpoint:
                os.makedirs(
                    os.path.join(root_path, "checkpoints", run_suffix), exist_ok=True
                )
                torch.save(
                    model.state_dict(),
                    os.path.join(root_path, "checkpoints", run_suffix, "model.pt"),
                )

    # set seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    with torch.inference_mode():
        losses, fno_times, nwp_times = autoregressive_inference(
            model,
            dataset,
            os.path.join(root_path, "figures", run_suffix),
            dist_manager=dist_manager,
            nsteps=nsteps,
            autoreg_steps=10,
        )
        metrics["loss_mean"] = np.mean(losses)
        metrics["loss_std"] = np.std(losses)
        metrics["fno_time_mean"] = np.mean(fno_times)
        metrics["fno_time_std"] = np.std(fno_times)
        metrics["nwp_time_mean"] = np.mean(nwp_times)
        metrics["nwp_time_std"] = np.std(nwp_times)
        if train:
            metrics["training_time"] = training_time
            metrics[f"max_memory_allocated_gib"] = (
                torch.cuda.max_memory_allocated() * 1.0 / (1024**3)
            )
            metrics[f"max_memory_reserved_gib"] = (
                torch.cuda.max_memory_reserved() * 1.0 / (1024**3)
            )

    if dist_manager.rank == 0:
        with open(
            os.path.join(root_path, "output_data", run_suffix, "metrics.json"), "w"
        ) as f:
            json.dump(metrics, f)
        if train:
            run.finish()

    DistributedManager.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_size", type=int, default=6)
    parser.add_argument("--dummy_data", action="store_true")
    parser.add_argument("--short_run", action="store_true")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--root_path", type=str, default=None)
    args = parser.parse_args()

    for k, v in vars(args):
        print(f"{k}: {v}")

    main(
        mesh_size=args.mesh_size,
        dummy_data=args.dummy_data,
        short_run=args.short_run,
        seed=args.seed,
        root_path=args.root_path,
    )
