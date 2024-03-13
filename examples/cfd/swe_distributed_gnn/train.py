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

import os
import time

import torch
from torch.utils.data import DataLoader
from torch.cuda import amp

import numpy as np

import json

import matplotlib.pyplot as plt

from pde_dataset import PdeDataset

from modulus.models.graphcast.graph_cast_net import GraphCastNet

# wandb logging
import wandb
wandb.login()


def l2loss_sphere(solver, prd, tar, relative=False, squared=True):
    loss = ((prd - tar)**2).mean()
    return loss


# rolls out the FNO and compares to the classical solver
def autoregressive_inference(model,
                             dataset,
                             path_root,
                             nsteps,
                             autoreg_steps=10,
                             nskip=1,
                             plot_channel=1,
                             nics=20):

    #model.eval()

    losses = np.zeros(nics)
    fno_times = np.zeros(nics)
    nwp_times = np.zeros(nics)

    for iic in range(nics):
        ic = dataset.solver.random_initial_condition(mach=0.1)
        inp_mean = dataset.inp_mean
        inp_var = dataset.inp_var

        prd = (dataset.solver.spec2grid(ic) - inp_mean) / torch.sqrt(inp_var)
        prd = prd.unsqueeze(0)
        uspec = ic.clone()

        # ML model
        start_time = time.time()
        for i in range(1, autoreg_steps+1):
            # evaluate the ML model
            prd = model(prd)

            if iic == nics-1 and nskip > 0 and i % nskip == 0:

                # do plotting
                fig = plt.figure(figsize=(7.5, 6))
                dataset.solver.plot_griddata(prd[0, plot_channel], fig, vmax=4, vmin=-4)
                plt.savefig(path_root+'_pred_'+str(i//nskip)+'.png')
                plt.clf()

        fno_times[iic] = time.time() - start_time

        # classical model
        start_time = time.time()
        for i in range(1, autoreg_steps+1):
            
            # advance classical model
            uspec = dataset.solver.timestep(uspec, nsteps)

            if iic == nics-1 and i % nskip == 0 and nskip > 0:
                ref = (dataset.solver.spec2grid(uspec) - inp_mean) / torch.sqrt(inp_var)

                fig = plt.figure(figsize=(7.5, 6))
                dataset.solver.plot_griddata(ref[plot_channel], fig, vmax=4, vmin=-4)
                plt.savefig(path_root+'_truth_'+str(i//nskip)+'.png')
                plt.clf()

        nwp_times[iic] = time.time() - start_time

        # ref = (dataset.solver.spec2grid(uspec) - inp_mean) / torch.sqrt(inp_var)
        ref = dataset.solver.spec2grid(uspec)
        prd = prd * torch.sqrt(inp_var) + inp_mean
        losses[iic] = l2loss_sphere(dataset.solver, prd, ref, relative=True).item()
        

    return losses, fno_times, nwp_times

# convenience function for logging weights and gradients
def log_weights_and_grads(model, iters=1):
    """
    Helper routine intended for debugging purposes
    """
    root_path = os.path.join(os.path.dirname(__file__), "weights_and_grads")

    weights_and_grads_fname = os.path.join(root_path, f"weights_and_grads_step{iters:03d}.tar")
    print(weights_and_grads_fname)

    weights_dict = {k:v for k,v in model.named_parameters()}
    grad_dict = {k:v.grad for k,v in model.named_parameters()}

    store_dict = {'iteration': iters, 'grads': grad_dict, 'weights': weights_dict}
    torch.save(store_dict, weights_and_grads_fname)


# training function
def train_model(model,
                dataloader,
                optimizer,
                gscaler,
                scheduler=None,
                nepochs=10,
                nfuture=0,
                num_examples=512,
                num_valid=8,
                loss_fn='l2',
                enable_amp=False,
                log_grads=0):

    train_start = time.time()

    # count iterations
    iters = 0

    for epoch in range(nepochs):

        # time each epoch
        epoch_start = time.time()

        dataloader.dataset.set_initial_condition('random')
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

            if log_grads and iters % log_grads == 0:
                log_weights_and_grads(model, iters=iters)

            gscaler.step(optimizer)
            gscaler.update()

            iters += 1

        acc_loss = acc_loss / len(dataloader.dataset)

        dataloader.dataset.set_initial_condition('random')
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

        print(f'--------------------------------------------------------------------------------')
        print(f'Epoch {epoch} summary:')
        print(f'time taken: {epoch_time}')
        print(f'accumulated training loss: {acc_loss}')
        print(f'relative validation loss: {valid_loss}')

        if wandb.run is not None:
            current_lr = optimizer.param_groups[0]['lr']
            wandb.log({"loss": acc_loss, "validation loss": valid_loss, "learning rate": current_lr})


    train_time = time.time() - train_start

    print(f'--------------------------------------------------------------------------------')
    print(f'done. Training took {train_time}.')
    return valid_loss

def main(train=True, load_checkpoint=False, enable_amp=False, log_grads=0):

    # set seed
    torch.manual_seed(333)
    torch.cuda.manual_seed(333)

    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #if torch.cuda.is_available():
    #    torch.cuda.set_device(device.index)

    # 1 hour prediction steps
    dt = 1*3600
    dt_solver = 150
    nsteps = dt//dt_solver
    dataset = PdeDataset(dt=dt, nsteps=nsteps, dims=(256, 512), device=device, normalize=True)
    # There is still an issue with parallel dataloading. Do NOT use it at the moment     
    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, persistent_workers=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0, persistent_workers=False)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # prepare dicts containing models and corresponding metrics
    metrics = {}

    model = GraphCastNet(
            meshgraph_path="./icospheres.json",
            static_dataset_path=None,
            input_res=(256, 512),
            input_dim_grid_nodes=3,
            input_dim_mesh_nodes=3,
            input_dim_edges=4,
            output_dim_grid_nodes=3,
            processor_layers=15,
            hidden_dim=128,
            partition_size=world_size,
            partition_group_name=None,
            expect_partitioned_input=True,
            produce_aggregated_output=True,
        ).to(device)

    # iterate over models and train each model
    root_path = os.path.dirname(__file__)
    print(model)

    num_params = count_parameters(model)
    print(f'number of trainable params: {num_params}')
    metrics['num_params'] = num_params

    if load_checkpoint:
        model.load_state_dict(torch.load(os.path.join(root_path, 'checkpoints')))

    # run the training
    if train:
        run = wandb.init(project="swe", group='SWE_GC', name='SWE_GC' + '_' + str(time.time()))

        # optimizer:
        optimizer = torch.optim.Adam(model.parameters(), lr=1E-4)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        scheduler = None
        gscaler = amp.GradScaler(enabled=enable_amp)

        start_time = time.time()

        print(f'Training, single step')
        train_model(model, dataloader, optimizer, gscaler, scheduler, nepochs=10, loss_fn='l2', enable_amp=enable_amp, log_grads=log_grads)

        training_time = time.time() - start_time

        run.finish()
        os.makedirs(os.path.join(root_path, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(root_path, 'figures'), exist_ok=True)
        os.makedirs(os.path.join(root_path, 'output_data'), exist_ok=True)
        #torch.save(model.state_dict(), os.path.join(root_path, 'checkpoints'))

    # set seed
    torch.manual_seed(333)
    torch.cuda.manual_seed(333)

    with torch.inference_mode():
        losses, fno_times, nwp_times = autoregressive_inference(model, dataset, os.path.join(root_path,'figures/'), nsteps=nsteps, autoreg_steps=10)
        metrics['loss_mean'] = np.mean(losses)
        metrics['loss_std'] = np.std(losses)
        metrics['fno_time_mean'] = np.mean(fno_times)
        metrics['fno_time_std'] = np.std(fno_times)
        metrics['nwp_time_mean'] = np.mean(nwp_times)
        metrics['nwp_time_std'] = np.std(nwp_times)
        if train:
            metrics['training_time'] = training_time

    with open(os.path.join(root_path, 'output_data/metrics.json'), 'w') as f:
        json.dump(metrics, f)


if __name__ == "__main__":
    main(train=True, load_checkpoint=False, enable_amp=False, log_grads=0)
