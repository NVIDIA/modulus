# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Main training loop."""

import os
import time
import copy
import json
from utils.YParams import YParams
import pickle
import psutil
import numpy as np
import torch
import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc
from utils.diffusions.generate import edm_sampler
import utils.diffusions.networks
import utils.diffusions.losses
from utils.diffusions.power_ema import sigma_rel_to_gamma
from utils.data_loader_hrrr_era5 import get_data_loader, get_dataset, worker_init
import utils.img_utils
from torchvision import transforms
import matplotlib.pyplot as plt
from networks.swinv2_hrrr import swinv2net
import wandb
from utils.wandb import get_wandb_names
from utils.cluster_paths import adjust_cluster_paths
from utils.spectrum import compute_ps1d
from torch.nn.utils import clip_grad_norm_
#----------------------------------------------------------------------------

def get_pretrained_regression_net(basepath, regression_net_type, device, invariant_tensor=None):

    if regression_net_type == 'swin':

        from utils.simple_load_yaml import simple_load_yaml

        hyperparams = simple_load_yaml(os.path.join(basepath, 'hyperparams.yaml'))
        hyperparams.era5_img_size = hyperparams.hrrr_img_size #TODO fix this issue in the original hyperparam dump
        hyperparams.mask_ratio = 0
        hyperparams.nonzero_mask = False
        swin = swinv2net(hyperparams).to(device)
        checkpoint_path = os.path.join(basepath, 'training_checkpoints', 'ckpt.tar')

        # load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        for key in list(checkpoint['model_state'].keys()):
            if 'module.' in key:
                checkpoint['model_state'][key.replace('module.', '')] = checkpoint['model_state'].pop(key)

        swin.load_state_dict(checkpoint['model_state'])

        return swin
    
    elif regression_net_type == 'unet':

        from utils.diffusions.networks import EasyRegression

        hyperparams = YParams(os.path.join("config", 'hrrr_swin.yaml'), 'regression_a2a_v3_1_exclude_w')
        net_name = "song-unet-regression"
        resolution = 512
        target_channels = len(['u10m', 'v10m', 't2m', 'msl', 'u1', 'u2', 'u3', 'u4', 'u5', 'u6', 'u7', 'u8', 'u9', 'u10', 'u11', 'u13', 'u15', 'u20', 'u25', 'u30', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'v13', 'v15', 'v20', 'v25', 'v30', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't13', 't15', 't20', 't25', 't30', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10', 'q11', 'q13', 'q15', 'q20', 'q25', 'q30', 'z1', 'z2', 'z3', 'z4', 'z5', 'z6', 'z7', 'z8', 'z9', 'z10', 'z11', 'z13', 'z15', 'z20', 'z25', 'z30', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p13', 'p15', 'p20', 'refc']) 
        conditional_channels = target_channels + len(hyperparams.invariants) + 26

        net = utils.diffusions.networks.get_preconditioned_architecture(
        name=net_name,
        resolution=resolution,
        target_channels=target_channels,
        conditional_channels=conditional_channels,
        label_dim=0,
        spatial_embedding=hyperparams.spatial_pos_embed,
        attn_resolutions=hyperparams.attn_resolutions,
    )

        resume_pkl = os.path.join(basepath, "0-regression_a2a_v3_1_exclude_w-hrrr-gpus32/network-snapshot-012280.pkl")

        with open(resume_pkl, 'rb') as f:
            data = pickle.load(f)
        net.load_state_dict(data['net'].state_dict(), strict=True)

        latent_shape = [target_channels, 512, 640]

        net = EasyRegression(net, latent_shape)

        return net.to(device)

    elif regression_net_type == 'unet2':

        from utils.diffusions.networks import EasyRegressionV2

        hyperparams = YParams(os.path.join("config", 'hrrr_swin.yaml'), 'regression_a2a_v3_1_exclude_w_v2_noskip')
        net_name = "song-unet-regression-v2"
        resolution = 512
        target_channels = len(['u10m', 'v10m', 't2m', 'msl', 'u1', 'u2', 'u3', 'u4', 'u5', 'u6', 'u7', 'u8', 'u9', 'u10', 'u11', 'u13', 'u15', 'u20', 'u25', 'u30', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'v13', 'v15', 'v20', 'v25', 'v30', 't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't13', 't15', 't20', 't25', 't30', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10', 'q11', 'q13', 'q15', 'q20', 'q25', 'q30', 'z1', 'z2', 'z3', 'z4', 'z5', 'z6', 'z7', 'z8', 'z9', 'z10', 'z11', 'z13', 'z15', 'z20', 'z25', 'z30', 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'p10', 'p11', 'p13', 'p15', 'p20', 'refc']) 
        conditional_channels = target_channels + len(hyperparams.invariants) + 26

        net = utils.diffusions.networks.get_preconditioned_architecture(
        name=net_name,
        resolution=resolution,
        target_channels=target_channels,
        conditional_channels=conditional_channels,
        label_dim=0,
        spatial_embedding=hyperparams.spatial_pos_embed,
        attn_resolutions=hyperparams.attn_resolutions,
    )

        resume_pkl = os.path.join(basepath, "0-regression_a2a_v3_1_exclude_w_v2_noskip-hrrr-gpus32/network-snapshot-001023.pkl")

        with open(resume_pkl, 'rb') as f:
            data = pickle.load(f)
        net.load_state_dict(data['net'].state_dict(), strict=True)

        latent_shape = [target_channels, 512, 640]

        net = EasyRegressionV2(net)

        return net.to(device)

def training_loop(
    run_dir             = '.',              # Output directory.
    optimizer_kwargs    = {},               # Options for optimizer.
    seed                = 0,                # Global random seed.
    ema_sigma_rel       = [0.05, 0.1],      # Controls EMA profiles of model weight snapshots saved (sigma_rel from EDMv2)
    ema_freq_kimg       = 10,               # Frequency to save snapshots for EMA profiles (see EDMv2)
    lr_rampup_kimg      = 2000,            # Learning rate ramp-up duration.
    loss_scaling        = 1,                # Loss scaling factor for reducing FP16 under/overflows.
    kimg_per_tick       = 50,               # Interval of progress prints.
    snapshot_ticks      = 50,               # How often to save network snapshots, None = disable.
    state_dump_ticks    = 500,              # How often to dump training state, None = disable.
    resume_pkl          = None,             # Start from the given network snapshot, None = random initialization.
    resume_state_dump   = None,             # Start from the given training state, None = reset training state.
    resume_kimg         = 0,                # Start from the given training progress.
    cudnn_benchmark     = True,             # Enable torch.backends.cudnn.benchmark?
    device              = torch.device('cuda'),
    config_file         = None,
    config_name         = None,
    log_to_wandb        = False,
):
    params = YParams(config_file, config_name)
    params = adjust_cluster_paths(params)
    batch_size = params.batch_size
    local_batch_size = batch_size // dist.get_world_size()
    img_per_tick = params.img_per_tick
    use_regression_net = params.use_regression_net    
    residual = params.residual
    log_scale_residual = params.log_scale_residual
    previous_step_conditioning = params.previous_step_conditioning
    tendency_normalization = params.tendency_normalization
    pure_diffusion = params.pure_diffusion
    loss_type = params.loss
    if loss_type == "regression":
        train_regression_unet = True
        net_name = "song-unet-regression"
    elif loss_type == 'regression_v2':
        train_regression_unet = True
        net_name = "song-unet-regression-v2"
        print("Using regression_v2")
    elif loss_type == "edm":
        train_regression_unet = False
        net_name = "ddpmpp-cwb-v0"
        gamma1, gamma2 = [sigma_rel_to_gamma(x) for x in ema_sigma_rel]
        ema_freq_kimg = params.ema_freq_kimg


    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    if resume_state_dump is not None:
        print('Resuming from state dump:', resume_state_dump)
        print('Resuming from kimg:', resume_kimg)
        print('Resuming from pkl:', resume_pkl)
    
    # Load dataset.
    dist.print0('Loading dataset...')
    # hard code this name

    #load pretrained regression net
    if use_regression_net:
        regression_net = get_pretrained_regression_net(params.regression_model_basepath, params.regression_net_type, device)

    total_kimg = params.total_kimg


    dataset_train = get_dataset(params, train=True)
    dataset_valid = get_dataset(params, train=False)
    #hrrr_channels = dataset_train.hrrr_channels.values.tolist()
    base_hrrr_channels, kept_hrrr_channels = dataset_train._get_hrrr_channel_names()

    #hrrr_channels = hrrr_channels[:-1] #remove the last channel vil. TODO: fix this in the dataset
    hrrr_channels = kept_hrrr_channels
    
    diffusion_channels = params.diffusion_channels
    if diffusion_channels == 'all':
        diffusion_channels = hrrr_channels
    input_channels = params.input_channels
    diffusion_channel_indices = [hrrr_channels.index(channel) for channel in diffusion_channels]
    dist.print0('diffusion_channel_indices', diffusion_channel_indices)
    if input_channels == 'all':
        input_channel_indices = [hrrr_channels.index(channel) for channel in hrrr_channels]
        input_channels = hrrr_channels
    else:
        input_channel_indices = [hrrr_channels.index(channel) for channel in input_channels]
    
    sampler = misc.InfiniteSampler(dataset=dataset_train, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
    valid_sampler = misc.InfiniteSampler(dataset=dataset_valid, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=local_batch_size,
        num_workers=params.num_data_workers,
        sampler=sampler,
        worker_init_fn=worker_init,
        drop_last=True,
        pin_memory=torch.cuda.is_available()
    )

    valid_data_loader = torch.utils.data.DataLoader(
        dataset=dataset_valid,
        batch_size=local_batch_size,
        num_workers=params.num_data_workers,
        sampler=valid_sampler,
        drop_last=True,
        pin_memory=torch.cuda.is_available()
    )

    dataset_iterator = iter(data_loader)
    valid_dataset_iterator = iter(valid_data_loader)


    # Construct network.
    dist.print0('Constructing network...')
    resolution = params.crop_size if params.crop_size is not None else 512
    target_channels = len(diffusion_channels)
    if tendency_normalization or train_regression_unet:
        conditional_channels = len(input_channels) + 26 # 26 is the number of era5 channels
    else:
        conditional_channels = len(input_channels) if not previous_step_conditioning else 2*len(input_channels)
    
    if pure_diffusion:
        conditional_channels = len(input_channels) + 26 # 26 is the number of era5 channels

    if len(params.invariants) > 0:
        conditional_channels += len(params.invariants)
        invariant_array = dataset_train._get_invariants()
        invariant_tensor = torch.from_numpy(invariant_array).to(device)

    if params.linear_grid:
        conditional_channels += 2
        dims = params.hrrr_img_size
        grid_x, grid_y = torch.meshgrid(torch.linspace(0, 1, dims[0]), torch.linspace(0, 1, dims[1]))
        grid = torch.stack((grid_x, grid_y), dim=0).to(device)
        if len(params.invariants) > 0:
            invariant_tensor = torch.cat((invariant_tensor, grid), dim=0)
        else:
            invariant_tensor = grid        
    
    if not train_regression_unet:
        if not pure_diffusion:
            if params.regression_net_type in ['unet', 'unet2']:
                #required only for inference of pretrained regression net
                regression_net.set_invariant(invariant_tensor)

    if params.tendency_normalization:
        tendency_stds = torch.from_numpy(dataset_train.tendency_stds_hrrr).to(device)
        input_mean = torch.from_numpy(dataset_train.means_hrrr).to(device) 
        input_std = torch.from_numpy(dataset_train.stds_hrrr).to(device)
        input_mean = input_mean[input_channel_indices]
        input_std = input_std[input_channel_indices]
        tendency_stds = tendency_stds[input_channel_indices]
        assert input_channel_indices == diffusion_channel_indices
    
    label_dim = 0

    dist.print0("hrrr_channels", kept_hrrr_channels)
    dist.print0("target_channels for diffusion", target_channels)
    dist.print0("conditional_channels for diffusion", conditional_channels)

    net = utils.diffusions.networks.get_preconditioned_architecture(
        name=net_name,
        resolution=resolution,
        target_channels=target_channels,
        conditional_channels=conditional_channels,
        label_dim=0,
        spatial_embedding=params.spatial_pos_embed,
        attn_resolutions=params.attn_resolutions,
    )

    if not params.loss in ['regression', 'regression_v2']: assert net.sigma_min < net.sigma_max
    net.train().requires_grad_(True).to(device)

    # Setup optimizer.
    dist.print0('Setting up optimizer...')
    if params.loss == 'regression':
        loss_fn = utils.diffusions.losses.RegressionLoss()
    elif params.loss == 'regression_v2':
        loss_fn = utils.diffusions.losses.RegressionLossV2()
    elif params.loss == 'edm':
        loss_fn = utils.diffusions.losses.EDMLoss(P_mean=params.P_mean)
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs) # subclass of torch.optim.Optimizer
    augment_pipe = None
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device], broadcast_buffers=False)
    if loss_type == 'edm':
        ema1 = copy.deepcopy(net).eval().requires_grad_(False)
        ema2 = copy.deepcopy(net).eval().requires_grad_(False)

    total_steps = 0

    # Resume training from previous snapshot.
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier() # rank 0 goes first
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier() # other ranks follow
        misc.copy_params_and_buffers(src_module=data['net'], dst_module=net, require_all=False)
        del data # conserve memory
    if resume_state_dump:
        dist.print0(f'Loading training state from "{resume_state_dump}"...')
        data = torch.load(resume_state_dump, map_location=torch.device('cpu'))
        misc.copy_params_and_buffers(src_module=data['net'], dst_module=net, require_all=True)
        if loss_type == 'edm':
            misc.copy_params_and_buffers(src_module=data['ema1'], dst_module=ema1, require_all=True)
            misc.copy_params_and_buffers(src_module=data['ema2'], dst_module=ema2, require_all=True)
        total_steps = data['total_steps']
        optimizer.load_state_dict(data['optimizer_state'])
        del data # conserve memory

    # Train.
    dist.print0(f'Training for {total_kimg} kimg...')
    dist.print0()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None
    wandb_logs = {}

    def transform(x):
        return utils.img_utils.image_to_crops(x, resolution, resolution)

    while True:

        # Accumulate gradients.
        optimizer.zero_grad(set_to_none=True)
        batch = next(dataset_iterator)
        hrrr_0 = batch['hrrr'][0].to(device).to(torch.float32)
        hrrr_1 = batch['hrrr'][1].to(device).to(torch.float32)

        if use_regression_net:
            era5 = batch['era5'][0].to(device).to(torch.float32)

            if previous_step_conditioning:
                with torch.no_grad():
                    reg_out = regression_net(hrrr_0, era5, mask=None)
                    hrrr_0 = torch.cat((hrrr_0[:, input_channel_indices, :, :], reg_out[:, input_channel_indices, :, :]), dim=1)
                    if residual:
                        hrrr_1 = hrrr_1 - reg_out
                    if log_scale_residual:
                        hrrr_1 = torch.sign(hrrr_1) * torch.log1p(torch.abs(hrrr_1))
                    del reg_out
            else:
                with torch.no_grad():
                    hrrr_0 = regression_net(hrrr_0, era5, mask=None)
                    if residual:
                        hrrr_1 = hrrr_1 - hrrr_0
                    if log_scale_residual:
                        hrrr_1 = torch.sign(hrrr_1) * torch.log1p(torch.abs(hrrr_1))

        elif tendency_normalization: 

            assert diffusion_channel_indices == input_channel_indices

            era5 = batch['era5'][0].to(device).to(torch.float32)

            hrrr_0 = torch.cat((hrrr_0[:, input_channel_indices, :, :], era5), dim=1)
        
        elif train_regression_unet:

            assert diffusion_channel_indices == input_channel_indices
            assert use_regression_net == False

            era5 = batch['era5'][0].to(device).to(torch.float32)

            hrrr_0 = torch.cat((hrrr_0[:, input_channel_indices, :, :], era5), dim=1)

        elif pure_diffusion:

            if residual:

                hrrr_1 = hrrr_1 - hrrr_0

            era5 = batch['era5'][0].to(device).to(torch.float32)
            hrrr_0 = torch.cat((hrrr_0[:, input_channel_indices, :, :], era5), dim=1)

        hrrr_1 = hrrr_1[:, diffusion_channel_indices, :, :] #targets of the diffusion model
        
        if params.crop_size is not None:
            hrrr_1 = transform(hrrr_1) 

        if len(params.invariants) > 0:
            invariant_tensor_ = invariant_tensor.unsqueeze(0)
            invariant_tensor_ = invariant_tensor.repeat(hrrr_0.shape[0],1,1,1)
            hrrr_0 = torch.cat((hrrr_0, invariant_tensor_), dim=1)

        if params.crop_size is not None:
            hrrr_0 = transform(hrrr_0) 
        loss = loss_fn(net=ddp, x=hrrr_1, condition=hrrr_0, augment_pipe=augment_pipe)
        channelwise_loss = loss.mean(dim=(0, 2, 3))
        channelwise_loss_dict = { f"ChLoss/{diffusion_channels[i]}": channelwise_loss[i].item() for i in range(target_channels) }
        training_stats.report('Loss/loss', loss.mean())
        loss_value = loss.sum() / target_channels
        if log_to_wandb:
            wandb_logs['channelwise_loss'] = channelwise_loss_dict
            #TODO: aggregate channelwise loss across all gpus. Training_stats.report only works for scalar values
    
        loss_value.backward()

        if params.clip_grad_norm is not None:

            torch.nn.utils.clip_grad_norm_(net.parameters(), params.clip_grad_norm)

        # Update weights.
        for g in optimizer.param_groups:
            g['lr'] = optimizer_kwargs['lr'] * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)
            if log_to_wandb:
                wandb_logs['lr'] = g['lr']
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        optimizer.step()

        # Update weight snapshots (EDMv2 Eq. 2)
        effective_batch_size = (batch_size//local_batch_size) * hrrr_0.shape[0]
        total_steps += 1
        if loss_type == 'edm':
            for p_ema1, p_ema2, p_net in zip(ema1.parameters(), ema2.parameters(), net.parameters()):
                beta1, beta2 = (1 - 1./total_steps)**(gamma1 + 1), (1 - 1./total_steps)**(gamma2 + 1)
                theta1 = beta1*p_ema1 + (1-beta1)*p_net.detach()
                theta2 = beta2*p_ema2 + (1-beta2)*p_net.detach()
                p_ema1.copy_(theta1)
                p_ema2.copy_(theta2)
        
            # Save snapshots at fixed time intervals
            ema_freq = max(ema_freq_kimg*1000//effective_batch_size, 1)
            #print('STEP', total_steps, ema_freq, effective_batch_size)
            if total_steps % ema_freq == 0:
                data = dict(ema1=ema1, ema2=ema2, ema_sigma_rel=ema_sigma_rel)
                for key, value in data.items():
                    if isinstance(value, torch.nn.Module):
                        value = copy.deepcopy(value).eval().requires_grad_(False)
                        misc.check_ddp_consistency(value)
                        data[key] = value.cpu().to(torch.float16)
                    del value # conserve memory
                if dist.get_rank() == 0:
                    with open(os.path.join(run_dir, f'ema-snapshot-{total_steps:08d}.pkl'), 'wb') as f:
                        pickle.dump(data, f)
                del data # conserve memory

            

        # Perform maintenance tasks once per tick.
        cur_nimg += effective_batch_size
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + img_per_tick ):
            continue


        # make inference
        if cur_tick % params.validate_every == 0:

            batch = next(valid_dataset_iterator)

            with torch.no_grad():
                n = 1 
                hrrr_0, hrrr_1 = batch['hrrr']
                hrrr_0 = hrrr_0.to(torch.float32).to(device)
                hrrr_1 = hrrr_1.to(torch.float32).to(device)

                if len(params.invariants) > 0:
                    invariant_tensor_ = invariant_tensor.unsqueeze(0)
                

                if use_regression_net:
                    with torch.no_grad():
                        era5 = batch['era5'][0].to(device).to(torch.float32)
                        if previous_step_conditioning:
                            reg_out = regression_net(hrrr_0, era5, mask=None)
                            hrrr_0 = torch.cat((hrrr_0[:, input_channel_indices, :, :], reg_out[:, input_channel_indices, :, :]), dim=1)
                            latents = torch.randn_like(hrrr_1[0:n, diffusion_channel_indices, :, :])
                            if residual:
                                loss_target = hrrr_1 - reg_out #required for valid loss calculation
                            else:
                                loss_target = hrrr_1
                            if len(params.invariants) > 0:
                                output_images = edm_sampler(net, latents=latents, condition=torch.cat((hrrr_0, invariant_tensor_), dim=1))
                                valid_loss = loss_fn(net=ddp, x=loss_target[0:n, diffusion_channel_indices], condition=torch.cat((hrrr_0, invariant_tensor_), dim=1), augment_pipe=augment_pipe)
                            else:
                                output_images = edm_sampler(net, latents=latents, condition=hrrr_0)
                                valid_loss = loss_fn(net=ddp, x=loss_target[0:n, diffusion_channel_indices], condition=hrrr_0, augment_pipe=augment_pipe)
                            if residual:
                                output_images += reg_out[0:n, diffusion_channel_indices, :, :]
                            del reg_out
                        else:
                            hrrr_0 = regression_net(hrrr_0[0:n], era5[0:n], mask=None)
                            latents = torch.randn_like(hrrr_1[0:n, diffusion_channel_indices, :, :])
                            if residual:
                                loss_target = hrrr_1 - hrrr_0
                            else:
                                loss_target = hrrr_1
                            if len(params.invariants) > 0:
                                output_images = edm_sampler(net, latents=latents, condition=torch.cat((hrrr_0, invariant_tensor_), dim=1))
                                valid_loss = loss_fn(net=ddp, x=loss_target[0:n, diffusion_channel_indices], condition=torch.cat((hrrr_0, invariant_tensor_), dim=1), augment_pipe=augment_pipe)
                            else:
                                output_images = edm_sampler(net, latents=latents, condition=hrrr_0)
                                valid_loss = loss_fn(net=ddp, x=loss_target[0:n, diffusion_channel_indices], condition=hrrr_0, augment_pipe=augment_pipe)
                            if residual:
                                if log_scale_residual:
                                    output_images = torch.sign(output_images) * (torch.exp(torch.abs(output_images)) - 1)
                                output_images += hrrr_0[0:n, diffusion_channel_indices, :, :]

                elif train_regression_unet:

                    assert use_regression_net == False, "use_regression_net must be False when training regression unet"
                    assert input_channel_indices == diffusion_channel_indices, "input_channel_indices must be equal to diffusion_channel_indices when training regression unet"

                    if len(params.invariants) > 0:
                        condition = torch.cat( (hrrr_0[0:n, input_channel_indices, :, :], era5[0:n], invariant_tensor_), dim=1) 
                    else:
                        condition = torch.cat( (hrrr_0[0:n, input_channel_indices, :, :], era5[0:n]), dim=1)

                    latents = torch.zeros_like(hrrr_1[0:n, diffusion_channel_indices, :, :], device=hrrr_1.device)
                    rnd_normal = torch.randn([latents.shape[0], 1, 1, 1], device=latents.device)
                    sigma = (rnd_normal * 1.2 - 1.2).exp() #this isn't used by the code
                    if loss_type == 'regression_v2':
                        output_images = net(sigma=sigma, condition=condition)                   
                    elif loss_type == 'regression':
                        output_images = net(x=latents, sigma=sigma, condition=condition)
                    valid_loss = loss_fn(net=ddp, x=hrrr_1[0:n, diffusion_channel_indices, :, :], condition=condition, augment_pipe=augment_pipe)
                    channelwise_valid_loss = valid_loss.mean(dim=[0, 2, 3])
                    channelwise_valid_loss_dict = { f"ChLoss_valid/{diffusion_channels[i]}": channelwise_valid_loss[i].item() for i in range(target_channels) }
                    if log_to_wandb:
                        wandb_logs['channelwise_valid_loss'] = channelwise_valid_loss_dict

                elif tendency_normalization:

                    assert use_regression_net == False, "use_regression_net must be False when using tendency normalization"
                    assert input_channel_indices == diffusion_channel_indices, "input_channel_indices must be equal to diffusion_channel_indices when using tendency normalization"
                    era5 = batch['era5'][0].to(device).to(torch.float32)
                    if len(params.invariants) > 0:
                        condition = torch.cat( (hrrr_0[0:n, input_channel_indices, :, :], era5[0:n], invariant_tensor_), dim=1) 
                    else:
                        condition = torch.cat( (hrrr_0[0:n, input_channel_indices, :, :], era5[0:n]), dim=1)
                    latents = torch.randn_like(hrrr_1[0:n, diffusion_channel_indices, :, :])

                    output_images = edm_sampler(net, latents=latents, condition=condition)
                    #valid_loss = loss_fn(net=ddp, x=hrrr_1, condition=condition, augment_pipe=augment_pipe)

                elif pure_diffusion:
                    hrrr_0 = hrrr_0[:, input_channel_indices, :, :]
                    assert input_channel_indices == diffusion_channel_indices, "input_channel_indices must be equal to diffusion_channel_indices when training direct diffusion model"
                    assert use_regression_net == False, "use_regression_net must be False when training direct diffusion model"

                    era5 = batch['era5'][0].to(device).to(torch.float32)
                    
                    latents = torch.randn_like(hrrr_1[0:n, diffusion_channel_indices, :, :])
                    
                    if len(params.invariants) > 0:
                        output_images = edm_sampler(net, latents=latents, condition=torch.cat((hrrr_0, era5, invariant_tensor_), dim=1))
                        valid_loss = loss_fn(net=ddp, x=hrrr_1[0:n, diffusion_channel_indices], condition=torch.cat((hrrr_0, era5, invariant_tensor_), dim=1), augment_pipe=augment_pipe)
                    else:
                        output_images = edm_sampler(net, latents=latents, condition=torch.cat((hrrr_0, era5), dim=1))
                        valid_loss = loss_fn(net=ddp, x=hrrr_1[0:n, diffusion_channel_indices], condition=torch.cat(hrrr_0, era5), augment_pipe=augment_pipe)
                    if residual:
                        output_images += hrrr_0[0:n, diffusion_channel_indices, :, :]
                
                hrrr_1 = hrrr_1[0:n, diffusion_channel_indices, :, :]

                training_stats.report('Loss/valid_loss', valid_loss.mean())

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(' '.join(fields))

        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0('Aborting...')

        # Save network snapshot.
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            data = dict(net=net, loss_fn=loss_fn, augment_pipe=augment_pipe)
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value # conserve memory
            if dist.get_rank() == 0:
                with open(os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl'), 'wb') as f:
                    pickle.dump(data, f)
                
            del data # conserve memory
        
        if cur_tick % params.validate_every == 0:

            if log_to_wandb:
                if dist.get_rank() == 0:
                    print("logging to wandb")
                    wandb.log( wandb_logs, step=cur_nimg )

            if dist.get_rank() == 0:
            #TODO: improve the image saving and run_dir setup for thread safe image saving from all ranks
                    
                for i in range(output_images.shape[0]):
                    image = output_images[i].cpu().numpy()
                    #hrrr_channels = dataset_train.hrrr_channels
                    fields = ["u10m", "v10m", "t2m", "refc", "q1", "q5", "q10"]

                    # Compute spectral metrics
                    figs, spec_ratios = compute_ps1d(output_images[i], hrrr_1[i], fields, diffusion_channels)
                    if log_to_wandb:
                        wandb.log(spec_ratios, step=cur_nimg)
                        for figname, fig in figs.items():
                            wandb.log({figname : wandb.Image(fig)}, step=cur_nimg)
                    
                    for f_ in fields:

                        f_index = diffusion_channels.index(f_)
                        image_dir = os.path.join(run_dir, "images", f_)
                        generated = image[f_index]
                        truth = hrrr_1[i, f_index].cpu().numpy()

                        fig, (a, b) = plt.subplots(1, 2)
                        im = a.imshow(generated)
                        a.set_title("generated, {}.png".format(f_) )
                        plt.colorbar(im, fraction=0.046, pad=0.04)
                        im = b.imshow(truth)
                        b.set_title("truth")
                        plt.colorbar(im, fraction=0.046, pad=0.04)
                        os.makedirs(image_dir, exist_ok=True)
                        plt.savefig(os.path.join(image_dir, f"{cur_tick}_{i}_{f_}.png"))
                        plt.close('all')

                        specfig = 'PS1D_'+f_
                        figs[specfig].savefig(os.path.join(image_dir, f"{cur_tick}{i}{f_}_spec.png"))
                        plt.close(figs[specfig])

                        #log the images to wandb
                        if log_to_wandb:
                            #log fig to wandb
                            wandb.log( {f"generated_{f_}": fig}, step=cur_nimg )


        # Save full dump of the training state.
        if loss_type == 'edm':
            if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and cur_tick != 0 and dist.get_rank() == 0:
                torch.save(dict(net=net,
                                ema1=ema1,
                                ema2=ema2, 
                                optimizer_state=optimizer.state_dict(),
                                total_steps=total_steps), os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt'))
        else:
            if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and cur_tick != 0 and dist.get_rank() == 0:
                torch.save(dict(net=net,
                                optimizer_state=optimizer.state_dict(),
                                total_steps=total_steps), os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt'))

        # Update logs.
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
            
            stats_dict = dict(training_stats.default_collector.as_dict(), timestamp=time.time())
            if True:
                wandb_logs['loss'] = stats_dict['Loss/loss']['mean']
                wandb_logs['valid_loss'] = stats_dict['Loss/valid_loss']['mean']
                print("loss: ", wandb_logs['loss'])
                print("valid_loss: ", wandb_logs['valid_loss'])
            stats_jsonl.write(json.dumps(stats_dict) + '\n')
            stats_jsonl.flush()
        dist.update_progress(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    dist.print0()
    dist.print0('Exiting...')
