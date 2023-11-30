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

"""Main training loop."""

import os
import sys
import time
import copy
import json
import pickle
import psutil
import numpy as np
import torch
import dnnlib
from torch.nn.parallel import DistributedDataParallel
from torch_utils import training_stats
from torch_utils import misc

from modulus.distributed import DistributedManager
from modulus.launch.logging import PythonLogger, RankZeroLoggingWrapper

#weather related
from .YParams import YParams
#from .dataset_old import Era5Dataset, CWBDataset, CWBERA5DatasetV2, ZarrDataset
from .dataset import Era5Dataset, CWBDataset, CWBERA5DatasetV2, ZarrDataset

import glob

#----------------------------------------------------------------------------

def training_loop(
    run_dir             = '.',      # Output directory.
    dataset_kwargs      = {},       # Options for training set.
    data_loader_kwargs  = {},       # Options for torch.utils.data.DataLoader.
    network_kwargs      = {},       # Options for model and preconditioning.
    loss_kwargs         = {},       # Options for loss function.
    optimizer_kwargs    = {},       # Options for optimizer.
    augment_kwargs      = None,     # Options for augmentation pipeline, None = disable.
    seed                = 0,        # Global random seed.
    batch_size          = 512,      # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU, None = no limit.
    total_kimg          = 200000,   # Training duration, measured in thousands of training images.
    ema_halflife_kimg   = 500,      # Half-life of the exponential moving average (EMA) of model weights.
    ema_rampup_ratio    = 0.05,     # EMA ramp-up coefficient, None = no rampup.
    lr_rampup_kimg      = 10000,    # Learning rate ramp-up duration.
    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    kimg_per_tick       = 50,       # Interval of progress prints.
    snapshot_ticks      = 50,       # How often to save network snapshots, None = disable.
    state_dump_ticks    = 500,      # How often to dump training state, None = disable.
    resume_pkl          = None,     # Start from the given network snapshot, None = random initialization.
    resume_state_dump   = None,     # Start from the given training state, None = reset training state.
    resume_kimg         = 0,        # Start from the given training progress.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    device              = torch.device('cuda'),
    data_type           = None,
    data_config         = None,
    task                = None,
):
    
    # Instantiate distributed manager.
    dist = DistributedManager()

    # Initialize logger.
    logger = PythonLogger(name="training_loop")  # General python logger
    logger0 = RankZeroLoggingWrapper(logger, dist)
    logger.file_logging(file_name="training_loop.log")

    # Initialize.
    start_time = time.time()

    device = dist.device
    np.random.seed((seed * dist.world_size + dist.rank) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.world_size
    logger0.info(f'batch_gpu: {batch_gpu}')
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.world_size

    '''
    # Load dataset: cifar10
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # subclass of training.dataset.Dataset
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.rank, num_replicas=dist.world_size, seed=seed)
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))
    '''
    
    # Load dataset: weather
    logger0.info('Loading dataset...')
    yparams = YParams(data_type + '.yaml', config_name=data_config)

    
    if data_type == 'era5': 
        dataset_obj = Era5Dataset(yparams, yparams.train_data_path, train=True, task=task)
        worker_init_fn = None
    elif data_type == 'cwb':
        dataset_obj = CWBDataset(yparams, yparams.train_data_path, train=True, task=task)
        worker_init_fn = None
    elif data_type == 'era5-cwb-v1':
        #filelist = os.listdir(path=yparams.cwb_data_dir + '/2018') 
        #filelist = [name for name in filelist if "2018" in name]
        filelist = []
        for root, dirs, files in os.walk(yparams.cwb_data_dir):
            for file in files:
                if '2022' not in file:
                    filelist.append(file)
        dataset_obj = CWBERA5DatasetV2(yparams, filelist=filelist, chans=list(range(20)), train=True, task=task)  
        worker_init_fn = dataset_obj.worker_init_fn 
    elif data_type == 'era5-cwb-v2':
        dataset_obj = ZarrDataset(yparams, yparams.train_data_path, train=True)
        worker_init_fn = None
    elif data_type == 'era5-cwb-v3':
        dataset_obj = ZarrDataset(yparams, yparams.train_data_path, train=True)
        #worker_init_fn = dataset_obj.worker_init_fn 
        worker_init_fn = None

    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.rank, num_replicas=dist.world_size, seed=seed)
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, worker_init_fn=worker_init_fn, **data_loader_kwargs))

    
    img_in_channels = len(yparams.in_channels)   #noise + low-res input
    if yparams.add_grid:
            img_in_channels = img_in_channels + yparams.N_grid_channels
        
    img_out_channels = len(yparams.out_channels)
    
    # if use_mean_input:  #add it to the args and store_true in yaml file
    #     img_in_channels = img_in_channels + yparams.N_grid_channels + img_out_channels
    
    # Construct network.
    logger0.info('Constructing network...')
    #interface_kwargs = dict(img_resolution=dataset_obj.resolution, img_channels=dataset_obj.num_channels, label_dim=dataset_obj.label_dim)    #cifar10
    interface_kwargs = dict(img_resolution=yparams.crop_size_x, img_channels=img_out_channels, img_in_channels=img_in_channels, img_out_channels=img_out_channels, label_dim=0)    #weather
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs) # subclass of torch.nn.Module
    net.train().requires_grad_(True).to(device)
    
    # if dist.rank == 0:
    #     with torch.no_grad():
    #         img_clean = torch.zeros([batch_gpu, img_out_channels, net.img_resolution, net.img_resolution], device=device)
    #         img_lr = torch.zeros([batch_gpu, img_in_channels, net.img_resolution, net.img_resolution], device=device)
    #         sigma = torch.ones([batch_gpu], device=device)
    #         labels = torch.zeros([batch_gpu, net.label_dim], device=device)
    #         misc.print_module_summary(net, [img_clean, img_lr, sigma, labels], max_nesting=2)
    
    #import pdb; pdb.set_trace()
    #breakpoint()
            
    # params = net.parameters()
    # print('************************************')
    # print('dist.rank', dist.rank)
    # print('net.parameters()', net.parameters())
    # for idx, param in enumerate(net.parameters()):
    #     if idx == 230:
    #         print(f"Parameter {idx}: {param.stride()}")
    #         print(f"Parameter {idx}: {param.shape}")
    #         break
    # print('************************************')
    

    # Setup optimizer.
    logger0.info('Setting up optimizer...')
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs) # training.loss.(VP|VE|EDM)Loss
    optimizer = dnnlib.util.construct_class_by_name(params=net.parameters(), **optimizer_kwargs) # subclass of torch.optim.Optimizer
    augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs) if augment_kwargs is not None else None # training.augment.AugmentPipe
    if dist.world_size > 1:
        ddp = DistributedDataParallel(net, device_ids=[dist.local_rank], broadcast_buffers=True, output_device=dist.device, find_unused_parameters=dist.find_unused_parameters)
    ema = copy.deepcopy(net).eval().requires_grad_(False)   
    
    
    # Import autoresume module
    #print('os.environ', print(os.environ))
    # sys.path.append(os.environ.get('SUBMIT_SCRIPTS', '.'))
    SUBMIT_SCRIPTS = '/lustre/fsw/adlr/adlr-others/gpeled/adlr-utils/release/cluster-interface/latest'
    sys.path.append(SUBMIT_SCRIPTS)
    #sync autoresums across gpus ...
    AutoResume = None
    try:
        from userlib.auto_resume import AutoResume
        AutoResume.init()
    except ImportError:
        logger0.warning('AutoResume not imported')
        
    # Resume training from previous snapshot.
    if resume_pkl is not None:
        logger0.info(f'Loading network weights from "{resume_pkl}"...')
        if dist.rank != 0:
            torch.distributed.barrier() # rank 0 goes first
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.rank == 0)) as f:
            data = pickle.load(f)
        if dist.rank == 0:
            torch.distributed.barrier() # other ranks follow
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=net, require_all=False)
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=ema, require_all=False)
        del data # conserve memory
    if resume_state_dump:
        logger0.info(f'Loading training state from "{resume_state_dump}"...')
        data = torch.load(resume_state_dump, map_location=torch.device('cpu'))
        misc.copy_params_and_buffers(src_module=data['net'], dst_module=net, require_all=True)
        #dist.print0('data-optimizer', data['optimizer_state'])
        #import pdb; pdb.set_trace()
        optimizer.load_state_dict(data['optimizer_state'])
        del data # conserve memory
        
    
    # #check num params per gpu
    # with open(f"params_{dist.rank}.txt", "w") as fo:
    #     dist.print0(net.parameters())
    #     for param in net.parameters():
    #         dist.print0(param.size())
    #         #fo.write(f"{name}\t{param.size()}\n")
    # import pdb; pdb.set_trace()
        
    # Train.
    logger0.info(f'Training for {total_kimg} kimg...')
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    # dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None
    while True:

        # Accumulate gradients.
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(ddp, (round_idx == num_accumulation_rounds - 1)):
                
                # Fetch training data: weather
                img_clean, img_lr, labels = next(dataset_iterator)
                
                # dist.print0(img_clean.shape)
                # dist.print0('max-clean', torch.max(img_clean))
                # dist.print0('min-clean', torch.min(img_clean))
                # dist.print0('mean-clean', torch.mean(img_clean))
                # dist.print0('std-clean', torch.std(img_clean))    
                # dist.print0(img_lr.shape)
                # dist.print0('max-lr', torch.max(img_lr))
                # dist.print0('min-lr', torch.min(img_lr))
                # dist.print0('mean-lr', torch.mean(img_lr))
                # dist.print0('std-lr', torch.std(img_lr))   
                # import pdb; pdb.set_trace()          

                # Normalization: weather (normalized already in the dataset)
                img_clean = img_clean.to(device).to(torch.float32).contiguous()   #[-4.5, +4.5]
                img_lr = img_lr.to(device).to(torch.float32).contiguous() 
                labels = labels.to(device).contiguous() 

                # # Fetch training data: cifar10
                # images, labels = next(dataset_iterator)
                # # Normalization: cifar10 (normalized already in the dataset)
                # images = images.to(device).to(torch.float32) / 127.5 - 1
                # labels = labels.to(device)

                loss = loss_fn(net=ddp, img_clean=img_clean, img_lr=img_lr, labels=labels, augment_pipe=augment_pipe)
                training_stats.report('Loss/loss', loss)
                loss.sum().mul(loss_scaling / batch_gpu_total).backward()
                


        # Update weights.
        for g in optimizer.param_groups:
            g['lr'] = optimizer_kwargs['lr'] * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)
        for param in net.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)
        optimizer.step()

        # Update EMA.
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
        ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        for p_ema, p_net in zip(ema.parameters(), net.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        # Perform maintenance tasks once per tick.
        cur_nimg += batch_size
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

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
        logger0.info(' '.join(fields))
                
        ckpt_dir = run_dir
        
        logger0.info(f'AutoResume.termination_requested(): {AutoResume.termination_requested()}')
        logger0.info(f'AutoResume: {AutoResume}')

        if AutoResume.termination_requested():
            AutoResume.request_resume()
            logger0.info("Training terminated. Returning")
            done = True
            #print('dist.rank', dist.rank)
            #with open(os.path.join(os.path.split(ckpt_dir)[0],'resume.txt'), "w") as f: 
            with open(os.path.join(ckpt_dir,'resume.txt'), "w") as f:
                f.write(os.path.join(ckpt_dir, f'training-state-{cur_nimg//1000:06d}.pt'))
                logger0.info(os.path.join(ckpt_dir, f'training-state-{cur_nimg//1000:06d}.pt'))
                f.close()
                #return 0

        # Check for abort.
        logger0.info(f'dist.should_stop(): {dist.should_stop()}')
        logger0.info(f'done: {done}')

        # if (not done) and dist.should_stop():
        #     done = True
        #     dist.print0()
        #     dist.print0('Aborting...')

        # Save network snapshot.
        if (snapshot_ticks is not None) and (done or cur_tick % snapshot_ticks == 0):
            data = dict(ema=ema, loss_fn=loss_fn, augment_pipe=augment_pipe, dataset_kwargs=dict(dataset_kwargs))
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value # conserve memory
            if dist.rank == 0:
                with open(os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl'), 'wb') as f:
                    pickle.dump(data, f)
            del data # conserve memory

        # Save full dump of the training state.
        #if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and cur_tick != 0 and dist.rank == 0:
        if (state_dump_ticks is not None) and (done or cur_tick % state_dump_ticks == 0) and dist.rank == 0:
            torch.save(dict(net=net, optimizer_state=optimizer.state_dict()), os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt'))

        # Update logs.
        training_stats.default_collector.update()
        if dist.rank == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
            stats_jsonl.write(json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n')
            stats_jsonl.flush()
        # dist.update_progress(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break
            
    
    # Done.
    logger0.info('Exiting...')

#----------------------------------------------------------------------------
