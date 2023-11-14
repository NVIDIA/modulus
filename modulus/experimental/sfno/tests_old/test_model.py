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

import os, sys
sys.path.insert(0, os.path.join(sys.path[0], '..'))

import numpy as np
import argparse
import logging

# gpu info
import pynvml

# torch related tools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp

from modulus.experimental.sfno.utils import logging_utils
from modulus.experimental.sfno.utils.YParams import YParams
from modulus.experimental.sfno.networks.preprocessor import get_preprocessor
from modulus.experimental.sfno.networks.models import get_model
from modulus.experimental.sfno.utils.dataloader import get_dataloader
from apex import optimizers

# distributed modules
from modulus.experimental.sfno.utils import comm
import torch.distributed as dist
import torch.distributed.autograd as dist_autograd
from modulus.experimental.sfno.mpu.mappings import init_gradient_reduction_hooks
from modulus.experimental.sfno.mpu.helpers import sync_params

from modulus.experimental.sfno.utils.losses import LossHandler
from modulus.experimental.sfno.utils.metric import MetricsHandler
from modulus.experimental.sfno.networks.helpers import count_parameters
from modulus.experimental.sfno.mpu.mappings import scatter_to_parallel_region, gather_from_parallel_region

class ModelTester():

    def _update_parameters(self, params):
        """
        This could be moved potentially. The idea is to process params and handle the logics for params
        """

        params.in_channels = self.train_dataset.in_channels
        params.N_in_channels = len(self.train_dataset.in_channels)
        params.out_channels = self.train_dataset.out_channels
        params.N_out_channels = len(self.train_dataset.out_channels)

        params.img_shape_x = self.train_dataset.img_shape_x
        params.img_shape_y = self.train_dataset.img_shape_y
        
        params.img_crop_shape_x = self.train_dataset.img_crop_shape_x
        params.img_crop_shape_y = self.train_dataset.img_crop_shape_y
        params.img_crop_offset_x = self.train_dataset.img_crop_offset_x
        params.img_crop_offset_y = self.train_dataset.img_crop_offset_y 

        params.img_local_shape_x = self.train_dataset.img_local_shape_x
        params.img_local_shape_y = self.train_dataset.img_local_shape_y
        params.img_local_offset_x = self.train_dataset.img_local_offset_x
        params.img_local_offset_y = self.train_dataset.img_local_offset_y
        
        # sanitization:
        if not hasattr(params, 'add_zenith'):
            params["add_zenith"] = False

        # input channels
        # zenith channel is appended to all the samples, so we need to do it here
        if params.add_zenith:
            params.N_in_channels += 1

        if params.n_history >= 1:
            params.N_in_channels = (params.n_history + 1) * params.N_in_channels

        # these are static and the same for all samples in the same time history
        if params.add_grid:
            params.N_in_channels += 2

        if params.add_orography:
            params.N_in_channels += 1
            
        if params.add_landmask:
            params.N_in_channels += 2

        # target channels
        params.N_target_channels = (params.n_future + 1) * params.N_out_channels

        # MISC parameters
        if not hasattr(params, 'num_visualization_workers'):
            params['num_visualization_workers'] = 1

        if not hasattr(params, 'log_video'):
            params['log_video'] = 0

        # automatically detect wind channels and keep track of them 
        if hasattr(params, 'channel_names') and not hasattr(params, 'wind_channels'):
            channel_names = params.channel_names
            channel_dict = { channel_names[ch] : ch for ch in set(params.in_channels + params.out_channels)}
            wind_channels = []
            for chn, ch in channel_dict.items():
                if chn[0] == 'u':
                    vchn = 'v' + chn[1:]
                    if vchn in channel_dict.keys():
                        # wind_channels.append(ch, channel_dict[vchn])
                        wind_channels = wind_channels + [ch, channel_dict[vchn]]
            params['wind_channels'] = wind_channels

        return params

    def _get_time_stats(self):

        # get some stats: make data shared with tensor from the class
        _, out_scale = self.train_dataloader.get_output_normalization()
        mult_cpu = torch.from_numpy(out_scale)[0, :, 0, 0]

        # compute
        if os.path.isfile(self.params.time_means_path):

            # full bias and scale
            in_bias, in_scale = self.train_dataloader.get_input_normalization()
            in_bias = in_bias[0, ...] #np.load(self.params.global_means_path)[0, self.params.out_channels]
            in_scale =  in_scale[0, ...] #np.load(self.params.global_stds_path)[0, self.params.out_channels]

            # we need this window
            start_x = self.params.img_crop_offset_x
            end_x = start_x + self.params.img_crop_shape_x
            start_y = self.params.img_crop_offset_y
            end_y = start_y + self.params.img_crop_shape_y

            # now we crop the time means
            time_means = np.load(self.params.time_means_path)[0, self.params.out_channels, start_x:end_x, start_y:end_y]
            clim = torch.as_tensor((time_means - in_bias) / in_scale, dtype=torch.float32)

        elif self.params.enable_synthetic_data:
            clim = torch.zeros([self.params.N_out_channels, self.params.img_crop_shape_x, self.params.img_crop_shape_y], dtype=torch.float32, device=self.device)

        else:
            raise IOError(f'stats file {self.params.global_means_path} or {self.params.global_stds_path} not found')

        return mult_cpu, clim
    
    def __init__(self, params, world_rank):

        self.params = None
        self.world_rank = world_rank
        self.data_parallel_rank = comm.get_rank("data")
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        else:
            self.device = torch.device("cpu")

        # nvml stuff
        if params.log_to_screen:
            pynvml.nvmlInit()
            self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self.device.index)

        # set amp_parameters
        self.amp_enabled = (params.amp_mode != "none")
        self.amp_dtype = torch.float16 if (params.amp_mode == "fp16") else torch.bfloat16 if (params.amp_mode == "bf16") else None

        # data loader
        if params.log_to_screen:
            logging.info('initializing data loader')
        self.train_dataloader, self.train_dataset, self.train_sampler = get_dataloader(
            params,
            params.train_data_path,
            train=True,
            device=self.device
            )


        if params.log_to_screen:
            logging.info('data loader initialized')

        # update params
        params = self._update_parameters(params)

        # save params
        self.params = params

        # init preprocessor and model
        self.model = get_model(params).to(self.device)
        self.preprocessor = get_preprocessor(params).to(self.device)

        # if model-parallelism is enabled, we need to sure that shared weights are matching across ranks
        # as random seeds might get out of sync during initialization
        if comm.get_size("model") > 1:
            sync_params(self.model)

        # define process group for DDP, we might need to override that
        if dist.is_initialized() and not params.disable_ddp:
            ddp_process_group = comm.get_group("data")

        # print model
        if self.world_rank == 0:
            print(self.model)

        # metrics handler
        mult_cpu, clim = self._get_time_stats()
        self.metrics = MetricsHandler(self.params, mult_cpu, clim, self.device)

        # loss handler
        self.loss_obj = LossHandler(self.params)
        self.loss_obj = self.loss_obj.to(self.device)
        if self.params.enable_nhwc:
            self.loss_obj = self.loss_obj.to(memory_format=torch.channels_last)

        self.capturable_optimizer = False
        betas = (params.optimizer_beta1, params.optimizer_beta2)
        if params.optimizer_type == 'FusedAdam':
            logging.info("using FusedAdam")
            self.optimizer = optimizers.FusedAdam(self.model.parameters(),
                                                  betas = betas,
                                                  lr = params.lr,
                                                  weight_decay = params.weight_decay)
        elif params.optimizer_type == 'FusedLAMB':
            try:
                import doesnotexist
                from apex.optimizers import FusedMixedPrecisionLamb
                logging.info("using FusedMixedPrecisionLamb")
                self.optimizer = FusedMixedPrecisionLamb(self.model.parameters(),
                                                         betas = betas,
                                                         lr = params.lr,
                                                         weight_decay = params.weight_decay,
                                                         max_grad_norm = params.optimizer_max_grad_norm)
                self.capturable_optimizer = True
            except ImportError:
                logging.info("using FusedLAMB")
                self.optimizer = optimizers.FusedLAMB(self.model.parameters(),
                                                      betas = betas,
                                                      lr = params.lr,
                                                      weight_decay = params.weight_decay,
                                                      max_grad_norm = params.optimizer_max_grad_norm)
        else:
            logging.info("using Adam")
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr = params.lr)

        if params.scheduler == 'ReduceLROnPlateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.2, patience=5, mode='min')
        elif params.scheduler == 'CosineAnnealingLR':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=params.scheduler_T_max)
        else:
            self.scheduler = None
        if params.lr_warmup_steps > 0:
            from utils.warmup_scheduler import WarmupScheduler
            self.scheduler = WarmupScheduler(self.scheduler, num_warmup_steps = params.lr_warmup_steps, start_lr = params.lr_start)

        self.gscaler = amp.GradScaler(enabled = (self.amp_dtype == torch.float16))

        # we need this further down
        capture_stream = None
        if dist.is_initialized() and not params.disable_ddp:
            capture_stream = torch.cuda.Stream()
            parameter_size_mb = count_parameters(self.model, self.device) * 4 / float(1024 * 1024)
            reduction_size_mb = int((parameter_size_mb / params.parameters_reduction_buffer_count) * 1.05) #math.ceil(parameter_size_mb / 2.)
            with torch.cuda.stream(capture_stream):
                self.model = init_gradient_reduction_hooks(self.model,
                                                           device_ids = [self.device.index],
                                                           output_device = [self.device.index],
                                                           bucket_cap_mb = reduction_size_mb,
                                                           broadcast_buffers = True,
                                                           find_unused_parameters = False,
                                                           gradient_as_bucket_view = True,
                                                           static_graph = params.checkpointing > 0)
                capture_stream.synchronize()

                # we need to set up some additional gradient reductions
                #if params.model_parallel_size > 1:
                #    init_additional_parameters_reductions(self.model)

            # capture stream sync
            capture_stream.synchronize()

        # jit compile
        inp_shape = (self.params.batch_size, self.params.N_in_channels, self.params.img_local_shape_x, self.params.img_local_shape_y)

        # self._compile_model(inp_shape)
        # if not self.loss_obj.is_distributed():
        #     self.loss_obj = torch.jit.script(self.loss_obj)
        self.model_train = self.model
        self.model_eval = self.model

        # graph capture
        self.graph = None
        if params.cuda_graph_mode != "none":
            tar_shape = (self.params.batch_size, self.params.N_target_channels, self.params.img_local_shape_x, self.params.img_local_shape_y)
            self._capture_model(capture_stream, inp_shape, tar_shape, num_warmup_steps=20)

        # reload checkpoints
        self.iters = 0
        self.startEpoch = 0
        if params.finetune and not params.resuming:
            assert(params.pretrained_checkpoint_path is not None), "Error, please specify a valid pretrained checkpoint path"
            self.restore_checkpoint(params.pretrained_checkpoint_path)
        
        if params.resuming:
            self.restore_checkpoint(params.checkpoint_path)

        self.epoch = self.startEpoch

        # if params.log_to_screen:
        #   logging.info(self.model)

        # counting runs a reduction so we need to count on all ranks before printing on rank 0
        pcount = count_parameters(self.model, self.device)
        if params.log_to_screen:
            logging.info("Number of trainable model parameters: {}".format(pcount))

        # for scattering and gathering
        self.nlat = self.params.img_shape_x
        self.nlon = self.params.img_shape_y
        comm_size_h = comm.get_size("h")
        comm_size_w = comm.get_size("w")
        latdist = (self.nlat + comm_size_h - 1) // comm_size_h
        self.nlatpad = latdist * comm_size_h - self.nlat
        londist = (self.nlon + comm_size_w - 1) // comm_size_w
        self.nlonpad = londist * comm_size_w - self.nlon

    @torch.jit.ignore
    def _scatter_hw(self, x: torch.Tensor) -> torch.Tensor:
        # start by correctly padding the domain
        xp = F.pad(x, [0, self.nlonpad, 0, self.nlatpad])
        xh = scatter_to_parallel_region(xp, -2, "h")
        x = scatter_to_parallel_region(xh, -1, "w")
        
        return x

    @torch.jit.ignore
    def _gather_hw(self, x: torch.Tensor) -> torch.Tensor:
	    # gather the data over the spatial communicator
        xh = gather_from_parallel_region(x, -2, "h")
        xw = gather_from_parallel_region(xh, -1, "w")
        x = xw[..., :self.nlat, :self.nlon]
        return x

    def run_test(self):
        if self.params.log_to_screen:
            # log memory usage so far
            all_mem_gb = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle).used / (1024. * 1024. * 1024.)
            max_mem_gb = torch.cuda.max_memory_allocated(device=self.device) / (1024. * 1024. * 1024.)
            logging.info(f"Scaffolding memory high watermark: {all_mem_gb} GB ({max_mem_gb} GB for pytorch)")
            # announce training start
            logging.info("Starting Forward pass...")

        # perform a barrier here to make sure everybody is ready
        if dist.is_initialized():
            dist.barrier(device_ids=[self.device.index])

        batch_size = self.params.batch_size
        in_chans = self.params.N_in_channels
        out_chans = self.params.N_out_channels

        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        sample = torch.randn(batch_size, in_chans, self.nlat, self.nlon, device=self.device)
        print(f"INPUT | rank: {self.world_rank}, shape: {sample.shape}, min: {torch.min(sample)}, max: {torch.max(sample)}")
        sample = self._scatter_hw(sample)

        torch.manual_seed(1)
        torch.cuda.manual_seed(1)
        target = torch.randn(batch_size, in_chans, self.nlat, self.nlon, device=self.device)
        target = self._scatter_hw(target)

        print(f"ENCODER0 WEIGHT | rank: {self.world_rank}, weight first: {self.model.module.encoder[0].weight[0,0,0]}, weight last: {self.model.module.encoder[0].weight[-1,-1,-1]}")
        print(f"ENCODER0 WEIGHT | rank: {self.world_rank}, sum: {torch.sum(self.model.module.encoder[0].weight)}, min: {torch.min(self.model.module.encoder[0].weight)}, max: {torch.max(self.model.module.encoder[0].weight)}")
        print(f"POSEMBED WEIGHT | rank: {self.world_rank}, weight first: {self.model.module.pos_embed[0,0,0,0]}, weight last: {self.model.module.pos_embed[-1,-1,-1,-1]}")
        print(f"DECODER0 WEIGHT | rank: {self.world_rank}, weight first: {self.model.module.decoder[0].weight[0,0,0]}, weight last: {self.model.module.decoder[0].weight[-1,-1,-1]}")
        # print(f"MLP WEIGHT      | rank: {self.world_rank}, weight first: {self.model.module.blocks[0].mlp.fwd[0].weight[0,0]}, weight last: {self.model.module.blocks[0].mlp.fwd[0].weight[-1,-1]}")

        output = self.model(sample)

        with amp.autocast(enabled = self.amp_enabled, dtype = self.amp_dtype):
            prediction = self.model(sample)
            loss = self.loss_obj(prediction, target, sample)

        output = self._gather_hw(prediction)
        print(f"OUTPUT | rank: {self.world_rank}, shape: {output.shape}, min: {torch.min(output)}, max: {torch.max(output)}")

        print(f"LOSS | rank: {self.world_rank}, loss: {loss}")

        # self.gscaler.scale(loss).backward()
        loss.backward()

        # torch.cuda.synchronize(device=self.device)
        # if dist.is_initialized():
        #     dist.barrier(device_ids=[self.device.index], async_op=False)

        # with dist_autograd.context() as context_id:
        #     prediction = self.model(sample)
        #     loss = self.loss_obj(prediction, target, sample)
        #     dist_autograd.backward(context_id, [loss])

        print(f"ENCODER0 GRAD | rank: {self.world_rank}, weight first: {self.model.module.encoder[0].weight.grad.cpu()[0,0,0]}, weight last: {self.model.module.encoder[0].weight.grad.cpu()[-1,-1,-1]}")
        print(f"POSEMBED GRAD | rank: {self.world_rank}, weight first: {self.model.module.pos_embed.grad.cpu()[0,0,0,0]}, weight last: {self.model.module.pos_embed.grad.cpu()[-1,-1,-1,-1]}")
        print(f"DECODER0 GRAD | rank: {self.world_rank}, weight first: {self.model.module.decoder[0].weight.grad.cpu()[0,0,0]}, weight last: {self.model.module.decoder[0].weight.grad.cpu()[-1,-1,-1]}")
        # print(f"FNO GRAD      | rank: {self.world_rank}, weight first: {self.model.module.blocks[0].filter.filter.weight.grad.cpu()[0,0,0]}, weight last: {self.model.module.blocks[0].filter.filter.weight.grad.cpu()[-1,-1,-1]}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--matmul_parallel_size", default=1, type=int, help="Matmul parallelism dimension, only applicable to AFNO")
    parser.add_argument("--h_parallel_size", default=1, type=int, help="Spatial parallelism dimension in h")
    parser.add_argument("--w_parallel_size", default=1, type=int, help="Spatial parallelism dimension in w")
    parser.add_argument("--parameters_reduction_buffer_count", default=1, type=int, help="How many buffers will be used (approximately) for weight gradient reductions.") 
    parser.add_argument("--run_num", default='00', type=str)
    parser.add_argument("--yaml_config", default='./config/afnonet.yaml', type=str)
    parser.add_argument("--batch_size", default=-1, type=int, help="Switch for overriding batch size in the configuration file.")
    parser.add_argument("--config", default='default', type=str)
    parser.add_argument("--enable_synthetic_data", action='store_true')
    parser.add_argument("--amp_mode", default='none', type=str, choices=["none", "fp16", "bf16"], help="Specify the mixed precision mode which should be used.")
    parser.add_argument("--jit_mode", default='none', type=str, choices=["none", "script", "trace"], help="Specify if and how to use torch jit.") 
    parser.add_argument("--cuda_graph_mode", default='none', type=str, choices=["none", "fwdbwd", "step"], help="Specify which parts to capture under cuda graph")
    parser.add_argument("--enable_benchy", action='store_true')
    parser.add_argument("--disable_ddp", action='store_true')
    parser.add_argument("--enable_nhwc", action='store_true')
    parser.add_argument("--checkpointing_level", default=0, type=int)
    parser.add_argument("--epsilon_factor", default = 0, type = float)
    parser.add_argument("--split_data_channels", action='store_true')
    parser.add_argument("--print_timings_frequency", default=-1, type=int, help="Frequency at which to print timing information")

    # multistep stuff
    parser.add_argument("--multistep_count", default=1, type=int, help="Number of autoregressive training steps. A value of 1 denotes conventional training")

    # parse
    args = parser.parse_args()

    # parse parameters
    params = YParams(os.path.abspath(args.yaml_config), args.config)
    params['epsilon_factor'] = args.epsilon_factor

    # distributed
    params["matmul_parallel_size"] = args.matmul_parallel_size
    params["h_parallel_size"] = args.h_parallel_size
    params["w_parallel_size"] = args.w_parallel_size

    params["model_parallel_sizes"] = [args.h_parallel_size, args.w_parallel_size, args.matmul_parallel_size]
    params["model_parallel_names"] = ["h", "w", "matmul"]
    params["parameters_reduction_buffer_count"] = args.parameters_reduction_buffer_count

    # make sure to reconfigure logger after the pytorch distributed init
    comm.init(params, verbose=False)

    world_rank = comm.get_world_rank()

    # update parameters
    params["world_size"] = comm.get_world_size()
    if args.batch_size > 0:
        params.batch_size = args.batch_size
    params['global_batch_size'] = params.batch_size
    assert (params['global_batch_size'] % comm.get_size("data") == 0), f"Error, cannot evenly distribute {params['global_batch_size']} across {comm.get_size('data')} GPU."
    params['batch_size'] = int(params['global_batch_size'] // comm.get_size("data"))

    # optimizer params
    if "optimizer_max_grad_norm" not in params:
        params["optimizer_max_grad_norm"] = 1.

    # set device
    torch.cuda.set_device(comm.get_local_rank())
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # DEBUG
    #torch.autograd.set_detect_anomaly(True)
    # DEBUG

    # Set up directory
    expDir = os.path.join(params.exp_dir, args.config, str(args.run_num))
    if world_rank == 0:
        logging.info(f'writing output to {expDir}')
        if not os.path.isdir(expDir):
            os.makedirs(expDir, exist_ok=True)
            os.makedirs(os.path.join(expDir, 'training_checkpoints'), exist_ok=True)

    params['experiment_dir'] = os.path.abspath(expDir)
    params['checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/ckpt_mp{mp_rank}.tar')
    params['best_checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/best_ckpt_mp{mp_rank}.tar')

    # Do not comment this line out please:
    # check if all files are there
    args.resuming = True
    for mp_rank in range(comm.get_size("model")):
        checkpoint_fname = params.checkpoint_path.format(mp_rank=mp_rank)
        args.resuming = args.resuming and os.path.isfile(checkpoint_fname)

    params['resuming'] = args.resuming
    params['amp_mode'] = args.amp_mode
    params['jit_mode'] = args.jit_mode
    params['cuda_graph_mode'] = args.cuda_graph_mode
    params['enable_benchy'] = args.enable_benchy
    params['disable_ddp'] = args.disable_ddp
    params['enable_nhwc'] = args.enable_nhwc
    params['checkpointing'] = args.checkpointing_level
    params['enable_synthetic_data'] = args.enable_synthetic_data
    params['split_data_channels'] = args.split_data_channels
    params['print_timings_frequency'] = args.print_timings_frequency
    params['multistep_count'] = args.multistep_count
    params['n_future'] = args.multistep_count - 1 # note that n_future counts only the additional samples

    # wandb configuration
    if params['wandb_name'] is None:
        params['wandb_name'] = args.config + '_' + str(args.run_num)
    if params['wandb_group'] is None:
        params['wandb_group'] = "era5_wind" + args.config

    if world_rank==0:
        logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, 'out.log'))
        logging_utils.log_versions()
        params.log()

    params['log_to_screen'] = (world_rank==0) and params['log_to_screen']

    #params['in_channels'] = np.array(params['in_channels'])
    #params['out_channels'] = np.array(params['out_channels'])
    #params['N_in_channels'] = len(params['in_channels'])
    #params['N_out_channels'] = len(params['out_channels'])

    # instantiate trainer object
    test = ModelTester(params, world_rank)
    test.run_test()
