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

import os
import gc
import time
import numpy as np
from tqdm import tqdm
import json

# gpu info
import pynvml

# torch
import torch
import torch.cuda.amp as amp

import logging
import wandb

from modulus.experimental.sfno.networks.models import get_model
from modulus.experimental.sfno.utils.dataloader import get_dataloader
from modulus.experimental.sfno.mpu.mappings import init_gradient_reduction_hooks
from modulus.experimental.sfno.mpu.helpers import sync_params, gather_uneven
from modulus.experimental.sfno.utils.losses import LossHandler
from modulus.experimental.sfno.utils.metric import MetricsHandler

# distributed computing stuff
from modulus.experimental.sfno.utils import comm
from modulus.experimental.sfno.utils import visualize
import torch.distributed as dist

# for the manipulation of state dict
from collections import OrderedDict

# for counting model parameters
from modulus.experimental.sfno.networks.helpers import count_parameters

# Custom optimizers to to enable optimization of complex parameters
from modulus.experimental.sfno.third_party.torch.optim.adam import Adam as CustomAdam
from modulus.experimental.sfno.third_party.torch.optim.adamw import AdamW as CustomAdamW


# profile stuff
from ctypes import cdll
libcudart = cdll.LoadLibrary('libcudart.so')

def cudaProfilerStart(device, enabled=True):
    # global barrier to help profile alignment
    if dist.is_initialized():
        dist.barrier(device_ids=[device.index])
    if enabled:
        libcudart.cudaProfilerStart()

def cudaProfilerStop(device, enabled=True):
    if enabled:
        libcudart.cudaProfilerStop()
    # global barrier to help profile alignment
    if dist.is_initialized():
        dist.barrier(device_ids=[device.index])

class Trainer():

    # jit stuff
    def _compile_model(self, inp_shape):

        if self.params.jit_mode == "script":
            if dist.is_initialized() and not self.params.disable_ddp:
                self.model.module = torch.jit.script(self.model.module)
            else:
                self.model = torch.jit.script(self.model)
            self.model_train = self.model
            self.model_eval = self.model

        elif self.params.jit_mode == "inductor":
            self.model = torch.compile(self.model)
            self.model_train = self.model
            self.model_eval = self.model

        else:
            self.model_train = self.model
            self.model_eval = self.model

        return


    # graph stuff
    def _capture_model(self, capture_stream, inp_shape, tar_shape, num_warmup_steps=20):

        matmul_comm_size = comm.get_size("matmul")

        # modify inp shape due to model parallelism
        if self.params.split_data_channels:
            inp_shape_eff = (inp_shape[0],
                             (inp_shape[1] + matmul_comm_size - 1) // matmul_comm_size,
                             inp_shape[2],
                             inp_shape[3])

            tar_shape_eff = (tar_shape[0],
                             (tar_shape[1] + matmul_comm_size - 1) // matmul_comm_size,
                             tar_shape[2],
                             tar_shape[3])
        else:
            inp_shape_eff = (inp_shape[0],
                             inp_shape[1],
                             inp_shape[2],
                             inp_shape[3])

            tar_shape_eff = (tar_shape[0],
                             tar_shape[1],
                             tar_shape[2],
                             tar_shape[3])

        #print(inp_shape_eff, tar_shape_eff)
        self.static_inp = torch.zeros(inp_shape_eff, dtype=torch.float32, device=self.device)
        self.static_tar = torch.zeros(tar_shape_eff, dtype=torch.float32, device=self.device)

        if self.params.enable_nhwc:
            self.static_inp = self.static_inp.to(memory_format=torch.channels_last)
            self.static_tar = self.static_tar.to(memory_format=torch.channels_last)

        # set to train
        self._set_train()

        # do capture
        if capture_stream is None:
            capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            for _ in range(num_warmup_steps):
                self.model_train.zero_grad(set_to_none=True)

                # FW
                with amp.autocast(enabled = self.amp_enabled, dtype = self.amp_dtype):
                    self.static_pred = self.model_train(self.static_inp).to(self.device)
                    self.static_loss = self.loss_obj(self.static_pred, self.static_tar, self.static_inp)

                # BW
                self.gscaler.scale(self.static_loss).backward()

            # sync here
            capture_stream.synchronize()

            gc.collect()
            torch.cuda.empty_cache()

            # create graph
            self.graph = torch.cuda.CUDAGraph()

            # zero grads before capture:
            self.model_train.zero_grad(set_to_none=True)

            # start capture
            self.graph.capture_begin()

            # FW
            with amp.autocast(enabled = self.amp_enabled, dtype = self.amp_dtype):
                self.static_pred = self.model_train(self.static_inp)
                self.static_loss = self.loss_obj(self.static_pred, self.static_tar, self.static_inp)

            # BW
            self.gscaler.scale(self.static_loss).backward()

            # end capture
            self.graph.capture_end()

        torch.cuda.current_stream().wait_stream(capture_stream)

        return

    def _get_time_stats(self):

        # get some stats: make data shared with tensor from the class
        _, out_scale = self.train_dataloader.get_output_normalization()
        mult_cpu = torch.from_numpy(out_scale)[0, :, 0, 0]

        # compute
        if self.params.enable_synthetic_data:
            clim = torch.zeros([self.params.N_out_channels, self.params.img_crop_shape_x, self.params.img_crop_shape_y], dtype=torch.float32, device=self.device)

        else:
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


        return mult_cpu, clim


    def _update_parameters(self, params):
        """
        This could be moved potentially. The idea is to process params and handle the logics for params
        """

        params.N_in_channels =len(self.valid_dataset.in_channels)
        params.N_out_channels = len(self.valid_dataset.out_channels)

        params.img_shape_x = self.valid_dataset.img_shape_x
        params.img_shape_y = self.valid_dataset.img_shape_y
        
        params.img_crop_shape_x = self.valid_dataset.img_crop_shape_x
        params.img_crop_shape_y = self.valid_dataset.img_crop_shape_y
        params.img_crop_offset_x = self.valid_dataset.img_crop_offset_x
        params.img_crop_offset_y = self.valid_dataset.img_crop_offset_y 

        params.img_local_shape_x = self.valid_dataset.img_local_shape_x
        params.img_local_shape_y = self.valid_dataset.img_local_shape_y
        params.img_local_offset_x = self.valid_dataset.img_local_offset_x
        params.img_local_offset_y = self.valid_dataset.img_local_offset_y

        # derived quantities
        params["N_in_predicted_channels"] = params.N_in_channels
        
        # sanitization:
        if not hasattr(params, 'add_zenith'):
            params["add_zenith"] = False

        # input channels
        # zenith channel is appended to all the samples, so we need to do it here
        if params.add_zenith:
            params.N_in_channels += 1
            
        if params.n_history >= 1:
            params.N_in_channels = (params.n_history + 1) * params.N_in_channels
            params.N_in_predicted_channels *= (params.n_history + 1)

        # these are static and the same for all samples in the same time history
        if params.add_grid:
            n_grid_chan = 2
            if (params.gridtype == "sinusoidal") and hasattr(params, "grid_num_frequencies"):
                n_grid_chan *= params.grid_num_frequencies
            params.N_in_channels += n_grid_chan

        if params.add_orography:
            params.N_in_channels += 1
            
        if params.add_landmask:
            params.N_in_channels += 2

        # target channels
        params.N_target_channels = (params.n_future + 1) * params.N_out_channels

        # MISC parameters
        if not hasattr(params, "history_normalization_mode"):
            params["history_normalization_mode"] = "none"

        if not hasattr(params, "num_visualization_workers"):
            params["num_visualization_workers"] = 1

        if not hasattr(params, "log_video"):
            params["log_video"] = 0

        if not hasattr(params, "load_checkpoint"):
            params["load_checkpoint"] = "legacy"
        
        if not hasattr(params, "save_checkpoint"):
            params["save_checkpoint"] = "legacy"

        return params

    def __del__(self):
        if self.params.log_to_wandb:
            wandb.finish()


    def __init__(self, params, world_rank):

        self.params = None
        self.world_rank = world_rank
        self.data_parallel_rank = comm.get_rank("data")
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        else:
            self.device = torch.device("cpu")

        # get logger
        if params.log_to_screen:
            self.logger = logging.getLogger()

        # nvml stuff
        if params.log_to_screen:
            pynvml.nvmlInit()
            self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self.device.index)

        # set amp_parameters
        if hasattr(params, "amp_mode") and (params.amp_mode != "none"):
            self.amp_enabled = True
            if params.amp_mode == "fp16":
                self.amp_dtype = torch.float16
            elif params.amp_mode == "bf16":
                self.amp_dtype = torch.bfloat16
            else:
                raise ValueError(f"Unknown amp mode {params.amp_mode}")

            if params.log_to_screen:
                self.logger.info("Enabling automatic mixed precision in {params.amp_mode}.")
        else:
            self.amp_enabled = False
            self.amp_dtype = torch.float32

        # profiling stuff
        self.capture_range_start = params.capture_range_start
        self.capture_range_stop = params.capture_range_stop
        self.profile_this_rank = world_rank in params.capture_ranks
        self.capture_mode = params.capture_mode
        self.is_finished = False

        if params.log_to_wandb:
            # login first:
            wandb.login()
            # init
            wandb.init(dir=params.wandb_dir,
                       config=params,
                       name=params.wandb_name,# if not params.resuming else None,
                       group=params.wandb_group,# if not params.resuming else None,
                       project=params.wandb_project,
                       entity=params.wandb_entity,
                       resume=params.resuming)

        # data loader
        if params.log_to_screen:
            self.logger.info('initializing data loader')
        self.train_dataloader, self.train_dataset, self.train_sampler = get_dataloader(
            params,
            params.train_data_path,
            train=True,
            device=self.device
            )
        self.valid_dataloader, self.valid_dataset = get_dataloader(
            params,
            params.valid_data_path,
            train=False,
            device=self.device
            )


        if params.log_to_screen:
            self.logger.info('data loader initialized')

        # update params
        params = self._update_parameters(params)

        # save params
        self.params = params

        # init preprocessor and model

        # save the modified params to a json file to make it easier to load for inference later on
        # This should happen immediately before ``get_model`` is called.
        if self.world_rank == 0:
            config_path = os.path.join(params.experiment_dir, "config.json")
            with open(config_path, "w") as f:
                json.dump(params.to_dict(), f)

        self.model = get_model(params).to(self.device)
        self.preprocessor = self.model.preprocessor
        #self.preprocessor = get_preprocessor(params).to(self.device)

        # if model-parallelism is enabled, we need to sure that shared weights are matching across ranks
        # as random seeds might get out of sync during initialization
        # DEBUG: this also needs to be fixed in NCCL
        #if comm.get_size("model") > 1:
        sync_params(self.model, mode='broadcast')

        # define process group for DDP, we might need to override that
        if dist.is_initialized() and not params.disable_ddp:
            ddp_process_group = comm.get_group("data")

        if hasattr(params, "log_to_wandb") and params.log_to_wandb:
            wandb.watch(self.model)

        # print model
        if self.params.log_to_screen:
            self.logger.info(f"\n{self.model}")

        # register ANNA hooks
        self.enable_anna = params.enable_anna
        if params.enable_anna:
            if params.log_to_screen:
                print("Registering ANNA hooks")
            from pyt_hooks import PytHooks
            # insert the following two lines after model is defined
            hooks = PytHooks()
            hooks.register_hooks(self.model) 

        # metrics handler
        mult_cpu, clim = self._get_time_stats()
        self.metrics = MetricsHandler(self.params, mult_cpu, clim, self.device)
        self.metrics.initialize_buffers()

        # loss handler
        self.loss_obj = LossHandler(self.params)
        self.loss_obj = self.loss_obj.to(self.device)
        if self.params.enable_nhwc:
            self.loss_obj = self.loss_obj.to(memory_format=torch.channels_last)


        if not params.resuming:
            if params.nettype == 'unet':
                self.model.apply(self.model.get_weights_function(params.weight_init))

        self.capturable_optimizer = False
        betas = (params.optimizer_beta1, params.optimizer_beta2)
        all_parameters = self.model.parameters()
        if params.optimizer_type == 'Adam':
            if params.log_to_screen:
                self.logger.info("using Adam")
            self.optimizer = CustomAdam(all_parameters, betas = betas,
                                        lr = params.lr,
                                        weight_decay = params.weight_decay,
                                        foreach = True)
        elif params.optimizer_type == 'AdamW':
            if params.log_to_screen:
                self.logger.info("using AdamW")
            self.optimizer = CustomAdamW(all_parameters, betas = betas,
                                        lr = params.lr,
                                        weight_decay = params.weight_decay,
                                        foreach = True)
        elif params.optimizer_type == 'FusedLAMB':
            try:
                import doesnotexist
                from apex.optimizers import FusedMixedPrecisionLamb
                if params.log_to_screen:
                    self.logger.info("using FusedMixedPrecisionLamb")
                self.optimizer = FusedMixedPrecisionLamb(all_parameters,
                                                         betas = betas,
                                                         lr = params.lr,
                                                         weight_decay = params.weight_decay,
                                                         max_grad_norm = params.optimizer_max_grad_norm)
                self.capturable_optimizer = True
            except ImportError:
                if params.log_to_screen:
                    self.logger.info("using FusedLAMB")
                from apex.optimizers import FusedLAMB
                self.optimizer = FusedLAMB(all_parameters,
                                           betas = betas,
                                           lr = params.lr,
                                           weight_decay = params.weight_decay,
                                           max_grad_norm = params.optimizer_max_grad_norm)
        elif params.optimizer_type == 'SGD':
            if params.log_to_screen:
                self.logger.info("using SGD")
            self.optimizer = torch.optim.SGD(all_parameters,
                                             lr = params.lr,
                                             weight_decay = params.weight_decay,
                                             momentum = params.momentum)
        else:
            raise ValueError(f"Unknown optimizer type {params.optimizer_type}")

        if params.scheduler == 'ReduceLROnPlateau':
            if not hasattr(params, "scheduler_mode"):
                params["scheduler_mode"] = "min"
            if params.skip_validation:
                raise ValueError(f"Error, you cannot skip validation when using ReduceLROnPlateau scheduler.")
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.2, patience=5, mode='min')
        elif params.scheduler == 'CosineAnnealingLR':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=params.scheduler_T_max)
        else:
            self.scheduler = None

        # warmup scheduler
        if params.lr_warmup_steps > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer,
	                                                         start_factor = params.lr_start,
                                                                 end_factor = 1.0,
                                                                 total_iters = params.lr_warmup_steps)
            
            self.scheduler = torch.optim.lr_scheduler.SequentialLR(self.optimizer, [warmup_scheduler, self.scheduler],
                                                                   milestones = [params.lr_warmup_steps])
            
        # gradient scaler
        self.gscaler = amp.GradScaler(enabled = (self.amp_dtype == torch.float16))

        capture_stream = None
        if dist.is_initialized() and not params.disable_ddp:
            if self.device.type == "cuda":
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

            # capture stream sync
            if capture_stream is not None:
                capture_stream.synchronize()

        # lets get one sample from the dataloader:
        # get sample and map to gpu
        iterator = iter(self.train_dataloader)
        data = next(iterator)
        gdata = map(lambda x: x.to(self.device, dtype = torch.float32), data)
        # extract unpredicted features
        inp, tar = self.preprocessor.cache_unpredicted_features(*gdata)
        # flatten
        inp = self.preprocessor.flatten_history(inp)
        tar = self.preprocessor.flatten_history(tar)
        # get shapes
        inp_shape = inp.shape
        tar_shape = tar.shape

        self._compile_model(inp_shape)
        if not self.loss_obj.is_distributed():
            self.loss_obj = torch.jit.script(self.loss_obj)

        # graph capture
        self.graph = None
        if params.cuda_graph_mode != "none":
            #tar_shape = (self.params.batch_size, self.params.N_target_channels, self.params.img_local_shape_x, self.params.img_local_shape_y)
            self._capture_model(capture_stream, inp_shape, tar_shape, num_warmup_steps=20)

        # visualization wrapper:
        plot_list = [{"name": "windspeed_uv10", "functor": "lambda x: np.sqrt(np.square(x[0, ...]) + np.square(x[1, ...]))", "diverging": False}]
        out_bias, out_scale = self.train_dataloader.get_output_normalization()
        self.visualizer = visualize.VisualizationWrapper(params.log_to_wandb,
                                                         path = None,
                                                         prefix = None,
                                                         plot_list = plot_list,
                                                         lat = self.valid_dataset.grid_converter.get_dst_coords()[0].cpu().numpy(),
                                                         lon = self.valid_dataset.grid_converter.get_dst_coords()[1].cpu().numpy() - np.pi,
                                                         scale = out_scale[0, ...],
                                                         bias = out_bias[0, ...],
                                                         num_workers = params.num_visualization_workers)
        # allocate pinned tensors for faster copy:
        if self.device.type == "cuda":
            self.viz_stream = torch.cuda.Stream()
        else:
            self.viz_stream = None
        
        pin_memory = self.device.type == "cuda"
        self.viz_prediction_cpu = torch.empty(( (params.N_target_channels // (params.n_future + 1)), params.img_shape_x, params.img_shape_y), device='cpu', pin_memory=pin_memory)
        self.viz_target_cpu = torch.empty(( (params.N_target_channels // (params.n_future + 1)), params.img_shape_x, params.img_shape_y), device='cpu', pin_memory=pin_memory)

        # reload checkpoints
        self.iters = 0
        self.startEpoch = 0
        if params.finetune and not params.resuming:
            assert(params.pretrained_checkpoint_path is not None), "Error, please specify a valid pretrained checkpoint path"
            self.restore_checkpoint(params.pretrained_checkpoint_path,
                                    checkpoint_mode = params.load_checkpoint,
                                    load_optimizer = params.load_optimizer,
                                    load_scheduler = params.load_scheduler,
                                    load_counters = params.load_counters)
        
        if params.resuming:
            self.restore_checkpoint(params.checkpoint_path,
                                    checkpoint_mode = params.load_checkpoint,
                                    load_optimizer = params.load_optimizer,
                                    load_scheduler = params.load_scheduler,
                                    load_counters = params.load_counters)

        self.epoch = self.startEpoch

        # if params.log_to_screen:
        #   self.logger.info(self.model)

        # counting runs a reduction so we need to count on all ranks before printing on rank 0
        pcount = count_parameters(self.model, self.device)
        if params.log_to_screen:
            self.logger.info("Number of trainable model parameters: {}".format(pcount))

    def train(self):
        # log parameters
        if self.params.log_to_screen:
            # log memory usage so far
            all_mem_gb = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle).used / (1024. * 1024. * 1024.)
            max_mem_gb = torch.cuda.max_memory_allocated(device=self.device) / (1024. * 1024. * 1024.)
            self.logger.info(f"Scaffolding memory high watermark: {all_mem_gb} GB ({max_mem_gb} GB for pytorch)")
            # announce training start
            self.logger.info("Starting Training Loop...")

        # perform a barrier here to make sure everybody is ready
        if dist.is_initialized():
            dist.barrier(device_ids=[self.device.index])

        try:
            torch.cuda.reset_peak_memory_stats(self.device)
        except ValueError:
            pass

        training_start = time.time()
        best_valid_loss = 1.e6
        for epoch in range(self.startEpoch, self.params.max_epochs):

            if dist.is_initialized() and self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)

            # start timer
            epoch_start = time.time()

            # train
            train_time, train_data_gb, train_logs = self.train_one_epoch()

            if self.is_finished:
                if dist.is_initialized():
                    dist.barrier(device_ids=[self.device.index])
                return

            # eval
            if not self.params.skip_validation:
                valid_time, viz_time, valid_logs = self.validate_one_epoch(epoch)
            else:
                valid_time = 0
                viz_time = 0
                valid_logs = {"base": {}, "metrics": {}}

            if self.is_finished:
                if dist.is_initialized():
                    dist.barrier(device_ids=[self.device.index])
                return

            if self.params.scheduler == 'ReduceLROnPlateau':
                self.scheduler.step(valid_logs['base']['validation loss'])
            elif self.scheduler is not None:
                self.scheduler.step()
                #if self.epoch >= self.params.max_epochs:
                #  self.logger.info("Terminating training after reaching params.max_epochs while LR scheduler is set to CosineAnnealingLR")
                #  exit()

            if self.params.log_to_wandb:
                for pg in self.optimizer.param_groups:
                    lr = pg['lr']
                wandb.log({'learning rate': lr}, step=self.epoch)

            if (self.data_parallel_rank == 0) and (self.params.save_checkpoint != 'none'):
                #checkpoint at the end of every epoch
                self.save_checkpoint(self.params.checkpoint_path, checkpoint_mode=self.params['save_checkpoint'])
                best_checkpoint_path = self.params.best_checkpoint_path.format(mp_rank=comm.get_rank("model"))
                best_checkpoint_saved = os.path.isfile(best_checkpoint_path)
                if (not self.params.skip_validation) and ((not best_checkpoint_saved) or (valid_logs["base"]['validation loss'] <= best_valid_loss)):
                    self.save_checkpoint(self.params.best_checkpoint_path, checkpoint_mode=self.params['save_checkpoint'])
                    best_valid_loss = valid_logs["base"]['validation loss']

            # end timer
            epoch_end = time.time()

            # create timing logs:
            timing_logs = {"epoch time [s]": epoch_end - epoch_start,
                           "training time [s]": train_time,
                           "validation time [s]": valid_time,
                           "visualization time [s]": viz_time,
                           "training step time [ms]": (train_time / train_logs["train_steps"]) * 10**3,
                           "minimal IO rate [GB/s]": train_data_gb/train_time}

            # log metrics:
            self.log_epoch(train_logs, valid_logs, timing_logs)
            
        # training done
        training_end = time.time()
        if self.params.log_to_screen:
            self.logger.info('Total training time is {:.2f} sec'.format(training_end-training_start))

        return

    def _set_train(self):
        self.model.train()
        self.loss_obj.train()

    def _set_eval(self):
        self.model.eval()
        self.loss_obj.eval()

    def train_one_epoch(self):
        self.epoch += 1
        total_data_bytes = 0
        self._set_train()

        train_steps = 0
        train_start = time.perf_counter_ns()
        with torch.autograd.profiler.emit_nvtx(enabled=True, record_shapes=self.enable_anna):

            torch.cuda.nvtx.range_push("training")
            
            for data in tqdm(self.train_dataloader, desc="Training progress  ", disable=not self.params.log_to_screen):
                train_steps += 1
                self.iters += 1

                if (self.capture_mode == "training") and (self.iters == self.capture_range_start):
                    cudaProfilerStart(self.device, self.profile_this_rank)
                    print("Starting Training Profiling...")

                # step range
                torch.cuda.nvtx.range_push(f"train step {self.iters}")
                
                # map to device
                gdata = map(lambda x: x.to(self.device, dtype = torch.float32), data)

                # do preprocessing
                inp, tar = self.preprocessor.cache_unpredicted_features(*gdata)
                inp = self.preprocessor.flatten_history(inp)
                tar = self.preprocessor.flatten_history(tar)
            
                # assuming float32
                total_data_bytes += (torch.numel(inp) + torch.numel(tar)) * 4
            
                if self.graph is not None:
                    self.static_inp.copy_(inp)
                    self.static_tar.copy_(tar)
                    self.graph.replay()
                    loss = self.static_loss
                else:
                    self.model_train.zero_grad(set_to_none=True)
                    with amp.autocast(enabled = self.amp_enabled, dtype = self.amp_dtype):
                        pred = self.model_train(inp)
                        loss = self.loss_obj(pred, tar, inp)

                    self.gscaler.scale(loss).backward()

                # perform weight update
                self.gscaler.step(self.optimizer)
                self.gscaler.update()
            
                if (self.params.print_timings_frequency > 0) and (self.iters % self.params.print_timings_frequency == 0) and self.params.log_to_screen:
                    running_train_time = time.perf_counter_ns() - train_start
                    print(f"Average step time after step {self.iters}: {running_train_time / float(train_steps) * 10**(-6):.1f} ms")
                    print(f"Average effective io rate after step {self.iters}: {total_data_bytes * float(comm.get_world_size()) / (float(running_train_time) * 10**(-9) * 1024. * 1024. * 1024.):.2f} GB/s")
                    print(f"Current loss {loss.item()}")

                # close range
                torch.cuda.nvtx.range_pop()

                # stop profiling
                if (self.capture_mode == "training") and (self.iters >= self.capture_range_stop):
                    cudaProfilerStop(self.device, self.profile_this_rank)
                    print("Training Profiling Stopped")
                    self.is_finished = True
                    break

            # add the eval loss to logs
            logs = {'loss': loss}

            if dist.is_initialized():
                for key in sorted(logs.keys()):
                    dist.all_reduce(logs[key].detach(), op=dist.ReduceOp.AVG, group=comm.get_group("data"))
                    logs[key] = logs[key].item()

            # add train steps to log
            logs["train_steps"] = train_steps

            # global sync is in order
            if dist.is_initialized():
                dist.barrier(device_ids=[self.device.index])

            # finalize timers
            train_end = time.perf_counter_ns()
            train_time = (train_end - train_start) * 10**(-9)
            total_data_gb = (total_data_bytes / (1024. * 1024. * 1024.)) * float(comm.get_world_size())

            # end profiling range
            torch.cuda.nvtx.range_pop()

        return train_time, total_data_gb, logs

  
    def validate_one_epoch(self, epoch):
        self._set_eval()

        # clear cache
        torch.cuda.empty_cache()
        
        self.metrics.zero_buffers()

        visualize = self.params.log_video and (epoch % self.params.log_video == 0)

        # start the timer
        valid_start = time.time()

        with torch.autograd.profiler.emit_nvtx(enabled=True, record_shapes=False):

            torch.cuda.nvtx.range_push("validation")

            with torch.inference_mode():
                with torch.no_grad():
                    
                    eval_steps = 0
                    for data in tqdm(self.valid_dataloader, desc="Validation progress", disable=not self.params.log_to_screen):
                        eval_steps += 1

                        # profile validation
                        if (self.capture_mode == "validation") and (eval_steps == self.capture_range_start):
                            cudaProfilerStart(self.device, self.profile_this_rank)
                            print("Starting Validation Profiling...")

                        # nvtx range
                        torch.cuda.nvtx.range_push(f"eval step {eval_steps}")
                        
                        # map to gpu
                        gdata = map(lambda x: x.to(self.device, dtype = torch.float32), data)
                        
                        # preprocess
                        inp, tar = self.preprocessor.cache_unpredicted_features(*gdata)
                        inp = self.preprocessor.flatten_history(inp)
                        
                        # split list of targets
                        tarlist = torch.split(tar, 1, dim=1)
                        inpt = inp
                        for idt, targ in enumerate(tarlist):

                            # flatten history of the target
                            targ = self.preprocessor.flatten_history(targ)
                            
                            # FW pass
                            with amp.autocast(enabled = self.amp_enabled, dtype = self.amp_dtype):
                                
                                pred = self.model_eval(inpt)
                                loss = self.loss_obj(pred, targ, inpt)
                        
                            if eval_steps <= 1 and visualize:
                                pred_single = pred[0:1, ...]
                                targ_single = targ[0:1, ...]
                                pred_gather = torch.squeeze(self.metrics._gather_input(pred_single), dim=0)
                                targ_gather = torch.squeeze(self.metrics._gather_input(targ_single), dim=0)
                                if self.viz_stream is not None:
                                    self.viz_stream.wait_stream(torch.cuda.current_stream())
                                with torch.cuda.stream(self.viz_stream):
                                    self.viz_prediction_cpu.copy_(pred_gather, non_blocking=True)
                                    self.viz_target_cpu.copy_(targ_gather, non_blocking=True)
                                if self.viz_stream is not None:
                                    torch.cuda.current_stream().wait_stream(self.viz_stream)

                                # copy to fp32
                                pred_cpu = self.viz_prediction_cpu.to(torch.float32).numpy()
                                targ_cpu = self.viz_target_cpu.to(torch.float32).numpy()
                                tag = f"step{eval_steps}_time{str(idt).zfill(3)}"
                                self.visualizer.add(tag, pred_cpu, targ_cpu)


                            # put in the metrics handler
                            self.metrics.update(pred, targ, loss, idt)
            
                            # append history
                            inpt = self.preprocessor.append_history(inpt, pred, idt)

                        # nvtx range
                        torch.cuda.nvtx.range_pop()

                        # exit eval
                        if (self.capture_mode == "validation") and (eval_steps >= self.capture_range_stop):
                            cudaProfilerStop(self.device, self.profile_this_rank)
                            print("Validation Profiling Stopped")
                            self.is_finished = True
                            break

                    # create final logs
                    logs = self.metrics.finalize()

            # finalize plotting
            viz_time = time.perf_counter_ns()
            if visualize:
                self.visualizer.finalize()
            viz_time = (time.perf_counter_ns() - viz_time) * 10**(-9)

            # global sync is in order
            if dist.is_initialized():
                dist.barrier(device_ids=[self.device.index])

            # timer
            valid_time = time.time() - valid_start

            # close range
            torch.cuda.nvtx.range_pop()

        return valid_time, viz_time, logs
  
    def test_model_output(self, model):
        ''' helper to test checkpointing'''
        inp_shape = (self.params.batch_size, self.params.N_in_channels, self.params.img_shape_local_x, self.params.img_shape_local_y)
        matmul_comm_size = comm.get_size("matmul")
        
        # modify inp shape due to model parallelism
        if self.params.split_data_channels:
            inp_shape_eff = (inp_shape[0],
                            (inp_shape[1] + matmul_comm_size - 1) // matmul_comm_size,
                            inp_shape[2],
                            inp_shape[3])
        else:
            inp_shape_eff = (inp_shape[0],
                            inp_shape[1],
                            inp_shape[2],
                            inp_shape[3])

        random_tensor = os.path.join(self.params.experiment_dir, "random_tensor{}.npy".format(comm.get_rank("model")))
        if not os.path.exists(random_tensor):
            y = torch.rand(inp_shape_eff, dtype=torch.float).cpu().numpy()
            np.save(random_tensor, y)
        
        y = torch.from_numpy(np.load(random_tensor)).type(torch.float).to(self.device)
        out = model(y).detach().cpu().numpy()
        random_output = os.path.join(self.params.experiment_dir, "random_output{}.npy".format(comm.get_rank("model")))
        if os.path.exists(random_output):
            out_old = np.load(random_output)
            diff = (out - out_old).flatten()
            self.logger.info("Diff metrics: norm = {}, max = {}, min = {}".format(np.linalg.norm(diff), np.max(diff), np.min(diff)))
        np.save(random_output, out)


    def log_epoch(self, train_logs, valid_logs, timing_logs):

        # separator
        separator = "".join(["-" for _ in range(50)])
        print_prefix = "    "
        
        def get_pad(nchar):
            return "".join([" " for x in range(nchar)])
        
        if self.params.log_to_screen:
            # header:
            self.logger.info(separator)
            self.logger.info(f'Epoch {self.epoch} summary:')
            self.logger.info(f'Performance Parameters:')
            self.logger.info(print_prefix + 'training steps: {}'.format(train_logs["train_steps"]))
            self.logger.info(print_prefix + 'validation steps: {}'.format(valid_logs['base']["validation steps"]))
            all_mem_gb = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle).used / (1024. * 1024. * 1024.)
            self.logger.info(print_prefix + f'memory footprint [GB]: {all_mem_gb:.2f}')
            for key in timing_logs.keys():
                self.logger.info(print_prefix + key + ": {:.2f}".format(timing_logs[key]))
            
            #self.logger.info('Time taken for training in epoch {} is {:.2f} sec ({} steps)'.format(epoch + 1, time.time()-start, train_logs["train_steps"]))
            #self.logger.info('Time taken for validation in epoch {} is {:.2f} sec ({} steps)'.format(epoch + 1, valid_time, valid_logs['base']["validation steps"]))
            #self.logger.info('Effective training IO rate for epoch {} is {:.2f} GB/s'.format(epoch + 1, train_data_gb/tr_time))
            #all_mem_gb = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle).used / (1024. * 1024. * 1024.)
            #max_mem_gb = torch.cuda.max_memory_allocated(device=self.device) / (1024. * 1024. * 1024.)
            #self.logger.info(f'Memory high watermark: {all_mem_gb:.2f} GB ({max_mem_gb:.2f} GB for pytorch)')    
            
            # compute padding:
            print_list = ['training loss', 'validation loss', 'validation L1'] + list(valid_logs['metrics'].keys())
            max_len = max([len(x) for x in print_list])
            pad_len = [max_len - len(x) for x in print_list]
            # validation summary
            self.logger.info('Metrics:')
            self.logger.info(print_prefix + 'training loss: {}{}'.format(get_pad(pad_len[0]), train_logs['loss']))
            self.logger.info(print_prefix + 'validation loss: {}{}'.format(get_pad(pad_len[1]), valid_logs['base']['validation loss']))
            self.logger.info(print_prefix + 'validation L1: {}{}'.format(get_pad(pad_len[2]), valid_logs['base']['validation L1']))
            for idk, key in enumerate(print_list[3:], start=3):
                value = valid_logs['metrics'][key]
                self.logger.info(f'{print_prefix}{key}: {get_pad(pad_len[idk])}{value}')
            self.logger.info(separator)

        if self.params.log_to_wandb:
            wandb.log(train_logs, step=self.epoch)
            wandb.log(valid_logs['base'], step=self.epoch)
            wandb.log(valid_logs['metrics'], step=self.epoch) 

        return


    def save_checkpoint(self, checkpoint_path, model=None, checkpoint_mode="flexible"):
        """ We intentionally require a checkpoint_dir to be passed
            in order to allow Ray Tune to use this function """

        if self.params.log_to_screen:
            self.logger.info(f"Writing checkpoint to {checkpoint_path} ({checkpoint_mode} format)")

        if not model:
            model = self.model

        with torch.no_grad():
            # legacy mode
            if checkpoint_mode == "legacy":

                # start timer
                store_start = time.time()
                checkpoint_fname = checkpoint_path.format(mp_rank=comm.get_rank("model"))
                store_dict = {'iters': self.iters, 'epoch': self.epoch, 'model_state': model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'params': self.params}
                if self.scheduler is not None:
                    store_dict['scheduler_state_dict'] = self.scheduler.state_dict()
                torch.save(store_dict, checkpoint_fname)

                # stop timer
                store_stop = time.time()

                # report time
                if self.params.log_to_screen:
                    self.logger.info(f"Save checkpoint (legacy): {(store_stop - store_start):.2f} sec ({sys.getsizeof(store_dict)/(1024.**3)}) GB")

            elif checkpoint_mode == "flexible":

                # clear cache
                torch.cuda.empty_cache()

                # start timer
                collect_start = time.time()
                
                # state_dict = model.state_dict()
                state_dict = OrderedDict()
                

                for k, v in self.model.named_parameters():
                        
                    weight = v.clone()
                    if hasattr(v, "sharded_dims_mp"):
                        
                        # gather the weight across all sharded dimensions
                        for d, group in enumerate(v.sharded_dims_mp):
                            if group is not None:
                                weight = gather_uneven(weight, d, group)

                    state_dict[k] = weight.to("cpu")

                # stop timer
                collect_stop = time.time()

                # print collect time
                if self.params.log_to_screen:
                    self.logger.info(f"Collect checkpoint (flexible): {(collect_stop - collect_start):.2f} sec.")
                
                # start timer:
                store_start = time.time()
                
                checkpoint_fname = checkpoint_path.format(mp_rank=0)
                store_dict = {'iters': self.iters, 'epoch': self.epoch, 'model_state': state_dict, 'params': self.params}
                if self.optimizer is not None:
                    store_dict['optimizer_state_dict'] = self.optimizer.state_dict()

                if self.scheduler is not None:
                    store_dict['scheduler_state_dict'] = self.scheduler.state_dict()

                # in flexible mode only rank 0 needs to save the data to disk
                if self.world_rank == 0:
                    torch.save(store_dict, checkpoint_fname, _use_new_zipfile_serialization=False)

                # wait for group
                if dist.is_initialized() and (comm.get_size("model") > 1):
                    dist.barrier(device_ids=[self.device.index], group=comm.get_group("model"))
                    
                # stop timer
                store_stop = time.time()

                if self.params.log_to_screen:
                    self.logger.info(f"Save checkpoint (flexible): {(store_stop - store_start):.2f} sec")
            else:
                raise ValueError(f"Unknown checkoint mode {checkpoint_mode}.")
                
        
    def restore_checkpoint(self,
                           checkpoint_path,
                           checkpoint_mode = "flexible",
                           load_optimizer = False,
                           load_scheduler = False,
                           load_counters = False
                           ):
        """ We intentionally require a checkpoint_dir to be passed
            in order to allow Ray Tune to use this function """
    
        # legacy mode
        if checkpoint_mode == "legacy":
            
            checkpoint_fname = checkpoint_path.format(mp_rank=comm.get_rank("model"))
            if self.params.log_to_screen:
                self.logger.info("Loading checkpoint %s in legacy mode"%checkpoint_fname)
            checkpoint = torch.load(checkpoint_fname, map_location='cpu')

            # this is reworked to avoid loading modules related to the SHT
            state_dict = checkpoint['model_state']
            if not isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, "module.") 
            self.model.load_state_dict(state_dict, strict=True)

            # If finetuning, restore checkpoint does not load optimizer state, instead uses config specified lr.
            if load_optimizer:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if load_scheduler and (self.scheduler is not None):
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            if load_counters:
                self.iters = checkpoint['iters']
                self.startEpoch = checkpoint['epoch']

        # new flexible mode allows to load models in arbitrary model-parallel configurations
        elif checkpoint_mode == "flexible":

            # when loading the weights in flexble mode we exclusively use mp_rank=0 and load them onto the cpu
            checkpoint_fname = checkpoint_path.format(mp_rank=0)
            if self.params.log_to_screen:
                self.logger.info("Loading checkpoint %s in flexible mode"%checkpoint_fname)
            checkpoint = torch.load(checkpoint_fname, map_location='cpu')
            
            # this is reworked to avoid loading modules related to the SHT
            state_dict = checkpoint['model_state']

            with torch.inference_mode():
                with torch.no_grad():
                    for k,v in self.model.named_parameters():
                        if k in state_dict.keys():
                            weight = state_dict[k]

                            if hasattr(v, "sharded_dims_mp"):
                                for d, group in enumerate(v.sharded_dims_mp):
                                    
                                    # continue if there is nothing to do
                                    if (group is None) or (comm.get_size(group) == 1):
                                        continue

                                    shard_size = (weight.shape[d] + comm.get_size(group) - 1) // comm.get_size(group)
                                    weight = torch.split(weight, split_size_or_sections=shard_size, dim=d)[comm.get_rank(group)]

                            v.copy_(weight)                                
                                                        
                        else:
                            # put a warning here
                            self.logger.warn(f"missing {k}")

            # If finetuning, restore checkpoint does not load optimizer state, instead uses config specified lr.
            if load_optimizer:
                # not loading optimzer as momentum tensor shapes might have changed
                # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                raise NotImplementedError("Error, restoring optimizer not supported for flexible checkpoint format yet")

            if load_scheduler and (self.scheduler is not None):
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            if load_counters:
                self.iters = checkpoint['iters']
                self.startEpoch = checkpoint['epoch']

        else:
            raise ValueError(f"Unknown checkoint mode {checkpoint_mode}.")

        # let's clean up
        gc.collect()
        torch.cuda.empty_cache()
        
        # sync the device
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()

        return

