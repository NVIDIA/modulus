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
import time

import numpy as np
from tqdm import tqdm
import pynvml

import torch
import torch.cuda.amp as amp

# from torch.nn.parallel import DistributedDataParallel

import logging
import wandb

from modulus.experimental.sfno.utils.dataloader import get_dataloader
from modulus.experimental.sfno.networks.models import get_model
from modulus.experimental.sfno.utils.trainer import Trainer
from modulus.experimental.sfno.utils.losses import LossHandler
from modulus.experimental.sfno.utils.metric import MetricsHandler

# distributed computing stuff
from modulus.experimental.sfno.utils import comm
from modulus.experimental.sfno.utils import visualize
import torch.distributed as dist


class Inferencer(Trainer):
    """
    Inferencer class holding all the necessary information to perform inference. Design is similar to Trainer, however only keeping the necessary information.
    """

    def __init__(self, params, world_rank):

        # init the trainer
        # super().__init__(params, world_rank, job_type="inference")

        self.params = None
        self.world_rank = world_rank

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
                self.logger.info(f"Enabling automatic mixed precision in {params.amp_mode}.")
        else:
            self.amp_enabled = False
            self.amp_dtype = torch.float32

        # resuming needs is set to False so loading checkpoints does not attempt to set the optimizer state
        params["resuming"] = False

        if hasattr(params, "log_to_wandb") and params.log_to_wandb:
            # login first:
            wandb.login()
            # init
            wandb.init(dir=params.experiment_dir,
                       config=params,
                       name=params.wandb_name,# if not params.resuming else None,
                       group=params.wandb_group,# if not params.resuming else None,
                       project=params.wandb_project,
                       entity=params.wandb_entity,
                       resume=params.resuming)

        # data loader
        if params.log_to_screen:
            self.logger.info("initializing data loader")

        if not hasattr(params, "multifiles"):
            params["multifiles"] = False
        if not hasattr(params, "enable_synthetic_data"):
            params["enable_synthetic_data"] = False
        if not hasattr(params, "amp"):
            params["enable_synthetic_data"] = False

        # although it is called validation dataloader here, the file path is taken from inf_data_path to perform inference on the 
        # out of sample dataset
        self.valid_dataloader, self.valid_dataset = get_dataloader(
            params, 
            params.inf_data_path, 
            train=False, 
            final_eval=True,
            device=self.device
        )
        if params.log_to_screen:
            self.logger.info('data loader initialized')

        # update params
        params = self._update_parameters(params)

        # save params
        self.params = params

        self.model = get_model(params).to(self.device)
        self.preprocessor = self.model.preprocessor

        # print model
        if self.world_rank == 0:
            print(self.model)

        self.restore_checkpoint(params.checkpoint_path,
                                checkpoint_mode=params.load_checkpoint)

        # metrics handler
        mult_cpu, clim = self._get_time_stats()
        self.metrics = MetricsHandler(self.params, mult_cpu, clim, self.device)
        self.metrics.initialize_buffers()

        # loss handler
        self.loss_obj = LossHandler(self.params)
        self.loss_obj = self.loss_obj.to(self.device)

        
    def _autoregressive_inference(self, data, compute_metrics=False, output_data=False, output_channels=[0,1]):

        # map to gpu
        gdata = map(lambda x: x.to(self.device, dtype = torch.float32), data)

        # preprocess
        inp, tar = self.preprocessor.cache_unpredicted_features(*gdata)
        inp = self.preprocessor.flatten_history(inp)

        # split list of targets
        tarlist = torch.split(tar, 1, dim=1)

        # do autoregression
        inpt = inp
        for idt, targ in enumerate(tarlist):
            
            # flatten history of the target
            targ = self.preprocessor.flatten_history(targ)
        
            # FW pass
            with amp.autocast(enabled = self.amp_enabled, dtype = self.amp_dtype):
                pred = self.model(inpt)
                loss = self.loss_obj(pred, targ, inpt)

            # put in the metrics handler
            if compute_metrics:
                self.metrics.update(pred, targ, loss, idt)

            if output_data:    
                self.pred_outputs.append(pred[:, output_channels].cpu())
                self.targ_outputs.append(targ[:, output_channels].cpu())

            # append history
            inpt = self.preprocessor.append_history(inpt, pred, idt)

        return
    
    def inference_single(self, ic=0, compute_metrics=False, output_data=False, output_channels=[0,1]):
        """
        Runs the model in autoregressive inference mode on a single initial condition.
        """

        self._set_eval()

        # clear cache
        torch.cuda.empty_cache()

        # initialize metrics buffers
        if compute_metrics:
            self.metrics.zero_buffers()

        if output_data:
            self.targ_outputs = []
            self.pred_outputs = []

        with torch.inference_mode():
            with torch.no_grad():

                data = self.valid_dataset[ic]
                # add batch dimension - this is necessary as we do not use the dataloader here
                data = map(lambda x: x.unsqueeze(0), data)

                self._autoregressive_inference(data, compute_metrics=compute_metrics, output_data=output_data, output_channels=output_channels)

        result = []
        if output_data:
            targ = torch.stack(self.targ_outputs, dim=0)
            pred = torch.stack(self.pred_outputs, dim=0)
            result = result + [targ, pred]

        # create final logs
        if compute_metrics: 
            logs, acc_curves, rmse_curves = self.metrics.finalize(final_inference=True)
            result = result + [logs, acc_curves.cpu(), rmse_curves.cpu()]

        return tuple(result)

    def inference_epoch(self):
        """
        Runs the model in autoregressive inference mode on the entire validation dataset. Computes metrics and scores the model.
        """
        
        # set to eval
        self._set_eval()

        # clear cache
        torch.cuda.empty_cache()

        # initialize metrics buffers
        self.metrics.zero_buffers()

        with torch.inference_mode():
            with torch.no_grad():

                eval_steps = 0
                for data in tqdm(self.valid_dataloader, desc="Scoring progress", disable=not self.params.log_to_screen):
                    eval_steps += 1

                    self._autoregressive_inference(data, compute_metrics=True, output_data=False, output_channels=False)

        # create final logs
        logs, acc_curves, rmse_curves = self.metrics.finalize(final_inference=True)

        # save the acc curve
        if self.world_rank == 0:

            np.save(os.path.join(self.params.experiment_dir, 'acc_curves.npy'), acc_curves.cpu().numpy())
            np.save(os.path.join(self.params.experiment_dir, 'rmse_curves.npy'), rmse_curves.cpu().numpy())

            # visualize the result and log it to wandb. The dummy epoch 0 is used for logging to wandb
            visualize.plot_rollout_metrics(acc_curves, rmse_curves, self.params, epoch=0, model_name=self.params.nettype)
 
        # global sync is in order
        if dist.is_initialized():
            dist.barrier(device_ids=[self.device.index])

        return logs

    def log_score(self, scoring_logs, scoring_time):

        # separator
        separator = "".join(["-" for _ in range(50)])
        print_prefix = "    "
        
        def get_pad(nchar):
            return "".join([" " for x in range(nchar)])
        
        if self.params.log_to_screen:
            # header:
            self.logger.info(separator)
            self.logger.info(f'Scoring summary:')
            self.logger.info('Total scoring time is {:.2f} sec'.format(scoring_time))
            
            # compute padding:
            print_list = list(scoring_logs['metrics'].keys())
            max_len = max([len(x) for x in print_list])
            pad_len = [max_len - len(x) for x in print_list]
            # validation summary
            self.logger.info('Metrics:')
            for idk, key in enumerate(print_list):
                value = scoring_logs['metrics'][key]
                self.logger.info(f'{print_prefix}{key}: {get_pad(pad_len[idk])}{value}')
            self.logger.info(separator)

        return

    def score_model(self):

        # log parameters
        if self.params.log_to_screen:
            # log memory usage so far
            all_mem_gb = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle).used / (1024. * 1024. * 1024.)
            max_mem_gb = torch.cuda.max_memory_allocated(device=self.device) / (1024. * 1024. * 1024.)
            self.logger.info(f"Scaffolding memory high watermark: {all_mem_gb} GB ({max_mem_gb} GB for pytorch)")
            # announce training start
            self.logger.info("Starting Scoring...")

        # perform a barrier here to make sure everybody is ready
        if dist.is_initialized():
            dist.barrier(device_ids=[self.device.index])

        try:
            torch.cuda.reset_peak_memory_stats(self.device)
        except ValueError:
            pass

        # start timer
        scoring_start = time.time()

        scoring_logs = self.inference_epoch()

        # end timer
        scoring_end = time.time()

        self.log_score(scoring_logs, scoring_end-scoring_start)

        return





