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

import torch
import wandb

# distributed computing stuff
from modulus.experimental.sfno.utils import comm
from modulus.experimental.sfno.utils.metrics.functions import GeometricL1, GeometricRMSE, GeometricACC, Quadrature
import torch.distributed as dist
from modulus.experimental.sfno.mpu.mappings import gather_from_parallel_region

class MetricsHandler():
    """
    Handler object which takes care of computation of metrics. Keeps buffers for the computation of 
    """

    def __init__(self, params, mult, clim, device,
                rmse_var_names = ['u10m', 't2m', 'u500', 'z500', 'r500', 'q500'],
                acc_vars_names = ['u10m', 't2m', 'u500', 'z500', 'r500', 'q500'],
                acc_auc_var_names = ['u10m', 't2m', 'u500', 'z500', 'r500', 'q500']):

        self.device = device
        self.log_to_screen = params.log_to_screen
        self.log_to_wandb = params.log_to_wandb
        self.channel_names = params.channel_names

        # set a stream
        if self.device.type == "cuda":
            self.stream = torch.cuda.Stream()
        else:
            self.stream = None

        # determine effective time interval and number of steps per day:
        self.dtxdh = params.dt * params.dhours
        self.dd = 24 // self.dtxdh
        
        # select the vars which are actually present
        rmse_var_names = [x for x in rmse_var_names if x in self.channel_names]
        acc_vars_names = [x for x in acc_vars_names if x in self.channel_names]
        acc_auc_var_names = [x for x in acc_auc_var_names if x in self.channel_names]

        # now create an inverse mapping
        rmse_vars = { var_name : self.channel_names.index(var_name) for var_name in rmse_var_names }
        acc_vars = { var_name : self.channel_names.index(var_name) for var_name in acc_vars_names }
        acc_auc_vars = { var_name : self.channel_names.index(var_name) for var_name in acc_auc_var_names }

        self.rmse_vars = rmse_vars
        self.acc_vars = acc_vars
        self.acc_auc_vars = acc_auc_vars

        self.split_data_channels = params.split_data_channels

        # get some stats: make data shared with tensor from the class
        self.mult = mult.to(self.device)

        # how many steps to run in acc curve
        self.valid_autoreg_steps = params.valid_autoreg_steps
    
        # climatology for autoregressive ACC
        self.simpquad = Quadrature(self.valid_autoreg_steps, 1./float(self.valid_autoreg_steps+1), self.device)
        clim = torch.unsqueeze(clim, 0)
        self.clim = clim.to(self.device, dtype=torch.float32)
        matmul_comm_size = comm.get_size("matmul")

        # get global and local output channels
        self.N_out_channels = params.N_out_channels
        if self.split_data_channels:
            self.out_channels_local = (params.N_out_channels + matmul_comm_size - 1) // matmul_comm_size
            # split channel-wise climatology by matmul parallel rank
            mprank = comm.get_rank("matmul")
            self.clim = torch.split(self.clim, self.out_channels_local, dim=1)[mprank].contiguous()
        else:
            self.out_channels_local = params.N_out_channels

        # store shapes
        self.img_shape = (params.img_shape_x, params.img_shape_y)
        self.crop_shape = (params.img_crop_shape_x, params.img_crop_shape_y)
        self.crop_offset = (params.img_crop_offset_x, params.img_crop_offset_y)         

        # grid renaming
        quadrature_rule_type = "naive"
        if params.model_grid_type == "legendre_gauss":
            quadrature_rule_type = "legendre-gauss"
        
        # set up handles
        self.l1_handle = GeometricL1(quadrature_rule_type,
                                     img_shape=self.img_shape,
                                     crop_shape=self.crop_shape,
                                     crop_offset=self.crop_offset,
                                     normalize=True,
                                     channel_reduction='mean',
                                     batch_reduction='sum').to(self.device)
        self.l1_handle = torch.compile(self.l1_handle, mode='max-autotune-no-cudagraphs')
        
        self.rmse_handle = GeometricRMSE(quadrature_rule_type,
                                         img_shape=self.img_shape,
                                         crop_shape=self.crop_shape,
                                         crop_offset=self.crop_offset,
                                         normalize=True,
                                         channel_reduction='none',
                                         batch_reduction='none').to(self.device)
        self.rmse_handle = torch.compile(self.rmse_handle, mode='max-autotune-no-cudagraphs')
        
        self.acc_handle = GeometricACC(quadrature_rule_type,
                                       img_shape=self.img_shape,
                                       crop_shape=self.crop_shape,
                                       crop_offset=self.crop_offset,
                                       normalize=True,
                                       channel_reduction='none',
                                       batch_reduction='sum').to(self.device)
        self.acc_handle = torch.compile(self.acc_handle, mode='max-autotune-no-cudagraphs')
        
        self.do_gather_input = False
        if (comm.get_size("h") * comm.get_size("w") > 1):
            self.do_gather_input = True

            
    @torch.jit.ignore
    def _gather_input(self, x: torch.Tensor) -> torch.Tensor:
        """gather and crop the data"""

        xh = gather_from_parallel_region(x, -2, "h")
        xw = gather_from_parallel_region(xh, -1, "w")
        
        x = xw[...,
               self.crop_offset[0]:self.crop_offset[0]+self.crop_shape[0],
               self.crop_offset[1]:self.crop_offset[1]+self.crop_shape[1]]
        
        return x
            

    def initialize_buffers(self):
        """initialize buffers for computing ACC, RMSE, ACC AUC curves"""

        # initialize buffers for the validation metrics
        self.valid_buffer = torch.zeros((3), dtype=torch.float32, device=self.device)
        self.valid_loss = self.valid_buffer[0].view(-1)
        self.valid_l1 = self.valid_buffer[1].view(-1)
        self.valid_steps = self.valid_buffer[2].view(-1)
    
        # we need these buffers 
        self.acc_curve = torch.zeros( (self.out_channels_local, self.valid_autoreg_steps+1), dtype=torch.float32, device=self.device)
        self.rmse_curve = torch.zeros( (self.out_channels_local, self.valid_autoreg_steps+1), dtype=torch.float32, device=self.device)
        self.acc_counter = torch.zeros( (self.valid_autoreg_steps+1), dtype=torch.float32, device=self.device)

        # create CPU copies for all the buffers
        pin_memory = self.device.type == "cuda"
        self.valid_buffer_cpu = torch.zeros((3), dtype=torch.float32, device='cpu', pin_memory=pin_memory)
        self.acc_curve_cpu = torch.zeros( (self.out_channels_local, self.valid_autoreg_steps+1), dtype=torch.float32, device='cpu', pin_memory=pin_memory)
        self.acc_auc_cpu = torch.zeros( (self.out_channels_local), dtype=torch.float32, device='cpu', pin_memory=pin_memory)
        self.rmse_curve_cpu = torch.zeros( (self.out_channels_local, self.valid_autoreg_steps+1), dtype=torch.float32, device='cpu', pin_memory=pin_memory)


    def zero_buffers(self):
        """set buffers to zero"""

        with torch.inference_mode():
            with torch.no_grad():
                self.valid_buffer.fill_(0)
                self.acc_curve.fill_(0)
                self.rmse_curve.fill_(0)
                self.acc_counter.fill_(0)
        return

        
    def update(self, prediction, target, loss, idt):
        """update function to update buffers on each autoregressive rollout step"""

        if self.do_gather_input:
            prediction = self._gather_input(prediction)
            target = self._gather_input(target)

        # update parameters
        self.acc_curve[:, idt] += self.acc_handle(prediction - self.clim, target - self.clim)
        self.rmse_curve[:, idt] += self.mult * torch.sum(self.rmse_handle(prediction, target), dim=0)
        self.acc_counter[idt] += 1

        # only update this at the first step
        if idt == 0:
            self.valid_steps += 1.
            self.valid_loss += loss
            self.valid_l1 += self.l1_handle(prediction, target)

        return

            
    def finalize(self, final_inference=False):
        """Finalize routine to gather all of the metrics to rank 0 and assemble logs"""

        # sync here
        if dist.is_initialized():
            dist.barrier(device_ids=[self.device.index])
        
        with torch.no_grad():

            valid_steps_local = int(self.valid_steps.item())
            
            if dist.is_initialized():
                dist.all_reduce(self.valid_buffer, op=dist.ReduceOp.SUM, group = comm.get_group("data"))
                dist.all_reduce(self.acc_curve, op=dist.ReduceOp.SUM, group = comm.get_group("data"))
                dist.all_reduce(self.rmse_curve, op=dist.ReduceOp.SUM, group = comm.get_group("data"))
                dist.all_reduce(self.acc_counter, op=dist.ReduceOp.SUM, group = comm.get_group("data"))

            # gather from matmul parallel ranks
            if self.split_data_channels:
                # reduce l1
                dist.all_reduce(valid_l1, op=dist.ReduceOp.AVG, group=comm.get_group("matmul"))

                # gather acc curves
                acc_curve_list = torch.split(torch.zeros( (self.N_out_channels, self.valid_autoreg_steps+1), dtype=torch.float32, device=self.device), self.out_channels_local, dim=0)
                acc_curve_list = [x.contiguous() for x in acc_curve_list]
                acc_curve_list[comm.get_rank("matmul")] = self.acc_curve
                dist.all_gather(acc_curve_list, self.acc_curve, group=comm.get_group("matmul"))
                self.acc_curve = torch.cat(acc_curve_list, dim=0)

                # gather acc curves
                rmse_curve_list = torch.split(torch.zeros( (self.N_out_channels, self.valid_autoreg_steps+1), dtype=torch.float32, device=self.device), self.out_channels_local, dim=0)
                rmse_curve_list = [x.contiguous() for x in rmse_curve_list]
                rmse_curve_list[comm.get_rank("matmul")] = self.rmse_curve
                dist.all_gather(rmse_curve_list, self.rmse_curve, group=comm.get_group("matmul"))
                self.rmse_curve = torch.cat(rmse_curve_list, dim=0)
            
            # divide by number of steps
            self.valid_buffer[0:2] = self.valid_buffer[0:2] / self.valid_buffer[2]

            # Pull out autoregessive acc values
            self.acc_curve /= self.acc_counter
            self.rmse_curve /= self.acc_counter

            # compute auc:
            acc_auc = self.simpquad(self.acc_curve, dim=1)
            
            # copy buffers to cpu
            # sync on stream
            if self.stream is not None:
                self.stream.wait_stream(torch.cuda.current_stream())

            # schedule copy
            with torch.cuda.stream(self.stream):
                self.valid_buffer_cpu.copy_(self.valid_buffer, non_blocking=True)
                self.acc_curve_cpu.copy_(self.acc_curve, non_blocking=True)
                self.rmse_curve_cpu.copy_(self.rmse_curve, non_blocking=True)
                self.acc_auc_cpu.copy_(acc_auc, non_blocking=True)
            
            # wait for stream
            if self.stream is not None:
                self.stream.synchronize()
                
            # prepare logs with the minimum content
            valid_buffer_arr = self.valid_buffer_cpu.numpy()
            logs = {'base': {'validation steps' : valid_steps_local,
                             'validation loss': valid_buffer_arr[0],
                             'validation L1': valid_buffer_arr[1]},
                    "metrics": {}}

            # scalar metrics
            valid_rmse_arr = self.rmse_curve_cpu[:, 0].numpy()
            for var_name, var_idx in self.rmse_vars.items():
                logs["metrics"]['validation ' + var_name] = valid_rmse_arr[var_idx]

            acc_auc_arr = self.acc_auc_cpu.numpy()
            for var_name, var_idx in self.acc_auc_vars.items():
                logs["metrics"]['ACC AUC ' + var_name] = acc_auc_arr[var_idx]

            # table
            table_data = []
            acc_curve_arr = self.acc_curve_cpu.numpy()
            for var_name, var_idx in self.acc_vars.items():

                # create table
                for d in range(0, self.valid_autoreg_steps+1):
                    table_data.append(["ACC", f"{var_name}", (d+1) * self.dtxdh, acc_curve_arr[var_idx, d]])

            rmse_curve_arr = self.rmse_curve_cpu.numpy()
            for var_name, var_idx in self.rmse_vars.items():

                # create table
                for d in range(0, self.valid_autoreg_steps+1):
                    table_data.append(["RMSE", f"{var_name}", (d+1) * self.dtxdh, rmse_curve_arr[var_idx, d]])

            # add table
            logs["metrics"]["rollouts"] = wandb.Table(data = table_data, columns = ["metric type", "variable name", "time [h]", "value"])

        self.logs = logs

        if final_inference:
            return logs, self.acc_curve, self.rmse_curve
        else:
            return logs
