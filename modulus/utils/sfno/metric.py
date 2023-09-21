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
import torch.distributed as dist

# distributed computing stuff
from modulus.utils.sfno.distributed import comm
from modulus.utils.sfno.distributed.mappings import gather_from_parallel_region

# we need those here:
from modulus.utils.sfno.metrics.weighted_acc_rmse import (
    Quadrature,
    l1_torch_distributed,
    l1_torch_local,
    lat_torch,
    latitude_weighting_factor_torch,
    weighted_acc_torch_distributed,
    weighted_acc_torch_local,
    weighted_rmse_torch_distributed,
    weighted_rmse_torch_local,
)


class MetricsHandler:
    """
    A class that handles metrics for model training and validation. It calculates
    metrics such as root mean square error (RMSE)and  anomaly correlation coefficient
    (ACC) for specified variables.
    """

    # TODO not to hardcode channel indices
    def __init__(
        self,
        params,
        mult,
        clim,
        device,
        rmse_var_names=["u10m", "t2m", "u500", "z500", "r500"],
        acc_vars_names=["u10m", "t2m", "u500", "z500", "r500"],
        acc_auc_var_names=["u10m", "t2m", "u500", "z500", "r500"],
    ):  # pragma: no cover

        self.device = device
        self.log_to_screen = params.log_to_screen
        self.log_to_wandb = params.log_to_wandb
        self.ifs_acc_path = params.ifs_acc_path
        self.channel_names = params.channel_names

        # set a stream
        self.stream = torch.cuda.Stream()

        # select the vars which are actually present
        rmse_var_names = [x for x in rmse_var_names if x in self.channel_names]
        acc_vars_names = [x for x in acc_vars_names if x in self.channel_names]
        acc_auc_var_names = [x for x in acc_auc_var_names if x in self.channel_names]

        # now create an inverse mapping
        rmse_vars = {
            var_name: self.channel_names.index(var_name) for var_name in rmse_var_names
        }
        acc_vars = {
            var_name: self.channel_names.index(var_name) for var_name in acc_vars_names
        }
        acc_auc_vars = {
            var_name: self.channel_names.index(var_name)
            for var_name in acc_auc_var_names
        }

        self.rmse_vars = rmse_vars
        self.acc_vars = acc_vars
        self.acc_auc_vars = acc_auc_vars

        self.split_data_channels = params.split_data_channels

        # get some stats: make data shared with tensor from the class
        self.mult = mult.to(self.device)

        # how many steps to run in acc curve
        self.valid_autoreg_steps = params.valid_autoreg_steps

        # climatology for autoregressive ACC
        self.simpquad = Quadrature(
            self.valid_autoreg_steps,
            1.0 / float(self.valid_autoreg_steps + 1),
            self.device,
        )
        clim = torch.unsqueeze(clim, 0)
        self.clim = clim.to(self.device, dtype=torch.float32)
        matmul_comm_size = comm.get_size("matmul")

        # get global and local output channels
        self.N_out_channels = params.N_out_channels
        if self.split_data_channels:
            self.out_channels_local = (
                params.N_out_channels + matmul_comm_size - 1
            ) // matmul_comm_size
            # split channel-wise climatology by matmul parallel rank
            mprank = comm.get_rank("matmul")
            self.clim = torch.split(self.clim, self.out_channels_local, dim=1)[
                mprank
            ].contiguous()
        else:
            self.out_channels_local = params.N_out_channels

        # compute latitude weighting factor
        # lat_local = lat_torch(torch.arange(start=params.img_local_offset_x, end=(params.img_local_offset_x + params.img_local_shape_x), device=self.device), params.img_crop_shape_x)
        lat_full = lat_torch(
            torch.arange(start=0, end=params.img_crop_shape_x, device=self.device),
            params.img_crop_shape_x,
        )
        lat_norm = torch.sum(torch.cos(torch.deg2rad(lat_full)))
        # lwf = latitude_weighting_factor_torch(lat_local, params.img_crop_shape_x, lat_norm)
        # using the global one for now as we are gathering
        lwf = latitude_weighting_factor_torch(lat_full, params.img_shape_x, lat_norm)[
            params.img_crop_offset_x : params.img_crop_offset_x
            + params.img_crop_shape_x
        ]
        self.latitude_weights = torch.reshape(lwf, (1, 1, -1, 1))
        self.img_shape = (params.img_crop_shape_x, params.img_crop_shape_y)

        # which rmse compute handle do we need to use:
        if params.multigrid_mode == "distributed":
            self.metric_correction_factor = 1.0 / float(comm.get_size("matmul"))
            self.l1_handle = l1_torch_distributed
            self.weighted_rmse_handle = weighted_rmse_torch_distributed
            self.weighted_acc_handle = weighted_acc_torch_distributed
        else:
            self.metric_correction_factor = 1.0
            self.l1_handle = l1_torch_local
            self.weighted_rmse_handle = weighted_rmse_torch_local
            self.weighted_acc_handle = weighted_acc_torch_local

        self.do_gather_input = False
        if comm.get_size("h") * comm.get_size("w") > 1:
            self.do_gather_input = True

    @torch.jit.ignore
    def _gather_input(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        """helper that gathers data from spatially distributed regions"""
        # combine data
        # h
        xh = gather_from_parallel_region(x, -2, "h")
        xw = gather_from_parallel_region(xh, -1, "w")

        # crop
        x = xw[:, :, : self.img_shape[0], : self.img_shape[1]]

        return x

    def initialize_buffers(self):  # pragma: no cover
        """Initialize buffers for the validation metrics"""
        self.valid_buffer = torch.zeros((3), dtype=torch.float32, device=self.device)
        self.valid_loss = self.valid_buffer[0].view(-1)
        self.valid_l1 = self.valid_buffer[1].view(-1)
        self.valid_steps = self.valid_buffer[2].view(-1)

        # we need these buffers
        self.valid_weighted_rmse = torch.zeros(
            (self.out_channels_local), dtype=torch.float32, device=self.device
        )
        self.acc_curve = torch.zeros(
            (self.out_channels_local, self.valid_autoreg_steps + 1),
            dtype=torch.float32,
            device=self.device,
        )
        self.acc_counter = torch.zeros(
            (self.valid_autoreg_steps + 1), dtype=torch.float32, device=self.device
        )

        # create CPU copies for all the buffers
        self.valid_buffer_cpu = torch.zeros(
            (3), dtype=torch.float32, device="cpu"
        ).pin_memory()
        self.valid_weighted_rmse_cpu = torch.zeros(
            (self.out_channels_local), dtype=torch.float32, device="cpu"
        ).pin_memory()
        self.acc_curve_cpu = torch.zeros(
            (self.out_channels_local, self.valid_autoreg_steps + 1),
            dtype=torch.float32,
            device="cpu",
        ).pin_memory()
        self.acc_auc_cpu = torch.zeros(
            (self.out_channels_local), dtype=torch.float32, device="cpu"
        ).pin_memory()

    def zero_buffers(self):  # pragma: no cover
        """Helper that zeros out buffers"""
        with torch.inference_mode():
            with torch.no_grad():
                self.valid_buffer.fill_(0)
                self.valid_weighted_rmse.fill_(0)
                self.acc_curve.fill_(0)
                self.acc_counter.fill_(0)
        return

    def update(self, prediction, target, loss, idt):  # pragma: no cover
        """Updates the validation metrics with the given prediction and target."""
        if self.do_gather_input:
            prediction = self._gather_input(prediction)
            target = self._gather_input(target)

        # store values for rmse:
        rmse_prediction = prediction
        rmse_target = target

        # update parameters
        self.acc_curve[:, idt] += (
            self.weighted_acc_handle(
                prediction - self.clim, target - self.clim, self.latitude_weights
            )
            * self.metric_correction_factor
        )
        self.acc_counter[idt] += 1
        if idt == 0:
            self.valid_steps += 1.0
            self.valid_loss += loss
            self.valid_l1 += (
                self.l1_handle(prediction, target) * self.metric_correction_factor
            )
            self.valid_weighted_rmse += (
                self.weighted_rmse_handle(
                    rmse_prediction, rmse_target, self.latitude_weights
                )
                * self.metric_correction_factor
            )
        return

    def finalize(self, final_inference=False):  # pragma: no cover
        """
        Finalizes the validation metrics after a validation run. It gathers the metrics
        across different processes, computes the final metrics, and prepares the logs.
        """
        # sync here
        if dist.is_initialized():
            dist.barrier(device_ids=[self.device.index])

        with torch.no_grad():

            valid_steps_local = int(self.valid_steps.item())

            if dist.is_initialized():
                dist.all_reduce(
                    self.valid_buffer,
                    op=dist.ReduceOp.SUM,
                    group=comm.get_group("data"),
                )
                dist.all_reduce(
                    self.valid_weighted_rmse,
                    op=dist.ReduceOp.SUM,
                    group=comm.get_group("data"),
                )
                dist.all_reduce(
                    self.acc_curve, op=dist.ReduceOp.SUM, group=comm.get_group("data")
                )
                dist.all_reduce(
                    self.acc_counter, op=dist.ReduceOp.SUM, group=comm.get_group("data")
                )

            # gather from matmul parallel ranks
            if self.split_data_channels:
                # gather rmse
                valid_weighted_rmse_list = torch.split(
                    torch.zeros(
                        (self.N_out_channels), dtype=torch.float32, device=self.device
                    ),
                    self.out_channels_local,
                    dim=0,
                )
                valid_weighted_rmse_list = [
                    x.contiguous() for x in valid_weighted_rmse_list
                ]
                valid_weighted_rmse_list[
                    comm.get_rank("matmul")
                ] = self.valid_weighted_rmse
                dist.all_gather(
                    valid_weighted_rmse_list,
                    self.valid_weighted_rmse,
                    group=comm.get_group("matmul"),
                )
                self.valid_weighted_rmse = torch.cat(valid_weighted_rmse_list, dim=0)
                # we need to reduce the l1 loss as well, since this is not encoded in the loss obj
                dist.all_reduce(
                    self.valid_l1, op=dist.ReduceOp.AVG, group=comm.get_group("matmul")
                )

                # gather acc curves
                acc_curve_list = torch.split(
                    torch.zeros(
                        (self.N_out_channels, self.valid_autoreg_steps + 1),
                        dtype=torch.float32,
                        device=self.device,
                    ),
                    self.out_channels_local,
                    dim=0,
                )
                acc_curve_list = [x.contiguous() for x in acc_curve_list]
                acc_curve_list[comm.get_rank("matmul")] = self.acc_curve
                dist.all_gather(
                    acc_curve_list, self.acc_curve, group=comm.get_group("matmul")
                )
                self.acc_curve = torch.cat(acc_curve_list, dim=0)

            # divide by number of steps
            self.valid_buffer[0:2] = self.valid_buffer[0:2] / self.valid_buffer[2]
            self.valid_weighted_rmse = (
                self.mult * self.valid_weighted_rmse / self.valid_buffer[2]
            )

            # Pull out autoregessive acc values
            self.acc_curve /= self.acc_counter

            # compute auc
            acc_auc = self.simpquad(self.acc_curve, dim=1)

            # copy buffers to cpu
            # sync on stream
            self.stream.wait_stream(torch.cuda.current_stream())

            # schedule copy
            with torch.cuda.stream(self.stream):
                self.valid_buffer_cpu.copy_(self.valid_buffer, non_blocking=True)
                self.valid_weighted_rmse_cpu.copy_(
                    self.valid_weighted_rmse, non_blocking=True
                )
                self.acc_curve_cpu.copy_(self.acc_curve, non_blocking=True)
                self.acc_auc_cpu.copy_(acc_auc, non_blocking=True)

            # wait for stream
            self.stream.synchronize()

            # prepare logs with the minimum content
            valid_buffer_arr = self.valid_buffer_cpu.numpy()
            logs = {
                "base": {
                    "validation steps": valid_steps_local,
                    "validation loss": valid_buffer_arr[0],
                    "validation L1": valid_buffer_arr[1],
                },
                "metrics": {},
            }

            valid_weighted_rmse_arr = self.valid_weighted_rmse_cpu.numpy()
            for var_name, var_idx in self.rmse_vars.items():
                logs["metrics"]["validation " + var_name] = valid_weighted_rmse_arr[
                    var_idx
                ]

            acc_curve_arr = self.acc_curve_cpu.numpy()
            for var_name, var_idx in self.acc_vars.items():
                logs["metrics"]["ACC time " + var_name] = acc_curve_arr[
                    var_idx, self.valid_autoreg_steps
                ]

            acc_auc_arr = self.acc_auc_cpu.numpy()
            for var_name, var_idx in self.acc_auc_vars.items():
                logs["metrics"]["ACC AUC " + var_name] = acc_auc_arr[var_idx]

        self.logs = logs

        if final_inference:
            return logs, self.acc_curve
        else:
            return logs
