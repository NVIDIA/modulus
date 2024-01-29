# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import xarray as xr
import numpy as np
import torch as th
import pandas as pd
from typing import DefaultDict, Optional, Sequence, Union


class TrailingAverageCoupler:
    def __init__(
        self,
        dataset: xr.Dataset,
        batch_size: int,
        variables: Sequence,
        presteps: int = 0,
        input_time_dim: int = 2,
        output_time_dim: int = 2,
        averaging_window: str = "24H",
        input_times: Sequence = [pd.Timedelta("24h"), pd.Timedelta("48H")],
        prepared_coupled_data=True,
    ):
        """
        coupler used to inferface two components of the earth system

        Trailing average coupler uses coupled input times as the right side of
        an averag that is taken over an "averaging_window" window size.

        :param dataset: xr.Dataset that holds coupled data
        :param batch_size: int that indicated the batch size during training.
               forecasting batch size should be 1
        :param variables: sequence of strings that indicate the coupled variable
               names in the dataset
        :param presteps: int the number of model steps used to initialize the
               hidden state. If not using a GRU, prestep is 0
        :param input_time_dim: int number of input times into the model
        :param output_time_dim: int number of output times for each model step
        :param averaging_window: period over which coupled data is averaged before
               sent back to model
        :param input_times: sequence of pandas Timedelta objects that indicate
               which times are to be coupled
        :param prepared_coupled_data: boolean. If True assumes data in dataset has
               been prepared approiately for training: averages have already been
               calculated so that each time step denotes the right side of a
               averaging_window window. This is highly remcommended for training
        """
        # extract important meta data from ds
        self.ds = dataset
        self.batch_size = batch_size
        self.spatial_dims = self.ds.inputs.shape[2:]
        self.variables = variables
        self.presteps = presteps
        self.input_time_dim = input_time_dim
        self.output_time_dim = output_time_dim
        self.input_times = [pd.Timedelta(t) for t in input_times]
        self.averaging_window = pd.Timedelta(averaging_window)
        self.output_channels = len(self.variables) * len(self.input_times)
        self._set_time_increments()
        self.coupled_integration_dim = self._compute_coupled_integration_dim()
        self.timevar_dim = self._compute_timevar_dim()
        self.coupled_inputs_shape = None
        self.scaling_dict = None
        self._coupled_offsets = None
        self.integrated_couplings = None

        if not prepared_coupled_data:
            print(
                "Assuming coupled data is not preprocessed, averaging fields in as designed in\
 TrailingAverageCoupler. See docs for specifics."
            )
            self._prepare_coupled_data()
        else:
            print(
                '**Assuming coupled data has been prepared properly, using coupled field[s] from\
 dataset "as-is"**'
            )

    def compute_coupled_indices(self, interval, data_time_step):
        """
        Called by CoupledDataset to compute static indices for training
        samples

        :param interval: int ratio of dataset timestep to model dt
        :param data_time_step: dataset timestep
        """

        # create array of static coupled offstes that accompany each batch
        self._coupled_offsets = np.empty(
            [self.batch_size, self.coupled_integration_dim, len(self.input_times)]
        )
        for b in range(self.batch_size):
            for i in range(self.coupled_integration_dim):
                self._coupled_offsets[b, i, :] = (
                    b
                    + ((self.input_time_dim * i) + 1) * interval
                    + np.array([ts / data_time_step for ts in self.input_times])
                )

        self._coupled_offsets = self._coupled_offsets.astype(int)

    def _prepare_coupled_data(self):
        # TODO: write function to lazily compute average as spcified in time scheme
        raise NotImplementedError("Data preparation not yet implemented")

    def set_scaling(self, scaling_da):
        coupled_scaling = scaling_da.sel(index=self.variables).rename(
            {"index": "channel_in"}
        )
        self.coupled_scaling = {
            "mean": np.expand_dims(coupled_scaling["mean"].to_numpy(), (0, 2, 3, 4)),
            "std": np.expand_dims(coupled_scaling["std"].to_numpy(), (0, 2, 3, 4)),
        }

    def _set_time_increments(self):
        # get the dt of the dataset
        dt = pd.Timedelta(
            self.ds.time[1].values - self.ds.time[0].values
        ).total_seconds()
        # assert that the time increments are divisible by the dt of the dataset
        if np.any([t.total_seconds() % dt != 0 for t in self.input_times]):
            raise ValueError(
                f"Coupled input times {self.input_times} \
({[t.total_seconds() for t in self.input_times]} in secs) are not divisible by dataset dt: {dt}"
            )
        self.time_increments = [t.total_seconds() / dt for t in self.input_times]

    def _compute_timevar_dim(self):
        return len(self.input_times) * len(self.variables)

    def _compute_coupled_integration_dim(self):
        return self.presteps + max(self.output_time_dim // self.input_time_dim, 1)

    def construct_integrated_couplings(
        self,
        batch,
        bsize,
    ):
        """
        Construct array of coupled inputs that includes values required for
        model integration steps.

        :param batch: indices of dataset sample dimension associated with
               current batch
        :param bsize: int batch size
        """

        # reset integrated couplings
        self.integrated_couplings = np.empty(
            (bsize, self.coupled_integration_dim, self.timevar_dim) + self.spatial_dims
        )

        # extract coupled variables and scale lazily
        ds = (
            self.ds.inputs.sel(channel_in=self.variables) - self.coupled_scaling["mean"]
        ) / self.coupled_scaling["std"]

        # use static offsets to create integrated coupling array
        for b in range(bsize):
            for i in range(self.coupled_integration_dim):
                coupling_temp = ds.isel(
                    time=batch["time"].start + self._coupled_offsets[b, i, :]
                )
                self.integrated_couplings[
                    b, i, :, :, :
                ] = coupling_temp.to_numpy().reshape(
                    (self.timevar_dim,) + coupling_temp.shape[2:]
                )
        return self.integrated_couplings.transpose((1, 0, 2, 3, 4, 5)).astype(
            "float32"
        )  # cast to
        # float32 for
        # pytroch compatibility.
