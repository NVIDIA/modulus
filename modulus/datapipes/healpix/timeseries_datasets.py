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

import logging
import os
import shutil
import time
from typing import DefaultDict, Optional, Sequence, Union

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr
from torch.utils.data import Dataset
from dask.diagnostics import ProgressBar
from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn.functional as F
import torch.distributed as dist

from modulus.datapipes.datapipe import Datapipe
from modulus.datapipes.meta import DatapipeMetaData

from modulus.utils.insolation import insolation
import modulus.datapipes.healpix.couplers as couplers 

logger = logging.getLogger(__name__)

@dataclass
class TimeSeriesMetaData(DatapipeMetaData):
    name: str = "TimeSeries"
    # Optimization
    auto_device: bool = False
    cuda_graphs: bool = False
    # Parallel
    ddp_sharding: bool = False

class TimeSeriesDataset(Dataset, Datapipe):
    def __init__(
            self,
            dataset: xr.Dataset,
            scaling: DictConfig,
            input_time_dim: int = 1,
            output_time_dim: int = 1,
            data_time_step: Union[int, str] = '3H',
            time_step: Union[int, str] = '6H',
            gap: Union[int, str, None] = None,
            batch_size: int = 32,
            drop_last: bool = False,
            add_insolation: bool = False,
            forecast_init_times: Optional[Sequence] = None
    ):
        """
        Dataset for sampling from continuous time-series data, compatible with pytorch data loading.

        :param dataset: xarray Dataset produced by one of the `open_*` methods herein
        :param scaling: dictionary containing scaling parameters for data variables
        :param input_time_dim: number of time steps in the input array
        :param output_time_dim: number of time steps in the output array
        :param data_time_step: either integer hours or a str interpretable by pandas: time between steps in the
            original data time series
        :param time_step: either integer hours or a str interpretable by pandas: desired time between effective model
            time steps
        :param gap: either integer hours or a str interpretable by pandas: time step between the last input time and
            the first output time. Defaults to `time_step`.
        :param batch_size: batch size
        :param drop_last: whether to drop the last batch if it is smaller than batch_size
        :param add_insolation: option to add prescribed insolation as a decoder input feature
        :param forecast_init_times: a Sequence of pandas Timestamps dictating the specific initialization times
            to produce inputs for. Note that providing this parameter configures the data loader to only produce
            this number of samples, and NOT produce any target array.
        """
        Datapipe.__init__(
            self,
            meta=TimeSeriesMetaData(),
        )
        self.ds = dataset
        self.scaling = OmegaConf.to_object(scaling)
        self.input_time_dim = input_time_dim
        self.output_time_dim = output_time_dim
        self.data_time_step = self._convert_time_step(data_time_step)
        self.time_step = self._convert_time_step(time_step)
        self.gap = self._convert_time_step(gap if gap is not None else time_step)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.add_insolation = add_insolation
        self.forecast_init_times = forecast_init_times
        self.forecast_mode = self.forecast_init_times is not None

        # Time stepping
        if (self.time_step % self.data_time_step).total_seconds() != 0:
            raise ValueError(f"'time_step' must be a multiple of 'data_time_step' "
                             f"(got {self.time_step} and {self.data_time_step}")
        if (self.gap % self.data_time_step).total_seconds() != 0:
            raise ValueError(f"'gap' must be a multiple of 'data_time_step' "
                             f"(got {self.gap} and {self.data_time_step}")
        self.interval = self.time_step // self.data_time_step

        # Find indices of init times for forecast mode
        if self.forecast_mode:
            if self.batch_size != 1:
                self.batch_size = 1
                logger.warning("providing 'forecast_init_times' to TimeSeriesDataset requires `batch_size=1`; "
                               "setting it now")
            self._forecast_init_indices = np.array(
                [int(np.where(self.ds['time'] == s)[0]) for s in self.forecast_init_times],
                dtype='int'
            ) - ((self.input_time_dim - 1) * self.interval)
        else:
            self._forecast_init_indices = None

        # Length of the data window needed for one sample.
        if self.forecast_mode:
            self._window_length = self.interval * (self.input_time_dim - 1) + 1
        else:
            self._window_length = (
                    self.interval * (self.input_time_dim - 1) + 1 +
                    (self.gap // self.data_time_step) +
                    self.interval * (self.output_time_dim - 1)  # first point is counted by gap
            )
        self._batch_window_length = self.batch_size + self._window_length - 1
        self._output_delay = self.interval * (self.input_time_dim - 1) + (self.gap // self.data_time_step)
        # Indices within a batch
        self._input_indices = [list(range(n, n + self.interval * self.input_time_dim, self.interval))
                               for n in range(self.batch_size)]
        self._output_indices = [list(range(n + self._output_delay,
                                           n + self.interval * self.output_time_dim + self._output_delay,
                                           self.interval))
                               for n in range(self.batch_size)]

        self.spatial_dims = (self.ds.dims['face'], self.ds.dims['height'], self.ds.dims['width'])

        self.input_scaling = None
        self.target_scaling = None
        self._get_scaling_da()

    def get_constants(self):
        # extract from ds:
        const = self.ds.constants.values

        # transpose to match new format:
        # [C, F, H, W] -> [F, C, H, W]
        const = np.transpose(const, axes=(1, 0, 2, 3))

        return const

    @staticmethod
    def _convert_time_step(dt):  # pylint: disable=invalid-name
        return pd.Timedelta(hours=dt) if isinstance(dt, int) else pd.Timedelta(dt)

    def _get_scaling_da(self):
        scaling_df = pd.DataFrame.from_dict(self.scaling).T
        scaling_df.loc['zeros'] = {'mean': 0., 'std': 1.}
        scaling_da = scaling_df.to_xarray().astype('float32')

        # REMARK: we remove the xarray overhead from these
        try:
            self.input_scaling = scaling_da.sel(index=self.ds.channel_in.values).rename({'index': 'channel_in'})
            self.input_scaling = {"mean": np.expand_dims(self.input_scaling["mean"].to_numpy(), (0, 2, 3, 4)),
                                  "std": np.expand_dims(self.input_scaling["std"].to_numpy(), (0, 2, 3, 4))}
        except (ValueError, KeyError):
            raise KeyError(f"one or more of the input data variables f{list(self.ds.channel_in)} not found in the "
                           f"scaling config dict data.scaling ({list(self.scaling.keys())})")
        try:
            self.target_scaling = scaling_da.sel(index=self.ds.channel_out.values).rename({'index': 'channel_out'})
            self.target_scaling = {"mean": np.expand_dims(self.target_scaling["mean"].to_numpy(), (0, 2, 3, 4)),
                                   "std": np.expand_dims(self.target_scaling["std"].to_numpy(), (0, 2, 3, 4))}
        except (ValueError, KeyError):
            raise KeyError(f"one or more of the target data variables f{list(self.ds.channel_out)} not found in the "
                           f"scaling config dict data.scaling ({list(self.scaling.keys())})")

    def __len__(self):
        if self.forecast_mode:
            return len(self._forecast_init_indices)
        length = (self.ds.dims['time'] - self._window_length + 1) / self.batch_size
        if self.drop_last:
            return int(np.floor(length))
        return int(np.ceil(length))

    def _get_time_index(self, item):
        start_index = self._forecast_init_indices[item] if self.forecast_mode else item * self.batch_size
        # TODO: I think this should be -1 and still work (currently missing the last sample in last batch)
        max_index = start_index + self._window_length if self.forecast_mode else \
            (item + 1) * self.batch_size + self._window_length
        if not self.drop_last and max_index > self.ds.dims['time']:
            batch_size = self.batch_size - (max_index - self.ds.dims['time'])
        else:
            batch_size = self.batch_size
        return (start_index, max_index), batch_size

    def _get_forecast_sol_times(self, item):
        time_index, _ = self._get_time_index(item)
        if self.forecast_mode:
            timedeltas = np.array(self._input_indices[0] + self._output_indices[0]) * self.data_time_step
            return self.ds.time[time_index[0]].values + timedeltas
        return self.ds.time[slice(*time_index)].values

    def __getitem__(self, item):

        #return self.inputs_result, self.target

        # start range
        torch.cuda.nvtx.range_push("TimeSeriesDataset:__getitem__")
        
        if item < 0:
            item = len(self) + item
        if item < 0 or item > len(self):
            raise IndexError(f"index {item} out of range for dataset with length {len(self)}")

        # remark: load first then normalize
        torch.cuda.nvtx.range_push("TimeSeriesDataset:__getitem__:load_batch")
        time_index, this_batch = self._get_time_index(item)
        batch = {'time': slice(*time_index)}
        load_time = time.time()


        input_array = self.ds['inputs'].isel(**batch).to_numpy()
        input_array = (input_array - self.input_scaling['mean']) / self.input_scaling['std']

        if not self.forecast_mode:
            target_array = self.ds['targets'].isel(**batch).to_numpy()
            target_array = (target_array - self.target_scaling['mean']) / self.target_scaling['std']
            
        logger.log(5, "loaded batch data in %0.2f s", time.time() - load_time)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("TimeSeriesDataset:__getitem__:process_batch")
        compute_time = time.time()
        # Insolation
        if self.add_insolation:
            sol = insolation(self._get_forecast_sol_times(item), self.ds.lat.values, self.ds.lon.values)[:, None]
            decoder_inputs = np.empty((this_batch, self.input_time_dim + self.output_time_dim, 1) +
                                      self.spatial_dims, dtype='float32')

        # Get buffers for the batches, which we'll fill in iteratively.
        inputs = np.empty((this_batch, self.input_time_dim, self.ds.dims['channel_in']) +
                          self.spatial_dims, dtype='float32')
        if not self.forecast_mode:
            targets = np.empty((this_batch, self.output_time_dim, self.ds.dims['channel_out']) +
                               self.spatial_dims, dtype='float32')

        # Iterate over valid sample windows
        for sample in range(this_batch):
            inputs[sample] = input_array[self._input_indices[sample]]
            if not self.forecast_mode:
                targets[sample] = target_array[self._output_indices[sample]]
            if self.add_insolation:
                decoder_inputs[sample] = sol if self.forecast_mode else \
                    sol[self._input_indices[sample] + self._output_indices[sample]]

        inputs_result = [inputs]
        if self.add_insolation:
            inputs_result.append(decoder_inputs)

        # we need to transpose channels and data:
        # [B, T, C, F, H, W] -> [B, F, T, C, H, W]
        inputs_result = [np.transpose(x, axes=(0, 3, 1, 2, 4, 5)) for x in inputs_result]
            
        if 'constants' in self.ds.data_vars:
            # Add the constants as [F, C, H, W]
            inputs_result.append(np.swapaxes(self.ds.constants.values, 0, 1))
            #inputs_result.append(self.ds.constants.values)
        logger.log(5, "computed batch in %0.2f s", time.time() - compute_time)
        torch.cuda.nvtx.range_pop()

        # finish range
        torch.cuda.nvtx.range_pop()


        if self.forecast_mode:
            return inputs_result

        # we also need to transpose targets
        targets = np.transpose(targets, axes=(0, 3, 1, 2, 4, 5))

        return inputs_result, targets

@dataclass
class CoupledTimeSeriesMetaData(DatapipeMetaData):
    name: str = "CoupledTimeSeries"
    # Optimization
    auto_device: bool = False
    cuda_graphs: bool = False
    # Parallel
    ddp_sharding: bool = False

class CoupledTimeSeriesDataset(TimeSeriesDataset, Datapipe):
    def __init__(
            self,
            dataset: xr.Dataset,
            scaling: DictConfig,
            input_variables: Sequence,
            input_time_dim: int = 1,
            presteps: int = 0,
            output_time_dim: int = 1,
            data_time_step: Union[int, str] = '3H',
            time_step: Union[int, str] = '6H',
            gap: Union[int, str, None] = None,
            batch_size: int = 32,
            drop_last: bool = False,
            add_insolation: bool = False,
            forecast_init_times: Optional[Sequence] = None,
            couplings: Sequence = []
    ):
        """
        Dataset for coupling TimesSeriesDataset with external inputs from various earth system 
        components

        :param dataset: xarray Dataset produced by one of the `open_*` methods herein
        :param scaling: dictionary containing scaling parameters for data variables
        :param input_variables: a sequence of variables that will be ingested in to model 
        :param input_time_dim: number of time steps in the input array
        :param presteps: number of steps to initialize GRU 
        :param output_time_dim: number of time steps in the output array
        :param data_time_step: either integer hours or a str interpretable by pandas: time between steps in the
            original data time series
        :param time_step: either integer hours or a str interpretable by pandas: desired time between effective model
            time steps
        :param gap: either integer hours or a str interpretable by pandas: time step between the last input time and
            the first output time. Defaults to `time_step`.
        :param batch_size: batch size
        :param drop_last: whether to drop the last batch if it is smaller than batch_size
        :param add_insolation: option to add prescribed insolation as a decoder input feature
        :param forecast_init_times: a Sequence of pandas Timestamps dictating the specific initialization times
            to produce inputs for. Note that providing this parameter configures the data loader to only produce
            this number of samples, and NOT produce any target array.
        :param couplings: a Sequence of dictionaries that define the mechanics of couplings with other earth system
            components 
        """
        Datapipe.__init__(
            self,
            meta=CoupledTimeSeriesMetaData(),
        )
        self.input_variables = input_variables 
        if couplings is not None:
            self.couplings = [
                getattr(couplers,c['coupler'])(
                    dataset,
                    **OmegaConf.to_object(DictConfig(c))['params'],
                    ) for c in couplings
            ]
        else: 
            self.couplings = None
        super().__init__(
            dataset=dataset,
            scaling=scaling,
            input_time_dim=input_time_dim,
            output_time_dim=output_time_dim,
            data_time_step=data_time_step,
            time_step=time_step,
            gap=gap,
            batch_size=batch_size,
            drop_last=drop_last,
            add_insolation=add_insolation,
            forecast_init_times=forecast_init_times,
        )
        # calculate static indices for coupling 
        for c in self.couplings:
            c.compute_coupled_indices(self.interval, self.data_time_step)
             
    def _get_scaling_da(self):
        scaling_df = pd.DataFrame.from_dict(self.scaling).T
        scaling_df.loc['zeros'] = {'mean': 0., 'std': 1.}
        scaling_da = scaling_df.to_xarray().astype('float32')

        for c in self.couplings:
            c.set_scaling(scaling_da)
        # REMARK: we remove the xarray overhead from these
        try:
            self.input_scaling = scaling_da.sel(index=self.input_variables).rename({'index': 'channel_in'})
            self.input_scaling = {"mean": np.expand_dims(self.input_scaling["mean"].to_numpy(), (0, 2, 3, 4)),
                                  "std": np.expand_dims(self.input_scaling["std"].to_numpy(), (0, 2, 3, 4))}
        except (ValueError, KeyError):
            raise KeyError(f"one or more of the input data variables f{list(self.ds.channel_in)} not found in the "
                           f"scaling config dict data.scaling ({list(self.scaling.keys())})")
        try:
            self.target_scaling = scaling_da.sel(index=self.input_variables).rename({'index': 'channel_out'})
            self.target_scaling = {"mean": np.expand_dims(self.target_scaling["mean"].to_numpy(), (0, 2, 3, 4)),
                                   "std": np.expand_dims(self.target_scaling["std"].to_numpy(), (0, 2, 3, 4))}
        except (ValueError, KeyError):
            raise KeyError(f"one or more of the target data variables f{list(self.ds.channel_out)} not found in the "
                           f"scaling config dict data.scaling ({list(self.scaling.keys())})")

    def __getitem__(self, item):

        #return self.inputs_result, self.target

        # start range
        torch.cuda.nvtx.range_push("CoupledTimeSeriesDataset:__getitem__")
        
        if item < 0:
            item = len(self) + item
        if item < 0 or item > len(self):
            raise IndexError(f"index {item} out of range for dataset with length {len(self)}")

        # remark: load first then normalize
        torch.cuda.nvtx.range_push("CoupledTimeSeriesDataset:__getitem__:load_batch")
        time_index, this_batch = self._get_time_index(item)
        batch = {'time': slice(*time_index)}
        load_time = time.time()


        input_array = self.ds['inputs'].sel(channel_in=self.input_variables).isel(**batch).to_numpy()
        # retrieve coupled inputs 
        if len(self.couplings) > 0:
            integrated_couplings = np.concatenate([c.construct_integrated_couplings(batch, this_batch)\
                                                   for c in self.couplings],
                                                   axis=2)
                                            

        input_array = (input_array - self.input_scaling['mean']) / self.input_scaling['std']
        if not self.forecast_mode:
            target_array = self.ds['targets'].isel(**batch).to_numpy()
            target_array = (target_array - self.target_scaling['mean']) / self.target_scaling['std']
            
        logger.log(5, "loaded batch data in %0.2f s", time.time() - load_time)
        torch.cuda.nvtx.range_pop()

        torch.cuda.nvtx.range_push("CoupledTimeSeriesDataset:__getitem__:process_batch")
        compute_time = time.time()
        # Insolation
        if self.add_insolation:
            sol = insolation(self._get_forecast_sol_times(item), self.ds.lat.values, self.ds.lon.values)[:, None]
            decoder_inputs = np.empty((this_batch, self.input_time_dim + self.output_time_dim, 1) +
                                      self.spatial_dims, dtype='float32')

        # Get buffers for the batches, which we'll fill in iteratively.
        inputs = np.empty((this_batch, self.input_time_dim, 
                           len(self.input_variables)) +
                          self.spatial_dims, dtype='float32')
        if not self.forecast_mode:
            targets = np.empty((this_batch, self.output_time_dim, self.ds.dims['channel_out']) +
                               self.spatial_dims, dtype='float32')

        # Iterate over valid sample windows
        for sample in range(this_batch):
            inputs[sample] = input_array[self._input_indices[sample]]
            if not self.forecast_mode:
                targets[sample] = target_array[self._output_indices[sample]]
            if self.add_insolation:
                decoder_inputs[sample] = sol if self.forecast_mode else \
                    sol[self._input_indices[sample] + self._output_indices[sample]]

        inputs_result = [inputs]
        if self.add_insolation:
            inputs_result.append(decoder_inputs)

        # we need to transpose channels and data:
        # [B, T, C, F, H, W] -> [B, F, T, C, H, W]
        
        inputs_result = [np.transpose(x, axes=(0, 3, 1, 2, 4, 5)) for x in inputs_result]
            
        if 'constants' in self.ds.data_vars:
            # Add the constants as [F, C, H, W]
            inputs_result.append(np.swapaxes(self.ds.constants.values, 0, 1))
        logger.log(5, "computed batch in %0.2f s", time.time() - compute_time)

        # append integrated couplings 
        inputs_result.append(integrated_couplings)

        torch.cuda.nvtx.range_pop()

        # finish range
        torch.cuda.nvtx.range_pop()

        if self.forecast_mode:
            return inputs_result

        # we also need to transpose targets
        targets = np.transpose(targets, axes=(0, 3, 1, 2, 4, 5))
        
        return inputs_result, targets
