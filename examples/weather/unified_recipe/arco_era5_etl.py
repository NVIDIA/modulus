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

import numpy as np
import zarr
import xarray as xr
import fsspec
from typing import Any, Iterable, List, Union, Tuple
from pathlib import Path
from tqdm import tqdm
from dask.diagnostics import ProgressBar
import datetime

class ARCOERA5ETL:
    """
    ETL script for converting ARCO ERA5 dataset loadable with Zarr datapipe
    """

    def __init__(
        self,
        unpredicted_variables: List[str],
        predicted_variables: List[str],
        dataset_filename: Union[str, Path] = "./data.zarr",
        fs: fsspec.filesystem = fsspec.filesystem("file"),
        transform: None = None,
        date_range: Tuple[str, str] = ("2000-01-01", "2001-01-01"),
        dt: int = 1, # 1 hour
    ):
        super().__init__()

        # Store parameters
        self.unpredicted_variables = unpredicted_variables
        self.predicted_variables = predicted_variables
        self.dataset_filename = dataset_filename
        self.fs = fs
        self.transform = transform
        self.date_range = date_range
        assert dt in [1, 3, 6, 12], "dt must be 1, 3, 6, or 12"
        self.dt = dt

        # Load arco xarray
        arco_fs = fsspec.filesystem('gs')
        self.arco_filename = 'gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3'
        self.arco_mapper = arco_fs.get_mapper(self.arco_filename)
        self.arco_era5 = xr.open_zarr(self.arco_mapper, consolidated=True)

        # Subset variables (this speeds up chunking)
        needed_variables = ["latitude", "longitude", "time", "level"]
        for variable in self.unpredicted_variables + self.predicted_variables:
            if not isinstance(variable, str):
                needed_variables.append(variable[0])
            else:
                needed_variables.append(variable)
        for variables in self.arco_era5.variables:
            if variables not in needed_variables:
                self.arco_era5 = self.arco_era5.drop_vars(variables)

        # Chunk data
        self.arco_era5 = self.arco_era5.sel(time=slice(datetime.datetime.strptime(date_range[0], "%Y-%m-%d"), datetime.datetime.strptime(date_range[1], "%Y-%m-%d")))
        self.arco_era5 = self.arco_era5.sel(time=self.arco_era5.time.dt.hour.isin(np.arange(0, 24, self.dt)))
        self.arco_era5 = self.arco_era5.chunk({"time": 1, "level": 1, "latitude": 721, "longitude": 1440})

        # Gather all predicted variables
        xarray_predicted_variables = []
        for variable in self.predicted_variables:
            if not isinstance(variable, str): # TODO: better way to check if list
                pressure_variable = self.arco_era5[variable[0]].sel(level=variable[1])
                pressure_variable = pressure_variable.drop("level")
                pressure_variable = pressure_variable.rename({"level": "predicted_channel"})
                xarray_predicted_variables.append(pressure_variable)
            else:
                single_variable = self.arco_era5[variable]
                single_variable = single_variable.expand_dims("predicted_channel", axis=1)
                xarray_predicted_variables.append(single_variable)

        # Gather all unpredicted variables
        xarray_unpredicted_variables = []
        for variable in self.unpredicted_variables:
            if not isinstance(variable, str): # TODO: better way to check if list
                pressure_variable = self.arco_era5[variable[0]].sel(level=variable[1])
                pressure_variable = pressure_variable.drop("level")
                pressure_variable = pressure_variable.rename({"level": "unpredicted_channel"})
                xarray_unpredicted_variables.append(pressure_variable)
            else:
                single_variable = self.arco_era5[variable]
                single_variable = single_variable.expand_dims("unpredicted_channel", axis=1)
                xarray_unpredicted_variables.append(single_variable)

        # Concatenate all variables 
        self.arco_era5_subset = xr.Dataset()
        self.arco_era5_subset["predicted"] = xr.concat(xarray_predicted_variables, dim="predicted_channel")
        self.arco_era5_subset["unpredicted"] = xr.concat(xarray_unpredicted_variables, dim="unpredicted_channel")
        self.arco_era5_subset['time'] = self.arco_era5['time']
        self.arco_era5_subset.drop_vars(['latitude', 'longitude']) # Maybe keep these?
        self.arco_era5_subset = self.arco_era5_subset.chunk({"time": 1, "predicted_channel": self.arco_era5_subset.predicted_channel.size, "unpredicted_channel": self.arco_era5_subset.unpredicted_channel.size})

    def __call__(self):
        """
        Generate the zarr array
        """

        # Check if already exists
        if self.fs.exists(self.dataset_filename):
            print(f"Zarr file {self.dataset_filename} already exists. Skipping.")
            return

        # Run transform if specified
        if self.transform is not None:
            self.arco_era5_subset = self.transform(self.arco_era5_subset)

        # Save
        mapper = self.fs.get_mapper(self.dataset_filename)
        delayed_obj = self.arco_era5_subset.to_zarr(mapper, consolidated=True, compute=False)

        # Wait for save to finish
        with ProgressBar():
            delayed_obj.compute()
