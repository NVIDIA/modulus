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
import tempfile
import xarray as xr
import datetime
import json
import calendar
from typing import List, Tuple, Dict, Union
import urllib3
import logging
import numpy as np
import fsspec
import matplotlib.pyplot as plt
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import queue
from enum import Enum
import time
try:
    import cdsapi
except ImportError:
    import warnings
    warnings.warn("cdsapi not installed. ERA5Mirror will not work.")

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Enum for status
class Status(Enum):
    NOT_STARTED = 0
    IN_PROGRESS = 1
    COMPLETED = 2

@dataclass
class ERA5DownloadChunk:
    """
    A class to represent a chunk of data to be downloaded from the Copernicus Climate Data Store (CDS).

    Attributes
    ----------
    variable : str
        The variable to download.
    year : int
        The year to download.
    month : int
        The month to download.
    hours : List[int]
        The hours to download.
    pressure_level : int
        The pressure level to download. If None, the variable is not a pressure level.
    status : str
        The status of the download. Can be 'not started', 'in progress', or 'completed'.
    """
    variable: str
    year: int
    month: int
    hours: List[int]
    pressure_level: int = None
    status: str = Status.NOT_STARTED

    def __str__(self):
        if self.pressure_level is None:
            return f"{self.variable}_{self.year}_{self.month}_{self.hours}"
        else:
            return f"{self.variable}_{self.pressure_level}_{self.year}_{self.month}_{self.hours}"

class ERA5Mirror:
    """
    A class to manage downloading ERA5 datasets. The datasets are downloaded from the Copernicus Climate Data Store (CDS) and stored in Zarr format on a desired filesystem.
    The data is downloaded in 1 month chunks and the file chunking and compression can be specified.
    When restarting the download or adding new variables, the metadata is used to determine which chunks have already been downloaded.
    Restarting the download with different start date, chunking, compression, or hours will throw an error.
    TODO: rewrite this to use Apache Beam

    Attributes
    ----------
    base_path : Path
        The path to the Zarr dataset.
    fs : fsspec.AbstractFileSystem
        The filesystem to use for the Zarr dataset. If None, the local filesystem will be used.
    chunking : List[int]
        The chunking to use for the Zarr dataset. Must be a list of 3 integers, 'time', 'latitude', and 'longitude'.
        Defaults to [1, 721, 1440].
    compression : str
        The compression used for the Zarr dataset. Defaults to 'none'.
    variables : List[Union[str, Tuple[str, List[int]]]]
        A list of variables to mirror, where each element can either be a string (single-level variable)
        or a tuple (variable with pressure level).
    date_range : Tuple[datetime.date, datetime.date]
        A tuple containing the start and end dates for the data to be mirrored. This will download and store every month in the range.
    dt : int
        Hours between each timestep in the data. Defaults to 1, meaning that the data will be stored at 1-hour intervals.
    num_workers : int
        The number of workers to use for downloading. Defaults to 4. This will be capped at the number of variables.
    progress_plot : str
        Path to save a plot of the download progress. Defaults to "./era5_mirror_progress.png". If None, no plot will be generated
    """

    def __init__(
        self,
        base_path: str,
        variables: List[Union[str, Tuple[str, int]]],
        date_range: Tuple[datetime.date, datetime.date],
        dt: int,
        fs: fsspec.AbstractFileSystem = None,
        chunking: List[int] = [1, 721, 1440],
        compression: str = "none",
        num_workers: int = 16,
        progress_plot: str = "./era5_mirror_progress.png",
        ):

        # Filesystem parameters
        self.base_path = base_path
        if fs is None:
            fs = fsspec.filesystem("file")
        self.fs = fs

        # Zarr parameters
        self.chunking = chunking
        self.compression = compression

        # Variable parameters (reformat so all elements are tuples)
        reformatted_variables = []
        for variable in variables:
            if isinstance(variable, str):
                reformatted_variables.append((variable, None))
            else:
                reformatted_variables.append(tuple(variable))
        self.variables = reformatted_variables

        # Set start and end dates
        self.start_date = date_range[0]
        self.end_date = date_range[1]

        # Set time interval
        self.dt = dt
        self.hours = list(range(0, 24, dt))
        assert 24 % dt == 0, "dt must be a factor of 24"

        # Set progress plot
        self.progress_plot = progress_plot

        # Set number of workers and make queue
        self.num_workers = min(len(self.variables), num_workers)
        self.queue = queue.Queue()
        self.sentinel = object() # Sentinel to stop workers

        # Create the base path if it doesn't exist
        if not self.fs.exists(self.base_path):
            self.fs.makedirs(self.base_path)
        if not self.fs.isdir(os.path.join(self.base_path, "pressure_levels")):
            self.fs.makedirs(os.path.join(self.base_path, "pressure_levels"))
        if not self.fs.isdir(os.path.join(self.base_path, "single_levels")):
            self.fs.makedirs(os.path.join(self.base_path, "single_levels"))

        # Check that current download parameters are compatible with existing dataset
        #self._check_dataset_compatibility()

        # Attempt to load all zarr arrays
        self.zarr_arrays = self._load_zarr_arrays()

        # Create list of download chunks
        self.download_chunks = self._create_download_chunks()

        # Plot progress
        if self.progress_plot is not None:
            self._plot_progress()

        # Start Download
        self._download()

    def _check_dataset_compatibility(self):
        """
        Check that the current download parameters are compatible with the existing dataset.
        """

        # Check if metadata file exists
        self.metadata_file = os.path.join(self.base_path, "metadata.json")

        # If not, create it with the current parameters
        if not self.fs.exists(self.metadata_file):
            metadata = {
                "chunking": self.chunking,
                "compression": self.compression,
                "variables": self.variables,
                "start_date": str(self.start_date),
                "end_date": str(self.end_date),
                "dt": self.dt,
            }
            with self.fs.open(self.metadata_file, "w") as f:
                json.dump(metadata, f)

        # If it does, check that the parameters match
        else:
            with self.fs.open(self.metadata_file, "r") as f:
                metadata = json.load(f)
            if metadata["chunking"] != self.chunking:
                raise ValueError(
                    "The chunking specified does not match the existing dataset."
                )
            if metadata["compression"] != self.compression:
                raise ValueError(
                    "The compression specified does not match the existing dataset."
                )
            if metadata["start_date"] != str(self.start_date):
                raise ValueError(
                    "The start date specified does not match the existing dataset."
                )
            if metadata["dt"] != self.dt:
                raise ValueError(
                    "The dt specified does not match the existing dataset."
                )

    def _load_zarr_arrays(self):
        """
        Attempt to load all zarr arrays in the dataset.
        """

        # Create dictionary of zarr arrays
        zarr_arrays = {}
        for variable, pressure_level in self.variables:
            # Get zarr path
            zarr_path = self._variable_to_zarr_name(variable, pressure_level)
            mapper = self.fs.get_mapper(zarr_path)

            # Try to open zarr array
            try:
                zarr_array = xr.open_zarr(mapper)
                zarr_arrays[(variable, pressure_level)] = zarr_array
            except:
                pass

        return zarr_arrays


    def _create_download_chunks(self):
        """
        Create a list of download chunks based on the start and end dates.
        """

        # Make dictionary of chunks to download where the key is the variable and the value is a list of chunks
        download_chunks = {}
        
        # Round dates to months
        current_date = self.start_date.replace(day=1)
        end_date = self.end_date.replace(day=1)

        # Loop over until we reach the end date
        while current_date <= end_date:

            # Create a list of tasks to download the data
            for variable, pressure_level in self.variables:

                # Check if chunk already exists
                status = Status.NOT_STARTED

                if (variable, pressure_level) in self.zarr_arrays:
                    zarr_array = self.zarr_arrays[(variable, pressure_level)]
                    check_date = datetime.date(current_date.year, current_date.month, 1)
                    if np.datetime64(check_date) in zarr_array.time.values:
                        status = Status.COMPLETED

                # Create tuple for download chunk
                chunk = ERA5DownloadChunk(
                        variable,
                        current_date.year,
                        current_date.month,
                        self.hours,
                        pressure_level,
                        status)

                # Add to dictionary
                if (variable, pressure_level) not in download_chunks:
                    download_chunks[(variable, pressure_level)] = [chunk]
                else:
                    download_chunks[(variable, pressure_level)].append(chunk)

            # Update the current date
            days_in_month = calendar.monthrange(
                year=current_date.year, month=current_date.month
            )[1]
            current_date += datetime.timedelta(days=days_in_month)

        return download_chunks
 
    def _download_chunk(
        self,
        dc: ERA5DownloadChunk,
    ):
        """
        Download ERA5 data for the specified variable, date range, hours, and pressure levels.

        Parameters
        ----------
        dc : ERA5DownloadChunk

        Returns
        -------
        xr.Dataset
            An xarray Dataset containing the downloaded data.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            # Get all days in the month
            days_in_month = calendar.monthrange(dc.year, dc.month)[1]

            # Make tmpfile to store the data
            output_file = os.path.join(
                tmpdir,
                str(dc),
            )

            # start the CDS API client (maybe need to move this outside the loop?)
            c = cdsapi.Client(quiet=True)

            # Setup the request parameters
            request_params = {
                "product_type": "reanalysis",
                "variable": dc.variable,
                "year": str(dc.year),
                "month": str(dc.month),
                "day": [f"{day:02d}" for day in range(1, days_in_month + 1)],
                "time": [f"{hour:02d}:00" for hour in dc.hours],
                "format": "netcdf",
            }
            if dc.pressure_level:
                request_params["pressure_level"] = [str(dc.pressure_level)]
                dataset_name = "reanalysis-era5-pressure-levels"
            else:
                dataset_name = "reanalysis-era5-single-levels"

            # Download the data
            c.retrieve(
                dataset_name,
                request_params,
                output_file,
            )

            # Open the downloaded data
            ds = xr.open_dataset(output_file)
        return ds

    def _variable_to_zarr_name(self, variable: str, pressure_level: int = None):
        # create zarr path for variable
        if pressure_level:
            zarr_path = f"{self.base_path}/pressure_levels/{variable}_pl{pressure_level}"
        else:
            zarr_path = f"{self.base_path}/single_levels/{variable}"
        zarr_path += ".zarr"
        return zarr_path

    def _download_and_upload_chunk(
        self,
        dc: ERA5DownloadChunk,
    ):
        """
        Downloads a chunk of ERA5 data for a specific variable and date range, and uploads it to a Zarr array.
        This downloads a 1-month chunk of data.

        Parameters
        ----------
        dc : ERA5DownloadChunk
            The download chunk to download and upload.
        """

        # Download the data
        ds = self._download_chunk(dc)

        # Create the Zarr path
        zarr_path = self._variable_to_zarr_name(dc.variable, dc.pressure_level)

        # Specify the chunking options
        chunking = {"time": self.chunking[0], "latitude": self.chunking[1], "longitude": self.chunking[2]}
        if "level" in ds.dims:
            chunking["level"] = 1

        # Re-chunk the dataset
        ds = ds.chunk(chunking)

        # Check if the Zarr dataset exists
        if self.fs.exists(zarr_path):
            mode = "a"
            append_dim = "time"
            create = False
        else:
            mode = "w"
            append_dim = None
            create = True

        # Upload the data to the Zarr dataset
        mapper = self.fs.get_mapper(zarr_path, create=create)
        ds.to_zarr(mapper, mode=mode, consolidated=True, append_dim=append_dim)

        # Update download chunk
        dc.status = Status.COMPLETED

    def _queue_batch_download(self):
        """
        Queues up batches of ERA5 data to download.
        """

        # Queue up all the chunks needed to download
        added = 0
        for variable, pressure_level in self.variables:
            for i in range(len(self.download_chunks[(variable, pressure_level)])):
                # Get the download chunk
                dc = self.download_chunks[(variable, pressure_level)][i]

                # if first chunk and not started, start it
                if (i == 0) and (dc.status == Status.NOT_STARTED):
                    # Download the first chunk
                    dc.status = Status.IN_PROGRESS
                    self.queue.put(dc)
                    added += 1

                # if not first chunk, check if previous chunk is completed
                if (i > 0):
                    prev_dc = self.download_chunks[(variable, pressure_level)][i - 1]
                    if (prev_dc.status == Status.COMPLETED) and (dc.status == Status.NOT_STARTED):
                        # Download the chunk
                        dc.status = Status.IN_PROGRESS
                        self.queue.put(dc)
                        added += 1

        return added

    def _download_worker(self):
        """
        Worker function for downloading ERA5 data.
        """

        while True:
            # Get the download chunk
            dc = self.queue.get()

            # Break if there are no more chunks to download
            if dc is self.sentinel:
                self.queue.task_done()
                break

            # Download the chunk
            try:
                # Start the download
                self._download_and_upload_chunk(dc)

                # Update the progress if needed
                if self.progress_plot:
                    self._plot_progress()
            finally:
                self.queue.task_done()

    def _download(
        self,
    ):

        # Start the download workers
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Start the workers
            futures = [executor.submit(self._download_worker) for _ in range(self.num_workers)]

            # Keep queueing up chunks until all chunks are downloaded
            while any(future.running() for future in futures):
                while True:
                    # Sleep for a bit to avoid overloading the CDS API
                    time.sleep(1)

                    # Check if any workers are available
                    if self.queue.qsize() < self.num_workers:
                        # Queue up more chunks
                        jobs_added = self._queue_batch_download()
                        if jobs_added > 0:
                            print(f"Added {jobs_added} jobs to the queue")
                            print(f"Queue size: {self.queue.qsize()}")
                            print(f"Progress: {self.progress * 100:.2f} %")

                    # Break if no more chunks were added
                    if (jobs_added == 0) and all(future.running() for future in futures):
                        break

            # Send the sentinel to stop the workers
            for _ in range(self.num_workers):
                self.queue.put(self.sentinel)

            # Wait for the workers to finish
            for future in futures:
                future.result()

    @property
    def progress(self):
        """
        Returns the progress of the download.
        """
        # Generate array for status of download
        first_variable, first_pressure_level = list(self.download_chunks.keys())[0]
        nr_chunks = len(self.download_chunks[(first_variable, first_pressure_level)])
        nr_finished = 0
        total_size = 0
        for i, (variable, pressure_level) in enumerate(self.variables):
            for j, dc in enumerate(self.download_chunks[(variable, pressure_level)]):
                if dc.status == Status.COMPLETED:
                    nr_finished += 1
                total_size += 1
        return nr_finished / total_size

    def _plot_progress(
            self,
        ):
        """
        Plot the progress of the download.
        x-axis: chunk number (time)
        y-axis: variable
        color: green if completed, red if not started, yellow if in progress
        """

        # Generate array for status of download
        first_variable, first_pressure_level = list(self.download_chunks.keys())[0]
        nr_chunks = len(self.download_chunks[(first_variable, first_pressure_level)])
        status = np.zeros((len(self.variables), nr_chunks))
        for i, (variable, pressure_level) in enumerate(self.variables):
            for j, dc in enumerate(self.download_chunks[(variable, pressure_level)]):
                if dc.status == Status.COMPLETED:
                    status[i, j] = 1.0
                elif dc.status == Status.NOT_STARTED:
                    status[i, j] = 0.0
                elif dc.status == Status.IN_PROGRESS:
                    status[i, j] = 0.5

        # Plot the data
        size = max(nr_chunks, len(self.variables))
        fig, ax = plt.subplots(figsize=(10 * nr_chunks / size + 5, 10 * len(self.variables) / size + 5))
        ax.imshow(status, cmap="RdYlGn", aspect=True, vmin=0, vmax=1)
        ax.set_xticks(np.arange(nr_chunks)[::12])
        ax.set_xticklabels([str(dc.year) + '-' + str(dc.month) for dc in self.download_chunks[(first_variable, first_pressure_level)]][::12])
        ax.set_yticks(np.arange(len(self.variables)))
        ax.set_yticklabels([f"{variable} {pressure_level}" for variable, pressure_level in self.variables])
        ax.set_xlabel("Chunk")
        ax.set_ylabel("Variable")
        ax.set_title(f"Download Progress: {self.progress * 100:.2f} %")
        ax.set_aspect('equal', 'box')
        plt.savefig(self.progress_plot)
        plt.close()
