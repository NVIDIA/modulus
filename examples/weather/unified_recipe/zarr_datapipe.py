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
import h5py
import time
import fsspec
#import zarr
from tqdm import tqdm

#from modulus.experimental.datapipes.zarr.temporal import TemporalZarrDatapipe

#from experiment import IOExperiment
from fancy_bar_plot import fancy_bar_plot


def main():

    ## Get experiment
    #experiment = IOExperiment(
    #        base_path=cfg.base_path,
    #        filetype="zarr",
    #        device="cpu",
    #        compression_algorithm="none",
    #        batch_codec=False,
    #        zarr_loading="kvikio",
    #        chunking=(1, 32, 721, 1440),
    #)

    ## Get zarr array
    #experiment.save_data(era5_h5["fields"])
    #zarr_path = experiment.save_path
    # Save zarr array
    #zarr.array(era5_h5["fields"]))

    # Make zarr array
    zarr_path = "./era5_benchmarks/array.zarr"
    #era5_h5 = h5py.File(os.path.join(cfg.base_path, "era5_data.h5"), "r")
    #zarr_array = zarr.array(era5_h5["fields"], chunks=[1, 32, 721, 1440], compressor=None)
    #print(zarr_array.info)
    #zarr.save_array(zarr_path, zarr_array, compressor=None)
    #print("Saved zarr array")

    # Open zarr array using fsspec caching in memory
    local_fs = fsspec.filesystem(
            "simplecache",
            target_protocol="file",
            cache_storage="memory")

    # Map the local filesystem to memory
    fs_map = local_fs.get_mapper(zarr_path)

    # Now open the zarr array 
    zarr_array = zarr.open_array(fs_map, mode="r")

    # Make Dali datapipe
    datapipe = TemporalZarrDatapipe(
            zarr_array=zarr_array,
            batch_size=8,
            stride=1,
            num_steps=2,
            gpu_decompression=False,
    )

    # Run benchmark
    tic = time.time()
    nbytes_loaded = 0.0
    for i in tqdm(range(zarr_array.shape[0])):
        #nbytes_loaded += batch[0]['sequence'].nbytes
        array = zarr_array[i]
        nbytes_loaded += array.nbytes
        print(nbytes_loaded)
    elapsed = time.time() - tic
    print(f"First epoch GB/s: {nbytes_loaded / elapsed / 1e9}")

    tic = time.time()
    nbytes_loaded = 0.0
    for i in tqdm(range(zarr_array.shape[0])):
        #nbytes_loaded += batch[0]['sequence'].nbytes
        array = zarr_array[i]
        nbytes_loaded += array.nbytes
    elapsed = time.time() - tic
    print(f"Second epoch GB/s: {nbytes_loaded / elapsed / 1e9}")


if __name__ == "__main__":
    main()
