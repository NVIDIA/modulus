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
import sys
import shutil
import glob
import argparse as ap 

import dask.diagnostics    

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import modulus.experimental.sfno.datasets.era5


def main(args):
    
    # get files
    files = glob.glob(os.path.join(args.input_dir, "*.h5"))

    for ifname in files:
        array = datasets.era5.open_34_vars(ifname)
        ds = array.to_dataset()

        #construct output file name
        ofname = os.path.join(args.output_dir, os.path.basename(ifname).replace(".h5", ".zarr"))
        
        if os.path.exists(ofname):
            if not args.overwrite:
                print(f"File {ofname} already exists, skipping.", flush=True)
                continue
            else:
                shutil.rmtree(ofname)

        # save as zarr
        print(f"Converting {ifname} -> {ofname}", flush=True)
        with dask.diagnostics.ProgressBar():
            ds.to_zarr(ofname)


if __name__ == "__main__":
    # argparse
    parser = ap.ArgumentParser()
    parser.add_argument("--input_dir", type=str, help="Directory with input files.", required=True)
    parser.add_argument("--output_dir", type=str, help="Directory for output files.", required=True)
    parser.add_argument("--overwrite", action='store_true')
    args = parser.parse_args()

    main(args)
        
