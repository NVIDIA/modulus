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
import h5py as h5
import tqdm

def main(args):

    # set chunksize
    if args.chunksize == "auto":
        chunksize = True
    elif args.chunksize == "none":
        chunksize = None
    elif "MB" in args.chunksize:
        chunksize = int(args.chunksize.replace("MB", "")) * 1024 * 1024
    else:
        raise ValueError(f"Error, chunksize {args.chunksize} not supported.")
    
    # get files
    files = glob.glob(os.path.join(args.input_dir, "*.h5"))

    for ifname in files:

        #construct output file name
        ofname = os.path.join(args.output_dir, os.path.basename(ifname))

        # check if output file exists
        if os.path.exists(ofname):
            if not args.overwrite:
                print(f"File {ofname} already exists, skipping.", flush=True)
                continue
            else:
                shutil.rmtree(ofname)

        print(f"Converting {ifname} -> {ofname}", flush=True)
        with h5.File(ifname, 'r') as fin:
            data = fin["fields"][...]

            if args.transpose:
                data = np.transpose(data, (0, 2, 3, 1))

            with h5.File(ofname, 'w') as fout:
                fout.create_dataset("fields", data.shape, dtype=data.dtype, chunks=chunksize)
                # write data
                fout["fields"] = data[...]


if __name__ == "__main__":
    # argparse
    parser = ap.ArgumentParser()
    parser.add_argument("--input_dir", type=str, help="Directory with input files.", required=True)
    parser.add_argument("--output_dir", type=str, help="Directory for output files.", required=True)
    parser.add_argument("--chunksize", type=str, default="auto", help="Default chunksize.")
    parser.add_argument("--transpose", action='store_true')
    parser.add_argument("--overwrite", action='store_true')
    args = parser.parse_args()

    main(args)
        
