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
import numpy as np
import h5py as h5
from tqdm import tqdm

def main(args):
    
    # get files
    files = glob.glob(os.path.join(args.input_dir, "*.h5"))

    # create access control list
    fcpl = h5.h5p.create(h5.h5p.FILE_CREATE)
    fcpl.set_userblock(max(512, args.align_size_bytes))
    fapl = h5.h5p.create(h5.h5p.FILE_ACCESS)
    fapl.set_fapl_direct(args.align_size_bytes, args.block_size_bytes, 0)

    # iterate over files
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

        if not args.verify_integrity:
            print(f"Converting {ifname} -> {ofname}", flush=True)
            with h5.File(ifname, 'r') as fin:
                data_shape = fin["fields"].shape
                dtype = fin["fields"].dtype
            
                # create output file
                fid = h5.h5f.create(ofname.encode("ascii"), flags=h5.h5f.ACC_TRUNC, fcpl=fcpl, fapl=fapl)
                fout = h5.Group(fid)
                fout.create_dataset("fields", data_shape, dtype=dtype)

                # read and write loop:
                for idx in tqdm(range(0, data_shape[0], args.batch_size)):
                    start = idx
                    end = min(start + args.batch_size, data_shape[0])
                    data = fin["fields"][start:end, ...]
                    
                    if args.transpose:
                        data = np.transpose(data, (0, 2, 3, 1))
                        
                    # write data
                    fout["fields"][start:end, ...] = data[...]

                # close file
                fid.close()

        else:
            print(f"Checking {ifname}", flush=True)
            fin = h5.File(ifname, 'r', driver="direct")
            dset = fin["fields"]
            print(f"Shape: {dset.shape}", flush=True)
            
            fin.close()

    # close file list:
    fapl.close()
    fcpl.close()


if __name__ == "__main__":
    # argparse
    parser = ap.ArgumentParser()
    parser.add_argument("--input_dir", type=str, help="Directory with input files.", required=True)
    parser.add_argument("--output_dir", type=str, help="Directory for output files.", required=True)
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for number of rows to convert at a time.")
    parser.add_argument("--align_size_bytes", type=int, default=4096, help="Default alignment size in bytes.")
    parser.add_argument("--block_size_bytes", type=int, default=0, help="Default block size in bytes.")
    parser.add_argument("--verify_integrity", action='store_true')
    parser.add_argument("--transpose", action='store_true')
    parser.add_argument("--overwrite", action='store_true')
    args = parser.parse_args()

    main(args)
        
