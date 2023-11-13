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
import time
import pickle
import numpy as np
import h5py as h5
from math import ceil
import argparse as ap
from itertools import groupby, accumulate
import operator
from bisect import bisect_right

# MPI
from mpi4py import MPI
from mpi4py.util import dtlib

# we need the parser
import json

def allgather_safe(comm, obj):
    
    # serialize the stuff
    fdata = pickle.dumps(obj, protocol = pickle.HIGHEST_PROTOCOL)
    
    #total size
    comm_size = comm.Get_size()
    num_bytes = len(fdata)
    total_bytes = num_bytes * comm_size

    #chunk by ~1GB:
    gigabyte = 1024*1024*1024
    
    # determine number of chunks
    num_chunks = (total_bytes + gigabyte - 1) // gigabyte
    
    # determine local chunksize
    chunksize = (num_bytes + num_chunks - 1) // num_chunks
    
    # datatype stuff
    datatype = MPI.BYTE
    np_dtype = dtlib.to_numpy_dtype(datatype)
    
    # gather stuff
    # prepare buffers:
    sendbuff = np.frombuffer(memoryview(fdata), dtype=np_dtype, count=num_bytes)
    recvbuff = np.empty((comm_size * chunksize), dtype=np_dtype)
    resultbuffs = np.split(np.empty(num_bytes * comm_size, dtype=np_dtype), comm_size)
    
    # do subsequent gathers
    for i in range(0, num_chunks):
        # create buffer views
        start = i * chunksize
        end = min(start + chunksize, num_bytes)
        eff_bytes = end - start
        sendbuffv = sendbuff[start:end]
        recvbuffv = recvbuff[0:eff_bytes*comm_size]
        
        # perform allgather on views
        comm.Allgather([sendbuffv, datatype], [recvbuffv, datatype])
        
        # split result buffer for easier processing
        recvbuff_split = np.split(recvbuffv, comm_size)
        for j in range(comm_size):
            resultbuffs[j][start:end] = recvbuff_split[j][...]
    results = [x.tobytes() for x in resultbuffs]

    # unpickle:
    results = [pickle.loads(x) for x in results]
    
    return results
            

def _get_slices(lst):
    for a, b in groupby(enumerate(lst), lambda pair: pair[1] - pair[0]):
        b = list(b)
        # add one to the upper boundary since we want to convert to slices later
        yield slice(b[0][1], b[-1][1] + 1)


def get_file_stats(filename,
                   indexlist,
                   batch_size=8):

    # preprocess indexlist into slices:
    slices = list(_get_slices(indexlist))

    count = 0
    mins = []
    maxs = []
    with h5.File(filename, 'r') as f:
        for slc in slices:

            # create batch
            slc_start = slc.start
	    slc_stop = slc.stop
            for batch_start in range(slc_start, slc_stop, batch_size):
		batch_stop = min(batch_start+batch_size, slc_stop)
                sub_slc = slice(batch_start, batch_stop)
            
                data = f['fields'][sub_slc, ...]

                # counts
                count += data.shape[0] * data.shape[2] * data.shape[3]
            
                # min/max
                mins.append(np.min(data, axis=(0,2,3)))
                maxs.append(np.max(data, axis=(0,2,3)))
    
    # concat and take min/max
    mins = np.min(np.stack(mins, axis=1), axis=1)
    maxs = np.max(np.stack(maxs, axis=1), axis=1)

    return count, mins, maxs


def get_file_histograms(filename, indexlist,
                        minvals, maxvals, nbins,
                        batch_size=8):
    # preprocess indexlist into slices:
    slices = list(_get_slices(indexlist))

    histograms = None
    with h5.File(filename, 'r') as f:
        for slc in slices:

            # create batch
            slc_start = slc.start
	    slc_stop = slc.stop
            for batch_start in range(slc_start, slc_stop, batch_size):
		batch_stop = min(batch_start+batch_size, slc_stop)
                sub_slc = slice(batch_start, batch_stop)
            
                data = f['fields'][sub_slc, ...]

                # get histograms along channel axis
                datalist = np.split(data, data.shape[1], axis=1)

                # generate histograms
                tmphistograms = [np.histogram(x, bins=nbins, range=(minval, maxval)) for x,minval,maxval in zip(datalist, minvals, maxvals)]

                if histograms is None:
                    histograms = tmphistograms
                else:
                    histograms = [(x[0]+y[0], x[1]) for x,y in zip(histograms, tmphistograms)]

    return histograms


def get_wind_channels(channel_names):
    # find the pairs in the channel names and alter the stats accordingly
    channel_dict = { channel_names[ch] : ch for ch in set(range(len(channel_names)))}

    uchannels = []
    vchannels = []
    for chn, ch in channel_dict.items():
        if chn[0] == 'u':
            vchn = 'v' + chn[1:]
            if vchn in channel_dict.keys():
                vch = channel_dict[vchn]

                uchannels.append(ch)
                vchannels.append(vch)
    
    return uchannels, vchannels


def main(args):

    # get comm
    comm = MPI.COMM_WORLD.Dup()
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    # get files
    filelist = None
    data_shape = None
    num_samples = None
    wind_channels = None
    if comm_rank == 0:
        filelist = sorted([os.path.join(args.input_dir, x) for x in os.listdir(args.input_dir)])
        if not filelist:
            raise FileNotFoundError(f"Error, directory {args.input_dir} is empty.")

        # open the first file to check for stats
        num_samples = []
        for filename in filelist:
            with h5.File(filename, 'r') as f:
                data_shape = f['fields'].shape
                num_samples.append(data_shape[0])

        # open metadata file
        with open(args.metadata_file, 'r') as f:
            metadata = json.load(f)

        channel_names = metadata['coords']['channel']
        wind_channels = get_wind_channels(channel_names)

    # communicate the files
    filelist = comm.bcast(filelist, root=0)
    num_samples = comm.bcast(num_samples, root=0)
    data_shape = comm.bcast(data_shape, root=0)
    wind_channels = comm.bcast(wind_channels, root=0)

    # DEBUG
    filelist = filelist[:5]
    num_samples = num_samples[:5]
    # DEBUG
    
    # get file offsets
    num_samples_total = sum(num_samples)
    num_channels = data_shape[1]
    height, width = (data_shape[2], data_shape[3])

    if comm_rank == 0:
        print(f"Found {len(filelist)} files with a total of {num_samples_total} samples. Each sample has the shape {num_channels}x{height}x{width} (CxHxW).")
    
    # do the sharding:
    num_samples_chunk = (num_samples_total + comm_size - 1) // comm_size
    samples_start = num_samples_chunk * comm_rank
    samples_end = min([samples_start + num_samples_chunk, num_samples_total])
    sample_offsets = list(accumulate(num_samples, operator.add))[:-1]
    sample_offsets.insert(0, 0)

    # offsets has order like:
    #[0, 1460, 2920, ...]
    
    # convert list of indices to files and ranges in files:
    mapping = {}
    for idx in range(samples_start, samples_end):
        # compute indices
        file_idx = bisect_right(sample_offsets, idx) - 1
        #file_idx = idx // num_samples_per_file
        local_idx = idx - sample_offsets[file_idx]
        
        # lookup
        filename = filelist[file_idx]
        if filename in mapping:
            mapping[filename].append(local_idx)
        else:
            mapping[filename] = [local_idx]

    # just do be on the safe side, sort again
    mapping = {k: sorted(v) for k,v in mapping.items()}
    
    # compute local stats
    start = time.time()
    mins = []
    maxs = []
    count = 0
    for filename, indices in mapping.items():
        tmpcount, tmpmins, tmpmaxs = get_file_stats(filename, indices)
        mins.append(tmpmins)
        maxs.append(tmpmaxs)
        count += tmpcount
    mins = np.min(np.stack(mins, axis=1), axis=1)
    maxs = np.max(np.stack(maxs, axis=1), axis=1)
    duration = time.time() - start
        
    # wait for everybody else
    print(f"Rank {comm_rank} stats done. Duration for {(samples_end - samples_start)} samples: {duration:.2f}s", flush=True)
    comm.Barrier()

    # now gather the stats from all nodes: we need to do that safely
    countlist = allgather_safe(comm, count)
    minmaxlist = allgather_safe(comm, [mins, maxs])

    # compute global min and max and count
    count = sum(countlist)
    mins = np.min(np.stack([x[0] for x in minmaxlist], axis=1), axis=1).tolist()
    maxs = np.max(np.stack([x[1] for x in minmaxlist], axis=1), axis=1).tolist()
    
    if comm_rank == 0:
        print(f"Data range overview on {count} datapoints:")
        for c,mi,ma in zip(channel_names, mins, maxs):
            print(f"{c}: min = {mi}, max = {ma}")

    # set nbins to sqrt(count) if smaller than one
    if args.nbins <= 0:
        nbins = np.sqrt(count)
    else:
        nbins = args.nbins
            
    # wait for rank 0 to finish
    comm.Barrier()

    # now create histograms:
    start = time.time()
    histograms = None
    for filename, indices in mapping.items():
        tmphistograms = get_file_histograms(filename, indices, mins, maxs, nbins)
        
        if histograms is None:
            histograms = tmphistograms
        else:
            histograms = [(x[0]+y[0], x[1]) for x,y in zip(histograms, tmphistograms)]
            
    duration = time.time() - start

    # wait for everybody else
    print(f"Rank {comm_rank} histograms done. Duration for {(samples_end - samples_start)} samples: {duration:.2f}s", flush=True)
    comm.Barrier()
    
    # now gather the stats from all nodes: we need to do that safely
    histogramlist = allgather_safe(comm, histograms)
    
    # combine the results
    histograms = histogramlist[0]
    for tmphistograms in histogramlist[1:]:
        histograms = [(x[0]+y[0], x[1]) for x,y in zip(histograms, tmphistograms)]
    
    if comm_rank == 0:
        edges = np.stack([e for _,e in histograms], axis=0)
        data = np.stack([h for h,_ in histograms], axis=0)

        outfilename = os.path.join(args.output_dir, "histograms.h5")
        with h5.File(outfilename, "w") as f:
            f["edges"] = edges
            f["data"] = data

    # wait for everybody to finish
    comm.Barrier()

if __name__ == "__main__":
    # argparse
    parser = ap.ArgumentParser()
    parser.add_argument("--input_dir", type=str, help="Directory with input files.", required=True)
    parser.add_argument("--metadata_file", type=str, help="File containing dataset metadata.", required=True)
    parser.add_argument("--output_dir", type=str, help="Directory for saving stats files.", required=True)
    parser.add_argument("--nbins", type=int, default=100, help="Number of bins for histograms")
    args = parser.parse_args()
    
    main(args)




