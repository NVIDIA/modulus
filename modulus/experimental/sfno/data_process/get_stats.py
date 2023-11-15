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
import time
import pickle
import numpy as np
import h5py as h5
import math
import argparse as ap
from itertools import groupby, accumulate
import operator
from bisect import bisect_right
from glob import glob

# MPI
from mpi4py import MPI
from mpi4py.util import dtlib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from modulus.experimental.sfno.utils.grids import GridQuadrature

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


def welford_combine(stats1, stats2):
    # update time means first:
    stats = {}

    for k in stats1.keys():
        s_a = stats1[k]
        s_b = stats2[k]

        # update stats
        n_a = s_a["counts"]
        n_b = s_b["counts"]
        n_ab = n_a + n_b

        if s_a["type"] == "min":
            values = np.minimum(s_a["values"], s_b["values"])
        elif s_a["type"] == "max":
            values = np.maximum(s_a["values"], s_b["values"])
        elif s_a["type"] == "mean":
            mean_a = s_a["values"]
            mean_b = s_b["values"]
            values = (mean_a * float(n_a) + mean_b * float(n_b)) / float(n_ab)
        elif s_a["type"] == "meanvar":
            mean_a = s_a["values"][0]
            mean_b = s_b["values"][0]
            m2_a = s_a["values"][1]
            m2_b = s_b["values"][1]
            delta = mean_b - mean_a
            
            values = [(mean_a * float(n_a) + mean_b * float(n_b)) / float(n_ab),
                      m2_a + m2_b + delta * delta * float(n_a * n_b) / float(n_ab)]

        stats[k] = {"counts": n_ab,
                    "type": s_a["type"],
                    "values": values}

    return stats


def get_file_stats(filename,
                   indexlist,
                   wind_indices,
                   quadrature,
                   batch_size=16):

    # preprocess indexlist into slices:
    slices = list(_get_slices(indexlist))

    stats = None
    with h5.File(filename, 'r') as f:
        for slc in slices:
            
            # create batch
            slc_start = slc.start
            slc_stop = slc.stop
            for batch_start in range(slc_start, slc_stop, batch_size):
                batch_stop = min(batch_start+batch_size, slc_stop)
                sub_slc = slice(batch_start, batch_stop)
                
                # get slice
                data = f['fields'][sub_slc, ...]
            
                # min/max first:
                counts_time = data.shape[0]
                counts = counts_time * data.shape[2] * data.shape[3]

                # compute mean and variance
                tdata = torch.from_numpy(data)
                tmean = torch.mean(quadrature(tdata), keepdims=False, dim=0).reshape(1, -1, 1, 1)
                tvar = torch.mean(quadrature(torch.square(tdata - tmean)), keepdims=False, dim=0).reshape(1, -1, 1, 1)

                # time diffs
                tdiff = tdata[1:, ...] - tdata[:-1, ...]
                tdiffmean = torch.mean(quadrature(tdiff), keepdims=False, dim=0).reshape(1, -1, 1, 1)
                tdiffvar = torch.mean(quadrature(torch.square(tdiff - tdiffmean)), keepdims=False, dim=0).reshape(1, -1, 1, 1)

                # fill the dict
                tmpstats = dict(maxs = {"values": np.max(data, keepdims=True, axis = (0, 2, 3)),
                                        "type": "max",
                                        "counts": counts},
                                mins = {"values": np.min(data, keepdims=True, axis = (0, 2, 3)),
                                        "type": "min",
                                        "counts": counts},
                                time_means = {"values": np.mean(data, keepdims=True, axis = 0),
                                              "type": "mean",
                                              "counts": counts_time},
                                global_meanvar = {"values": [tmean.numpy(), float(counts) * tvar.numpy()],
                                                  "type": "meanvar",
                                                  "counts": counts})
                if counts_time > 1:
                    tmpstats["time_diff_meanvar"] = {"values": [tdiffmean.numpy(),
                                                                float(counts_time-1) * 4. * np.pi * tdiffvar.numpy()],
                                                     "type": "meanvar",
                                                     "counts": (counts_time-1) * data.shape[2] * data.shape[3]}
                else:
                    tmpstats["time_diff_meanvar"] = {"values": [0., 1.], "type": "meanvar", "counts": 0}

                if wind_indices is not None:
                    wind_data = np.sqrt(data[:, wind_indices[0]]**2 + data[:, wind_indices[1]]**2)
                    tmpstats["wind_meanvar"] = {"values": [np.zeros((1, len(wind_indices[0]), 1, 1)),
                                                           float(counts) * np.var(wind_data, keepdims=True, axis = (0, 2 ,3))],
                                                "type": "meanvar",
                                                "counts": counts}

                if stats is not None:
                    stats = welford_combine(stats, tmpstats)
                else:
                    stats = tmpstats

    return stats

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


def collective_reduce(comm, stats):
    statslist = allgather_safe(comm, stats)
    stats = statslist[0]
    for tmpstats in statslist[1:]:
        stats = welford_combine(stats, tmpstats)

    return stats


def binary_reduce(comm, stats):
    csize = comm.Get_size()
    crank = comm.Get_rank()
    
    # check for power of two
    assert((csize & (csize-1) == 0) and csize != 0)

    # how many steps do we need:
    nsteps = int(math.log(csize,2))

    # init step 1
    recv_ranks = range(0,csize,2)
    send_ranks = range(1,csize,2)

    for step in range(nsteps):
        for rrank,srank in zip(recv_ranks, send_ranks):
            if crank == rrank:
                rstats = comm.recv(source=srank, tag=srank)
                stats = welford_combine(stats, rstats)
            elif crank == srank:
                comm.send(stats, dest=rrank, tag=srank)

        # wait for everyone being ready before doing the next epoch
        comm.Barrier()

        # shrink the list
        if (step < nsteps-1):
            recv_ranks = recv_ranks[0::2]
            send_ranks = recv_ranks[1::2]

    return stats


def main(args):

    # get comm
    comm = MPI.COMM_WORLD.Dup()
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    # create group comm
    group_id = comm_rank % args.group_size
    group_rank = comm_rank // args.group_size
    group_comm = comm.Split(color=group_id, key=group_rank)

    # create intergroup comm
    intergroup_comm = comm.Split(color=group_rank, key=group_id)

    # get files
    filelist = None
    data_shape = None
    num_samples = None
    wind_channels = None
    if comm_rank == 0:
        #filelist = sorted([os.path.join(args.input_dir, x) for x in os.listdir(args.input_dir) if os.path.isfile(os.path.join(args.input_dir, x))])
        filelist = sorted(glob(os.path.join(args.input_dir, "*.h5")))
        if not filelist:
            raise FileNotFoundError(f"Error, directory {args.input_dir} is empty.")

        # open the first file to check for stats
        num_samples = []
        for filename in filelist:
            with h5.File(filename, 'r') as f:
                data_shape = f['fields'].shape
                num_samples.append(data_shape[0])

    if args.wind_angle:
        channel_names = ['u10', 'v10', 't2m', 'sp', 'msl', 't850', 'u1000', 'v1000', 'z1000', 'u850', 'v850', 'z850', 'u500', 'v500', 'z500', 't500', \
                         'z50', 'r500', 'r850', 'tcwv', 'u100m', 'v100m', 'u250', 'v250', 'z250', 't250', 'u100', 'v100', 'z100', 't100', 'u900', 'v900', \
                         'z900', 't900']
        wind_channels = get_wind_channels(channel_names)

    # communicate the files
    filelist = comm.bcast(filelist, root=0)
    num_samples = comm.bcast(num_samples, root=0)
    data_shape = comm.bcast(data_shape, root=0)
    wind_channels = comm.bcast(wind_channels, root=0)

    # DEBUG
    #filelist = filelist[:2]
    #num_samples = num_samples[:2]
    # DEBUG
    
    # get file offsets
    num_samples_total = sum(num_samples)
    num_channels = data_shape[1]
    height, width = (data_shape[2], data_shape[3])

    # quadrature:
    quadrature = GridQuadrature(args.quadrature_rule, (height, width),
                                crop_shape=None, crop_offset=(0, 0),
                                normalize=True, pole_mask=None)

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
        local_idx = idx - sample_offsets[file_idx]
        
        # lookup
        filename = filelist[file_idx]
        if filename in mapping:
            mapping[filename].append(local_idx)
        else:
            mapping[filename] = [local_idx]

    # just do be on the safe side, sort again
    mapping = {k: sorted(v) for k,v in mapping.items()}

    # initialize arrays
    stats = dict(global_meanvar = {"type": "meanvar", "counts": 0, "values": [np.zeros((1, num_channels, 1, 1)), np.zeros((1, num_channels, 1, 1))]},
                 mins = {"type": "min", "counts": 0, "values": np.zeros((1, num_channels, 1, 1))},
                 maxs = {"type": "max", "counts": 0, "values": np.zeros((1, num_channels, 1, 1))},
                 time_means = {"type": "mean", "counts": 0, "values": np.zeros((1, num_channels, height, width))},
                 time_diff_meanvar = {"type": "meanvar", "counts": 0, "values": [np.zeros((1, num_channels, 1, 1)), np.zeros((1, num_channels, 1, 1))]})

    if wind_channels is not None:
        num_wind_channels = len(wind_channels[0])
        stats["wind_meanvar"] = {"type": "meanvar", "counts": 0, "values": [np.zeros((1, num_wind_channels, 1, 1)), np.zeros((1, num_wind_channels, 1, 1))]}
    
    # compute local stats
    start = time.time()
    for filename, indices in mapping.items():
        tmpstats = get_file_stats(filename, indices, wind_channels, quadrature, args.batch_size)
        stats = welford_combine(stats, tmpstats)
    duration = time.time() - start
        
    # wait for everybody else
    print(f"Rank {comm_rank} done. Duration for {(samples_end - samples_start)} samples: {duration:.2f}s", flush=True)
    group_comm.Barrier()

    # now gather the stats across group:
    stats = collective_reduce(group_comm, stats)
    intergroup_comm.Barrier()
    if group_rank == 0:
        print(f"Group {group_id} done.", flush=True)

    # now, do binary reduction orthogonal to groups
    stats = binary_reduce(intergroup_comm, stats)

    # wait for everybody
    comm.Barrier()
    
    if comm_rank == 0:
        # compute global stds:
        stats["global_meanvar"]["values"][1] = np.sqrt(stats["global_meanvar"]["values"][1] / float(stats["global_meanvar"]["counts"]))
        stats["time_diff_meanvar"]["values"][1] = np.sqrt(stats["time_diff_meanvar"]["values"][1] / float(stats["time_diff_meanvar"]["counts"]))

        # overwrite the wind channels
        if wind_channels is not None:
            stats["wind_meanvar"]["values"][1] = np.sqrt(stats["wind_meanvar"]["values"][1] / float(stats["wind_meanvar"]["counts"]))
            stats["global_meanvar"]["values"][0][: , wind_channels[0]] = stats["wind_meanvar"]["values"][0]
            stats["global_meanvar"]["values"][0][: , wind_channels[1]] = stats["wind_meanvar"]["values"][0]
            stats["global_meanvar"]["values"][1][: , wind_channels[0]] = stats["wind_meanvar"]["values"][1]
            stats["global_meanvar"]["values"][1][: , wind_channels[1]] = stats["wind_meanvar"]["values"][1]


        # save the stats
        np.save(os.path.join(args.output_dir, 'global_means.npy'), stats["global_meanvar"]["values"][0])
        np.save(os.path.join(args.output_dir, 'global_stds.npy'), stats["global_meanvar"]["values"][1])
        np.save(os.path.join(args.output_dir, 'mins.npy'), stats["mins"]["values"])
        np.save(os.path.join(args.output_dir, 'maxs.npy'), stats["maxs"]["values"])
        np.save(os.path.join(args.output_dir, 'time_means.npy'), stats["time_means"]["values"])
        np.save(os.path.join(args.output_dir, 'time_diff_means.npy'), stats["time_diff_meanvar"]["values"][0])
        np.save(os.path.join(args.output_dir, 'time_diff_stds.npy'), stats["time_diff_meanvar"]["values"][1])

        print("means: ", stats["global_meanvar"]["values"][0])
        print("stds: ", stats["global_meanvar"]["values"][1])

    # wait for rank 0 to finish
    comm.Barrier()


if __name__ == "__main__":
    # argparse
    parser = ap.ArgumentParser()
    parser.add_argument("--input_dir", type=str, help="Directory with input files.", required=True)
    parser.add_argument("--output_dir", type=str, help="Directory for saving stats files.", required=True)
    parser.add_argument("--group_size", type=int, default=8, help="Size of collective reduction groups.")
    parser.add_argument("--quadrature_rule", type=str, default="naive", choices=["naive", "clenshaw-curtiss", "gauss-legendre"], help="Specify quadrature_rule for spatial averages.")
    parser.add_argument('--wind_angle', action='store_true')
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size used for reading chunks from a file at a time to avoid OOM errors.")
    args = parser.parse_args()
    
    main(args)




