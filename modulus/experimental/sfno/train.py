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
import numpy as np
import argparse
import torch
import logging
from modulus.experimental.sfno.utils import logging_utils
from modulus.experimental.sfno.utils.YParams import YParams

# distributed computing stuff
from modulus.experimental.sfno.utils import comm

# import trainer
from modulus.experimental.sfno.utils.parse_dataset_metada import parse_dataset_metadata
from modulus.experimental.sfno.utils.trainer import Trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--fin_parallel_size", default=1, type=int, help="Input feature paralellization")
    parser.add_argument("--fout_parallel_size", default=1, type=int, help="Output feature paralellization")
    parser.add_argument("--h_parallel_size", default=1, type=int, help="Spatial parallelism dimension in h")
    parser.add_argument("--w_parallel_size", default=1, type=int, help="Spatial parallelism dimension in w")
    parser.add_argument("--parameters_reduction_buffer_count", default=1, type=int, help="How many buffers will be used (approximately) for weight gradient reductions.") 
    parser.add_argument("--run_num", default='00', type=str)
    parser.add_argument("--yaml_config", default='./config/afnonet.yaml', type=str)
    parser.add_argument("--batch_size", default=-1, type=int, help="Switch for overriding batch size in the configuration file.")
    parser.add_argument("--config", default='default', type=str)
    parser.add_argument("--enable_synthetic_data", action='store_true')
    parser.add_argument("--amp_mode", default='none', type=str, choices=["none", "fp16", "bf16"], help="Specify the mixed precision mode which should be used.")
    parser.add_argument("--jit_mode", default='none', type=str, choices=["none", "script", "inductor"], help="Specify if and how to use torch jit.") 
    parser.add_argument("--cuda_graph_mode", default='none', type=str, choices=["none", "fwdbwd", "step"], help="Specify which parts to capture under cuda graph")
    parser.add_argument("--enable_benchy", action='store_true')
    parser.add_argument("--disable_ddp", action='store_true')
    parser.add_argument("--enable_odirect", action='store_true')
    parser.add_argument("--checkpointing_level", default=0, type=int, help="How aggressively checkpointing is used")
    parser.add_argument("--epsilon_factor", default = 0, type = float)
    parser.add_argument("--split_data_channels", action='store_true')
    parser.add_argument("--print_timings_frequency", default=-1, type=int, help="Frequency at which to print timing information")
    parser.add_argument("--skip_validation", action='store_true', help="Flag to allow validation skipping, useful for profiling and debugging")
    parser.add_argument("--mode", default='train', type=str, choices=["train", "test"], help="Run training or perform a test")
    
    # checkpoint format
    parser.add_argument("--save_checkpoint", default='legacy', choices=['none', 'flexible', 'legacy'], type=str, help="Format in which to save checkpoints.")
    parser.add_argument("--load_checkpoint", default='legacy', choices=['flexible', 'legacy'], type=str, help="Format in which to load checkpoints.")
    # multistep stuff
    parser.add_argument("--multistep_count", default=1, type=int, help="Number of autoregressive training steps. A value of 1 denotes conventional training")
    
    # parse
    args = parser.parse_args()

    # check whether the right h5py package is installed
    odirect_env_var_name = "ENABLE_H5PY_ODIRECT"
    if args.enable_odirect and os.environ.get(odirect_env_var_name, "False").lower() != "true":
        raise RuntimeError(f"Error, {odirect_env_var_name} h5py with MPI support is not installed. Please refer to README for instructions on how to install it.")

    # parse parameters
    params = YParams(os.path.abspath(args.yaml_config), args.config)
    params['epsilon_factor'] = args.epsilon_factor

    # distributed
    params["fin_parallel_size"] = args.fin_parallel_size
    params["fout_parallel_size"] = args.fout_parallel_size
    params["h_parallel_size"] = args.h_parallel_size
    params["w_parallel_size"] = args.w_parallel_size

    params["model_parallel_sizes"] = [args.h_parallel_size, args.w_parallel_size, args.fin_parallel_size, args.fout_parallel_size]
    params["model_parallel_names"] = ["h", "w", "fin", "fout"]
    params["parameters_reduction_buffer_count"] = args.parameters_reduction_buffer_count

    # checkpoint format
    params["load_checkpoint"] = args.load_checkpoint
    params["save_checkpoint"] = args.save_checkpoint

    # make sure to reconfigure logger after the pytorch distributed init
    comm.init(params, verbose=False)

    # update parameters
    world_rank = comm.get_world_rank()
    params["world_size"] = comm.get_world_size()
    if args.batch_size > 0:
        params.batch_size = args.batch_size
    params['global_batch_size'] = params.batch_size
    assert (params['global_batch_size'] % comm.get_size("data") == 0), f"Error, cannot evenly distribute {params['global_batch_size']} across {comm.get_size('data')} GPU."
    params['batch_size'] = int(params['global_batch_size'] // comm.get_size("data"))

    # optimizer params
    if "optimizer_max_grad_norm" not in params:
        params["optimizer_max_grad_norm"] = 1.

    # set device
    torch.cuda.set_device(comm.get_local_rank())
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Set up directory
    expDir = os.path.join(params.exp_dir, args.config, str(args.run_num))
    if world_rank == 0:
        logging.info(f'writing output to {expDir}')
        if not os.path.isdir(expDir):
            os.makedirs(expDir, exist_ok=True)
            os.makedirs(os.path.join(expDir, 'training_checkpoints'), exist_ok=True)
            os.makedirs(os.path.join(expDir, 'wandb'), exist_ok=True)

    params['experiment_dir'] = os.path.abspath(expDir)
    params['checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/ckpt_mp{mp_rank}.tar')
    params['best_checkpoint_path'] = os.path.join(expDir, 'training_checkpoints/best_ckpt_mp{mp_rank}.tar')

    # check if all files are there - do not comment out.
    resuming = True
    for mp_rank in range(comm.get_size("model")):
        checkpoint_fname = params.checkpoint_path.format(mp_rank=mp_rank)
        if params["load_checkpoint"] == "legacy" or mp_rank < 1:
            resuming = resuming and os.path.isfile(checkpoint_fname)

    params['resuming'] = resuming
    params['amp_mode'] = args.amp_mode
    params['jit_mode'] = args.jit_mode
    params['cuda_graph_mode'] = args.cuda_graph_mode
    params['enable_benchy'] = args.enable_benchy
    params['disable_ddp'] = args.disable_ddp
    params['skip_validation'] = args.skip_validation
    params['enable_odirect'] = args.enable_odirect
    params['checkpointing'] = args.checkpointing_level
    params['enable_synthetic_data'] = args.enable_synthetic_data
    params['split_data_channels'] = args.split_data_channels
    params['print_timings_frequency'] = args.print_timings_frequency
    params['multistep_count'] = args.multistep_count
    params['n_future'] = args.multistep_count - 1 # note that n_future counts only the additional samples

    # set default wandb dir
    if not hasattr(params, 'wandb_dir') or params['wandb_dir'] is None:
        params['wandb_dir'] = expDir

    if world_rank==0:
        logging_utils.config_logger()
        logging_utils.log_to_file(logger_name=None, log_filename=os.path.join(expDir, 'out.log'))
        logging_utils.log_versions()
        params.log(logging.getLogger())

    # determine logging
    params['log_to_wandb'] = (world_rank==0) and params['log_to_wandb']
    params['log_to_screen'] = (world_rank==0) and params['log_to_screen']

    # parse dataset metadata                                                                                                                                             
    if 'metadata_json_path' in params:
        params, _ = parse_dataset_metadata(params['metadata_json_path'], params=params)
    else:
        raise RuntimeError(f"Error, please specify a dataset descriptor file in json format")

    # instantiate trainer / inference / ensemble object
    if args.mode == "train":
        trainer = Trainer(params, world_rank)
        trainer.train()
    elif args.mode == "test":
        params["nettype"] = "debug"
        trainer = Trainer(params, world_rank)
        trainer.test_autoregression_pipeline()
    else:
        raise ValueError(f"Unknown training mode {args.mode}")
