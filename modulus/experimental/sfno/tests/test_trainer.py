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

# import os, sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import tempfile
# from typing import List, Optional

# import unittest

# import torch

# from modulus.experimental.sfno.utils.trainer import Trainer
# from modulus.experimental.sfno.utils.YParams import YParams, ParamsBase
# from modulus.experimental.sfno.inference.inferencer import Inferencer
# from modulus.experimental.sfno.networks.model_package import load_model_package

# from modulus.experimental.sfno.tests.testutils import get_default_parameters, init_dataset
# from modulus.experimental.sfno.tests.testutils import H5_PATH

# def init_params(exp_path: str,
#                 train_path: str,
#                 valid_path: str,
#                 stats_path: str,
#                 batch_size: int,
#                 n_history: int,
#                 n_future: int,
#                 normalization: str,
#                 num_data_workers: int,
# ):

#     # instantiate params base  
#     params = get_default_parameters()

#     # init paths
#     params.train_data_path = train_path
#     params.valid_data_path = valid_path
#     params.min_path = os.path.join(stats_path, "mins.npy")
#     params.max_path = os.path.join(stats_path, "maxs.npy")
#     params.time_means_path = os.path.join(stats_path, "time_means.npy")
#     params.global_means_path = os.path.join(stats_path, "global_means.npy")
#     params.global_stds_path = os.path.join(stats_path, "global_stds.npy")
#     params.time_diff_means_path = os.path.join(stats_path, "time_diff_means.npy")
#     params.time_diff_stds_path = os.path.join(stats_path, "time_diff_stds.npy")

#     # experiment direction
#     params.experiment_dir = exp_path

#     # checkpoint locations
#     params.checkpoint_path = os.path.join(exp_path, 'training_checkpoints/ckpt_mp{mp_rank}.tar')
#     params.best_checkpoint_path = os.path.join(exp_path, 'training_checkpoints/best_ckpt_mp{mp_rank}.tar')

#     # general parameters
#     params.dhours = 24
#     params.h5_path = H5_PATH
#     params.n_history = n_history
#     params.n_future = n_future
#     params.batch_size = batch_size
#     params.normalization = normalization

#     # performance parameters
#     params.num_data_workers = num_data_workers

#     # logging options
#     params.log_to_screen = False
#     params.log_to_wandb = False
#     params.log_video = 0

#     # test architecture
#     params.nettype = 'sfno'

#     # losss
#     params.loss = 'geometric l2'
#     params.lr = 5E-4
#     params.weight_decay = 0.0

#     # optimizer
#     params.optimizer_type = 'AdamW'
#     params.optimizer_beta1 = 0.9
#     params.optimizer_beta2 = 0.95
#     params.optimizer_max_grad_norm = 32

#     # job size
#     params.max_epochs = 1
#     params.n_train_samples = 10
#     params.n_eval_samples = 2

#     # scheduler
#     params.scheduler = 'CosineAnnealingLR'
#     params.scheduler_T_max = 10
#     params.lr_warmup_steps = 0

#     # other
#     params.finetune = False

#     return params

# class TestTrainer(unittest.TestCase):

#     def setUp(self, path: Optional[str] = "/tmp"):

#         self.device = torch.device('cpu')

#         # create temporary directory
#         self.tmpdir = tempfile.TemporaryDirectory(dir=path)
#         tmp_path = self.tmpdir.name

#         exp_path = os.path.join(tmp_path, "experiment_dir")
#         os.mkdir(exp_path)
#         os.mkdir(os.path.join(exp_path, "training_checkpoints"))

#         # init datasets and stats
#         train_path, num_train, valid_path, num_valid, stats_path = init_dataset(tmp_path)

#         # init parameters
#         self.params = init_params(exp_path,
#                                   train_path,
#                                   valid_path,
#                                   stats_path,
#                                   batch_size = 2,
#                                   n_history = 0,
#                                   n_future = 0,
#                                   normalization = 'zscore',
#                                   num_data_workers = 1)

#         self.params.multifiles = True
#         self.params.num_train = num_train
#         self.params.num_valid = num_valid

#         # this is also correct for most cases:
#         self.params.io_grid = [1, 1, 1]
#         self.params.io_rank = [0, 0, 0]

#         # self.num_steps = 5
#         self.params.print_timings_frequency = 0

#     def test_training(self):
#         self.trainer = Trainer(self.params, 0)
#         self.trainer.train()

# if __name__ == '__main__':
#     unittest.main()
