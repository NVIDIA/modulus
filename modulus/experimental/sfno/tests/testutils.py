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

# import os
# from typing import List, Optional

# import numpy as np
# import h5py as h5

# from modulus.experimental.sfno.utils.YParams import ParamsBase

# H5_PATH = "fields"
# NUM_CHANNELS = 4
# IMG_SIZE_H = 181
# IMG_SIZE_W = 360
# CHANNEL_NAMES = ['u10m', 't2m', 'u500', 'z500']

# def get_default_parameters():

#     # instantiate parameters
#     params = ParamsBase()

#     # dataset related
#     params.dt = 1
#     params.n_history = 0
#     params.n_future = 0
#     params.normalization = 'zscore'
#     params.data_grid_type = 'equiangular'
#     params.model_grid_type = 'equiangular'
#     params.sht_grid_type = 'legendre-gauss'
    
#     params.resuming = False
#     params.amp_mode = 'none'
#     params.jit_mode = 'none'
#     params.cuda_graph_mode = 'none'
#     params.enable_benchy = False
#     params.disable_ddp = False
#     params.enable_nhwc = False
#     params.checkpointing = 0
#     params.enable_synthetic_data = False
#     params.split_data_channels = False

#     # dataloader related
#     params.in_channels = list(range(NUM_CHANNELS))
#     params.out_channels = list(range(NUM_CHANNELS))
#     params.channel_names = [CHANNEL_NAMES[i] for i in range(NUM_CHANNELS)]

#     params.batch_size = 1
#     params.valid_autoreg_steps = 0
#     params.num_data_workers = 1
#     params.multifiles = True
#     params.io_grid = [1, 1, 1]
#     params.io_grid = [0, 0, 0]

#     # extra channels
#     params.add_grid = False
#     params.add_zenith = False
#     params.add_orography = False
#     params.add_landmask = False

#     return params

# def init_dataset(path: str,
#                  num_samples_per_year: Optional[int]=365,
#                  num_channels: Optional[int]=NUM_CHANNELS,
#                  img_size_h: Optional[int]=IMG_SIZE_H,
#                  img_size_w: Optional[int]=IMG_SIZE_W):
    
#     test_path = os.path.join(path, "test")
#     os.mkdir(test_path)

#     train_path = os.path.join(path, "train")
#     os.mkdir(train_path)

#     stats_path = os.path.join(path, "stats")
#     os.mkdir(stats_path)

#     # rng:
#     rng = np.random.default_rng(seed=333)
    
#     # create training files
#     num_train = 0
#     for y in [2016, 2017]:
#         data_path = os.path.join(train_path, f"{y}.h5")
#         with h5.File(data_path, "w") as hf:
#             hf.create_dataset(H5_PATH, shape=(num_samples_per_year, num_channels, img_size_h, img_size_w))
#             hf[H5_PATH][...] = rng.random((num_samples_per_year, num_channels, img_size_h, img_size_w), dtype=np.float32)
#         num_train += num_samples_per_year

#     # create validation files
#     num_test = 0
#     for	y in [2019]:
#         data_path = os.path.join(test_path, f"{y}.h5")
#         with h5.File(data_path, "w") as hf:
#             hf.create_dataset(H5_PATH, shape=(num_samples_per_year, num_channels, img_size_h, img_size_w))
#             hf[H5_PATH][...] =	rng.random((num_samples_per_year, num_channels, img_size_h, img_size_w), dtype=np.float32)
#         num_test += num_samples_per_year

#     # create stats files
#     np.save(os.path.join(stats_path, "mins.npy"),
#             np.zeros((1, num_channels, 1, 1), dtype=np.float64))
    
#     np.save(os.path.join(stats_path, "maxs.npy"),
#             np.ones((1, num_channels, 1, 1), dtype=np.float64))

#     np.save(os.path.join(stats_path, "time_means.npy"),
#             np.zeros((1, num_channels, img_size_h, img_size_w), dtype=np.float64))

#     np.save(os.path.join(stats_path, "global_means.npy"),
#             np.zeros((1, num_channels, 1, 1), dtype=np.float64))

#     np.save(os.path.join(stats_path, "global_stds.npy"),
#             np.ones((1, num_channels, 1, 1), dtype=np.float64))

#     np.save(os.path.join(stats_path, "time_diff_means.npy"),
#             np.zeros((1, num_channels, 1, 1), dtype=np.float64))

#     np.save(os.path.join(stats_path, "time_diff_stds.npy"),
#             np.ones((1, num_channels, 1, 1), dtype=np.float64))
    
#     return train_path, num_train, test_path, num_test, stats_path
