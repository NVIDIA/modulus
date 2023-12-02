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
# import glob
# import tempfile
# from typing import List, Optional

# import unittest
# import torch
# import numpy as np
# import h5py as h5

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from modulus.experimental.sfno.utils.dataloader import get_dataloader

# from modulus.experimental.sfno.tests.testutils import get_default_parameters, init_dataset
# from modulus.experimental.sfno.tests.testutils import H5_PATH, NUM_CHANNELS, IMG_SIZE_H, IMG_SIZE_W

# def get_sample(path: str, idx):
#     files = sorted(glob.glob(os.path.join(path, "*.h5")))
#     h5file = h5.File(files[0],'r')
#     num_samples_per_file = h5file[H5_PATH].shape[0]
#     h5file.close()
#     file_id = idx // num_samples_per_file
#     file_index = idx % num_samples_per_file

#     with h5.File(files[file_id],'r') as f:
#         data = f[H5_PATH][file_index, ...]

#     return data


# def init_params(train_path: str,
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

#     # general parameters
#     params.dhours = 24
#     params.h5_path = H5_PATH
#     params.n_history = n_history
#     params.n_future = n_future
#     params.batch_size = batch_size
#     params.normalization = normalization

#     # performance parameters
#     params.num_data_workers = num_data_workers

#     return params
    

# class TestMultifiles(unittest.TestCase):

#     def setUp(self, path: Optional[str] = "/tmp"):

#         self.device = torch.device('cpu')

#         # create temporary directory
#         self.tmpdir = tempfile.TemporaryDirectory(dir=path)
#         tmp_path = self.tmpdir.name

#         # init datasets and stats
#         train_path, num_train, valid_path, num_valid, stats_path = init_dataset(tmp_path)

#         # init parameters
#         self.params = init_params(train_path,
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

#         self.num_steps = 5

    
#     def test_shapes_and_sample_counts(self):
    
#         # create dataloaders
#         valid_loader, valid_dataset = get_dataloader(self.params, self.params.valid_data_path, train=False, device=self.device)
                
#         # do tests
#         num_valid_steps = self.params.num_valid // self.params.batch_size
#         for idt, token in enumerate(valid_loader):
#             self.assertEqual(len(token), 2)
#             inp, tar = token

#             self.assertEqual(tuple(inp.shape), (self.params.batch_size, 1, NUM_CHANNELS, IMG_SIZE_H, IMG_SIZE_W))
#             self.assertEqual(tuple(tar.shape), (self.params.batch_size, 1, NUM_CHANNELS, IMG_SIZE_H, IMG_SIZE_W))
            
#         self.assertEqual((idt+1), num_valid_steps)


#     def test_content(self):
        
#         # create dataloaders
#         valid_loader, valid_dataset = get_dataloader(self.params, self.params.valid_data_path, train=False, device=self.device)

#         # do tests
#         for idt, token in enumerate(valid_loader):
#             # get loader samples
#             inp, tar = token

#             # get test samples
#             off = self.params.batch_size * idt
#             inp_res = []
#             tar_res = []
#             for b in range(self.params.batch_size):
#                 inp_res.append(get_sample(self.params.valid_data_path, off+b))
#                 tar_res.append(get_sample(self.params.valid_data_path, off+b+1))
#             test_inp = np.squeeze(np.stack(inp_res, axis=0))
#             test_tar = np.squeeze(np.stack(tar_res, axis=0))

#             inp = np.squeeze(inp.cpu().numpy())
#             tar = np.squeeze(tar.cpu().numpy())

#             self.assertTrue(np.allclose(inp, test_inp))
#             self.assertTrue(np.allclose(tar, test_tar))

#             if idt > self.num_steps:
#                 break

#     def test_history(self):
#         # set history:
#         self.params.n_history = 3
        
#         # create dataloaders
#         valid_loader, valid_dataset = get_dataloader(self.params, self.params.valid_data_path, train=False, device=self.device)

#         # do tests
#         for idt, token in enumerate(valid_loader):
#             self.assertEqual(len(token), 2)
#             inp, tar = token

#             self.assertEqual(tuple(inp.shape), (self.params.batch_size, self.params.n_history+1, NUM_CHANNELS, IMG_SIZE_H, IMG_SIZE_W))
#             self.assertEqual(tuple(tar.shape), (self.params.batch_size, 1, NUM_CHANNELS, IMG_SIZE_H, IMG_SIZE_W))

#             if idt > self.num_steps:
#                 break

#     def test_future(self):
#         # set future:
#         self.params.n_future = 3

# 	# create dataloaders
#         train_loader, train_dataset, _ = get_dataloader(self.params, self.params.train_data_path, train=True, device=self.device)

# 	# do tests
#         for idt, token in enumerate(train_loader):
#             self.assertEqual(len(token), 2)
#             inp, tar = token

#             self.assertEqual(tuple(inp.shape), (self.params.batch_size, 1, NUM_CHANNELS, IMG_SIZE_H, IMG_SIZE_W))
#             self.assertEqual(tuple(tar.shape), (self.params.batch_size, self.params.n_future+1, NUM_CHANNELS, IMG_SIZE_H, IMG_SIZE_W))

#             if idt > self.num_steps:
#                 break

#     def test_autoreg(self):
#         self.params.valid_autoreg_steps = 3

#         # create dataloaders
#         valid_loader, valid_dataset = get_dataloader(self.params, self.params.valid_data_path, train=False, device=self.device)

#         # do tests
#         for idt, token in enumerate(valid_loader):
#             self.assertEqual(len(token), 2)
#             inp, tar = token

#             self.assertEqual(tuple(inp.shape), (self.params.batch_size, 1, NUM_CHANNELS, IMG_SIZE_H, IMG_SIZE_W))
#             self.assertEqual(tuple(tar.shape), (self.params.batch_size, self.params.valid_autoreg_steps+1, NUM_CHANNELS, IMG_SIZE_H, IMG_SIZE_W))

#             if idt > self.num_steps:
#                 break

#     def test_distributed(self):
#         self.params.io_grid = [1, 2, 1]
#         self.params.io_rank = [0, 1, 0]

#         valid_loader, valid_dataset = get_dataloader(self.params, self.params.valid_data_path, train=False, device=self.device)

#         off_x = valid_dataset.img_local_offset_x
#         off_y = valid_dataset.img_local_offset_y
#         range_x = valid_dataset.img_local_shape_x
#         range_y = valid_dataset.img_local_shape_y
        
#         # do tests
#         num_steps = 3
#         for idt, token in enumerate(valid_loader):
#             # get loader samples
#             inp, tar = token

#             self.assertEqual(tuple(inp.shape), (self.params.batch_size, 1, NUM_CHANNELS, range_x, range_y))
#             self.assertEqual(tuple(tar.shape), (self.params.batch_size, 1, NUM_CHANNELS, range_x, range_y))
            
#             # get test samples
#             off = self.params.batch_size * idt
#             inp_res = []
#             tar_res = []
#             for b in range(self.params.batch_size):
#                 tmp = get_sample(self.params.valid_data_path, off+b)
#                 inp_res.append(tmp[:, off_x:off_x+range_x, off_y:off_y+range_y])
#                 tmp = get_sample(self.params.valid_data_path, off+b+1)
#                 tar_res.append(tmp[:, off_x:off_x+range_x, off_y:off_y+range_y])

#             # stack
#             test_inp = np.squeeze(np.stack(inp_res, axis=0))
#             test_tar = np.squeeze(np.stack(tar_res, axis=0))
            
#             inp = np.squeeze(inp.cpu().numpy())
#             tar = np.squeeze(tar.cpu().numpy())

#             self.assertTrue(np.allclose(inp, test_inp))
#             self.assertTrue(np.allclose(tar, test_tar))

#             if idt > self.num_steps:
#                 break

# if __name__ == '__main__':
#     unittest.main()
    
