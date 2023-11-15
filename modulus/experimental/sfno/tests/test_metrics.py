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

# import unittest

# import numpy as np
# import torch
# import xarray as xr

# from modulus.experimental.sfno.utils.grids import GridQuadrature
# from modulus.experimental.sfno.utils.metrics.functions import GeometricL1, GeometricRMSE, GeometricACC

# try:
#     import xskillscore as xs
# except ImportError:
#     raise ImportError('xskillscore is not installed. Please install it with "pip install xskillscore"')

# quadrature_list = ['naive', 'clenshaw-curtiss', 'legendre-gauss']
# param_list = [(1, 10, 20, 101), (4, 21, 721, 1440)]

# class TestMetrics(unittest.TestCase):

#     def setUp(self):

#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         torch.manual_seed(333)
#         torch.cuda.manual_seed(333)

#     def test_weight_normalization(self):
#         for quadrature_type in quadrature_list:
#             for shape in param_list:
#                 with self.subTest(f"{quadrature_type} {shape}"):
#                     quad = GridQuadrature(quadrature_type, img_shape=(shape[2], shape[3]), normalize=True).to(self.device)
#                     flat_vector = torch.ones(shape, device=self.device)
#                     integral = torch.mean(quad(flat_vector)).item()
                    
#                     self.assertTrue(np.allclose(integral, 1., rtol=1e-5, atol=0))

                        
#     def test_weighted_rmse(self):
#         for quadrature_type in quadrature_list:
#             for shape in param_list:
#                 with self.subTest(f"{quadrature_type} {shape}"):
                    
#                     # rmse handle
#                     rmse_func = GeometricRMSE(quadrature_type, img_shape=(shape[2], shape[3]),
#                                               normalize=True,
#                                               channel_reduction='none',
#                                               batch_reduction='none').to(self.device)
                    
#                     # generate toy data
#                     A = torch.randn(*shape, device=self.device)
#                     B = torch.randn(*shape, device=self.device)
#                     rmse = torch.mean(rmse_func(A, B), dim=0).cpu().numpy()
                    
#                     lwf = torch.squeeze(rmse_func.quadrature.quad_weight).cpu().numpy()
#                     lwf = xr.DataArray(lwf, dims=['lat', 'lon'])
#                     A = xr.DataArray(A.cpu(), dims=['batch', 'channels', 'lat', 'lon'])
#                     B = xr.DataArray(B.cpu(), dims=['batch', 'channels', 'lat', 'lon'])
#                     rmse_xskillscore = xs.rmse(A, B, weights=lwf, dim=['lat', 'lon']).to_numpy().mean(axis=0)
    
#                     # relative difference
#                     rdiff = np.abs(rmse-rmse_xskillscore) / np.abs(rmse_xskillscore)
                    
#                     self.assertTrue(rdiff.mean() <= 1e-5)
#                     self.assertTrue(np.allclose(rmse, rmse_xskillscore, rtol=1e-5, atol=0))

                    
#     def test_l1(self):
#         for quadrature_type in quadrature_list:
#             for shape in param_list:
#                 with self.subTest(f"{quadrature_type} {shape}"):
                    
#                     # l1 handle
#                     l1_func = GeometricL1(quadrature_type, img_shape=(shape[2], shape[3]),
#                                           normalize=True,
#                                           channel_reduction='mean',
#                                           batch_reduction='mean').to(self.device)
                    
#                     # generate toy data
#                     A = torch.randn(*shape, device=self.device)
#                     B = torch.randn(*shape, device=self.device)
#                     l1 = l1_func(A, B).cpu().numpy()
                    
#                     lwf = l1_func.quadrature.quad_weight.cpu().numpy()
#                     lwf = xr.DataArray(np.tile(lwf, (shape[0], shape[1], 1, 1)), dims=['batch', 'channels', 'lat', 'lon'])
#                     A = xr.DataArray(A.cpu(), dims=['batch', 'channels', 'lat', 'lon'])
#                     B = xr.DataArray(B.cpu(), dims=['batch', 'channels', 'lat', 'lon'])
#                     l1_xskillscore = xs.mae(A, B, weights=lwf).to_numpy()
                    
#                     rdiff = np.abs(l1-l1_xskillscore) / np.abs(l1_xskillscore)
                    
#                     self.assertTrue(rdiff <= 1e-5)

                    
#     def test_weighted_acc(self):
#         for quadrature_type in quadrature_list:
#             for shape in param_list:
#                 with self.subTest(f"{quadrature_type} {shape}"):
    
#                     # ACC handle
#                     acc_func = GeometricACC(quadrature_type, img_shape=(shape[2], shape[3]),
#                                             normalize=True,
#                                             channel_reduction='none',
#                                             batch_reduction='mean').to(self.device)
                    
#                     # generate toy data
#                     A = torch.randn(*shape, device=self.device)
#                     B = torch.randn(*shape, device=self.device)
#                     A_mean = acc_func.quadrature(A).reshape(shape[0], shape[1], 1, 1)
#                     B_mean = acc_func.quadrature(B).reshape(shape[0], shape[1], 1, 1)
#                     acc = acc_func(A-A_mean, B-B_mean).cpu().numpy()
                    
#                     lwf = torch.squeeze(acc_func.quadrature.quad_weight).cpu().numpy()
#                     lwf = xr.DataArray(lwf, dims=['lat', 'lon'])
#                     A = xr.DataArray(A.cpu(), dims=['batch', 'channels', 'lat', 'lon'])
#                     B = xr.DataArray(B.cpu(), dims=['batch', 'channels', 'lat', 'lon'])
                    
#                     acc_xskillscore = xs.pearson_r(A, B, weights=lwf, dim=['lat', 'lon']).to_numpy().mean(axis=0)
                    
#                     rdiff = np.abs(acc-acc_xskillscore) / np.abs(acc_xskillscore)
                    
#                     self.assertTrue(rdiff.mean() <= 1e-5)
#                     self.assertTrue(np.allclose(acc, acc_xskillscore, rtol=1e-5, atol=0))


# if __name__ == '__main__':
#     unittest.main()
