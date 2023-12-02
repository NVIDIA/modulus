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
# from parameterized import parameterized
# import torch

# from modulus.experimental.sfno.networks.models import list_models, get_model

# from modulus.experimental.sfno.tests.testutils import get_default_parameters

# class TestModels(unittest.TestCase):

#     def setUp(self):
#         self.params = get_default_parameters()

#         self.params.history_normalization_mode = "none"

#         # generating the image logic that is typically used by the dataloader
#         self.params.img_shape_x = 32
#         self.params.img_shape_y = 64
#         self.params.N_in_channels = 4
#         self.params.N_out_channels = 2
#         self.params.img_local_shape_x = self.params.img_crop_shape_x = self.params.img_shape_x
#         self.params.img_local_shape_y = self.params.img_crop_shape_y = self.params.img_shape_y
#         self.params.img_local_offset_x = 0
#         self.params.img_local_offset_y = 0

#         # also set the batch size for testing
#         self.params.batch_size = 4

#     @parameterized.expand(list_models())
#     def test_model(self, nettype):
#         """
#         Tests initialization of all the models and the forward and backward pass
#         """
#         self.params.nettype = nettype
#         if nettype == "debug":
#             return
#         model = get_model(self.params)

#         inp_shape = (self.params.batch_size, self.params.N_in_channels, self.params.img_shape_x, self.params.img_shape_y)
#         out_shape = (self.params.batch_size, self.params.N_out_channels, self.params.img_shape_x, self.params.img_shape_y)
        
#         # prepare some dummy data
#         inp = torch.randn(*inp_shape)
#         inp.requires_grad = True

#         # forward pass and check shapes
#         out = model(inp)
#         self.assertEqual(out.shape, out_shape)

#         # backward pass and check gradients are not None
#         out = torch.sum(out)
#         out.backward()
#         self.assertTrue(inp.grad is not None)
#         self.assertEqual(inp.grad.shape, inp_shape)

# if __name__ == '__main__':
#     unittest.main()
