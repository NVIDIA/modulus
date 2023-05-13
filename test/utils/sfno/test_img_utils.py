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
import torch
import numpy as np
from modulus.utils.sfno.img_utils import PeriodicPad2d, reshape_fields


def test_PeriodicPad2d():
    pad_width = 1
    pad = PeriodicPad2d(pad_width)

    # Create a tensor with shape (batch_size, channels, height, width) = (1, 1, 3, 3)
    tensor = torch.Tensor([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]])

    padded_tensor = pad(tensor)

    # Check if padding is correctly applied
    assert padded_tensor.shape == torch.Size(
        [1, 1, 5, 5]
    ), "Padding was not applied correctly"


def test_reshape_fields():
    # Define a class to mock the params
    class MockParams:
        pass

    # Create a mock params object
    params = MockParams()
    params.in_channels = [0, 1]
    params.out_channels = [0, 1]
    params.min_path = "min_mock.npy"
    params.max_path = "max_mock.npy"
    params.global_means_path = "global_means_mock.npy"
    params.global_stds_path = "global_stds_mock.npy"
    params.normalization = None
    params.add_grid = False
    params.gridtype = None
    params.n_grid_channels = None
    params.roll = False

    # Create mock npy files
    np.save(params.min_path, np.zeros((1, 2)))
    np.save(params.max_path, np.ones((1, 2)))
    np.save(params.global_means_path, np.zeros((1, 2)))
    np.save(params.global_stds_path, np.ones((1, 2)))

    # Create a numpy array for the test
    img = np.ones((2, 2, 3, 3))  # shape (n_history+1, c, h, w)

    # Call the function under test
    reshaped_img = reshape_fields(
        img, "inp", None, None, 0, 0, params, 0, False, normalize=False
    )

    # Check if the output shape is as expected
    assert reshaped_img.shape == torch.Size(
        [4, 3, 3]
    ), "reshape_fields did not return the expected shape"

    # Remove mock npy files
    os.remove(params.min_path)
    os.remove(params.max_path)
    os.remove(params.global_means_path)
    os.remove(params.global_stds_path)
