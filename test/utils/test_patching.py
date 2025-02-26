# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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


import torch

from modulus.utils.patching import image_batching, image_fuse


def test_image_fuse_basic():
    # Basic test: No overlap, no boundary, one patch
    batch_size = 1
    for img_shape_y, img_shape_x in ((4, 4), (8, 4)):
        overlap_pix = 0
        boundary_pix = 0

        input_tensor = (
            torch.arange(1, 17).view(1, 1, img_shape_y, img_shape_x).cuda().float()
        )
        fused_image = image_fuse(
            input_tensor,
            img_shape_y,
            img_shape_x,
            batch_size,
            overlap_pix,
            boundary_pix,
        )
        assert fused_image.shape == (batch_size, 1, img_shape_y, img_shape_x)
        expected_output = input_tensor
        assert torch.allclose(
            fused_image, expected_output, atol=1e-5
        ), "Output does not match expected output."


def test_image_fuse_with_boundary():
    # Test with boundary pixels
    batch_size = 1
    img_shape_x = img_shape_y = 4
    overlap_pix = 0
    boundary_pix = 1

    input_tensor = torch.ones(1, 1, 6, 6).cuda().float()  # All ones for easy validation
    fused_image = image_fuse(
        input_tensor,
        img_shape_y,
        img_shape_x,
        batch_size,
        overlap_pix,
        boundary_pix,
    )
    assert fused_image.shape == (batch_size, 1, img_shape_y, img_shape_x)
    expected_output = (
        torch.ones(1, 1, 4, 4).cuda().float()
    )  # Expected output is just the inner 4x4 part
    assert torch.allclose(
        fused_image, expected_output, atol=1e-5
    ), "Output with boundary does not match expected output."


def test_image_fuse_with_multiple_batches():
    # Test with multiple batches
    batch_size = 2
    for img_shape_y, img_shape_x in ((4, 4), (8, 4)):
        overlap_pix = 0
        boundary_pix = 0

        input_tensor = (
            torch.cat(
                [
                    torch.arange(1, 17).view(1, 1, 4, 4),
                    torch.arange(17, 33).view(1, 1, 4, 4),
                ]
            )
            .cuda()
            .float()
        )
        input_tensor = input_tensor.repeat(2, 1, 1, 1)
        fused_image = image_fuse(
            input_tensor,
            img_shape_y,
            img_shape_x,
            batch_size,
            overlap_pix,
            boundary_pix,
        )
        assert fused_image.shape == (batch_size, 1, img_shape_y, img_shape_x)
        expected_output = (
            torch.cat(
                [
                    torch.arange(1, 17).view(1, 1, img_shape_y, img_shape_x),
                    torch.arange(17, 33).view(1, 1, img_shape_y, img_shape_x),
                ]
            )
            .cuda()
            .float()
        )
        assert torch.allclose(
            fused_image, expected_output, atol=1e-5
        ), "Output for multiple batches does not match expected output."


def test_image_batching_basic():
    # Test with no overlap, no boundary, no input_interp
    batch_size = 1
    patch_shape_x = patch_shape_y = 4
    overlap_pix = 0
    boundary_pix = 0

    input_tensor = torch.arange(1, 17).view(1, 1, 4, 4).cuda().float()
    batched_images = image_batching(
        input_tensor,
        patch_shape_y,
        patch_shape_x,
        overlap_pix,
        boundary_pix,
    )
    assert batched_images.shape == (batch_size, 1, patch_shape_y, patch_shape_x)
    expected_output = input_tensor
    assert torch.allclose(
        batched_images, expected_output, atol=1e-5
    ), "Batched images do not match expected output."


def test_image_batching_with_boundary():
    # Test with boundary pixels, no overlap, no input_interp
    patch_shape_x = patch_shape_y = 6
    overlap_pix = 0
    boundary_pix = 1

    input_tensor = torch.ones(1, 1, 4, 4).cuda().float()  # All ones for easy validation
    batched_images = image_batching(
        input_tensor,
        patch_shape_y,
        patch_shape_x,
        overlap_pix,
        boundary_pix,
    )
    assert batched_images.shape == (1, 1, patch_shape_x, patch_shape_y)
    expected_output = torch.ones(1, 1, 6, 6).cuda().float()
    assert torch.allclose(
        batched_images, expected_output, atol=1e-5
    ), "Batched images with boundary do not match expected output."


def test_image_batching_with_input_interp():
    # Test with input_interp tensor
    patch_shape_x = patch_shape_y = 4
    overlap_pix = 0
    boundary_pix = 0

    for img_shape_y, img_shape_x in ((4, 4), (16, 8)):
        batch_size = (img_shape_y // patch_shape_y) * (img_shape_x // patch_shape_x)
        input_tensor = (
            torch.arange(1, 17).view(1, 1, img_shape_y, img_shape_x).cuda().float()
        )
        input_interp = (
            torch.ones(1, 1, img_shape_y, img_shape_x).cuda().float()
        )  # All ones for easy validation
        batched_images = image_batching(
            input_tensor,
            patch_shape_y,
            patch_shape_x,
            overlap_pix,
            boundary_pix,
            input_interp=input_interp,
        )
        assert batched_images.shape == (batch_size, 2, patch_shape_x, patch_shape_y)
        expected_output = torch.cat((input_tensor, input_interp), dim=1)
        assert torch.allclose(
            batched_images, expected_output, atol=1e-5
        ), "Batched images with input_interp do not match expected output."
