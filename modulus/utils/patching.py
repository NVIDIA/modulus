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


import math
from typing import Optional, Tuple
import random
from abc import ABC, abstractmethod
import warnings

import torch
from torch import Tensor

"""
This module defines utilities, including classes and functions, for domain
decomposition.
"""


class BasePatching(ABC):
    """
    Abstract base class for image patching operations.

    This class provides a foundation for implementing various image patching
    strategies.
    It handles basic validation and provides abstract methods that must be
    implemented by subclasses.

    Parameters
    ----------
    img_shape : Tuple[int, int]
        The height and width of the input images (img_shape_y, img_shape_x).
    patch_shape : Tuple[int, int]
        The height and width of the patches (patch_shape_y, patch_shape_x) to
        extract.
    """

    def __init__(
        self,
        img_shape: Tuple[int, int],
        patch_shape: Tuple[int, int]
    ) -> None:
        # Make sure patches fit within the image
        if any(p > i for p, i in zip(self.patch_shape, self.img_shape)):
            warnings.warn(
                f"Patch shape ({self.patch_shape}) is larger than "
                f"image shape ({self.img_shape}). "
                f"Patches will be cropped to fit within the image."
            )
        self.img_shape = img_shape
        self.patch_shape = (
            min(p, i) for p, i in zip(self.patch_shape, self.img_shape)
        )

    @abstractmethod
    def apply(self, input: Tensor, **kwargs) -> Tensor:
        """
        Apply the patching operation to the input tensor.

        Parameters
        ----------
        input : Tensor
            Input tensor of shape (batch_size, channels, img_shape_y,
            img_shape_x).
        **kwargs : dict
            Additional keyword arguments specific to the patching
            implementation.

        Returns
        -------
        Tensor
            Patched tensor, shape depends on specific implementation.
        """
        pass

    def fuse(self, input: Tensor, **kwargs) -> Tensor:
        """
        Fuse patches back into a complete image.

        Parameters
        ----------
        input : Tensor
            Input tensor containing patches.
        **kwargs : dict
            Additional keyword arguments specific to the fusion implementation.

        Returns
        -------
        Tensor
            Fused tensor, shape depends on specific implementation.

        Raises
        ------
        NotImplementedError
            If the subclass does not implement this method.
        """
        raise NotImplementedError(
            "'fuse' method must be implemented in subclasses."
        )

    def global_index(self, batch_size: int) -> Tensor:
        """
        Returns a tensor containing the global indices for each patch.

        Global indices correspond to (y, x) global grid coordinates of each
        element within the original image (before patching). It is typically
        used to keep track of the original position of each patch in the
        original image.

        Parameters
        ----------
        batch_size : int
            The size of the input batch.

        Returns
        -------
        Tensor
            A tensor of shape (batch_size * self.patch_num, 2, patch_shape_y,
            patch_shape_x). `global_index[:, 0, :, :]` contains the
            y-coordinate (height), and `global_index[:, 1, :, :]` contains the
            x-coordinate (width).
        """
        Ny = torch.arange(self.img_shape[0]).int()
        Nx = torch.arange(self.img_shape[1]).int()
        grid = torch.stack(torch.meshgrid(Ny, Nx, indexing="ij"), dim=0)[
            None,
        ].expand(batch_size, -1, -1, -1)
        global_index = self.apply(grid)
        return global_index


class RandomPatching(BasePatching):
    """
    Class for randomly extracting patches from images.

    This class provides utilities to randomly extract patches from images
    represented as 4D tensors. It maintains a list of random patch indices
    that can be reset as needed.

    Parameters
    ----------
    img_shape : Tuple[int, int]
        The height and width of the input images (img_shape_y, img_shape_x).
    patch_shape : Tuple[int, int]
        The height and width of the patches (patch_shape_y, patch_shape_x) to
        extract.
    patch_num : int
        The number of patches to extract.

    Attributes
    ----------
    patch_indices : List[Tuple[int, int]]
        The indices of the patches to extract from the images. These indices
        correspond to the (y, x) coordinates of the lower left corner of each
        patch.

    See Also
    --------
    :class:`modulus.utils.patching.BasePatching`
        The base class providing the patching interface.
    :class:`modulus.utils.patching.DeterministicPatching`
        Alternative patching strategy using deterministic patch locations.
    """

    def __init__(
        self,
        img_shape: Tuple[int, int],
        patch_shape: Tuple[int, int],
        patch_num: int
    ) -> None:
        """
        Initialize the RandomPatching object with the provided image shape,
        patch shape, and number of patches to extract.

        Parameters
        ----------
        img_shape : Tuple[int, int]
            The height and width of the input images (img_shape_y,
            img_shape_x).
        patch_shape : Tuple[int, int]
            The height and width of the patches (patch_shape_y, patch_shape_x)
            to extract.
        patch_num : int
            The number of patches to extract.

        Returns
        -------
            None
        """
        super().__init__(img_shape, patch_shape)
        self.patch_num = patch_num
        # Generate the indices of the patches to extract
        self.reset_patches_indices()

    def reset_patch_indices(self) -> None:
        """
        Generate new random indices for the patches to extract. These indices
        correspond to the indices of the lower left corner of each patch.

        Returns
        -------
            None
        """
        self.patch_indices = [
            (
                random.randint(0, self.img_shape_y - self.patch_shape_y),
                random.randint(0, self.img_shape_x - self.patch_shape_x)
            ) for _ in range(self.patch_num)
        ]
        return

    def apply(
        self,
        input: Tensor,
        additional_input: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Applies the patching operation by extracting patches specified by
        `self.patch_indices` from the `input` Tensor. Extracted patches are
        batched along the first dimension of the output. The layout of the
        output assumes an outer index for the patches and an inner index for
        the batch elements, thats is for any i, `out[B * i: B * (i + 1)]`
        corresponds to the same patch exacted from each batch element of
        `input`.

        Arguments
        ---------
        input : Tensor
            The input tensor representing the full image with shape
            (batch_size, channels, img_shape_y, img_shape_x).
        additional_input : Optional[Tensor], optional
            If provided, it is concatenated to each patch along `dim=1`.
            Must have same batch size as `input`. Bilinear interpolation
            is used to interpolate `additional_input` onto a 2D grid of shape
            (patch_shape_y, patch_shape_x).

        Returns
        -------
        Tensor
            A tensor of shape (batch_size * self.patch_num, channels_out,
            patch_shape_y, patch_shape_x). If `additional_input` is provided,
            channels_out comprises channels from both `input` and
            `additional_input`.
        """
        B = input.shape[0]
        out = torch.zeros(
            B * self.patch_num,
            (input.shape[1] + (additional_input.shape[1]
                               if additional_input is not None else 0)),
            self.patch_shape[0],
            self.patch_shape[1],
            device=input.device
        )
        if additional_input is not None:
            add_input_interp = torch.nn.functional.interpolate(
                input=additional_input,
                size=self.patch_shape,
                mode="bilinear"
            )
        for i, (py, px) in enumerate(self.patch_indices):
            if additional_input is not None:
                out[B * i: B * (i + 1),] = torch.cat(
                    (
                        input[
                            :,
                            :,
                            py: py + self.patch_shape[0],
                            px: px + self.patch_shape[1],
                        ],
                        add_input_interp,
                    ),
                    dim=1,
                )
            else:
                out[B * i: B * (i + 1),] = input[
                    :,
                    :,
                    py: py + self.patch_shape[0],
                    px: px + self.patch_shape[1],
                ]
        return out


class DeterministicPatching(BasePatching):
    """
    Class for deterministically extracting patches from images.

    This class provides utilities to extract patches from images in a
    deterministic manner, with configurable overlap and boundary pixels.
    The patches are extracted in a grid-like pattern covering the entire image.

    Parameters
    ----------
    img_shape : Tuple[int, int]
        The height and width of the input images (img_shape_y, img_shape_x).
    patch_shape : Tuple[int, int]
        The height and width of the patches (patch_shape_y, patch_shape_x) to
        extract.
    overlap_pix : int, optional
        Number of pixels to overlap between adjacent patches, by default 0.
    boundary_pix : int, optional
        Number of pixels to crop as boundary from each patch, by default 0.

    Attributes
    ----------
    patch_num : int
        Total number of patches that will be extracted from the image,
        calculated as patch_num_x * patch_num_y.

    See Also
    --------
    :class:`modulus.utils.patching.BasePatching`
        The base class providing the patching interface.
    :class:`modulus.utils.patching.RandomPatching`
        Alternative patching strategy using random patch locations.
    """

    def __init__(
        self,
        img_shape: Tuple[int, int],
        patch_shape: Tuple[int, int],
        overlap_pix: int = 0,
        boundary_pix: int = 0
    ):
        super().__init__(img_shape, patch_shape)
        self.overlap_pix = overlap_pix
        self.boundary_pix = boundary_pix
        patch_num_x = math.ceil(
            img_shape[1] / (patch_shape[1] - overlap_pix - boundary_pix))
        patch_num_y = math.ceil(
            img_shape[0] / (patch_shape[0] - overlap_pix - boundary_pix))
        self.patch_num = patch_num_x * patch_num_y

    def apply(
        self,
        input: Tensor,
        additional_input: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Apply deterministic patching to the input tensor.

        Splits the input tensor into patches in a grid-like pattern. Can
        optionally concatenate additional interpolated data to each patch.

        Parameters
        ----------
        input : Tensor
            Input tensor of shape (batch_size, channels, img_shape_y,
            img_shape_x).
        additional_input : Optional[Tensor], optional
            Additional data to concatenate to each patch. Will be interpolated
            to match patch dimensions. Shape must be (batch_size,
            additional_channels, H, W), by default None.

        Returns
        -------
        Tensor
            Tensor containing patches with shape (batch_size * patch_num,
            channels [+ additional_channels], patch_shape_y, patch_shape_x).
            If additional_input is provided, its channels are concatenated
            along the channel dimension.

        See Also
        --------
        :func:`modulus.utils.patching.image_batching`
            The underlying function used to perform the patching operation.
        """
        if additional_input is not None:
            add_input_interp = torch.nn.functional.interpolate(
                input=additional_input,
                size=self.patch_shape,
                mode="bilinear"
            )
        else:
            add_input_interp = None
        out = image_batching(
            input=input,
            patch_shape_y=self.patch_shape[0],
            patch_shape_x=self.patch_shape[1],
            overlap_pix=self.overlap_pix,
            boundary_pix=self.boundary_pix,
            input_interp=add_input_interp
        )
        return out

    def fuse(
        self,
        input: Tensor,
        batch_size: int
    ) -> Tensor:
        """
        Fuse patches back into a complete image.

        Reconstructs the original image by stitching together patches,
        accounting for overlapping regions and boundary pixels. In overlapping
        regions, values are averaged.

        Parameters
        ----------
        input : Tensor
            Input tensor containing patches with shape (batch_size * patch_num,
            channels, patch_shape_y, patch_shape_x).
        batch_size : int
            The original batch size before patching.

        Returns
        -------
        Tensor
            Reconstructed image tensor with shape (batch_size, channels,
            img_shape_y, img_shape_x).

        See Also
        --------
        :func:`modulus.utils.patching.image_fuse`
            The underlying function used to perform the fusion operation.
        """
        out = image_fuse(
            input=input,
            patch_shape_y=self.patch_shape[0],
            patch_shape_x=self.patch_shape[1],
            batch_size=batch_size,
            overlap_pix=self.overlap_pix,
            boundary_pix=self.boundary_pix
        )
        return out


def image_batching(
    input: Tensor,
    patch_shape_y: int,
    patch_shape_x: int,
    overlap_pix: int,
    boundary_pix: int,
    input_interp: Optional[Tensor] = None,
) -> Tensor:
    """
    Splits a full image into a batch of patched images.

    This function takes a full image and splits it into patches, adding padding
    where necessary. It can also concatenate additional interpolated data to
    each patch if provided.

    Parameters
    ----------
    input : Tensor
        The input tensor representing the full image with shape (batch_size,
        channels, img_shape_y, img_shape_x).
    patch_shape_y : int
        The height (y-dimension) of each image patch.
    patch_shape_x : int
        The width (x-dimension) of each image patch.
    overlap_pix : int
        The number of overlapping pixels between adjacent patches.
    boundary_pix : int
        The number of pixels to crop as a boundary from each patch.
    input_interp : Optional[Tensor], optional
        Optional additional data to concatenate to each patch with shape
        (batch_size, interp_channels, patch_shape_y, patch_shape_x).
        By default None.

    Returns
    -------
    Tensor
        A tensor containing the image patches, with shape (total_patches *
        batch_size, channels [+ interp_channels], patch_shape_x,
        patch_shape_y).
    """
    # Infer sizes from input image
    batch_size, _, img_shape_y, img_shape_x = input.shape

    # Safety check: make sure patch_shapes are large enough to accommodate
    # overlaps and boundaries pixels
    if (patch_shape_x - overlap_pix - boundary_pix) < 1:
        raise ValueError(
            f"patch_shape_x must verify patch_shape_x ({patch_shape_x}) >= "
            f"1 + overlap_pix ({overlap_pix}) + boundary_pix ({boundary_pix})"
        )
    if (patch_shape_y - overlap_pix - boundary_pix) < 1:
        raise ValueError(
            f"patch_shape_y must verify patch_shape_y ({patch_shape_y}) >= "
            f"1 + overlap_pix ({overlap_pix}) + boundary_pix ({boundary_pix})"
        )

    patch_num_x = math.ceil(
        img_shape_x / (patch_shape_x - overlap_pix - boundary_pix))
    patch_num_y = math.ceil(
        img_shape_y / (patch_shape_y - overlap_pix - boundary_pix))
    padded_shape_x = (
        (patch_shape_x - overlap_pix - boundary_pix) * (patch_num_x - 1)
        + patch_shape_x
        + boundary_pix
    )
    padded_shape_y = (
        (patch_shape_y - overlap_pix - boundary_pix) * (patch_num_y - 1)
        + patch_shape_y
        + boundary_pix
    )
    pad_x_right = padded_shape_x - img_shape_x - boundary_pix
    pad_y_right = padded_shape_y - img_shape_y - boundary_pix
    input_padded = torch.zeros(
        input.shape[0], input.shape[1], padded_shape_y, padded_shape_x
    ).to(input.device)
    image_padding = torch.nn.ReflectionPad2d(
        (boundary_pix, pad_x_right, boundary_pix, pad_y_right)
    ).to(
        input.device
    )  # (padding_left,padding_right,padding_top,padding_bottom)
    input_padded = image_padding(input)
    patch_num = patch_num_x * patch_num_y
    if input_interp is not None:
        output = torch.zeros(
            patch_num * batch_size,
            input.shape[1] + input_interp.shape[1],
            patch_shape_y,
            patch_shape_x,
        ).to(input.device)
    else:
        output = torch.zeros(
            patch_num * batch_size,
            input.shape[1],
            patch_shape_y,
            patch_shape_x
        ).to(input.device)
    for x_index in range(patch_num_x):
        for y_index in range(patch_num_y):
            x_start = x_index * (patch_shape_x - overlap_pix - boundary_pix)
            y_start = y_index * (patch_shape_y - overlap_pix - boundary_pix)
            if input_interp is not None:
                output[
                    (x_index * patch_num_y + y_index)
                    * batch_size: (x_index * patch_num_y + y_index + 1)
                    * batch_size,
                ] = torch.cat(
                    (
                        input_padded[
                            :,
                            :,
                            y_start: y_start + patch_shape_y,
                            x_start: x_start + patch_shape_x,
                        ],
                        input_interp,
                    ),
                    dim=1,
                )
            else:
                output[
                    (x_index * patch_num_y + y_index)
                    * batch_size: (x_index * patch_num_y + y_index + 1)
                    * batch_size,
                ] = input_padded[
                    :,
                    :,
                    y_start:y_start + patch_shape_y,
                    x_start:x_start + patch_shape_x,
                ]
    return output


def image_fuse(
    input: Tensor,
    img_shape_y: int,
    img_shape_x: int,
    batch_size: int,
    overlap_pix: int,
    boundary_pix: int,
) -> Tensor:
    """
    Reconstructs a full image from a batch of patched images.

    This function takes a batch of image patches and reconstructs the full
    image by stitching the patches together. The function accounts for
    overlapping and boundary pixels, ensuring that overlapping areas are
    averaged.

    Parameters
    ----------
    input : Tensor
        The input tensor containing the image patches with shape (patch_num
        * batch_size, channels, patch_shape_y, patch_shape_x).
    img_shape_x : int
        The width (x-dimension) of the original full image.
    img_shape_y : int
        The height (y-dimension) of the original full image.
    batch_size : int
        The original batch size before patching.
    overlap_pix : int
        The number of overlapping pixels between adjacent patches.
    boundary_pix : int
        The number of pixels to crop as a boundary from each patch.

    Returns
    -------
    Tensor
        The reconstructed full image tensor with shape (batch_size, channels,
        img_shape_y, img_shape_x).

    """
    # Infer sizes from input image shape
    patch_shape_y, patch_shape_x = input.shape[2], input.shape[3]

    patch_num_x = math.ceil(
        img_shape_x / (patch_shape_x - overlap_pix - boundary_pix))
    patch_num_y = math.ceil(
        img_shape_y / (patch_shape_y - overlap_pix - boundary_pix))
    padded_shape_x = (
        (patch_shape_x - overlap_pix - boundary_pix) * (patch_num_x - 1)
        + patch_shape_x
        + boundary_pix
    )
    padded_shape_y = (
        (patch_shape_y - overlap_pix - boundary_pix) * (patch_num_y - 1)
        + patch_shape_y
        + boundary_pix
    )
    pad_x_right = padded_shape_x - img_shape_x - boundary_pix
    pad_y_right = padded_shape_y - img_shape_y - boundary_pix
    residual_x = patch_shape_x - pad_x_right  # residual pixels in the last patch
    residual_y = patch_shape_y - pad_y_right  # residual pixels in the last patch
    output = torch.zeros(
        batch_size,
        input.shape[1],
        img_shape_y,
        img_shape_x,
        device=input.device
    )
    one_map = torch.ones(
        1, 1, input.shape[2], input.shape[3], device=input.device)
    count_map = torch.zeros(
        1, 1, img_shape_y, img_shape_x, device=input.device
    )  # to count the overlapping times
    for x_index in range(patch_num_x):
        for y_index in range(patch_num_y):
            x_start = x_index * (patch_shape_x - overlap_pix - boundary_pix)
            y_start = y_index * (patch_shape_y - overlap_pix - boundary_pix)
            if (x_index == patch_num_x - 1) and (y_index != patch_num_y - 1):
                output[
                    :,
                    :,
                    y_start: y_start + patch_shape_y - 2 * boundary_pix,
                    x_start:
                ] += input[
                    (x_index * patch_num_y + y_index)
                    * batch_size: (x_index * patch_num_y + y_index + 1)
                    * batch_size,
                    :,
                    boundary_pix: patch_shape_y - boundary_pix,
                    boundary_pix: residual_x + boundary_pix,
                ]
                count_map[
                    :,
                    :,
                    y_start: y_start + patch_shape_y - 2 * boundary_pix,
                    x_start:
                ] += one_map[
                    :,
                    :,
                    boundary_pix: patch_shape_y - boundary_pix,
                    boundary_pix: residual_x + boundary_pix,
                ]
            elif (y_index == patch_num_y - 1) and ((x_index != patch_num_x - 1)):
                output[
                    :,
                    :,
                    y_start:,
                    x_start: x_start + patch_shape_x - 2 * boundary_pix
                ] += input[
                    (x_index * patch_num_y + y_index)
                    * batch_size: (x_index * patch_num_y + y_index + 1)
                    * batch_size,
                    :,
                    boundary_pix: residual_y + boundary_pix,
                    boundary_pix: patch_shape_x - boundary_pix,
                ]
                count_map[
                    :, :, y_start:, x_start: x_start + patch_shape_x - 2 * boundary_pix
                ] += one_map[
                    :,
                    :,
                    boundary_pix: residual_y + boundary_pix,
                    boundary_pix: patch_shape_x - boundary_pix,
                ]
            elif x_index == patch_num_x - 1 and y_index == patch_num_y - 1:
                output[:, :, y_start:, x_start:] += input[
                    (x_index * patch_num_y + y_index)
                    * batch_size: (x_index * patch_num_y + y_index + 1)
                    * batch_size,
                    :,
                    boundary_pix: residual_y + boundary_pix,
                    boundary_pix: residual_x + boundary_pix,
                ]
                count_map[:, :, y_start:, x_start:] += one_map[
                    :,
                    :,
                    boundary_pix: residual_y + boundary_pix,
                    boundary_pix: residual_x + boundary_pix,
                ]
            else:
                output[
                    :,
                    :,
                    y_start: y_start + patch_shape_y - 2 * boundary_pix,
                    x_start: x_start + patch_shape_x - 2 * boundary_pix,
                ] += input[
                    (x_index * patch_num_y + y_index)
                    * batch_size: (x_index * patch_num_y + y_index + 1)
                    * batch_size,
                    :,
                    boundary_pix: patch_shape_y - boundary_pix,
                    boundary_pix: patch_shape_x - boundary_pix,
                ]
                count_map[
                    :,
                    :,
                    y_start: y_start + patch_shape_y - 2 * boundary_pix,
                    x_start: x_start + patch_shape_x - 2 * boundary_pix,
                ] += one_map[
                    :,
                    :,
                    boundary_pix: patch_shape_y - boundary_pix,
                    boundary_pix: patch_shape_x - boundary_pix,
                ]
    return output / count_map
