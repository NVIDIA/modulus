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

import importlib
from dataclasses import dataclass
from typing import Union, Tuple, Literal

import torch

from modulus.models.meta import ModelMetaData
from modulus.models.module import Module

network_module = importlib.import_module("modulus.models.diffusion")


@dataclass
class MetaData(ModelMetaData):
    name: str = "UNet"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
    amp_cpu: bool = False
    amp_gpu: bool = True
    torch_fx: bool = False
    # Data type
    bf16: bool = True
    # Inference
    onnx: bool = False
    # Physics informed
    func_torch: bool = False
    auto_grad: bool = False


class UNet(Module):  # TODO a lot of redundancy, need to clean up
    """
    U-Net Wrapper for CorrDiff deterministic regression model.

    Parameters
    -----------
    img_resolution : Union[int, Tuple[int, int]]
        The resolution of the input/output image. If a single int is provided,
        then the image is assumed to be square.
    img_channels : int
         Number of color channels.
    img_in_channels : int
        Number of input color channels.
    img_out_channels : int
        Number of output color channels.
    use_fp16: bool, optional
        Execute the underlying model at FP16 precision, by default False.
    sigma_min: float, optional
        Minimum supported noise level, by default 0.
    sigma_max: float, optional
        Maximum supported noise level, by default float('inf').
    sigma_data: float, optional
        Expected standard deviation of the training data, by default 0.5.
    model_type: str, optional
        Class name of the underlying model. Must be one of the following:
        'SongUNet', 'SongUNetPosEmbd', 'SongUNetPosLtEmbd', 'DhariwalUNet'.
        Defaults to 'SongUNetPosEmbd'.
    **model_kwargs : dict
        Keyword arguments to create the underlying model.

    Examples
    ---------
    The model forward pass can be called with:
    >>> `model(x, img_lr, force_fp32=False, **forward_kwargs)`
    where:
        x, img_lr: torch.Tensor
            Passed as positional arguments to the underlying model. Refer
            to the documentation for the class specified by `model_type`.
        force_fp32: bool, optional
            If True, force conversion of inputs to torch.float32. Defaults to
            `False`.
        **forward_kwargs: dict, optional
            Additional keyword arguments to be passed to the underlying
            model's forward pass.

    See Also
    --------
    For possible `model_type` and their accepted `model_kwargs` and
    `forward_kwargs`, see fo example:
    :class:`~modulus.models.diffusion.SongUNetPosEmbd` or
    :class:`modulus.models.diffusion.SongUNetPosLtEmbd`.

    Reference
    ----------
    Reference: Mardani, M., Brenowitz, N., Cohen, Y., Pathak, J., Chen, C.Y.,
    Liu, C.C.,Vahdat, A., Kashinath, K., Kautz, J. and Pritchard, M., 2023.
    Generative Residual Diffusion Modeling for Km-scale Atmospheric Downscaling.
    arXiv preprint arXiv:2309.15214.
    """

    def __init__(
        self,
        img_resolution: Union[int, Tuple[int, int]],
        img_channels: int,
        img_in_channels: int,
        img_out_channels: int,
        use_fp16: bool = False,
        sigma_min: float = 0,
        sigma_max: float = float("inf"),
        sigma_data: float = 0.5,
        model_type: Literal[
            "SongUNetPosEmbd", "SongUNetPosLtEmbd",
            "SongUNet", "DhariwalUNet"] = "SongUNetPosEmbd",
        **model_kwargs: dict,
    ):
        super().__init__(meta=MetaData)

        self.img_channels = img_channels

        # for compatibility with older versions that took only 1 dimension
        if isinstance(img_resolution, int):
            self.img_shape_x = self.img_shape_y = img_resolution
        else:
            self.img_shape_x = img_resolution[0]
            self.img_shape_y = img_resolution[1]

        self.img_in_channels = img_in_channels
        self.img_out_channels = img_out_channels

        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        model_class = getattr(network_module, model_type)
        self.model = model_class(
            img_resolution=img_resolution,
            in_channels=img_in_channels + img_out_channels,
            out_channels=img_out_channels,
            **model_kwargs,
        )

    def forward(self, x, img_lr, force_fp32=False, **model_kwargs):
        # SR: concatenate input channels
        if img_lr is not None:
            x = torch.cat((x, img_lr), dim=1)

        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )

        F_x = self.model(
            x.to(dtype),  # (c_in * x).to(dtype),
            torch.zeros(x.shape[0], dtype=dtype, device=x.device),  # c_noise.flatten()
            class_labels=None,
            **model_kwargs,
        )

        if (F_x.dtype != dtype) and not torch.is_autocast_enabled():
            raise ValueError(
                f"Expected the dtype to be {dtype}, " f"but got {F_x.dtype} instead."
            )

        # skip connection - for SR there's size mismatch bewtween input and
        # output
        D_x = F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        """
        Convert a given sigma value(s) to a tensor representation.

        Parameters
        ----------
        sigma : Union[float list, torch.Tensor]
            The sigma value(s) to convert.

        Returns
        -------
        torch.Tensor
            The tensor representation of the provided sigma value(s).
        """
        return torch.as_tensor(sigma)


class StormCastUNet(Module):
    """
    U-Net wrapper for StormCast; used so the same Song U-Net network can be re-used for this model.

    Parameters
    -----------
    img_resolution : int or List[int]
        The resolution of the input/output image.
    img_channels : int
         Number of color channels.
    img_in_channels : int
        Number of input color channels.
    img_out_channels : int
        Number of output color channels.
    use_fp16: bool, optional
        Execute the underlying model at FP16 precision?, by default False.
    sigma_min: float, optional
        Minimum supported noise level, by default 0.
    sigma_max: float, optional
        Maximum supported noise level, by default float('inf').
    sigma_data: float, optional
        Expected standard deviation of the training data, by default 0.5.
    model_type: str, optional
        Class name of the underlying model, by default 'SongUNet'.
    **model_kwargs : dict
        Keyword arguments for the underlying model.

    """

    def __init__(
        self,
        img_resolution,
        img_in_channels,
        img_out_channels,
        use_fp16=False,
        sigma_min=0,
        sigma_max=float("inf"),
        sigma_data=0.5,
        model_type="SongUNet",
        **model_kwargs,
    ):
        super().__init__(meta=MetaData("StormCastUNet"))

        if isinstance(img_resolution, int):
            self.img_shape_x = self.img_shape_y = img_resolution
        else:
            self.img_shape_x = img_resolution[0]
            self.img_shape_y = img_resolution[1]

        self.img_in_channels = img_in_channels
        self.img_out_channels = img_out_channels

        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        model_class = getattr(network_module, model_type)
        self.model = model_class(
            img_resolution=img_resolution,
            in_channels=img_in_channels,
            out_channels=img_out_channels,
            **model_kwargs,
        )

    def forward(self, x, force_fp32=False, **model_kwargs):
        """Run a forward pass of the StormCast regression U-Net.

        Args:
            x (torch.Tensor): input to the U-Net
            force_fp32 (bool, optional): force casting to fp_32 if True. Defaults to False.

        Raises:
            ValueError: If input data type is a mismatch with provided options

        Returns:
            D_x (torch.Tensor): Output (prediction) of the U-Net
        """

        x = x.to(torch.float32)
        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )

        F_x = self.model(
            x.to(dtype),
            torch.zeros(x.shape[0], dtype=x.dtype, device=x.device),
            class_labels=None,
            **model_kwargs,
        )

        if (F_x.dtype != dtype) and not torch.is_autocast_enabled():
            raise ValueError(
                f"Expected the dtype to be {dtype}, but got {F_x.dtype} instead."
            )

        D_x = F_x.to(torch.float32)
        return D_x
