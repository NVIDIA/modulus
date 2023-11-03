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

from dataclasses import dataclass

import torch

from modulus.models.meta import ModelMetaData
from modulus.models.module import Module


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


class UNet(Module):
    """
    U-Net architecture.

    Parameters:
    -----------
    img_resolution : int
        The resolution of the input/output image.
    img_channels : int
         Number of color channels.
    img_in_channels : int
        Number of input color channels.
    img_out_channels : int
        Number of output color channels.
    label_dim : int, optional
        Number of class labels; 0 indicates an unconditional model. By default 0.
    use_fp16: bool, optional
        Execute the underlying model at FP16 precision?, by default False.
    sigma_min: float, optional
        Minimum supported noise level, by default 0.
    sigma_max: float, optional
        Maximum supported noise level, by default float('inf').
    sigma_data: float, optional
        Expected standard deviation of the training data, by default 0.5.
    model_type: str, optional
        Class name of the underlying model, by default 'DhariwalUNet'.
    **model_kwargs : dict
        Keyword arguments for the underlying model.


    Note:
    -----
    Reference: Mardani, M., Brenowitz, N., Cohen, Y., Pathak, J., Chen, C.Y.,
    Liu, C.C.,Vahdat, A., Kashinath, K., Kautz, J. and Pritchard, M., 2023.
    Generative Residual Diffusion Modeling for Km-scale Atmospheric Downscaling.
    arXiv preprint arXiv:2309.15214.
    """

    def __init__(
        self,
        img_resolution,
        img_channels,
        img_in_channels,
        img_out_channels,
        label_dim=0,
        use_fp16=False,
        sigma_min=0,
        sigma_max=float("inf"),
        sigma_data=0.5,
        model_type="DhariwalUNet",
        **model_kwargs,
    ):
        super().__init__()
        self.img_resolution = img_resolution
        self.img_channels = img_channels

        self.img_in_channels = img_in_channels
        self.img_out_channels = img_out_channels

        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.model = globals()[model_type](
            img_resolution=img_resolution,
            in_channels=img_in_channels + img_out_channels,
            out_channels=img_out_channels,
            label_dim=label_dim,
            **model_kwargs,
        )

    def forward(
        self, x, img_lr, sigma, class_labels=None, force_fp32=False, **model_kwargs
    ):

        # SR: concatenate input channels
        if img_lr is None:
            x = x
        else:
            x = torch.cat((x, img_lr), dim=1)

        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1, 1, 1)
        class_labels = (
            None
            if self.label_dim == 0
            else torch.zeros([1, self.label_dim], device=x.device)
            if class_labels is None
            else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        )
        dtype = (
            torch.float16
            if (self.use_fp16 and not force_fp32 and x.device.type == "cuda")
            else torch.float32
        )

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_skip = 0.0 * c_skip
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_out = torch.ones_like(c_out)
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_in = torch.ones_like(c_in)
        c_noise = sigma.log() / 4
        c_noise = 0.0 * c_noise

        F_x = self.model(
            (c_in * x).to(dtype),
            c_noise.flatten(),
            class_labels=class_labels,
            **model_kwargs,
        )

        if F_x.dtype != dtype:
            raise ValueError(
                f"Expected the dtype to be {dtype}, but got {F_x.dtype} instead."
            )

        # skip connection - for SR there's size mismatch bwtween input and output
        x = x[:, 0 : self.img_out_channels, :, :]
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
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
