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

import logging
from typing import Tuple, Union

import torch

Tensor = torch.Tensor
logger = logging.getLogger("__name__")


def dummy_loss_fn(data: Union[Tensor, Tuple[Tensor, ...]]):
    """Trivial summation loss for testing"""
    # Output of tensor
    if isinstance(data, torch.Tensor):
        loss = data.sum()
    # Output of tuple of tensors
    elif isinstance(data, tuple):
        # Loop through tuple of outputs
        loss = 0
        for data_tensor in data:
            # If tensor use allclose
            if isinstance(data_tensor, Tensor):
                loss = data_tensor.sum()
    else:
        logger.error(
            "Model returned invalid type for unit test, should be Tensor or Tuple[Tensor]"
        )
        loss = None
    return loss


def compare_output(
    output_1: Union[Tensor, Tuple[Tensor, ...]],
    output_2: Union[Tensor, Tuple[Tensor, ...]],
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> bool:
    """Compares model outputs and returns if they are the same

    Parameters
    ----------
    output_1 : Union[Tensor, Tuple[Tensor, ...]]
        Output one
    output_2 : Union[Tensor, Tuple[Tensor, ...]]
        Output two
    rtol : float, optional
        Relative tolerance of error allowed, by default 1e-5
    atol : float, optional
        Absolute tolerance of error allowed, by default 1e-5

    Returns
    -------
    bool
        If outputs are the same
    """
    # Output of tensor
    if isinstance(output_1, Tensor):
        return torch.allclose(output_1, output_2, rtol, atol)
    # Output of tuple of tensors
    elif isinstance(output_1, tuple):
        # Loop through tuple of outputs
        for i, (out_1, out_2) in enumerate(zip(output_1, output_2)):
            # If tensor use allclose
            if isinstance(out_1, Tensor):
                if not torch.allclose(out_1, out_2, rtol, atol):
                    logger.warning(f"Failed comparison between outputs {i}")
                    logger.warning(
                        f"Max Difference: {torch.amax(torch.abs(out_1 - out_2))}"
                    )
                    logger.warning(f"Difference: {out_1 - out_2}")
                    return False
            # Otherwise assume primative
            else:
                if not out_1 == out_2:
                    return False
    # Unsupported output type
    else:
        logger.error(
            "Model returned invalid type for unit test, should be Tensor or Tuple[Tensor]"
        )
        return False

    return True


def is_fusion_available(cls_name: str):
    """Check if certain APIs are available in nvfuser package."""

    try:
        return hasattr(__import__("nvfuser", fromlist=[cls_name]), cls_name)
    except ImportError:
        return False
