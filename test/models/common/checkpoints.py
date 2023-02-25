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
import modulus
import torch

from typing import Tuple
from pathlib import Path
from .utils import compare_output

Tensor = torch.Tensor
logger = logging.getLogger("__name__")


@torch.no_grad()
def validate_checkpoint(
    model_1: modulus.Module,
    model_2: modulus.Module,
    in_args: Tuple[Tensor] = (),
    rtol: float = 1e-5,
    atol: float = 1e-5,
) -> bool:
    """Check network's checkpoint safely saves and loads the state of the model

    This test will check if a model's state is fully saved in its checkpoint. Two
    seperately initialized models should be provided. One model will load the other's
    checkpoint and produce the same output.

    Parameters
    ----------
    model_1 : modulus.Module
        Modulus model to save checkpoint from
    model_2 : modulus.Module
        Modulus model to load checkpoint to
    in_args : Tuple[Tensor], optional
        Input arguments, by default ()
    rtol : float, optional
        Relative tolerance of error allowed, by default 1e-5
    atol : float, optional
        Absolute tolerance of error allowed, by default 1e-5

    Returns
    -------
    bool
        Test passed
    """
    # First check fail safes of save/load functions
    try:
        model_1.save("folder_does_not_exist/checkpoint.pth")
    except IOError:
        pass

    try:
        model_1.load("does_not_exist.pth")
    except IOError:
        pass

    # Now test forward passes
    output_1 = model_1.forward(*in_args)
    output_2 = model_2.forward(*in_args)

    # Model outputs should initially be different
    assert not compare_output(
        output_1, output_2, rtol, atol
    ), "Model outputs should initially be different"

    # Safe checkpoint from model 1 and load it into model 2
    model_1.save("checkpoint.pth")
    model_2.load("checkpoint.pth")
    # Forward with loaded checkpoint
    output_2 = model_2.forward(*in_args)
    # Delete checkpoint file (it should exist!)
    Path("checkpoint.pth").unlink(missing_ok=False)

    return compare_output(output_1, output_2, rtol, atol)
