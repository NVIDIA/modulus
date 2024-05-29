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
import numpy as np


def prepare_input(invar, cos_zenith=None, num_history=0, static_data=None, step=None):
    """Prepare input by adding history, cos zenith angle, and static data, if applicable"""
    # Add history
    if num_history > 0:
        # flatten the history dimension
        invar = invar.view(invar.size(0), -1, *(invar.size()[3:]))
    # Add cos zenith
    if cos_zenith is not None:
        cos_zenith = torch.squeeze(cos_zenith, dim=2)
        cos_zenith = torch.clamp(cos_zenith, min=0.0) - 1.0 / np.pi
        invar = torch.concat(
            (invar, cos_zenith[:, step - 1 : num_history + step, ...]), dim=1
        )  # TODO check
    # Add static data
    if static_data is not None:
        invar = torch.concat((invar, static_data), dim=1)
    return invar


def count_trainable_params(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in a model.

    Args:
        model (torch.nn.Module): Model to count parameters of.

    Returns:
        int: Number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
