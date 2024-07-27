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

from typing import Dict, Any, Tuple

import torch
import torch.nn as nn


class BaseModule(nn.Module):
    """Base module for models."""

    def __init__(self):
        super().__init__()
        self.device_indicator_param = nn.Parameter(torch.empty(0))

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.device_indicator_param.device


class BaseModel(BaseModule):
    """Base model class."""

    def data_dict_to_input(self, data_dict, **kwargs) -> Any:
        """Convert data dictionary to appropriate input for the model."""
        raise NotImplementedError

    def loss_dict(self, data_dict, **kwargs) -> Dict:
        """Compute the loss dictionary for the model."""
        raise NotImplementedError

    @torch.no_grad()
    def eval_dict(self, data_dict, **kwargs) -> Dict:
        """Compute the evaluation dictionary for the model."""
        raise NotImplementedError

    def image_pointcloud_dict(self, data_dict, datamodule) -> Tuple[Dict, Dict]:
        """Compute the image dict and pointcloud dict for the model."""
        raise NotImplementedError
