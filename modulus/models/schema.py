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

from typing import List, Optional
import pydantic
from enum import Enum


class Grid(Enum):
    grid_721x1440 = "721x1440"
    grid_720x1440 = "720x1440"


# Enum of channels
class ChannelSet(Enum):
    """An Enum of standard sets of channels

    These correspond to the post-processed outputs in .h5 files like this:

        73var: /lustre/fsw/sw_climate_fno/test_datasets/73var-6hourly
        34var: /lustre/fsw/sw_climate_fno/34Vars

    This concept is needed to map from integer channel numbers (e.g. [0, 1, 2]
    to physical variables).

    """

    var34 = "34var"
    var73 = "73var"


class Model(pydantic.BaseModel):
    """Metadata for using a ERA5 time-stepper model"""

    n_history: int
    channel_set: ChannelSet
    grid: Grid
    in_channels: List[int]
    out_channels: List[int]
    architecture: str = ""
    architecture_entrypoint: str = ""
