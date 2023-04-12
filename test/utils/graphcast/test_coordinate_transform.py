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


import torch
import pytest

from modulus.utils.graphcast.graph_utils import latlon2xyz, xyz2latlon


@pytest.mark.parametrize("latlon", [[-27.0, 48.0], [0, 0], [62.0, -45.0]])
def test_coordinate_transform(latlon):
    """Test coordinate transformation from latlon to xyz and back."""
    latlon = torch.tensor([latlon], dtype=torch.float)
    xyz = latlon2xyz(latlon)
    latlon_recovered = xyz2latlon(xyz)
    assert torch.allclose(
        latlon, latlon_recovered
    ), f"coordinate transformation failed, {latlon} != {latlon_recovered}"
