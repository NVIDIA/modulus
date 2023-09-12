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

from modulus.distributed.config import ProcessGroupNode, ProcessGroupConfig


def test_config():
    # Create model parallel group with data parallel as the orthogonal group
    mp = ProcessGroupNode("model_parallel", orthogonal_group="data_parallel")

    # Create the process group config with the highest level process group
    pg_config = ProcessGroupConfig(mp)

    # Create spatial and channel parallel sub-groups
    pg_config.add_node(ProcessGroupNode("spatial_parallel"), parent=mp)
    pg_config.add_node(ProcessGroupNode("channel_parallel"), parent="model_parallel")

    # Now check that the leaf nodes are correct
    assert sorted(pg_config.leaf_groups()) == ["channel_parallel", "spatial_parallel"]

    # Set leaf group sizes
    group_sizes = {"channel_parallel": 3, "spatial_parallel": 2}
    pg_config.set_leaf_group_sizes(group_sizes)  # Updates all parent group sizes too

    assert (
        pg_config.get_node("model_parallel").size == 6
    ), "Incorrect size for parent node"
