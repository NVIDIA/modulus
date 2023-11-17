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

from modulus.distributed import ProcessGroupConfig, ProcessGroupNode


def test_config():
    # Create world group that contains all processes that are part of this job
    world = ProcessGroupNode("world")

    # Create the process group config with the highest level process group
    config = ProcessGroupConfig(world)

    # Create model and data parallel sub-groups
    # Sub-groups of a single node are guaranteed to be orthogonal by construction
    config.add_node(ProcessGroupNode("model_parallel"), parent=world)
    config.add_node(ProcessGroupNode("data_parallel"), parent="world")

    # Create spatial and channel parallel sub-groups
    config.add_node(ProcessGroupNode("spatial_parallel"), parent="model_parallel")
    config.add_node(ProcessGroupNode("channel_parallel"), parent="model_parallel")

    # Now check that the leaf nodes are correct
    assert sorted(config.leaf_groups()) == [
        "channel_parallel",
        "data_parallel",
        "spatial_parallel",
    ]

    # Set leaf group sizes
    group_sizes = {"channel_parallel": 3, "spatial_parallel": 2, "data_parallel": 4}
    config.set_leaf_group_sizes(group_sizes)  # Update all parent group sizes too

    assert (
        config.get_node("model_parallel").size == 6
    ), "Incorrect size for 'model_parallel' parent node"

    assert config.get_node("world").size == 24, "Incorrect size for 'world' parent node"
