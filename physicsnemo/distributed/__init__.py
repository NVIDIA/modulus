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


from ._shard_tensor_spec import ShardTensorSpec
from .autograd import all_gather_v, gather_v, indexed_all_to_all_v, scatter_v
from .config import ProcessGroupConfig, ProcessGroupNode

# Load and register custom ops:
from .manager import (
    DistributedManager,
    PhysicsNeMoUndefinedGroupError,
    PhysicsNeMoUninitializedDistributedManagerWarning,
)
from .shard_tensor import ShardTensor, scatter_tensor
from .utils import (
    mark_module_as_shared,
    reduce_loss,
    unmark_module_as_shared,
)


def register_custom_ops():
    # These imports will register the custom ops with the ShardTensor class.
    # It's done here to avoid an import cycle.
    from .custom_ops import (
        sharded_mean_wrapper,
        unbind_rules,
    )
    from .shard_utils import register_shard_wrappers

    register_shard_wrappers()


# Custom ops are not ready to be enabled yet:
# register_custom_ops()
