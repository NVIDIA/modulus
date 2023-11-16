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

import torch.distributed as dist
from modulus.experimental.sfno.third_party.torch.distributed import utils

def dist_patch():
    """Monkey patching torch.distributed."""
    dist.utils._pack_kwargs = utils._pack_kwargs
    dist.utils._cast_forward_inputs = utils._cast_forward_inputs
    dist.utils._unpack_kwargs = utils._unpack_kwargs
    dist.utils._recursive_to = utils._recursive_to
    dist.utils._p_assert = utils._p_assert
    dist.utils._alloc_storage = utils._alloc_storage
    dist.utils._free_storage = utils._free_storage
    dist.utils._apply_to_tensors = utils._apply_to_tensors
    dist.utils._to_kwargs = utils._to_kwargs
    dist.utils._verify_param_shape_across_processes = utils._verify_param_shape_across_processes
    dist.utils._sync_module_states = utils._sync_module_states
    dist.utils._sync_params_and_buffers = utils._sync_params_and_buffers
    dist.utils._replace_by_prefix = utils._replace_by_prefix
    return dist
