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

import pkg_resources

# This model registry follows conventions similar to fsspec,
# https://github.com/fsspec/filesystem_spec/blob/master/fsspec/registry.py#L62C2-L62C2
# Tutorial on entrypoints: https://amir.rachum.com/blog/2017/07/28/python-entry-points/
def _construct_registry():
    """
    This function constructs the registry of all the models that are available.
    It does so by looking at the entrypoints in the setup.py file and any other
    entrypoints that are added by external packages.

    Note: This function is called only once when the modulus package is imported
    for the first time.

    Example:
    In 

    Returns:

    """

    model_registry = {}
    group = "modulus.models"
    entrypoints = pkg_resources.iter_entry_points(group)
    for entry_point in entrypoints:
        model_registry[entry_point.name] = entry_point
    return model_registry

_model_registry = _construct_registry()
