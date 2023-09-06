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


import pytest


def import_or_fail(module_names, config):
    if not isinstance(module_names, (list, tuple)):
        module_names = [module_names]  # allow single names

    for module_name in module_names:
        if config.getoption("--fail-on-missing-modules"):
            __import__(module_name)
        else:
            pytest.importorskip(module_name)
