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

import os
from functools import wraps

import pytest


def import_or_fail(module_names):
    def decorator(test_func):
        @pytest.mark.usefixtures("pytestconfig")
        @wraps(test_func)
        def wrapper(*args, **kwargs):
            pytestconfig = kwargs.get("pytestconfig")
            if pytestconfig is None:
                raise ValueError(
                    "pytestconfig must be passed as an argument when using the import_or_fail_decorator."
                )
            _import_or_fail(module_names, pytestconfig)

            return test_func(*args, **kwargs)

        return wrapper

    return decorator


def _import_or_fail(module_names, config):
    if not isinstance(module_names, (list, tuple)):
        module_names = [module_names]  # allow single names

    for module_name in module_names:
        if config.getoption("--fail-on-missing-modules"):
            __import__(module_name)
        else:
            pytest.importorskip(module_name)


def nfsdata_or_fail(test_func):
    @pytest.mark.usefixtures("pytestconfig")
    @wraps(test_func)
    def wrapper(*args, **kwargs):
        pytestconfig = kwargs.get("pytestconfig")
        if pytestconfig is None:
            raise ValueError(
                "pytestconfig must be passed as an argument when using the nfsdata_required_decorator."
            )
        _nfsdata_or_fail(pytestconfig)
        return test_func(*args, **kwargs)

    return wrapper


def _nfsdata_or_fail(config):
    if not os.path.exists("/data/nfs/modulus-data"):
        pytest.skip(
            "NFS volumes not set up with CI data repo. Run `make get-data` from the root directory of the repo"
        )
