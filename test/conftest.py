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

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--multigpu", action="store_true", default=False, help="run multigpu tests"
    )
    parser.addoption(
        "--fail-on-missing-modules",
        action="store_true",
        default=False,
        help="fail tests if required modules are missing",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "multigpu: mark test as multigpu to run")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--multigpu") and not config.getoption("-m"):
        skip_multigpu = pytest.mark.skip(reason="need --multigpu option to run")
        for item in items:
            if "multigpu" in item.keywords:
                item.add_marker(skip_multigpu)
