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

from modulus.utils.corrdiff import get_time_from_range


def test_default_interval():
    times_range = ["2024-01-01T00:00:00", "2024-01-01T01:00:00"]
    expected = ["2024-01-01T00:00:00", "2024-01-01T01:00:00"]
    result = get_time_from_range(times_range)
    assert result == expected


def test_hourly_interval():
    times_range = ["2024-01-01T00:00:00", "2024-01-01T03:00:00", 1]
    expected = [
        "2024-01-01T00:00:00",
        "2024-01-01T01:00:00",
        "2024-01-01T02:00:00",
        "2024-01-01T03:00:00",
    ]
    result = get_time_from_range(times_range)
    assert result == expected


def test_custom_interval():
    times_range = ["2024-01-01T00:00:00", "2024-01-01T03:00:00", 2]
    expected = ["2024-01-01T00:00:00", "2024-01-01T02:00:00"]
    result = get_time_from_range(times_range)
    assert result == expected


def test_no_interval_provided():
    times_range = ["2024-01-01T00:00:00", "2024-01-01T02:00:00"]
    expected = ["2024-01-01T00:00:00", "2024-01-01T01:00:00", "2024-01-01T02:00:00"]
    result = get_time_from_range(times_range)
    assert result == expected


def test_same_start_end_time():
    times_range = ["2024-01-01T00:00:00", "2024-01-01T00:00:00"]
    expected = ["2024-01-01T00:00:00"]
    result = get_time_from_range(times_range)
    assert result == expected
