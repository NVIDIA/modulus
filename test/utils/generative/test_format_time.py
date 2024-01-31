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


from modulus.utils.generative import format_time, format_time_brief


# Test format_time function
def test_format_time():
    assert format_time(59) == "59s"
    assert format_time(60) == "1m 00s"
    assert format_time(3599) == "59m 59s"
    assert format_time(3600) == "1h 00m 00s"
    assert format_time(86399) == "23h 59m 59s"
    assert format_time(86400) == "1d 00h 00m"
    assert format_time(90061) == "1d 01h 01m"
    assert format_time(100000) == "1d 03h 46m"


# Test format_time_brief function
def test_format_time_brief():
    assert format_time_brief(59) == "59s"
    assert format_time_brief(60) == "1m 00s"
    assert format_time_brief(3600) == "1h 00m"
    assert format_time_brief(86399) == "23h 59m"
    assert format_time_brief(86400) == "1d 00h"
    assert format_time_brief(86459) == "1d 00h"
    assert format_time_brief(90061) == "1d 01h"
    assert format_time_brief(100000) == "1d 03h"
