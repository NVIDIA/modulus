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

import datetime

import cftime


def convert_datetime_to_cftime(
    time: datetime.datetime, cls=cftime.DatetimeGregorian
) -> cftime.DatetimeGregorian:
    """Convert a Python datetime object to a cftime DatetimeGregorian object."""
    return cls(time.year, time.month, time.day, time.hour, time.minute, time.second)


def time_range(
    start_time: datetime.datetime,
    end_time: datetime.datetime,
    step: datetime.timedelta,
    inclusive: bool = False,
):
    """Like the Python `range` iterator, but with datetimes."""
    t = start_time
    while (t <= end_time) if inclusive else (t < end_time):
        yield t
        t += step
