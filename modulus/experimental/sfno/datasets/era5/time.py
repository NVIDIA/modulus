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

from typing import List
import os
import datetime


def filename_to_year(path: str) -> int:
    filename = os.path.basename(path)
    return int(filename[:4])


def datetime_range(
    year: int, time_step: datetime.timedelta, n: int
) -> List[datetime.datetime]:
    initial_time = datetime.datetime(year=year, month=1, day=1)
    return [initial_time + time_step * i for i in range(n)]
