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

import yaml
import datetime
import cftime
from training.time import convert_datetime_to_cftime

def test_datetime_yaml():
    dt = datetime.datetime(2011, 1, 1)
    s = dt.isoformat()
    loaded = yaml.safe_load(s)
    assert dt == loaded


def test_convert_to_cftime():
    dt = datetime.datetime(2011, 1, 1)
    expected = cftime.DatetimeGregorian(2011, 1, 1)
    assert convert_datetime_to_cftime(dt) == expected
