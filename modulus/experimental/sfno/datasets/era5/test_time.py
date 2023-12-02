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

import datetime
from modulus.experimental.sfno.datasets.era5 import time


def test_datetime_range():
    times = time.datetime_range(2018, datetime.timedelta(hours=6), 2)
    assert times == [datetime.datetime(2018, 1, 1, 0), datetime.datetime(2018, 1, 1, 6)]


def test_filename_to_year():
    assert 2018 == time.filename_to_year("some/long/path/2018.h5")
