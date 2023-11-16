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

# coding: utf-8
import training.YParams
import training.dataset
from training.time import convert_datetime_to_cftime
import datetime
import yaml
import random
import sys

p = training.YParams.YParams("era5-cwb-v3.yaml", "validation_small")
ds = training.dataset.get_zarr_dataset(p, train=False)
times = ds.time()
random.shuffle(times)
subset = times[:200]
subset = sorted([convert_datetime_to_cftime(t, cls=datetime.datetime) for t in subset])
yaml.safe_dump({"validation_big": {"times": subset}}, sys.stdout)
