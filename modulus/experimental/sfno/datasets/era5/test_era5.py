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

import pathlib
import h5py

from modulus.experimental.sfno.datasets import era5


def test_open_34_vars(tmp_path: pathlib.Path):
    path = tmp_path / "1979.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("fields", shape=[1, 34, 721, 1440])

    ds = era5.open_34_vars(path)
    # ensure that data can be grabbed
    ds.mean().compute()

    assert set(ds.coords) == {"time", "channel", "lat", "lon"}
