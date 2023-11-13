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

import numpy as np


def save_time_mean(params, path, nx, ny, channels):
    out_path = path / "time_mean.npy"
    stat = np.ones([1, channels, nx, ny])
    np.save(out_path.as_posix(), stat)
    params.time_means_path = out_path.as_posix()


def save_stats(params, path, channels):
    stat = np.ones([1, channels, 1, 1])

    mean_path = path / "mean.npy"
    std_path = path / "scale.npy"

    np.save(std_path.as_posix(), stat)
    np.save(mean_path.as_posix(), stat)

    params.global_means_path = mean_path.as_posix()
    params.global_stds_path = std_path.as_posix()
    params.min_path = mean_path.as_posix()
    params.max_path = mean_path.as_posix()

