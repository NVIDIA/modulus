# ignore_header_test
# ruff: noqa: E402

# © Copyright 2023 HP Development Company, L.P.
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


import torch


def l2_dist(pts1, pts2, reduction="mean"):
    """
    L2-loss compute, mean of all points' difference
    """
    l2_per_batch = torch.mean(torch.sum(torch.pow(pts1 - pts2, 2), -1), -1)
    if reduction == "mean":
        return torch.mean(l2_per_batch)
    else:
        return l2_per_batch
