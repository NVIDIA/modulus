# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
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

"""Optimizer partitioning used by the GeoTransolver comparison recipe."""

import torch

from physicsnemo.optim import CombinedOptimizer


def muon_adamw(params, lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-8):
    """Apply Muon to matrices and AdamW to other parameters, as in GeoTransolver."""
    params = list(params)
    return CombinedOptimizer(
        [
            torch.optim.Muon(
                [p for p in params if p.ndim == 2],
                lr=lr,
                weight_decay=weight_decay,
                adjust_lr_fn="match_rms_adamw",
            ),
            torch.optim.AdamW(
                [p for p in params if p.ndim != 2],
                lr=lr,
                weight_decay=weight_decay,
                betas=betas,
                eps=eps,
            ),
        ]
    )
