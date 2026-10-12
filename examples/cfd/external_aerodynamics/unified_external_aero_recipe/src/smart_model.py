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

"""Single-field SMART adapter for the unified recipe's tensor contract."""

from __future__ import annotations

import torch
from torch import Tensor

from physicsnemo.experimental.models.smart import SMART


class SMARTField(SMART):
    """Expose one SMART head as a tensor, with bounded evaluation query chunks.

    Geometry remains a separate full CFD boundary cloud. Only the initial
    latents and per-layer subsamples enter the SMART encoder. Query features
    use SMART's existing additive projections. The unified recipe provides
    normalization, losses, optimizer, metrics, and physical-unit export.

    Args:
        field: Enabled head, either ``surface`` or ``volume``.
        out_dim: Number of target channels in dataset declaration order.
        query_feature_channels: Number of normal or SDF/direction channels.
        eval_chunk_size: Maximum queries decoded at once during evaluation.
        **kwargs: Remaining SMART architecture arguments. Head channel counts
            are set by this adapter; parameter_channels defaults to three for
            the recipe's freestream velocity vector.
    """

    def __init__(
        self,
        field: str,
        out_dim: int,
        query_feature_channels: int,
        eval_chunk_size: int = 8192,
        **kwargs,
    ):
        if field not in ("surface", "volume"):
            raise ValueError("field must be surface or volume")
        if (
            isinstance(eval_chunk_size, bool)
            or not isinstance(eval_chunk_size, int)
            or eval_chunk_size <= 0
        ):
            raise ValueError("eval_chunk_size must be a positive integer")
        kwargs.setdefault("parameter_channels", 3)
        super().__init__(
            surface_channels=out_dim if field == "surface" else 0,
            volume_channels=out_dim if field == "volume" else 0,
            surface_query_feature_channels=query_feature_channels
            if field == "surface"
            else 0,
            volume_query_feature_channels=query_feature_channels
            if field == "volume"
            else 0,
            **kwargs,
        )
        self.field = field
        self.eval_chunk_size = eval_chunk_size

    def forward(
        self,
        geometry: Tensor,
        queries: Tensor,
        query_features: Tensor,
        parameters: Tensor,
        geometry_seed: Tensor,
    ) -> Tensor:
        """Predict ``(1, N, out_dim)`` with aligned ``(1, N, C)`` features.

        The collate adds a token dimension to global inputs: freestream
        parameters have shape ``(1, 1, 3)`` and the seed ``(1, 1, 1)``.
        A reader-derived case/epoch seed isolates geometry draws from query
        sampling, augmentation and global model RNG. Evaluation instead uses
        the core model's fixed eval_seed, including across query chunks.
        """
        if geometry.shape[0] != 1 or geometry_seed.numel() != 1:
            raise ValueError("The unified recipe requires one case per batch")
        if (
            queries.ndim != 3
            or query_features.ndim != 3
            or query_features.shape[:2] != queries.shape[:2]
        ):
            raise ValueError(
                "Query coordinates and features must have aligned (B, N, C) shapes"
            )
        parameters = parameters.reshape(1, -1) if self.parameter_channels else None
        generator = None
        if self.training:
            generator = torch.Generator(device=geometry.device).manual_seed(
                int(geometry_seed.item())
            )
        contexts = self.encode_geometry(geometry, parameters, generator=generator)
        if self.training or queries.shape[1] == 0:
            return self.decode_queries(
                queries,
                contexts,
                parameters,
                field=self.field,
                query_features=query_features,
            )
        return torch.cat(
            [
                self.decode_queries(
                    queries[:, start : start + self.eval_chunk_size],
                    contexts,
                    parameters,
                    field=self.field,
                    query_features=query_features[
                        :, start : start + self.eval_chunk_size
                    ],
                )
                for start in range(0, queries.shape[1], self.eval_chunk_size)
            ],
            dim=1,
        )
