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

from typing import Dict, Any, Tuple

from torch import Tensor
import torch
import torch.nn.functional as F

try:
    from openpoints.models import build_model_from_cfg
except ImportError:
    print(
        "OpenPoints not installed. Please run `git clone https://github.com/guochengqian/openpoints.git` to use this model."
    )

from src.networks.base_model import BaseModel
from src.networks.drivaer_base import DrivAerDragRegressionBase, DrivAerBase


class OpenPointBase(BaseModel):
    """
    OpnePointBase class
    """

    def __init__(self, **kwargs):
        BaseModel.__init__(self)
        self.openpoint_model = build_model_from_cfg(kwargs)

    def forward(self, x: Tensor):
        # Assert x is in cuda
        assert x.is_cuda
        return self.openpoint_model(x)


class OpenPointDrivAer(OpenPointBase, DrivAerDragRegressionBase):
    """
    OpenPointDrivAer class
    """

    def __init__(self, **kwargs):
        DrivAerDragRegressionBase.__init__(self)
        OpenPointBase.__init__(self, **kwargs)

    def data_dict_to_input(self, data_dict):
        return data_dict["cell_centers"].to(self.device)


class OpenPointPressureDragDrivAer(OpenPointBase, DrivAerBase):
    """
    OpenPointDrivAer class
    """

    def __init__(self, **kwargs):
        DrivAerBase.__init__(self)
        OpenPointBase.__init__(self, **kwargs)

        mlp_in_channels = kwargs.get("mlp_in_channels", 512)
        decoder_out_channels = kwargs.get("decoder_out_channels", 512)
        self.decoder_inputs = kwargs.get("decoder_inputs", "feats")
        # mlp for drag prediction
        self.drag_mlp = torch.nn.Sequential(
            torch.nn.Linear(mlp_in_channels, mlp_in_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(mlp_in_channels, 1),
        )
        self.pressure_mlp = torch.nn.Conv1d(decoder_out_channels, 1, 1)

    def data_dict_to_input(self, data_dict):
        return data_dict["cell_centers"].to(self.device)

    def forward(self, data: Tensor) -> Tuple[Tensor, Tensor]:
        assert data.is_cuda
        op = self.openpoint_model
        p, f = op.encoder.forward_seg_feat(data)
        # For the drag prediction, pool and mlp
        glob_feats = f
        if isinstance(f, list):
            glob_feats = f[-1]
        drag = self.drag_mlp(F.adaptive_avg_pool1d(glob_feats, 1).squeeze(-1))
        if hasattr(op, "decoder") and op.decoder is not None:
            if self.decoder_inputs == "feats":
                f = op.decoder(f).squeeze(-1)
            elif self.decoder_inputs == "both":
                f = op.decoder(p, f).squeeze(-1)
            else:
                raise NotImplementedError
        pressure = self.pressure_mlp(f)
        return pressure, drag


if __name__ == "__main__":
    model_cfg = {
        "NAME": "PointMLP",
        "in_channels": 3,
        "points": 1024,
        "num_classes": 1,
        "embed_dim": 64,
        "groups": 1,
        "res_expansion": 1.0,
        "activation": "relu",
        "bias": False,
        "use_xyz": False,
        "normalize": "anchor",
        "dim_expansion": [2, 2, 2, 2],
        "pre_blocks": [2, 2, 2, 2],
        "pos_blocks": [2, 2, 2, 2],
        "k_neighbors": [24, 24, 24, 24],
        "reducers": [2, 2, 2, 2],
    }
    model = OpenPointDrivAer(**model_cfg).to("cuda")
    print(model)
    B, N = 2, 2048
    x = torch.randn(B, N, 3).to("cuda")
    print(model(x).shape)
