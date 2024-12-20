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
from torch.testing import assert_close

from modulus.models.aimnet import modules
from modulus.models.aimnet.aimnet2 import AIMNet2


def get_test_data():
    """Returns fake data resembling caffeine molecule."""

    # CNCNC3ONCONC2H10
    atomic_numbers = torch.tensor(
        [6, 7, 6, 7, 6, 6, 6, 8, 7, 6, 8, 7, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ).unsqueeze(0)

    # coord = 6 * torch.rand(1, atomic_numbers.shape[0], 3) - 3
    coord = torch.tensor(
        [
            [-3.24598, -1.13535, 0.033],
            [-2.26635, -0.07913, 0.00424],
            [-2.52224, 1.26667, -0.00684],
            [-1.40901, 1.97386, -0.00991],
            [-0.41465, 1.04188, -0.0039],
            [-0.90883, -0.22681, -0.00072],
            [-0.08496, -1.38464, 0.00294],
            [-0.54245, -2.52372, 0.003],
            [1.27753, -1.07858, 0.00884],
            [1.82883, 0.21929, 0.00635],
            [3.05203, 0.38761, 0.00786],
            [0.93675, 1.29234, 0.00259],
            [1.42804, 2.65774, 0.00996],
            [2.22356, -2.17683, 0.02198],
            [-4.24963, -0.70415, 0.00614],
            [-3.11340, -1.70485, 0.9564],
            [-3.09644, -1.77493, -0.84056],
            [-3.52582, 1.67408, -0.00973],
            [2.52041, 2.69722, 0.03772],
            [1.03996, 3.17404, 0.89377],
            [1.08360, 3.16741, -0.89524],
            [1.73380, -3.15366, 0.02338],
            [2.85081, -2.09344, 0.91559],
            [2.86702, -2.10317, -0.86073],
        ]
    ).unsqueeze(0)

    return {"coord": coord, "numbers": atomic_numbers, "charge": torch.tensor([0.0])}


def create_model():
    model = AIMNet2(
        aev={
            "rc_s": 5.0,
            "nshifts_s": 16,
        },
        nfeature=16,
        d2features=True,
        ncomb_v=12,
        hidden=(
            [512, 380],
            [512, 380],
            [512, 380, 380],
        ),
        aim_size=256,
        outputs={
            "energy_mlp": modules.Output(
                n_in=256,
                n_out=1,
                key_in="aim",
                key_out="energy",
                mlp={
                    "activation_fn": torch.nn.GELU(),
                    "last_linear": True,
                    "hidden": [128, 128],
                },
            ),
            "atomic_shift": modules.AtomicShift(
                key_in="energy",
                key_out="energy",
            ),
            "atomic_sum": modules.AtomicSum(
                key_in="energy",
                key_out="energy",
            ),
            "lrcoulomb": modules.LRCoulomb(
                rc=4.6,
                key_in="charges",
                key_out="energy",
            ),
            # TODO(akamenev): need to verify the source and license of dftd3_data.pt checkpoint.
            # "dftd3": modules.DFTD3(
            #     s8=0.3908,
            #     a1=0.5660,
            #     a2=3.1280,
            #     chk_path=Path(__file__).parent / "data/dftd3_data.pt",
            # ),
        },
    )
    return model


def test_aimnet2_forward():
    torch.manual_seed(1)

    model = modules.core.Forces(create_model())
    in_data = get_test_data()
    x0 = model(in_data)

    # TODO(akamenev): update once DFTD3 is enabled.
    assert_close(x0["energy"].item(), -5.09827, atol=1e-5, rtol=1e-4)
