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

import torch
from torch import nn
from torch.optim import lr_scheduler as lrs

from modulus.utils.sfno.warmup_scheduler import WarmupScheduler


def test_warmup_scheduler():
    """test warmup scheduler"""

    param = nn.Parameter(torch.zeros((10), dtype=torch.float))
    opt = torch.optim.Adam([param], lr=0.5)

    start_lr = 0.01
    num_warmup = 10
    num_steps = 20

    main_scheduler = lrs.CosineAnnealingLR(opt, num_steps, eta_min=0)
    scheduler = WarmupScheduler(main_scheduler, num_warmup, start_lr)

    for epoch in range(num_steps + num_warmup):
        scheduler.step()

    sd = scheduler.state_dict()
    scheduler.load_state_dict(sd)
    assert torch.allclose(torch.tensor(scheduler.get_last_lr()[0]), torch.tensor(0.0))
