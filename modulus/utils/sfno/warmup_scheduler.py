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

from typing import List

from torch.optim import lr_scheduler as lrs


class WarmupScheduler(lrs._LRScheduler):  # pragma: no cover
    """Scheduler with linear warmup"""

    def __init__(self, scheduler, num_warmup_steps, start_lr):
        self.scheduler = scheduler
        self.num_warmup_steps = num_warmup_steps
        if not isinstance(start_lr, List):
            self.start_lrs = [start_lr]
        else:
            self.start_lrs = start_lr
        self.steps = 0

        # this is hacky but I don't see a better way of doing that
        self.end_lrs = self.scheduler.base_lrs
        for lr, group in zip(self.start_lrs, self.scheduler.optimizer.param_groups):
            group["lr"] = lr
        # self.scheduler.base_lrs = [group['initial_lr'] for group in self.scheduler.optimizer.param_groups]

        # init warmup scheduler:
        def linwarm(step, max_steps, slr, elr):
            if step <= max_steps:
                t = step / float(max_steps)
                res = t + (1 - t) * slr / elr
            else:
                res = 1.0
            return res

        self.warmup_scheduler = lrs.LambdaLR(
            self.scheduler.optimizer,
            lr_lambda=[
                lambda x: linwarm(x, self.num_warmup_steps, slr, elr)
                for slr, elr in zip(self.start_lrs, self.end_lrs)
            ],
        )
        self._last_lr = [
            group["lr"] for group in self.warmup_scheduler.optimizer.param_groups
        ]

    def step(self):  # pragma: no cover
        """Scheduler step"""
        shandle = (
            self.scheduler
            if self.steps >= self.num_warmup_steps
            else self.warmup_scheduler
        )
        shandle.step()
        self.steps += 1
        self._last_lr = [group["lr"] for group in shandle.optimizer.param_groups]

    def state_dict(self):  # pragma: no cover
        """Returns the scheduler's state dict."""
        state_dict = {
            key: value
            for key, value in self.__dict__.items()
            if key not in ("optimizer", "_schedulers")
        }
        state_dict["warmup_scheduler"] = self.warmup_scheduler.state_dict()
        state_dict["scheduler"] = self.scheduler.state_dict()

        return state_dict

    def load_state_dict(self, state_dict):  # pragma: no cover
        """Load the scheduler's state dict."""
        warmup_scheduler = state_dict.pop("warmup_scheduler")
        scheduler = state_dict.pop("scheduler")

        self.__dict__.update(state_dict)
        # Restore state_dict keys in order to prevent side effects
        # https://github.com/pytorch/pytorch/issues/32756
        state_dict["warmup_scheduler"] = warmup_scheduler
        state_dict["scheduler"] = scheduler

        self.warmup_scheduler.load_state_dict(warmup_scheduler)
        self.scheduler.load_state_dict(scheduler)
