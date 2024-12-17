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


from dataclasses import dataclass

from contextlib import nullcontext, ContextDecorator

from torch.profiler import ProfilerActivity

class Profiler(ContextDecorator):
    """
    Profiler Class to enable easy, simple to configure profiling tools in modulus.
    
    This is meant to be used as a reusable context manager for profile capturing.
    Integrate it into StreamCapture for simplest use, after configuration.  But, you
    don't have to integrate it: This profiler tool could be used to capture profiles from code or
    functions in other contexts.
    
    It is not a singleton class, so you could make more than one, but it IS meant to be
    reused in a StreamCapture - one per capture.  When used this way, it will track entrances/exits
    from the profiling context and decorate output traces accordingly.
    
    In other words, you could capture the training and validation loops separately, but the training loop
    will use the same profiler each iteration.
    
    It's suggested that you do not profile every time, since that adds some overhead.  Instead, use a shorter
    run for profiling and then disable the profiler entirely. (pass `enabled=False` in the constructor, which is default.)
    """
    def __init__(self,
        enabled: bool = False,
        torch_profile: bool = False,
        torch_output: str = "./profiler_output/",
        torch_profiler_activies: Tuple[int] = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
        include_nvtx: bool = False,
    ):
        
        if not enabled:
            # Do nothing ... 
            return
        
        # Configure line_profiler here:
        
        # Configure pytorch profiler here:
        
        # If the model is distributed, control the location of the output:
        if DistributedManager().distributed:
            self.output_location.append(f"/rank_{DistributedManager().rank}")

        # Configure nvtx context tagging here:

    @property
    def enabled(self):
        """
        Return true if profiling is enabled
        """
        return self._enabled

    def __enter__():
        """
        Enter profiling contexts 
        """
        
        # Activate pytorch profiling context
        # Set nvtx context based on name
        # Activate the line_profiler for use as a context
        
        # Capture each context in an exit stack that we'll back out of in the exit.
        
        pass
    
    def __exit__():
        pass

    def __call__(self, fn: Callable) -> Callable:
        """
        For using the Profiler as a decorator
        """
        self.function = fn
        
        @functools.wraps(fn)
        def decorated(*args: Any, **kwds: Any) -> Any:
            """Training step decorator function"""

            with torch.no_grad() if self.no_grad else nullcontext():
                if self.cuda_graphs_enabled:
                    self._cuda_graph_forward(*args, **kwds)
                else:
                    self._zero_grads()
                    self.output = self._amp_forward(*args, **kwds)

                if not self.eval:
                    # Update model parameters
                    self.scaler.step(self.optim)
                    self.scaler.update()

            return self.output

        return decorated

# Requirements:
#     - For the pytorch profiler
#         - Enable CPU and CUDA devices
#         - memory toggle on by default, can toggle off
#     - For the line_profiler:
#         - include this?
#     - nvtx
#         - Any annotations to insert?