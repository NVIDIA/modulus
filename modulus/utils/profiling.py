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

import os
import atexit

from dataclasses import dataclass
from typing import Tuple, Callable

try:
    import line_profiler
    import kernprof
    lp_available = True
except:
    lp_available = False

import warnings

from contextlib import nullcontext, ContextDecorator, ExitStack

from torch.profiler import   ProfilerActivity, record_function, profile

from modulus.distributed import DistributedManager

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
        enabled: bool = True,
        torch_profile: bool = False,
        output_dir: str = "./profiler_output/",
        torch_profiler_activities: Tuple[int] = [ProfilerActivity.CPU],
        record_shapes : bool = True,
        profile_memory : bool = True,
        include_nvtx: bool = False,
        line_profile: bool = True,
    ):
        
        self._enabled = enabled
        if not self.enabled:
            # Do nothing ... 
            return
        
        self.output_location = output_dir
        # If the model is distributed, control the location of the output:
        if DistributedManager().distributed:
            self.output_location += f"/rank_{DistributedManager().rank}"

        # Configure line_profiler here:
        if lp_available:
            print("Hooray")
        elif line_profile:
            if DistributedManager().rank == 0:
                warnings.warn("Line profiler failed to import - is it installed?  Disabling")
            line_profile = False
        
        if line_profile:
            import line_profiler
            self.line_profiler = line_profiler.LineProfiler()
        
        # Configure pytorch profiler here:
        if torch_profile:
            self.do_torch_profiling = True
            self.torch_prof = profile(activities = torch_profiler_activities, profile_memory=profile_memory, record_shapes=record_shapes)
        else:
            self.do_torch_profiling = False
            self.torch_prof = nullcontext()
            

        # Configure nvtx context tagging here:

        self.exit_stack = ExitStack()

        # Prevent double-finalization:
        self.finalized = False



    @property
    def enabled(self):
        """
        Return true if profiling is enabled
        """
        return self._enabled

    def __enter__(self):
        """
        Enter profiling contexts 
        """
        if not self.enabled: return
        # Activate pytorch profiling context
        # Set nvtx context based on name
        # Activate the line_profiler for use as a context
        
        # Capture each context in an exit stack that we'll back out of in the exit.
        
        self.exit_stack.enter_context(self.line_profiler.runctx())
        self.exit_stack.enter_context(self.torch_prof)
        
        
        return self
    
    def __exit__(self, *exc):
        """
        Clear out the exit stack
        """
        if not self.enabled: return
        
        self.exit_stack.close()
        
    # register the finalize function to dump output incase of ctrl+C, etc., killing the profiler early:
    def finalize(self, rank0_only = True):
        """
        Write profiling results to output
        """
        
        if not self.enabled: return
        
        # Prevent double finalization:
        if self.finalized: return
        
        # Make sure the output directory exists:
        os.makedirs(self.output_location, exist_ok=True)
        
        if self.do_torch_profiling:
            torch_outut_dir = self.output_location + "/torch/"
            os.makedirs(torch_outut_dir, exist_ok=True)
            
            print(self.torch_prof.key_averages())
            
            # Write out torch profiling results:
            print("Storing cpu times in output:")
            with open(torch_outut_dir + "cpu_time.txt", 'w') as cpu_times:
                times = self.torch_prof.key_averages().table()
                cpu_times.write(times)
            print("Done writing CPU Times")

            # print("Storing gpu times in output:")
            # with open(torch_outut_dir + "gpu_time.txt", 'w') as gpu_times:
            #     times = self.torch_prof.key_averages().table(sort_by="gpu_times_total", row_limit=25)
            #     print(times)
            #     gpu_times.write(times)
            # print("Done writing gpu Times")

            # Store the trace
            

        self.finalized = True
        
    def __del__(self,):
        """
        Clean up and ensure results are output, just in case:
        """
        
        if not self.finalized: self.finalize()

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