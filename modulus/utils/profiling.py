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

from dataclasses import dataclass, replace
from typing import Tuple, Callable, Optional


import warnings

from contextlib import nullcontext, ContextDecorator, ExitStack

import torch
from torch.profiler import   ProfilerActivity, record_function, profile

from modulus.distributed import DistributedManager

@dataclass
class ModulusProfilerConfig:
    torch_enabled:   bool = True
    output_dir:      str  = "./profiler_output/"
    torch_prof_activities: Optional[Tuple[int]] = None
    record_shapes :  bool = True
    profile_memory : bool = True
    with_stack:      bool = True
    with_trace:      bool = True
    include_nvtx:    bool = False
    
    #TODO: add cls method to export some basic default configs
    
    
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
        config: Optional[ModulusProfilerConfig] = None,
        **config_overrides,
    ):
        
        default_config = ModulusProfilerConfig()
        self.config = replace(default_config, **config_overrides) if config is None else replace(config, config_overrides)
        
        if self.config.torch_enabled or self.config.include_nvtx: self._enabled = True
        
        print(self.config)
        

        # If the model is distributed, control the location of the output:
        if DistributedManager().distributed:
            self.config.output_dir += f"/rank_{DistributedManager().rank}/"

        print(self.config)
        
        # Configure pytorch profiler here:
        # Set the default profiling activities if not set:
        if self.config.torch_enabled:
            if self.config.torch_prof_activities is None:
                torch_prof_activities = [ProfilerActivity.CPU]
                if torch.cuda.is_available(): 
                    torch_prof_activities.append(ProfilerActivity.CUDA)
                self.config.torch_prof_activities = torch_prof_activities
            self.torch_prof = profile(
                activities     = self.config.torch_prof_activities, 
                profile_memory = self.config.profile_memory, 
                record_shapes  = self.config.record_shapes,
                with_stack     = self.config.with_stack
            )
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
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        if self.config.torch_enabled:
            torch_output_dir = self.config.output_dir + "/torch/"
            os.makedirs(torch_output_dir, exist_ok=True)
            
            
            # Write out torch profiling results:
            print("Storing cpu times in output:")
            with open(torch_output_dir + "cpu_time.txt", 'w') as cpu_times:
                times = self.torch_prof.key_averages().table()
                cpu_times.write(times)
            print("Done writing CPU Times")

            print("Storing gpu times in output:")
            with open(torch_output_dir + "cuda_time.txt", 'w') as gpu_times:
                times = self.torch_prof.key_averages().table(sort_by="cuda_time_total")
                gpu_times.write(times)
            print("Done writing gpu Times")

            # Store the trace
            self.torch_prof.export_chrome_trace(torch_output_dir + "/trace.json")
            

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