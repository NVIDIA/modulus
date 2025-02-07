import os

from pathlib import Path

from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable

import contextlib


from dataclasses import dataclass, replace

from . core import _Profiler_Singleton, ModulusProfilerWrapper

import torch
from torch.profiler import   ProfilerActivity, record_function, profile, schedule

import functools

@dataclass
class TorchProfilerConfig:
    """
    Specific configuration for the pytorch profiler.
    """
    name:             str = "torch"
    torch_prof_activities: Optional[Tuple[int]] = None
    record_shapes :  bool = True
    profile_memory : bool = True
    with_stack:      bool = True
    profile_memory:  bool = True
    with_flops:      bool = True
    schedule:    Callable = None
    on_trace_ready_path: Optional[Path] = None
    
class TorchProfileWrapper(ModulusProfilerWrapper):
    __metaclass__ = _Profiler_Singleton
    
    _name : str = "torch"
    
    # Overload any of these:
    _is_context   = True
    _is_decorator = False
    
    def __init__(self, config: Optional[TorchProfilerConfig] = None, **config_overrides):
        
        default_config = TorchProfilerConfig()
        
        
        # Replace any overrides right into the config:
        if config is None: 
            self._config = replace(default_config, **config_overrides) 
        else:
            self._config = replace(config, **config_overrides)
        
        # Configure pytorch profiler here:
        # Set the default profiling activities if not set:
        if self._config.torch_prof_activities is None:
            torch_prof_activities = [ProfilerActivity.CPU]
            if torch.cuda.is_available(): 
                torch_prof_activities.append(ProfilerActivity.CUDA)
            self._config.torch_prof_activities = torch_prof_activities
        
        return
        
    
    def _standup(self):

        if self._config.on_trace_ready_path is not None:
            on_trace_ready = torch.profiler.tensorboard_trace_handler(self._config.on_trace_ready_path)
        else:
            on_trace_ready = None

        self._profiler = profile(
            activities     = self._config.torch_prof_activities, 
            profile_memory = self._config.profile_memory, 
            record_shapes  = self._config.record_shapes,
            with_stack     = self._config.with_stack,
            schedule       = self._config.schedule,
            with_flops     = self._config.with_flops,
            on_trace_ready = on_trace_ready
        )

        
        self._initialized = True

    def finalize(self, output_top : Path):
                
                
        if not self.enabled: return
        
        # Avoid finalizing if we never initialized:
        if not self.initialized: return
        
        # Prevent double finalization:
        if self.finalized: return
        # Get the output directory:
        out_top = self.output_dir(output_top)

        if self._profiler is not None:
            
            try:
                averages = self._profiler.key_averages()
            except AssertionError as e:
                # no averages recorded!
                averages = None

            # Write out torch profiling results:
            if averages:
                with open(out_top / Path("cpu_time.txt"), 'w') as cpu_times:
                    times = averages.table()
                    cpu_times.write(times)


                with open(out_top / Path("cuda_time.txt"), 'w') as gpu_times:
                    times = averages.table(sort_by="cuda_time_total")
                    gpu_times.write(times)

            if self._config.on_trace_ready_path is None:
                # Store the trace
                trace_path = out_top / Path("trace.json")
                self._profiler.export_chrome_trace(str(trace_path))
            
        # Make this profiler completed:
        self.finalized = True
            
    
    def __enter__(self):
        self._profiler.__enter__()
        
    def __exit__(self,*exc):
        self._profiler.__exit__(*exc)
    
    def step(self):
        self._profiler.step()


