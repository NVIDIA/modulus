import os

from pathlib import Path

from dataclasses import dataclass, replace
from typing import List, Tuple, Optional, Callable


from . core import _Profiler_Singleton, ModulusProfilerWrapper, annotate

import functools

try:
    from line_profiler import LineProfiler
    lp_avail = True
except ImportError as e:
    lp_avail = False

# TODO - if not available make all of this null

@dataclass
class LineProfileConfig:
    """
    Specific configuration for the pytorch profiler.
    """
    name:             str = "line_profiler"
    output_dir:      Path = "./profiler_output/"

    
class LineProfileWrapper(ModulusProfilerWrapper):
    __metaclass__ = _Profiler_Singleton
    
    _name : str = "line_profiler"
        
    def __init__(self, config: Optional[LineProfileConfig] = None, **config_overrides):
        
        default_config = LineProfileConfig()
        
        # Pytorch is a context and annotation but not a wrapper:
        self._is_context    = False
        self._is_annotation = False
        self._is_decorator  = True
        
        # Replace any overrides right into the config:
        if config is None: 
            self._config = replace(default_config, **config_overrides) 
        else:
            self._config = replace(config, **config_overrides)
        
        
    def _standup(self):
        # Nothing to do here ... 
        if lp_avail:
            self._profiler = line_profiler.LineProfiler()
        else:
            self._profiler = None
            self.enabled = False
        self._initialized = True

    def finalize(self, output_top : Path):
        
        
        if not self.enabled: return
        
        # Prevent double finalization:
        if self.finalized: return
        
        # Get the output directory:
        out_top = self.output_dir(output_top)

        with open(out_top / Path("profiler_stats.txt"), 'w') as stats:
            self._profiler.print_stats(stream=stats)
            # stats.write(self._profiler.print_stats)

        # Make this profiler completed:
        self.finalized = True
        

    def __call__(self, fn):
        """
        Function Decororator actions
        """

        f = self._profiler(fn)
        return f
        