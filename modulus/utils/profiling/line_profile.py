import os

from pathlib import Path

from dataclasses import dataclass, replace
from typing import List, Tuple, Optional, Callable


from . core import _Profiler_Singleton, ModulusProfilerWrapper

import functools

try:
    from line_profiler import LineProfiler
    lp_avail = True
except ImportError as e:
    lp_avail = False

import warnings

class LineProfileWrapper(ModulusProfilerWrapper, metaclass=_Profiler_Singleton):
    # __metaclass__ = _Profiler_Singleton
    
    _name : str = "line_profiler"
        
    def __init__(self, **config_overrides):
        
        
        # Pytorch is a context and annotation but not a wrapper:
        self._is_context    = False
        self._is_decorator  = True
        
    # def __repr__(self):
    #     return "LineProfilerWrapper"

    def _standup(self):
        # Nothing to do here ... 
        if lp_avail:
            self._profiler = LineProfiler()
        else:
            warnings.warn(
                "Line Profiler was requested by the modulus profiler but " \
                "isn't install.  Try `pip install line_profiler`.")
            self._profiler = None
            self.enabled = False
        self._initialized = True

    def finalize(self, output_top : Path):
        """
        Serialize the line_profiler output if necessary
        """        
        if not self.enabled: return
        
        # Avoid finalizing if we never initialized:
        if not self.initialized: return
        
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
        