

# class nvtx_annotate

# class nvtx_wrapper(ModulusProfilerWrapper)

import os

from pathlib import Path

from dataclasses import dataclass, replace
from typing import List, Tuple, Optional, Callable


from . core import _Profiler_Singleton, ModulusProfilerWrapper

import functools

try:
    import nvtx
    nvtx_avail = True
except ImportError as e:
    nvtx_avail = False




class nvtxWrapper(ModulusProfilerWrapper, metaclass=_Profiler_Singleton):
    
    _name : str = "nvtx"
        
    def __init__(self):
        
        
        # nvtx is an annotation.  But we can use it as a context
        # to automatically annotate every function.  This is expensive.
        self._is_context    = True
        self._is_decorator  = False
        
    def _standup(self):
        # Create a profiler instance but don't enable it yet
        if nvtx_avail:
            self.pr = nvtx.Profile()
            self.enabled = True
        else:
            self.enabled = False
        self._initialized = True

    def finalize(self, output_top : Path):
        
        # nvtx has negligible finalization here, it's all in nsys
        if not self.enabled: return
        # Prevent double finalization:
        if self.finalized: return
        
        
        # Make this profiler completed:
        self.finalized = True
        
    def __enter__(self):
        """
        Using nvtx in the profiler will enable auto-annotation
        See (https://nvtx.readthedocs.io/en/latest/auto.html) for
        more details
        """
        if self.enabled:
            self.pr.enable()
        
        return self
    
    def __exit__(self, *exc):
        """ 
        Disable the profiler
        """
        if self.enabled:
            self.pr.disable()