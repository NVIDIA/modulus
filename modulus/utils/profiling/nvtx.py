

# class nvtx_annotate

# class nvtx_wrapper(ModulusProfilerWrapper)

import os

from pathlib import Path

from dataclasses import dataclass, replace
from typing import List, Tuple, Optional, Callable


from . core import _Profiler_Singleton, ModulusProfilerWrapper, annotate

import functools

try:
    import nvtx
    nvtx_avail = True
except ImportError as e:
    nvtx_avail = False




class nvtxWrapper(ModulusProfilerWrapper, metaclass=_Profiler_Singleton):
    
    _name : str = "nvtx"
        
    annotate = nvtx.annotate
        
    def __init__(self):
        
        
        # Pytorch is a context and annotation but not a wrapper:
        self._is_context    = True
        self._is_annotation = True
        self._is_decorator  = False
        

        
    def _standup(self):
        # Nothing to do here ... 
        if nvtx_avail:
            pass
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