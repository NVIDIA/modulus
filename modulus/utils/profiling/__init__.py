
from . interface import Profiler

from . torch import TorchProfileWrapper, TorchProfilerConfig
from . line_profile import LineProfileWrapper
from . nvtx import nvtxWrapper


# Last, import the registry and add built in profilers::
from . core import ProfileRegistry

ProfileRegistry.register_profiler("torch", TorchProfileWrapper)
ProfileRegistry.register_profiler("line_profile", LineProfileWrapper)
ProfileRegistry.register_profiler("line_profiler", LineProfileWrapper)
ProfileRegistry.register_profiler("nvtx", nvtxWrapper)


from pathlib import Path

import atexit
p = Profiler()
atexit.register(p.finalize)


# convienence wrappers for profiling and annotation decorators:
annotate = p.annotate
profile  = p.__call__ 