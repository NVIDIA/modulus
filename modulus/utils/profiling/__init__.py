from . interface import Profiler

from . torch import TorchProfileWrapper, TorchProfilerConfig
from . line_profile import LineProfileWrapper


# Last, import the registry and add built in profilers::
from . core import ProfileRegistry

ProfileRegistry.register_profiler("torch", TorchProfileWrapper)
ProfileRegistry.register_profiler("line_profile", LineProfileWrapper)

import atexit
p = Profiler()
atexit.register(p.finalize)