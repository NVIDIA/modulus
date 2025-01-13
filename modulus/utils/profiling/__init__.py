from . interface import Profiler

from . torch import TorchProfileWrapper, TorchProfilerConfig


# Last, import the registry and add built in profilers::
from . core import ProfileRegistry

ProfileRegistry.register_profiler("torch", TorchProfileWrapper)