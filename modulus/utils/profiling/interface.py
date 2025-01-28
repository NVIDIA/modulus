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

import sys, os
from pathlib import Path

from dataclasses import dataclass
from typing import List, Optional, Callable


from contextlib import ExitStack, ContextDecorator

from dataclasses import dataclass
from modulus.distributed import DistributedManager

from . core import _Profiler_Singleton, ProfileRegistry

import wrapt


try:
    import nvtx
    nvtx_annotate = nvtx.annotate
except ImportError as e:
    nvtx_avail = False
    nvtx_annotate = ContextDecorator


    
class Profiler(metaclass=_Profiler_Singleton):
    

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
    
    # Keep a list of configured, singleton profilers
    _profilers = []
    
    _output_top : Path = Path("./modulus_profiling_outputs/")
    
    # A list of functions to capture for decoration _before_ all the profilers are initialized
    _decoration_registry = []
    
    # Control flow switches for whether the profiler
    # has been initialized (and can do annotations/decorations)
    # (And if not - they get deferred, see below)
    _initialized : bool = False
    
    # Control flow for wrapping up the profiler: closing contexts/
    # writing outputs, etc.  Only want to trigger this once
    _finalized : bool
    
    exit_stack = ExitStack()
    
    annotate = nvtx_annotate
    

    @property
    def initialized(self):
        return self._initialized
    
    @initialized.setter
    def initialized(self, value : bool):
        assert isinstance(value, bool)
        self._initialized = value
        
    @property
    def finalized(self):
        return self._finalized
    
    @finalized.setter
    def finalized(self, value : bool):
        assert isinstance(value, bool)
        self._finalized = value

    @property
    def output_dir(self):
        return self._output_top
    
    @output_dir.setter
    def output_dir(self, value):
        self._output_top = Path(value)

    def _standup(self):
        
        # Call _standup on all profilers, then set initialized to prevent
        # reinit of profilers and functions
        
        if self.initialized: return
        
        # Stand up all attached profilers.  After this, can't add more.
        for p in self._profilers:
            p._standup()
    

        for func in self._decoration_registry:
            decorated = self._decorate_function(func)
            self.replace_function(func, decorated)

        self._decoration_registry.clear()
        
        
        self.initialized = True
        
        
        
    def initialize(self):
        """
        Manually initialize the profiler interface
        """
        self._standup()

    def finalize(self):
        """
        finalize the profiler interface.  Writes data to file
        if necessary, automatically
        """
        
        
        for p in self._profilers:
            p.finalize(self.output_path)
            
        
    def step(self):
        """
        For all attached profiling tools, call step if it is available
        """
        for p in self._profilers:
            p.step()
        
    @property
    def enabled(self): 
        """
        Return true if profiling is enabled
        """
        enabled = any( [ p.enabled for p in self._profilers] )
        return enabled
    
    def __repr__(self):
        """
        Summarize the current profiling interface in a string
        """
        
        name = f"<Profiler at {hex(id(self))}>"
        
        if self.initialized:
            s = f"Activated Modulus {name} with [{' '.join([str(_P) for _P in self._profilers])}] profilers."
        else:
            s = f"Un-Activated Modulus {name}"
            
        return s
    
    def enable(self, profiler):
        """
        Enable a profiler.  The profiler can be an instance of a class 
        that derives from the profiler wrapper, or it can be a keyword that
        is registered with the profiler manager.
        
        """
        
        if self.initialized:
            raise Exception("Can not enable more profiling tools after the profiler interface is initialized")
        
        # Is it an instance of the right type? If not, find it:
        if not isinstance(profiler, _Profiler_Singleton):
            profiler = ProfileRegistry.get_profiler(profiler)
        
        # make sure the summoned profiler is enabled:
        profiler.enable()
        
        # Prevent double-adds:
        if profiler not in self._profilers:
            self._profilers.append(profiler)
        
        return profiler
    
    def get(self, profiler):
        """
        Use the profiler registry to access a profiler
        """
        
        profiler = ProfileRegistry.get_profiler(profiler)
        
        return profiler
    
    def __enter__(self):
        """
        Enter profiling contexts 
        """
        if not self.initialized:
            self._standup()
            
            
        assert self.initialized, "Can not enter a context with an uninitialized profiler"
        
        if not self.enabled: 
            # An initialized but _empty_ profiler, then.
            return self
        

        
        # Activate context for all attached profilers
        # Set nvtx context based on name
        # Activate the line_profiler for use as a context
        
        # Capture each context in an exit stack that we'll back out of in the exit.
        
        
        for p in self._profilers:
            if p.enabled and p.is_context:
                self.exit_stack.enter_context(p)
        
        
        return self
    
    def __exit__(self, *exc):
        """
        Clear out the exit stack
        """
        if not self.enabled: return
        
        self.exit_stack.close()
        
    def __del__(self,):
        """
        Clean up and ensure results are output, just in case:
        """
        try:
            self.finalize()
        except Exception as e:
            print("Profiler Interface failed to cleanup, please call finalize in your code!")

    
    def __call__(self, fn: Callable) -> Callable:
        """
        For using the Profiler as a decorator
        """
        
        # For the function decorator, we pass the decoration 
        # on to active profilers for them to decorate.
        # Fires in the order they were activated!

        return self._deferred_or_immediate_decoration(fn)


    def _deferred_or_immediate_decoration(self, func):
        
        
        if self.initialized:
            return self._decorate_function(func)
        else:
            self._decoration_registry.append(func)
            return func
        

    def replace_function(self, func, wrapped_func):
        
        module_name = func.__module__
        module = sys.modules[module_name]
        
        
        if '.' in func.__qualname__:
            
            qualname_parts = func.__qualname__.split(".")
            
            obj = module
            for part in qualname_parts[:-1]:
                obj = getattr(obj, part)
                
            setattr(obj, qualname_parts[-1], wrapped_func)
        
        else:
            setattr(module, func.__qualname__, wrapped_func)
            # In case this function was imported into the main namespace
            # (aka, in the executed script `from A import func as weird_name`), there
            # is a reference to it there too.  Capture it:
            __main__ = sys.modules["__main__"]
            
            for name, obj in vars(__main__).items():
                if obj is func:
                    setattr(__main__, name, wrapped_func)
            
            if hasattr(__main__, func.__qualname__):
                setattr(__main__, func.__qualname__, wrapped_func)
        

    def _decorate_function(self, func):
        for p in self._profilers:
            if p.enabled and p.is_decorator:
                func = p(func)
        return func

    @property
    def output_path(self):
        
        return self._output_top
    
    @output_path.setter
    def output_path(self, path : Path):
        # cast if necessary:
        path = Path(path)  
        self._output_top = path