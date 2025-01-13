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
from pathlib import Path

from dataclasses import dataclass
from typing import List, Optional, Callable


from contextlib import ContextDecorator, ExitStack

from dataclasses import dataclass
from modulus.distributed import DistributedManager
from abc import ABC, abstractmethod

from . core import _Profiler_Singleton, ProfileRegistry

class Profiler(ContextDecorator):
    __metaclass__ = _Profiler_Singleton
    

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
    
    _output_top : Path
    
    # A list of functions to capture for annotation _before_ all the profilers are initialized
    _annotation_captures = []
    # we have to return a callable immediately from annotations, so send a placeholder
    _annotation_outputs = []
    
    def __init__(self, output_dir_top = None):
        super().__init__()
        
        # Configure nvtx context tagging here:

        self.exit_stack = ExitStack()

        # Prevent double-finalization and initializations:
        self.initialized = False
        self.finalized = False
        
        if output_dir_top is None:
            self._output_top = Path(".").resolve()
        else:
            self._output_top = Path(output_dir_top).resolve()
        


    def _standup(self):
        
        # Call _standup on all profilers, then set initialized to prevent
        # reinit
        
        print("PROFILER BEING INITIALIZED")
        
        for p in self._profilers:
            p._standup()
            
            
        print(f"original outputs: {self._annotation_outputs}")
        # Annotate any missed functions:
        for i, (captures, outputs) in  enumerate(zip(self._annotation_captures, self._annotation_outputs)):
            fn, args, kwargs = captures
            print(f"Original output: {outputs}")
            real_annotation = self._do_actual_annotation(fn, *args, **kwargs)
            self._annotation_outputs[i] = real_annotation
            print(f"Updated output: {outputs}")
            
        print(f"updated outputs: {self._annotation_outputs}")
        
        self.initialized = True
        

    def _teardown(self):
        
        for p in self._profilers:
            p._teardown(self.output_path)
            
        
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
    
    
    def enable(self, profiler):
        """
        Enable a profiler.  The profiler can be an instance of a class 
        that derives from the profiler wrapper, or it can be a keyword that
        is registered with the profiler manager.
        
        """
        
        # Is it an instance of the right type? If not, find it:
        if not isinstance(profiler, _Profiler_Singleton):
            profiler = ProfileRegistry.get_profiler(profiler)
        
        # make sure the summoned profiler is enabled:
        profiler.enable()
        
        # Prevent double-adds:
        if profiler not in self._profilers:
            self._profilers.append(profiler)
            return
    
    def finalize(self):
        """
        Allow option to manually finalize
        before destruction.  Preferred option!
        """
        self._teardown()

    def __enter__(self):
        """
        Enter profiling contexts 
        """
        
        if not self.enabled: return
        
        if not self.initialized:
            self._standup()
        
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
        print("Exiting!")
        if not self.enabled: return
        
        self.exit_stack.close()
        
    def __del__(self,):
        """
        Clean up and ensure results are output, just in case:
        """
        
        if not self.finalized: self._teardown()

    
    def __call__(fn: Callable) -> Callable:
        """
        For using the Profiler as a decorator
        """
        
        # For the function decorator, we pass the decoration 
        # on to active profilers for them to decorate.
        # Fires in the order they were activated!
        
        for p in self._profilers:
            if p.enabled and p.is_decorator:
                fn = p(fn)
                
        return fn

    
    def annotate(self, **kwargs):
        """Two layers of functions here to allow kwargs to the decorator"""

        print(f"got kwargs: {kwargs}")
        
        def decorator(fn):
        
            print(f"got fn: {fn}")
            
            # A challenge here: we may have annotations called
            # before the profiler is fully configured.
            
            # So, we capture function names and calls if 
            # the profiler isn't yet initialized, 
            # and run the annotation if it has already been initialized.
            
            if not self.initialized:
                # send the actual function as the placeholder to prevent bugs:
                self._annotation_outputs.append(lambda *a, **kw : fn(*a, **kw))
                self._annotation_captures.append(
                    (
                        fn,
                        kwargs
                    )
                )
                return self._annotation_outputs[-1]
            
            else:
                return self._do_actual_annotation(fn, *args, **kwargs)
            
        return decorator
            
    def _do_actual_annotation(self, fn, *args, **kwargs):
        
        for p in self._profilers:
            if p.enabled and p.is_annotation:
                print(f"In real annotated, got {fn}")
                fn = p.annotate(fn, *args, **kwargs)
                print(f"In real annotated, returned {fn}")
                
        return fn


    @property
    def output_path(self):
        
        return self._output_top
    
    @output_path.setter
    def output_path(self, path : Path):
        # cast if necessary:
        path = Path(path)  
        self._output_top = path