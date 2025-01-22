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


from contextlib import ExitStack, contextmanager, ContextDecorator

from dataclasses import dataclass
from modulus.distributed import DistributedManager

from . core import _Profiler_Singleton, ProfileRegistry

import wrapt


import inspect

@wrapt.decorator
def universal(wrapped, instance, args, kwargs):
    print("ENTER UNIVERSAL")
    print(f"wrapped: {wrapped}")
    print(f"instance: {instance}")
    print(f"args: {args}")
    print(f"kwargs: {kwargs}")
    if instance is None:
        if inspect.isclass(wrapped):
            # Decorator was applied to a class.
            return wrapped(*args, **kwargs)
        else:
            # Decorator was applied to a function or staticmethod.
            return wrapped(*args, **kwargs)
    else:
        if inspect.isclass(instance):
            # Decorator was applied to a classmethod.
            return wrapped(*args, **kwargs)
        else:
            # Decorator was applied to an instancemethod.
            return wrapped(*args, **kwargs)
    print("REACHED THE END!")

class AnnotationManager:
    """
    Annotations must be context decorators.  To make this a little cleaner,
    this class is a manager just for annotations.  It provides the 
    interface for __enter__ / __exit__ / __call__ and also a stack of
    sub-annotations
    """
    _annotations = []
    
    def append(self, annotation):
        self._annotations.append(annotation)
    
    def annotate(self, func_or_context):
        
        for annotation in reversed(self.annotations):
            func_or_context = annotation(func_or_context)
            
        return func_or_context
    
    def __call__(self, func):

        return self.annotate(func)

    def __enter__(self):
                  
        self.exit_stack = ExitStack()
        for annotation in self._annotations:
            self.exit_stack.enter_context(annotation)
        return self
        
    def __exit__(self, *exc):
        
        return self.exit_stack.__exit__(*exc)
        
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
    
    # A list of functions to capture for annotation _before_ all the profilers are initialized
    _annotation_registry = []
    _decoration_registry = []
    
    # Control flow switches for whether the profiler
    # has been initialized (and can do annotations/decorations)
    # (And if not - they get deferred, see below)
    _initialized : bool = False
    
    # Control flow for wrapping up the profiler: closing contexts/
    # writing outputs, etc.  Only want to trigger this once
    _finalized : bool
    
    exit_stack = ExitStack()
    annotation_stack = AnnotationManager()
    

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
        
        print("Decoration registry: ", self._decoration_registry)
        print("Annotation registry: ", self._annotation_registry)
            
        # Annotate any missed functions:
        for func, args, kwargs in self._annotation_registry:
            annotated = self._annotate_function(func, *args, **kwargs)            
            self.replace_function(func, annotated)
            

        self._annotation_registry.clear()

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
            return
    
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

    
    
    # annotate = ContextDecorator()
    
    
    @wrapt.decorator
    def annotate(self, wrapped, instance, args, kwargs):
        print("ENTER UNIVERSAL")
        print(f"wrapped: {wrapped}")
        print(f"instance: {instance}")
        print(f"args: {args}")
        print(f"kwargs: {kwargs}")
        if instance is None:
            if inspect.isclass(wrapped):
                # Decorator was applied to a class.
                return wrapped(*args, **kwargs)
            else:
                # Decorator was applied to a function or staticmethod.
                return wrapped(*args, **kwargs)
        else:
            if inspect.isclass(instance):
                # Decorator was applied to a classmethod.
                return wrapped(*args, **kwargs)
            else:
                # Decorator was applied to an instancemethod.
                return wrapped(*args, **kwargs)
        print("REACHED THE END!")
    
    # @universal
    # def annotate(self, *args, **kwargs):
    #     pass
    
                
    # def annotate(self, *a_args, **a_kwargs):
    #     print(f"Enter annotate with args {a_args}, {a_kwargs}")
        
    #     @wrapt.decorator
    #     def wrapper(wrapped, instance, args, kwargs):
            
    #         # Before the function
    #         print(f"Before function '{wrapped.__name__}' with param={a_args} and {a_kwargs}")

    #         # Call the original function
    #         result = wrapped(*args, **kwargs)

    #         # After the function
    #         print(f"After function '{wrapped.__name__}' with param={param}")
    #         return result
        
    #     print(f"Returning {wrapper}")
    #     return wrapper
    
    # def annotate(self, *args, **kwargs):
    #     """
    #     This is a "annotation factory" to produce the right decorator or context
    #     depending on the presence and value of args and kwargs.
        
    #     It also has to *defer* decoration until the profiler is actually 
    #     initialized and all tools are enabled.
        
    #     Two layers of functions here to allow kwargs to the decorator.
        
    #     In general, the logic is:
    #     if len(args) > 0:
    #         if callable(args[0]):
    #             this is annotating a function.  
    #         else:
    #             this is a context
    #     else:
    #         this is a context (because len(args) is 0)
           
    #     to make this simpler, invert it:
    #     if len(args) == 0 or not callable(args[0]):
    #         this is a context to annotate
    #     else;
    #         this is a function
            
    #     """

    #     print("Enter annotate")
    #     print(f"args: {args}")
    #     print(f"kwargs: {kwargs}")

    #     print(f"len(args) > 0 {len(args) > 0}")
    #     # print(f"callable(args[0]): {callable(args[0])}")

    #     if len(args) == 0 or not callable(args[0]):
    #         print("got a context")
    #     else:
    #         print("Got a function")

    #     # Supporting only kwargs for function annotations
    #     # if len(args) > 0 then this must be a context annotation
    #     if len(args) > 0 and not callable(args[0]):
    #         print("Didn't get a function!")
    #         # Return the annotation _context_:
    #         for p in self._profilers:
    #             self.annotation_stack.append(p.annotate(*args, **kwargs))
            
    #         print(f"Returning {self.annotation_stack}")
    #         return self.annotation_stack
    #     else:
    #         print("Got a function!")
    #         # This must be a function:
    #         if len(args) == 1 and len(kwargs) == 0:
    #             return self._deferred_or_immediate_annotation(args[0])
    #         else:
    #             # Called as function decorator `annotate(arguments, kwargs...)`
            
    #             # TODO - functools wraps here
            
    #             def decorator(func):
    #                 return self._deferred_or_immediate_annotation(func,*args, **kwargs)
            
    #             return decorator
                


    def _deferred_or_immediate_decoration(self, func):
        
        
        if self.initialized:
            return self._decorate_function(func)
        else:
            self._decoration_registry.append(func)
            return func
            
        
    def _deferred_or_immediate_annotation(self, func, *args, **kwargs):
        """
        The role of this function is to decide whether to do the annotation immediately
        or to defer it until after initialization.
        """
        
        
        if self.initialized:
            return self._annotate_function(func, *args, **kwargs)
        else:
            # Capture the function but return it un-edited for now:
            # Registering function for later:
            self._annotation_registry.append((func, args, kwargs))
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
        
    
    def _annotate_function(self, func, *args, **kwargs):
        
        for p in self._profilers:
            if p.enabled and p.is_annotation:
                func = p.annotate(func, *args, **kwargs)
                
        return func

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