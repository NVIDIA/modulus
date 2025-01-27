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

from pathlib import Path

from dataclasses import dataclass, replace
from typing import List


from contextlib import ContextDecorator

from dataclasses import dataclass
from modulus.distributed import DistributedManager
from abc import ABC, abstractmethod



class ModulusProfilerWrapper(ContextDecorator):
    """
    Abstract class to wrap a profiler interface.
    
    Modulus provides some useful profiling tools and configurations
    out of the box, but also is designed to enable other tools to plug
    in seamlessly.  If you annotate / decorate / wrap your code, you
    can attach a new profiler to your runs and activate it as well.
    
    You can also quickly and efficiently swap profiling tools without
    modifying model code: deactivate one profiler, reactivate another,
    and modulus will automatically handle the details.
    """
    _enabled     : bool = False
    _initialized : bool = False
    _finalized   : bool = False
    
    _is_context    : bool = False
    _is_decorator  : bool = False

    # Name is both a singleton lookup and output directory top:
    _name : str = ""
    
    # Default "config" - not always needed but need to have the attribute defined
    _config = None
    
    def __init__():
        self._config = None
    
    
    @abstractmethod
    def step(self):
        """
        For all attached profiling tools, call step if it is available
        if self.config.enabled:
            self._profiler.step()
        """
        pass
    
    # Getters / Setter for enabled, initialized, finalized
    @property
    def enabled(self):
        return self._enabled
    
    @enabled.setter
    def enabled(self, value):
        assert isinstance(value, bool)
        self._enabled = value
    
    @property
    def finalized(self):
        return self._finalized
    
    @finalized.setter
    def finalized(self, value : bool):
        assert isinstance(value, bool)
        self._finalized = value
        
    @property
    def initialized(self):
        return self._initialized
    
    @initialized.setter
    def initialized(self, value : bool):
        assert isinstance(value, bool)
        self._initialized = value
    
    @property
    def is_decorator(self):
        """
        Flag to declare if this profiling instance supports function decoration
        """
        return self._is_decorator
    
    @property
    def is_context(self):
        """
        Flag to declare if this profiling instance supports context-based profiling
        """
        return self._is_context
    
    
    def enable(self):
        self._enabled = True
        
    
    def __enter__(self):
        pass
    
    def __exit__(self, *exc):
        pass
    
    def step(self):
        pass
    
    def output_dir(self, top : Path):
        
        out_dir = top / Path(self._name)
        
        # If the model is distributed, control the location of the output:
        if DistributedManager.is_initialized():
            if DistributedManager().distributed:
                out_dir = out_dir.joinpath(Path(f"rank_{DistributedManager().rank}/"))
        
        # Make the directory, if necessary:
        out_dir.mkdir(exist_ok=True, parents=True)
        return out_dir

    def _teardown(self, path : Path ):
        """
        Don't overload _teardown; instead put your logic in finalize.
        
        The role of this function is to cleanly call the end logic 
        with possible singleton instances floating around.
        """
        
        if self.finalized:
            return
        try:
            self._teardown()
        except:
            print("Error in finalization")
        finally:
            self.finalized = True
    
    def reconfigure(self, **config_overrides):
        if self._config is not None:
            self._config = replace(self._config, **config_overrides)    
    
    
class _Profiler_Singleton(type):
    """
    The profiling tools, in general, need to be instantiated from 
    arbitrary files.  This is especially true for decorators and context
    managers that may need to exist across files.
    
    To handle this, all profiler wrappers (which inherit from ModulusProfilerWrapper)
    are implemented as class-based singletons.  That means when you create a pytorch
    profiler, it is created one time and reused at every file that also creates one.
    
    To avoid initialization (and re-initialization) problems, the profiler wrappers don't 
    do any under-the-hood instantiation until the last minute.  Or until the `_standup()`
    method is explicitly called.
    
    See here for more details and example code on which this is based:
    https://stackoverflow.com/questions/6760685/what-is-the-best-way-of-implementing-singleton-in-python
    """
    _instances = {}
    
    def __call__(class_, *args, **kwargs):
        if class_ not in class_._instances:
            class_._instances[class_] = super().__call__(*args, **kwargs)
        return class_._instances[class_]



class ProfileRegistry:
    
    # Keep track of constructed instances.
    # Each one is a singleton by class type, but it's 
    # possible to use different classes
    _instances = []
    
    # Keep track of key-value pairs to make summoning profiler instances easier.
    # This is purely for ease-of-use
    _registry = {}
    
    @classmethod
    def get_profiler(cls, key):

        # Search by key:
        if key in cls._registry:
            return cls._registry[key]
    
        # Search by type, too (Meaning, the key can be the type itself):
        if key in cls._instances:
            return cls._instances[key]
    
        else:
            raise Exception(f"ProfilerRegistry has no profiler under the key {key}")
        
    @classmethod
    def register_profiler(cls, profiler_key, profiler_cls):
        
        
        # assert isinstance(profiler_cls, _Profiler_Singleton), "Can only register instances of Profiler_Singleton"
        
        # If the class is already registered, just add the mapping:
        if profiler_cls not in cls._instances:
            cls._instances.append(profiler_cls())
        
        if profiler_key not in cls._registry.keys():
            cls._registry[profiler_key] = profiler_cls()
        
        # Last, get upset if the key is here and it's trying to reregister for another class:
        else:
            # If it's a reregister of the same thing, no problem:
            if profiler_cls != cls._registry[profiler_key]:
                raise Exception("Profiler key already in use for different profiler!")
            