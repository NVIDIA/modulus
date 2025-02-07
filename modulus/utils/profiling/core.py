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
from typing import List, Any


from contextlib import ContextDecorator

from modulus.distributed import DistributedManager



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
    
    def __init__(self):
        self._config = None
    
    
    def step(self):
        """
        For all attached profiling tools, call step if it is available
        if self.config.enabled:
            self._profiler.step()
        """
        pass
    
    @property
    def enabled(self) -> bool:
        """Get whether the profiler is enabled.

        Returns:
            bool: True if profiler is enabled, False otherwise
        """
        return self._enabled
    
    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Set whether the profiler is enabled.

        Args:
            value (bool): True to enable profiler, False to disable
        """
        assert isinstance(value, bool)
        self._enabled = value
    
    @property
    def finalized(self) -> bool:
        """Get whether the profiler has been finalized.

        Returns:
            bool: True if profiler is finalized, False otherwise
        """
        return self._finalized
    
    @finalized.setter
    def finalized(self, value: bool) -> None:
        """Set whether the profiler has been finalized.

        Args:
            value (bool): True to mark as finalized, False otherwise
        """
        assert isinstance(value, bool)
        self._finalized = value
        
    @property
    def initialized(self) -> bool:
        """Get whether the profiler has been initialized.

        Returns:
            bool: True if profiler is initialized, False otherwise
        """
        return self._initialized
    
    @initialized.setter
    def initialized(self, value: bool) -> None:
        """Set whether the profiler has been initialized.

        Args:
            value (bool): True to mark as initialized, False otherwise
        """
        assert isinstance(value, bool)
        self._initialized = value
    
    @property
    def is_decorator(self):
        """
        Flag to declare if this profiling instance supports function decoration
        """
        return self._is_decorator
    
    @is_decorator.setter
    def is_decorator(self, value: bool) -> None:
        """Set whether the profiler supports function decoration.

        Args:
            value (bool): True to support function decoration, False otherwise
        """
        assert isinstance(value, bool)

    @property
    def is_context(self):
        """
        Flag to declare if this profiling instance supports context-based profiling
        """
        return self._is_context
    
    @is_context.setter
    def is_context(self, value: bool) -> None:
        """Set whether the profiler supports context-based profiling.

        Args:
            value (bool): True to support context-based profiling, False otherwise
        """ 
        assert isinstance(value, bool)
        self._is_context = value
    
    def enable(self) -> None:
        """Enable the profiler.
        
        Sets the internal enabled flag to True to activate profiling.
        """
        self._enabled = True
    
    def __enter__(self) -> None:
        """Enter the profiling context.
        
        Called when entering a 'with' block. Base implementation does nothing.
        """
        pass
    
    def __exit__(self, exc_type: type[BaseException] | None, 
                exc_val: BaseException | None, 
                exc_tb: Any) -> None:
        """Exit the profiling context.
        
        Called when exiting a 'with' block. Base implementation does nothing.

        Args:
            exc_type: The type of exception that occurred, if any
            exc_val: The exception instance that occurred, if any 
            exc_tb: The traceback of the exception that occurred, if any
        """
        pass
    

    def output_dir(self, top: Path) -> Path:
        """Creates and returns an output directory for profiling data.
        
        Creates a subdirectory under the given top directory using this profiler's name.
        If running in distributed mode, further organizes output by rank.
        
        Args:
            top: The root directory to create the output directory under
            
        Returns:
            Path: The created output directory path
        """
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
            self.finalize()
        except:
            print("Error in finalization")
        finally:
            self.finalized = True

    def reconfigure(self, **config_overrides: Any) -> None:
        """Reconfigures the profiler with new configuration values.
        
        Updates the profiler's configuration by replacing specified values with new ones.
        Only works if the profiler has an existing configuration.

        Args:
            **config_overrides: Keyword arguments specifying configuration values to override
        """
        if self._config is not None:
            self._config = replace(self._config, **config_overrides)


from threading import Lock

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
    _lock = Lock()


    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                # Double-checked locking pattern
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

    def _clear_instance(cls):
        """Clear the singleton instance (mainly for testing purposes)"""
        if cls in cls._instances:
            del cls._instances[cls]

class ProfileRegistry:
    
    # Keep track of constructed instances.
    # Each one is a singleton by class type, but it's 
    # possible to use different classes
    _instances = []
    
    # Keep track of key-value pairs to make summoning profiler instances easier.
    # This is purely for ease-of-use
    _registry = {}
    
    @classmethod
    def get_profiler(cls, key: str | type) -> Any:
        """Get a registered profiler instance by key or type.

        Args:
            key: The key or type used to register the profiler

        Returns:
            The registered profiler instance

        Raises:
            Exception: If no profiler is found for the given key
        """

        # Search by key:
        if key in cls._registry:
            return cls._registry[key]
    
        # Search by type, too (Meaning, the key can be the type itself):
        if key in cls._instances:
            # Find instance of matching type in list
            for instance in cls._instances:
                if instance == key:
                    return instance
    
        else:
            raise Exception(f"ProfilerRegistry has no profiler under the key {key}")
        
    @classmethod
    def register_profiler(cls, profiler_key: str | type, profiler_cls: type) -> None:
        """Register a profiler class with an optional key for later retrieval.
        
        Args:
            profiler_key: String key or type to register the profiler under
            profiler_cls: The profiler class to register
            
        Raises:
            Exception: If attempting to register a different profiler under an existing key
        """
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
            
    @classmethod
    def _clear(cls):
        """Clear the registry and instances."""
        cls._registry = {}
        cls._instances = []
