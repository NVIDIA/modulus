import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from modulus.utils.profiling import Profiler, profile
from modulus.utils.profiling.core import _Profiler_Singleton, ProfileRegistry
from modulus.utils.profiling.core import ModulusProfilerWrapper
from modulus.utils.profiling import _register_profilers

from dataclasses import dataclass, replace
from typing import Any, Optional, Tuple

# Mock config class for testing

@dataclass
class MockProfilerConfig:
    """
    Specific configuration for the pytorch profiler.
    """
    name: str = "mock"
    option1: bool = True
    option2: int = 42

# Mock profiler class for testing
class MockProfiler(ModulusProfilerWrapper, metaclass=_Profiler_Singleton):
    
    def __init__(self, config: Optional[MockProfilerConfig] = None, **config_overrides):
    
        default_config = MockProfilerConfig()
        
        # Replace any overrides right into the config:
        if config is None: 
            self._config = replace(default_config, **config_overrides) 
        else:
            self._config = replace(config, **config_overrides)
    
        self.is_context = True
        self.is_decorator = True
        
    def enable(self):
        self.enabled = True
        
    def _standup(self):
        self._initialized = True
        
    def __enter__(self):
        return self
        
    def __exit__(self, *args):
        pass

    def __call__(self, *args, **kwargs):
        pass

    def finalize(self, output_path):
        pass
        
    def step(self):
        pass


@pytest.fixture(autouse=True)
def reset_profiler():
    Profiler._clear_instance()
    ProfileRegistry._clear()
    _register_profilers()
    # Register the mock profiler
    ProfileRegistry.register_profiler("mock", MockProfiler)

def test_profiler_initialization():
    profiler = Profiler()
    assert not profiler.initialized
    assert not profiler.enabled
    
    # Enable a mock profiler
    profiler.enable("mock")
    assert profiler.enabled
    
    # Test initialization
    profiler.initialize()
    assert profiler.initialized
    
    # It's a singleton, so we should get the same instance
    mock_profiler = MockProfiler()
    assert mock_profiler._initialized


def test_profiler_context_manager():
    profiler = Profiler()

    profiler.enable("mock")
    
    # Not initialized yet
    assert not profiler.initialized

    with profiler as p:
        # Now it is initialized
        assert p.initialized
        assert p.enabled

def test_profiler_decoration():
    profiler = Profiler()
    mock_profiler = MockProfiler()
    profiler.enable(mock_profiler)
    
    @profile
    def test_function():
        return "test"
        
    # Function should be registered for decoration before initialization
    assert test_function in profiler._decoration_registry
    
    # After initialization, function should be decorated
    profiler.initialize()
    assert test_function not in profiler._decoration_registry

def test_profiler_config_update():
    profiler = Profiler()

    mock_profiler = profiler.get("mock")

    mock_profiler.reconfigure(option1=False)
    assert not mock_profiler._config.option1

    mock_profiler.reconfigure(option2=100)
    assert mock_profiler._config.option2 == 100

    profiler.enable(mock_profiler)


def test_output_path():
    profiler = Profiler()
    test_path = Path("/tmp/test_output")
    profiler.output_path = test_path
    assert profiler.output_path == test_path
    
    # Test string conversion
    profiler.output_dir = "/tmp/test_output2"
    assert isinstance(profiler.output_dir, Path)

def test_profiler_finalization():
    profiler = Profiler()
    mock_profiler = MockProfiler()
    profiler.enable(mock_profiler)
    
    # Mock the finalize method
    mock_profiler.finalize = MagicMock()
    
    profiler.finalize()
    mock_profiler.finalize.assert_called_once()

def test_profiler_step():
    profiler = Profiler()
    mock_profiler = MockProfiler()
    profiler.enable(mock_profiler)
    
    # Mock the step method
    mock_profiler.step = MagicMock()
    
    profiler.step()
    mock_profiler.step.assert_called_once()



def test_function_replacement():
    def original_function():
        return "original"
        
    def wrapped_function():
        return "wrapped"
        
    profiler = Profiler()
    with patch('sys.modules') as mock_modules:
        mock_module = MagicMock()
        mock_modules.__getitem__.return_value = mock_module
        original_function.__module__ = 'test_module'
        original_function.__qualname__ = 'original_function'
        
        profiler.replace_function(original_function, wrapped_function)
        
        # Check that the function was replaced in the module
        mock_module.original_function = wrapped_function