# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


"""Miscellaneous utility classes and functions."""


import contextlib
import ctypes
import fnmatch
import glob
import hashlib
import html
import importlib
import inspect
import io
import os
import pickle
import re
import shutil
import sys
import tempfile
import types
import urllib
import urllib.request
import uuid
import warnings
from typing import Any, List, Tuple, Union

import numpy as np
import requests
import torch


class EasyDict(dict):
    """
    Convenience class that behaves like a dict but allows access with the attribute
    syntax.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]


class StackedRandomGenerator:
    """
    Wrapper for torch.Generator that allows specifying a different random seed
    for each sample in a minibatch.
    """

    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [
            torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds
        ]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack(
            [torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators]
        )

    def randn_like(self, input):
        return self.randn(
            input.shape, dtype=input.dtype, layout=input.layout, device=input.device
        )

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack(
            [
                torch.randint(*args, size=size[1:], generator=gen, **kwargs)
                for gen in self.generators
            ]
        )


def parse_int_list(s):
    """
    Parse a comma separated list of numbers or ranges and return a list of ints.
    Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]
    """
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r"^(\d+)-(\d+)$")
    for p in s.split(","):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges


# Cache directories
# -------------------------------------------------------------------------------------

_dnnlib_cache_dir = None


def set_cache_dir(path: str) -> None:
    global _dnnlib_cache_dir
    _dnnlib_cache_dir = path


def make_cache_dir_path(*paths: str) -> str:
    if _dnnlib_cache_dir is not None:
        return os.path.join(_dnnlib_cache_dir, *paths)
    if "DNNLIB_CACHE_DIR" in os.environ:
        return os.path.join(os.environ["DNNLIB_CACHE_DIR"], *paths)
    if "HOME" in os.environ:
        return os.path.join(os.environ["HOME"], ".cache", "dnnlib", *paths)
    if "USERPROFILE" in os.environ:
        return os.path.join(os.environ["USERPROFILE"], ".cache", "dnnlib", *paths)
    return os.path.join(tempfile.gettempdir(), ".cache", "dnnlib", *paths)


# Small util functions
# -------------------------------------------------------------------------------------


def format_time(seconds: Union[int, float]) -> str:
    """Convert the seconds to human readable string with days, hours, minutes and seconds."""
    s = int(np.rint(seconds))

    if s < 60:
        return "{0}s".format(s)
    elif s < 60 * 60:
        return "{0}m {1:02}s".format(s // 60, s % 60)
    elif s < 24 * 60 * 60:
        return "{0}h {1:02}m {2:02}s".format(s // (60 * 60), (s // 60) % 60, s % 60)
    else:
        return "{0}d {1:02}h {2:02}m".format(
            s // (24 * 60 * 60), (s // (60 * 60)) % 24, (s // 60) % 60
        )


def format_time_brief(seconds: Union[int, float]) -> str:
    """Convert the seconds to human readable string with days, hours, minutes and seconds."""
    s = int(np.rint(seconds))

    if s < 60:
        return "{0}s".format(s)
    elif s < 60 * 60:
        return "{0}m {1:02}s".format(s // 60, s % 60)
    elif s < 24 * 60 * 60:
        return "{0}h {1:02}m".format(s // (60 * 60), (s // 60) % 60)
    else:
        return "{0}d {1:02}h".format(s // (24 * 60 * 60), (s // (60 * 60)) % 24)


def tuple_product(t: Tuple) -> Any:
    """Calculate the product of the tuple elements."""
    result = 1

    for v in t:
        result *= v

    return result


_str_to_ctype = {
    "uint8": ctypes.c_ubyte,
    "uint16": ctypes.c_uint16,
    "uint32": ctypes.c_uint32,
    "uint64": ctypes.c_uint64,
    "int8": ctypes.c_byte,
    "int16": ctypes.c_int16,
    "int32": ctypes.c_int32,
    "int64": ctypes.c_int64,
    "float32": ctypes.c_float,
    "float64": ctypes.c_double,
}


def get_dtype_and_ctype(type_obj: Any) -> Tuple[np.dtype, Any]:
    """
    Given a type name string (or an object having a __name__ attribute), return
    matching Numpy and ctypes types that have the same size in bytes.
    """
    type_str = None

    if isinstance(type_obj, str):
        type_str = type_obj
    elif hasattr(type_obj, "__name__"):
        type_str = type_obj.__name__
    elif hasattr(type_obj, "name"):
        type_str = type_obj.name
    else:
        raise RuntimeError("Cannot infer type name from input")

    assert type_str in _str_to_ctype.keys()

    my_dtype = np.dtype(type_str)
    my_ctype = _str_to_ctype[type_str]

    assert my_dtype.itemsize == ctypes.sizeof(my_ctype)

    return my_dtype, my_ctype


def is_pickleable(obj: Any) -> bool:  # TODO remove
    try:
        with io.BytesIO() as stream:
            pickle.dump(obj, stream)
        return True
    except:
        return False


# Functionality to import modules/objects by name, and call functions by name
# -------------------------------------------------------------------------------------


def get_module_from_obj_name(obj_name: str) -> Tuple[types.ModuleType, str]:
    """
    Searches for the underlying module behind the name to some python object.
    Returns the module and the object name (original name with module part removed).
    """

    # allow convenience shorthands, substitute them by full names
    obj_name = re.sub("^np.", "numpy.", obj_name)
    obj_name = re.sub("^tf.", "tensorflow.", obj_name)

    # list alternatives for (module_name, local_obj_name)
    parts = obj_name.split(".")
    name_pairs = [
        (".".join(parts[:i]), ".".join(parts[i:])) for i in range(len(parts), 0, -1)
    ]

    # try each alternative in turn
    for module_name, local_obj_name in name_pairs:
        try:
            module = importlib.import_module(module_name)  # may raise ImportError
            get_obj_from_module(module, local_obj_name)  # may raise AttributeError
            return module, local_obj_name
        except:
            pass

    # maybe some of the modules themselves contain errors?
    for module_name, _local_obj_name in name_pairs:
        try:
            importlib.import_module(module_name)  # may raise ImportError
        except ImportError:
            if not str(sys.exc_info()[1]).startswith(
                "No module named '" + module_name + "'"
            ):
                raise

    # maybe the requested attribute is missing?
    for module_name, local_obj_name in name_pairs:
        try:
            module = importlib.import_module(module_name)  # may raise ImportError
            get_obj_from_module(module, local_obj_name)  # may raise AttributeError
        except ImportError:
            pass

    # we are out of luck, but we have no idea why
    raise ImportError(obj_name)


def get_obj_from_module(module: types.ModuleType, obj_name: str) -> Any:
    """
    Traverses the object name and returns the last (rightmost) python object.
    """
    if obj_name == "":
        return module
    obj = module
    for part in obj_name.split("."):
        obj = getattr(obj, part)
    return obj


def get_obj_by_name(name: str) -> Any:
    """
    Finds the python object with the given name.
    """
    module, obj_name = get_module_from_obj_name(name)
    return get_obj_from_module(module, obj_name)


def call_func_by_name(*args, func_name: str = None, **kwargs) -> Any:
    """
    Finds the python object with the given name and calls it as a function.
    """
    assert func_name is not None
    func_obj = get_obj_by_name(func_name)
    assert callable(func_obj)
    return func_obj(*args, **kwargs)


def construct_class_by_name(*args, class_name: str = None, **kwargs) -> Any:
    """
    Finds the python class with the given name and constructs it with the given
    arguments.
    """
    return call_func_by_name(*args, func_name=class_name, **kwargs)


def get_module_dir_by_obj_name(obj_name: str) -> str:
    """
    Get the directory path of the module containing the given object name.
    """
    module, _ = get_module_from_obj_name(obj_name)
    return os.path.dirname(inspect.getfile(module))


def is_top_level_function(obj: Any) -> bool:
    """
    Determine whether the given object is a top-level function, i.e., defined at module
    scope using 'def'.
    """
    return callable(obj) and obj.__name__ in sys.modules[obj.__module__].__dict__


def get_top_level_function_name(obj: Any) -> str:
    """
    Return the fully-qualified name of a top-level function.
    """
    assert is_top_level_function(obj)
    module = obj.__module__
    if module == "__main__":
        module = os.path.splitext(os.path.basename(sys.modules[module].__file__))[0]
    return module + "." + obj.__name__


# File system helpers
# ------------------------------------------------------------------------------------------


def list_dir_recursively_with_ignore(
    dir_path: str, ignores: List[str] = None, add_base_to_relative: bool = False
) -> List[Tuple[str, str]]:
    """
    List all files recursively in a given directory while ignoring given file and
    directory names. Returns list of tuples containing both absolute and relative paths.
    """
    assert os.path.isdir(dir_path)
    base_name = os.path.basename(os.path.normpath(dir_path))

    if ignores is None:
        ignores = []

    result = []

    for root, dirs, files in os.walk(dir_path, topdown=True):
        for ignore_ in ignores:
            dirs_to_remove = [d for d in dirs if fnmatch.fnmatch(d, ignore_)]

            # dirs need to be edited in-place
            for d in dirs_to_remove:
                dirs.remove(d)

            files = [f for f in files if not fnmatch.fnmatch(f, ignore_)]

        absolute_paths = [os.path.join(root, f) for f in files]
        relative_paths = [os.path.relpath(p, dir_path) for p in absolute_paths]

        if add_base_to_relative:
            relative_paths = [os.path.join(base_name, p) for p in relative_paths]

        assert len(absolute_paths) == len(relative_paths)
        result += zip(absolute_paths, relative_paths)

    return result


def copy_files_and_create_dirs(files: List[Tuple[str, str]]) -> None:
    """
    Takes in a list of tuples of (src, dst) paths and copies files.
    Will create all necessary directories.
    """
    for file in files:
        target_dir_name = os.path.dirname(file[1])

        # will create all intermediate-level directories
        if not os.path.exists(target_dir_name):
            os.makedirs(target_dir_name)

        shutil.copyfile(file[0], file[1])


# URL helpers
# ------------------------------------------------------------------------------------------


def is_url(obj: Any, allow_file_urls: bool = False) -> bool:
    """
    Determine whether the given object is a valid URL string.
    """
    if not isinstance(obj, str) or not "://" in obj:
        return False
    if allow_file_urls and obj.startswith("file://"):
        return True
    try:
        res = requests.compat.urlparse(obj)
        if not res.scheme or not res.netloc or not "." in res.netloc:
            return False
        res = requests.compat.urlparse(requests.compat.urljoin(obj, "/"))
        if not res.scheme or not res.netloc or not "." in res.netloc:
            return False
    except:
        return False
    return True


def open_url(
    url: str,
    cache_dir: str = None,
    num_attempts: int = 10,
    verbose: bool = True,
    return_filename: bool = False,
    cache: bool = True,
) -> Any:
    """
    Download the given URL and return a binary-mode file object to access the data.
    This code handles unusual file:// patterns that
    arise on Windows:

    file:///c:/foo.txt

    which would translate to a local '/c:/foo.txt' filename that's
    invalid.  Drop the forward slash for such pathnames.

    If you touch this code path, you should test it on both Linux and
    Windows.

    Some internet resources suggest using urllib.request.url2pathname() but
    but that converts forward slashes to backslashes and this causes
    its own set of problems.
    """
    assert num_attempts >= 1
    assert not (return_filename and (not cache))

    # Doesn't look like an URL scheme so interpret it as a local filename.
    if not re.match("^[a-z]+://", url):
        return url if return_filename else open(url, "rb")

    if url.startswith("file://"):
        filename = urllib.parse.urlparse(url).path
        if re.match(r"^/[a-zA-Z]:", filename):
            filename = filename[1:]
        return filename if return_filename else open(filename, "rb")

    assert is_url(url)

    # Lookup from cache.
    if cache_dir is None:
        cache_dir = make_cache_dir_path("downloads")

    url_md5 = hashlib.md5(url.encode("utf-8")).hexdigest()
    if cache:
        cache_files = glob.glob(os.path.join(cache_dir, url_md5 + "_*"))
        if len(cache_files) == 1:
            filename = cache_files[0]
            return filename if return_filename else open(filename, "rb")

    # Download.
    url_name = None
    url_data = None
    with requests.Session() as session:
        if verbose:
            print("Downloading %s ..." % url, end="", flush=True)
        for attempts_left in reversed(range(num_attempts)):
            try:
                with session.get(url) as res:
                    res.raise_for_status()
                    if len(res.content) == 0:
                        raise IOError("No data received")

                    if len(res.content) < 8192:
                        content_str = res.content.decode("utf-8")
                        if "download_warning" in res.headers.get("Set-Cookie", ""):
                            links = [
                                html.unescape(link)
                                for link in content_str.split('"')
                                if "export=download" in link
                            ]
                            if len(links) == 1:
                                url = requests.compat.urljoin(url, links[0])
                                raise IOError("Google Drive virus checker nag")
                        if "Google Drive - Quota exceeded" in content_str:
                            raise IOError(
                                "Google Drive download quota exceeded -- please try again later"
                            )

                    match = re.search(
                        r'filename="([^"]*)"',
                        res.headers.get("Content-Disposition", ""),
                    )
                    url_name = match[1] if match else url
                    url_data = res.content
                    if verbose:
                        print(" done")
                    break
            except KeyboardInterrupt:
                raise
            except:
                if not attempts_left:
                    if verbose:
                        print(" failed")
                    raise
                if verbose:
                    print(".", end="", flush=True)

    # Save to cache.
    if cache:
        safe_name = re.sub(r"[^0-9a-zA-Z-._]", "_", url_name)
        safe_name = safe_name[: min(len(safe_name), 128)]
        cache_file = os.path.join(cache_dir, url_md5 + "_" + safe_name)
        temp_file = os.path.join(
            cache_dir, "tmp_" + uuid.uuid4().hex + "_" + url_md5 + "_" + safe_name
        )
        os.makedirs(cache_dir, exist_ok=True)
        with open(temp_file, "wb") as f:
            f.write(url_data)
        os.replace(temp_file, cache_file)  # atomic
        if return_filename:
            return cache_file

    # Return data as file object.
    assert not return_filename
    return io.BytesIO(url_data)


# ----------------------------------------------------------------------------
# Cached construction of constant tensors. Avoids CPU=>GPU copy when the
# same constant is used multiple times.

_constant_cache = dict()


def constant(value, shape=None, dtype=None, device=None, memory_format=None):
    """Cached construction of constant tensors"""
    value = np.asarray(value)
    if shape is not None:
        shape = tuple(shape)
    if dtype is None:
        dtype = torch.get_default_dtype()
    if device is None:
        device = torch.device("cpu")
    if memory_format is None:
        memory_format = torch.contiguous_format

    key = (
        value.shape,
        value.dtype,
        value.tobytes(),
        shape,
        dtype,
        device,
        memory_format,
    )
    tensor = _constant_cache.get(key, None)
    if tensor is None:
        tensor = torch.as_tensor(value.copy(), dtype=dtype, device=device)
        if shape is not None:
            tensor, _ = torch.broadcast_tensors(tensor, torch.empty(shape))
        tensor = tensor.contiguous(memory_format=memory_format)
        _constant_cache[key] = tensor
    return tensor


# ----------------------------------------------------------------------------
# Replace NaN/Inf with specified numerical values.

try:
    nan_to_num = torch.nan_to_num  # 1.8.0a0
except AttributeError:

    def nan_to_num(
        input, nan=0.0, posinf=None, neginf=None, *, out=None
    ):  # pylint: disable=redefined-builtin
        """Replace NaN/Inf with specified numerical values"""
        assert isinstance(input, torch.Tensor)
        if posinf is None:
            posinf = torch.finfo(input.dtype).max
        if neginf is None:
            neginf = torch.finfo(input.dtype).min
        assert nan == 0
        return torch.clamp(
            input.unsqueeze(0).nansum(0), min=neginf, max=posinf, out=out
        )


# ----------------------------------------------------------------------------
# Symbolic assert.

try:
    symbolic_assert = torch._assert  # 1.8.0a0 # pylint: disable=protected-access
except AttributeError:
    symbolic_assert = torch.Assert  # 1.7.0

# ----------------------------------------------------------------------------
# Context manager to temporarily suppress known warnings in torch.jit.trace().
# Note: Cannot use catch_warnings because of https://bugs.python.org/issue29672


@contextlib.contextmanager
def suppress_tracer_warnings():
    """
    Context manager to temporarily suppress known warnings in torch.jit.trace().
    Note: Cannot use catch_warnings because of https://bugs.python.org/issue29672
    """
    flt = ("ignore", None, torch.jit.TracerWarning, None, 0)
    warnings.filters.insert(0, flt)
    yield
    warnings.filters.remove(flt)


# ----------------------------------------------------------------------------
# Assert that the shape of a tensor matches the given list of integers.
# None indicates that the size of a dimension is allowed to vary.
# Performs symbolic assertion when used in torch.jit.trace().


def assert_shape(tensor, ref_shape):
    """
    Assert that the shape of a tensor matches the given list of integers.
    None indicates that the size of a dimension is allowed to vary.
    Performs symbolic assertion when used in torch.jit.trace().
    """
    if tensor.ndim != len(ref_shape):
        raise AssertionError(
            f"Wrong number of dimensions: got {tensor.ndim}, expected {len(ref_shape)}"
        )
    for idx, (size, ref_size) in enumerate(zip(tensor.shape, ref_shape)):
        if ref_size is None:
            pass
        elif isinstance(ref_size, torch.Tensor):
            with suppress_tracer_warnings():  # as_tensor results are registered as constants
                symbolic_assert(
                    torch.equal(torch.as_tensor(size), ref_size),
                    f"Wrong size for dimension {idx}",
                )
        elif isinstance(size, torch.Tensor):
            with suppress_tracer_warnings():  # as_tensor results are registered as constants
                symbolic_assert(
                    torch.equal(size, torch.as_tensor(ref_size)),
                    f"Wrong size for dimension {idx}: expected {ref_size}",
                )
        elif size != ref_size:
            raise AssertionError(
                f"Wrong size for dimension {idx}: got {size}, expected {ref_size}"
            )


# ----------------------------------------------------------------------------
# Function decorator that calls torch.autograd.profiler.record_function().


def profiled_function(fn):
    """Function decorator that calls torch.autograd.profiler.record_function()."""

    def decorator(*args, **kwargs):
        with torch.autograd.profiler.record_function(fn.__name__):
            return fn(*args, **kwargs)

    decorator.__name__ = fn.__name__
    return decorator


# ----------------------------------------------------------------------------
# Sampler for torch.utils.data.DataLoader that loops over the dataset
# indefinitely, shuffling items as it goes.


class InfiniteSampler(torch.utils.data.Sampler):
    """
    Sampler for torch.utils.data.DataLoader that loops over the dataset
    indefinitely, shuffling items as it goes.
    """

    def __init__(
        self, dataset, rank=0, num_replicas=1, shuffle=True, seed=0, window_size=0.5
    ):
        assert len(dataset) > 0
        assert num_replicas > 0
        assert 0 <= rank < num_replicas
        assert 0 <= window_size <= 1
        super().__init__(dataset)
        self.dataset = dataset
        self.rank = rank
        self.num_replicas = num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.window_size = window_size

    def __iter__(self):
        order = np.arange(len(self.dataset))
        rnd = None
        window = 0
        if self.shuffle:
            rnd = np.random.RandomState(self.seed)
            rnd.shuffle(order)
            window = int(np.rint(order.size * self.window_size))

        idx = 0
        while True:
            i = idx % order.size
            if idx % self.num_replicas == self.rank:
                yield order[i]
            if window >= 2:
                j = (i - rnd.randint(window)) % order.size
                order[i], order[j] = order[j], order[i]
            idx += 1


# ----------------------------------------------------------------------------
# Utilities for operating with torch.nn.Module parameters and buffers.


def params_and_buffers(module):
    """Get parameters and buffers of a nn.Module"""
    assert isinstance(module, torch.nn.Module)
    return list(module.parameters()) + list(module.buffers())


def named_params_and_buffers(module):
    """Get named parameters and buffers of a nn.Module"""
    assert isinstance(module, torch.nn.Module)
    return list(module.named_parameters()) + list(module.named_buffers())


@torch.no_grad()
def copy_params_and_buffers(src_module, dst_module, require_all=False):
    """Copy parameters and buffers from a source module to target module"""
    assert isinstance(src_module, torch.nn.Module)
    assert isinstance(dst_module, torch.nn.Module)
    src_tensors = dict(named_params_and_buffers(src_module))
    for name, tensor in named_params_and_buffers(dst_module):
        assert (name in src_tensors) or (not require_all)
        if name in src_tensors:
            tensor.copy_(src_tensors[name])


# ----------------------------------------------------------------------------
# Context manager for easily enabling/disabling DistributedDataParallel
# synchronization.


@contextlib.contextmanager
def ddp_sync(module, sync):
    """
    Context manager for easily enabling/disabling DistributedDataParallel
    synchronization.
    """
    assert isinstance(module, torch.nn.Module)
    if sync or not isinstance(module, torch.nn.parallel.DistributedDataParallel):
        yield
    else:
        with module.no_sync():
            yield


# ----------------------------------------------------------------------------
# Check DistributedDataParallel consistency across processes.


def check_ddp_consistency(module, ignore_regex=None):
    """Check DistributedDataParallel consistency across processes."""
    assert isinstance(module, torch.nn.Module)
    for name, tensor in named_params_and_buffers(module):
        fullname = type(module).__name__ + "." + name
        if ignore_regex is not None and re.fullmatch(ignore_regex, fullname):
            continue
        tensor = tensor.detach()
        if tensor.is_floating_point():
            tensor = nan_to_num(tensor)
        other = tensor.clone()
        torch.distributed.broadcast(tensor=other, src=0)
        assert (tensor == other).all(), fullname


# ----------------------------------------------------------------------------
# Print summary table of module hierarchy.


def print_module_summary(module, inputs, max_nesting=3, skip_redundant=True):
    """Print summary table of module hierarchy."""
    assert isinstance(module, torch.nn.Module)
    assert not isinstance(module, torch.jit.ScriptModule)
    assert isinstance(inputs, (tuple, list))

    # Register hooks.
    entries = []
    nesting = [0]

    def pre_hook(_mod, _inputs):
        nesting[0] += 1

    def post_hook(mod, _inputs, outputs):
        nesting[0] -= 1
        if nesting[0] <= max_nesting:
            outputs = list(outputs) if isinstance(outputs, (tuple, list)) else [outputs]
            outputs = [t for t in outputs if isinstance(t, torch.Tensor)]
            entries.append(EasyDict(mod=mod, outputs=outputs))

    hooks = [mod.register_forward_pre_hook(pre_hook) for mod in module.modules()]
    hooks += [mod.register_forward_hook(post_hook) for mod in module.modules()]

    # Run module.
    outputs = module(*inputs)
    for hook in hooks:
        hook.remove()

    # Identify unique outputs, parameters, and buffers.
    tensors_seen = set()
    for e in entries:
        e.unique_params = [t for t in e.mod.parameters() if id(t) not in tensors_seen]
        e.unique_buffers = [t for t in e.mod.buffers() if id(t) not in tensors_seen]
        e.unique_outputs = [t for t in e.outputs if id(t) not in tensors_seen]
        tensors_seen |= {
            id(t) for t in e.unique_params + e.unique_buffers + e.unique_outputs
        }

    # Filter out redundant entries.
    if skip_redundant:
        entries = [
            e
            for e in entries
            if len(e.unique_params) or len(e.unique_buffers) or len(e.unique_outputs)
        ]

    # Construct table.
    rows = [
        [type(module).__name__, "Parameters", "Buffers", "Output shape", "Datatype"]
    ]
    rows += [["---"] * len(rows[0])]
    param_total = 0
    buffer_total = 0
    submodule_names = {mod: name for name, mod in module.named_modules()}
    for e in entries:
        name = "<top-level>" if e.mod is module else submodule_names[e.mod]
        param_size = sum(t.numel() for t in e.unique_params)
        buffer_size = sum(t.numel() for t in e.unique_buffers)
        output_shapes = [str(list(t.shape)) for t in e.outputs]
        output_dtypes = [str(t.dtype).split(".")[-1] for t in e.outputs]
        rows += [
            [
                name + (":0" if len(e.outputs) >= 2 else ""),
                str(param_size) if param_size else "-",
                str(buffer_size) if buffer_size else "-",
                (output_shapes + ["-"])[0],
                (output_dtypes + ["-"])[0],
            ]
        ]
        for idx in range(1, len(e.outputs)):
            rows += [
                [name + f":{idx}", "-", "-", output_shapes[idx], output_dtypes[idx]]
            ]
        param_total += param_size
        buffer_total += buffer_size
    rows += [["---"] * len(rows[0])]
    rows += [["Total", str(param_total), str(buffer_total), "-", "-"]]

    # Print table.
    widths = [max(len(cell) for cell in column) for column in zip(*rows)]
    print()
    for row in rows:
        print(
            "  ".join(
                cell + " " * (width - len(cell)) for cell, width in zip(row, widths)
            )
        )
    print()
    return outputs


# ----------------------------------------------------------------------------
