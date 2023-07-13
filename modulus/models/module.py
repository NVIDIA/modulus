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

import os
import json
import torch
import logging
import inspect
import importlib
import entrypoints

from typing import Union
from pathlib import Path
import modulus
from modulus.models.meta import ModelMetaData
from modulus.registry import Registry


class Module(torch.nn.Module):
    """The base class for all network models in Modulus.

    This should be used as a direct replacement for torch.nn.module

    Parameters
    ----------
    meta : ModelMetaData, optional
        Meta data class for storing info regarding model, by default None
    """

    # Define a class attribute to store dynamically created classes
    _dynamically_created_classes = {}
    _file_extension = ".mdlus"
    __version__ = "0.1.0" # Used for file versioning and is not the same as modulus version

    def __new__(cls, *args, **kwargs):
        out = super().__new__(cls)
        sig = inspect.signature(cls.__init__)
        bound_args = sig.bind_partial(
            *([None] + list(args)), **kwargs
        )  # Add None to account for self
        bound_args.apply_defaults()
        bound_args.arguments.pop("self", None)

        out._args = {
            "__name__": cls.__name__,
            "__module__": cls.__module__,
            "__args__": {cls.__name__: {k: v for k, v in bound_args.arguments.items()}},
        }
        return out

    def __init__(self, meta=None):
        super().__init__()
        self.meta = meta
        self.register_buffer("device_buffer", torch.empty(0))
        self._setup_logger()

    def _setup_logger(self):
        self.logger = logging.getLogger("core.module")
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            f"[%(asctime)s - %(levelname)s] %(message)s", datefmt="%H:%M:%S"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.WARNING)

    @classmethod
    def instantiate(cls, arg_dict):
        _mod = importlib.import_module(arg_dict["__module__"])
        _cls_name = arg_dict["__name__"]

        # Add a check if the class is one of the dynamically created ones
        if _cls_name in cls._dynamically_created_classes:
            _cls = cls._dynamically_created_classes[_cls_name]
        else:
            _cls = getattr(_mod, _cls_name)

        cls_args = arg_dict["__args__"]
        all_args = {}
        for args in cls_args.values():
            all_args.update(args)

        return _cls(**all_args)

    def debug(self):
        """Turn on debug logging"""
        self.logger.handlers.clear()
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            f"[%(asctime)s - %(levelname)s - {self.meta.name}] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.DEBUG)
        # TODO: set up debug log
        # fh = logging.FileHandler(f'modulus-core-{self.meta.name}.log')

    def save(
        self, file_name: Union[str, None] = None, save_git_hash: bool = False
    ) -> None:
        """Simple utility for saving just the model

        Parameters
        ----------
        file_name : Union[str,None], optional
            File name to save model weight to. When none is provide it will default to
            the model's name set in the meta data, by default None
        save_git_hash : bool, optional
            Whether to save the git hash of the current commit, by default False

        Raises
        ------
        IOError
            If file_name provided has a parent path that does not exist
        """

        if file_name is not None and not file_name.endswith(self._file_extension):
            raise ValueError(
                f"File name must end with {self._file_extension} extension"
            )

        directory = Path(file_name)
        if not directory.parents[0].is_dir():
            raise IOError(
                f"Model checkpoint parent directory {directory.parents[0]} not found"
            )
        if not directory.is_dir():
            os.makedirs(directory)

        torch.save(self.state_dict(), directory.joinpath("model.pt"))

        with open(directory.joinpath("args.json"), "w") as f:
            json.dump(self._args, f)
            if file_name is None:
                file_name = self.meta.name + ".pt"

        # Save the modulus version and git hash (if available)
        metadata_info = {"modulus_version": modulus.__version__,
                         "mdlus_file_version": self.__version__}

        if save_git_hash:
            import git

            repo = git.Repo(search_parent_directories=True)
            try:
                metadata_info["git_hash"] = repo.head.object.hexsha
            except git.InvalidGitRepositoryError:
                metadata_info["git_hash"] = None

        with open(directory.joinpath("metadata.json"), "w") as f:
            json.dump(metadata_info, f)

    @staticmethod
    def _check_checkpoint(file_name: str) -> bool:
        if not file_name.endswith(Module._file_extension):
            raise ValueError(
                f"File name must end with {Module._file_extension} extension"
            )

        if file_name is None:
            raise ValueError("File name must be provided to load the model")

        directory = Path(file_name)
        if not directory.is_dir():
            raise IOError(f"Model directory {directory} not found")

        if not directory.joinpath("args.json").exists():
            raise IOError(
                f"Model checkpoint {directory.joinpath('model.json')} not found"
            )

        if not directory.joinpath("metadata.json").exists():
            raise IOError(
                f"Model checkpoint {directory.joinpath('metadata.json')} not found"
            )

        if not directory.joinpath("model.pt").exists():
            raise IOError(
                f"Model checkpoint {directory.joinpath('model.pt')} not found"
            )

        # Check if the checkpoint version is compatible with the current version
        with open(directory.joinpath("metadata.json"), "r") as f:
            metadata_info = json.load(f)
            if metadata_info['mdlus_file_version'] != Module.__version__:
                raise IOError(
                    f"Model checkpoint version {metadata_info['mdlus_file_version']} is not compatible with current version {Module.__version__}"
                )

        return directory

    def load(self, file_name: str) -> None:
        """Simple utility for loading the model weights from checkpoint

        Parameters
        ----------
        file_name : str
            Checkpoint file name

        Raises
        ------
        IOError
            If file_name provided does not exist or is not a valid checkpoint
        """

        directory = Module._check_checkpoint(file_name)

        model_dict = torch.load(
            directory.joinpath("model.pt"), map_location=self.device
        )
        self.load_state_dict(model_dict)

    @classmethod
    def from_checkpoint(cls, file_name: str) -> None:
        """Simple utility for constructing a model from a checkpoint

        Parameters
        ----------
        file_name : str
            Checkpoint file name

        Returns
        -------
        Module

        Raises
        ------
        IOError
            If file_name provided does not exist or is not a valid checkpoint
        """

        directory = Module._check_checkpoint(file_name)

        with open(directory.joinpath("args.json"), "r") as f:
            args = json.load(f)

        model = cls.instantiate(args)

        model_dict = torch.load(
            directory.joinpath("model.pt"), map_location=model.device
        )
        model.load_state_dict(model_dict)

        return model

    @staticmethod
    def from_torch(torch_model_class, meta=None):
        """Construct a Modulus module from a PyTorch module

        Parameters
        ----------
        torch_model_class : torch.nn.Module
            PyTorch module class
        meta : ModelMetaData, optional
            Meta data for the model, by default None

        Returns
        -------
        Module
        """

        # Define an internal class as before
        class ModulusModel(Module):
            def __init__(self, *args, **kwargs):
                super().__init__(meta=meta)
                self.inner_model = torch_model_class(*args, **kwargs)

            def forward(self, x):
                return self.inner_model(x)

        # Get the argument names and default values of the PyTorch model's init method
        init_argspec = inspect.getfullargspec(torch_model_class.__init__)
        model_argnames = init_argspec.args[1:]  # Exclude 'self'
        model_defaults = init_argspec.defaults or []
        defaults_dict = dict(
            zip(model_argnames[-len(model_defaults) :], model_defaults)
        )

        # Define the signature of new init
        params = [inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
        params += [
            inspect.Parameter(
                argname,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=defaults_dict.get(argname, inspect.Parameter.empty),
            )
            for argname in model_argnames
        ]
        init_signature = inspect.Signature(params)

        # Replace ModulusModel.__init__ signature with new init signature
        ModulusModel.__init__.__signature__ = init_signature

        # Generate a unique name for the created class
        new_class_name = f"{torch_model_class.__name__}ModulusModel"
        ModulusModel.__name__ = new_class_name

        # Add this class to the dict of dynamically created classes
        ModulusModel.register(ModulusModel, new_class_name)

        return ModulusModel

    @classmethod
    def register(cls, model, name: Union[str, None] = None):
        """
        Registers a model under a specific name.

        Parameters
        ----------
        model
            The model to be registered.
        name : Union[str,None], optional
            The name to register the model under. If none is provided, the model's name
        """

        # If no name provided, use the model's name
        if name is None:
            name = model.__name__

        # Check if name already in use
        if name in Module._dynamically_created_classes:
            raise ValueError(f"Name {name} already in use")

        # Add this class to the dict of dynamically created classes
        Module._dynamically_created_classes[name] = model

    @classmethod
    def factory(cls, name: str):
        """
        Returns a registered model given its name.

        Parameters
        ----------
        name : str
            The name of the registered model.

        Returns
        -------
        model
            The registered model.

        Raises
        ------
        KeyError
            If no model is registered under the provided name.
        """

        if name in Module._dynamically_created_classes:
            return Module._dynamically_created_classes[name]

        group = "modulus.models"
        try:
            entry_point = entrypoints.get_group_named(group)[name]
            model_class = entry_point.load()
            return model_class
        except KeyError:
            raise KeyError(f"No model is registered under the name {name}")

    @classmethod
    def _clear_dynamically_created_classes(cls):
        cls._dynamically_created_classes = {}

    @property
    def device(self) -> torch.device:
        """Get device model is on

        Returns
        -------
        torch.device
            PyTorch device
        """
        return self.device_buffer.device

    def num_parameters(self) -> int:
        """Gets the number of learnable parameters"""
        count = 0
        for name, param in self.named_parameters():
            count += param.numel()
        return count
