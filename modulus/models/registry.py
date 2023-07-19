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

import pkg_resources
from typing import List, Union

# This model registry follows conventions similar to fsspec,
# https://github.com/fsspec/filesystem_spec/blob/master/fsspec/registry.py#L62C2-L62C2
# Tutorial on entrypoints: https://amir.rachum.com/blog/2017/07/28/python-entry-points/
class ModelRegistry:
    _instance = None
    _model_registry = {}

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(ModelRegistry, cls).__new__(cls)
            cls._model_registry = cls._construct_registry()
        return cls._instance

    @classmethod
    def _construct_registry(cls):
        registry = {}
        group = "modulus.models"
        entrypoints = pkg_resources.iter_entry_points(group)
        for entry_point in entrypoints:
            registry[entry_point.name] = entry_point
        return registry

    @classmethod
    def register(cls, model, name: Union[str, None] = None):
        # If no name provided, use the model's name
        if name is None:
            name = model.__name__

        # Check if name already in use
        if name in cls._model_registry:
            raise ValueError(f"Name {name} already in use")

        # Add this class to the dict of model registry
        cls._model_registry[name] = model

    @classmethod
    def list_models(cls) -> List[str]:
        return list(cls._model_registry.keys())

    @classmethod
    def __clear_model_registry(cls):
        # NOTE: This is only used for testing purposes
        cls._model_registry = {}

    @classmethod
    def __restore_model_registry(cls):
        # NOTE: This is only used for testing purposes
        cls._model_registry = cls._construct_registry()

registry = ModelRegistry()
