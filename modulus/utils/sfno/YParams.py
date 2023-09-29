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

import json
import logging

from ruamel.yaml import YAML


class ParamsBase:
    """Convenience wrapper around a dictionary. Allows referring to dictionary items
    as attributes, and tracking which attributes are modified.
    """

    def __init__(self):
        self._original_attrs = None
        self.params = {}
        self._original_attrs = list(self.__dict__)

    def __getitem__(self, key):
        return self.params[key]

    def __setitem__(self, key, val):
        self.params[key] = val
        self.__setattr__(key, val)

    def __contains__(self, key):
        return key in self.params

    def get(self, key, default=None):
        """Get a parameter value"""
        if hasattr(self, key):
            return getattr(self, key)
        else:
            return self.params.get(key, default)

    def to_dict(self):
        """Return a dictionary representation of the parameters"""
        new_attrs = {
            key: val
            for key, val in vars(self).items()
            if key not in self._original_attrs
        }
        return {**self.params, **new_attrs}

    @staticmethod
    def from_json(path: str) -> "ParamsBase":
        """Load parameters from a json file"""
        with open(path) as f:
            c = json.load(f)
        params = ParamsBase()
        params.update_params(c)
        return params

    def update_params(self, config):
        """Update parameters from a dictionary"""
        for key, val in config.items():
            if val == "None":
                val = None
            self.params[key] = val
            self.__setattr__(key, val)


class YParams(ParamsBase):
    """Open parameters stored with ``config_name`` in the yaml file ``yaml_filename``"""

    def __init__(self, yaml_filename, config_name, print_params=False):
        super().__init__()
        self._yaml_filename = yaml_filename
        self._config_name = config_name
        if print_params:
            print("------------------ Configuration ------------------")

        with open(yaml_filename) as _file:
            d = YAML().load(_file)[config_name]

        self.update_params(d)

        if print_params:
            for key, val in d.items():
                print(key, val)
            print("---------------------------------------------------")

    def log(self):
        """Log the parameters to the console"""
        logging.info("------------------ Configuration ------------------")
        logging.info("Configuration file: " + str(self._yaml_filename))
        logging.info("Configuration name: " + str(self._config_name))
        for key, val in self.to_dict().items():
            logging.info(str(key) + " " + str(val))
        logging.info("---------------------------------------------------")
