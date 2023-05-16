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

"""Model packaging utilities

This module supports Packages stored in a directory structure.

This directory structure should contain a ``metadata.json`` file and data files
(e.g. model checkpoints) required to instantiate a model

The `metadata.json` file contains data necessary to use the model for forecasts::

    {
        "architecture": "sfno_73ch",
        "n_history": 0,
        "channel_set": "73var",
        "grid": "721x1440",
        "in_channels": [
            0,
            1
        ],
        "out_channels": [
            0,
            1
        ]
    }

Its schema is provided by the :py:class:`modulus.models.schema.Model`.

"""
import os

from modulus.models import schema
from modulus.utils import filesystem


METADATA = "metadata.json"


class Package:
    """A model package

    Simple file system operations and quick metadata access

    """

    def __init__(self, root: str, seperator: str):
        self.root = root
        self.seperator = seperator

    def get(self, path, recursive: bool = False):
        return filesystem.download_cached(self._fullpath(path), recursive=recursive)

    def _fullpath(self, path):
        return self.root + self.seperator + path

    def metadata(self) -> schema.Model:
        metadata_path = self._fullpath(METADATA)
        local_path = filesystem.download_cached(metadata_path)
        with open(local_path) as f:
            return schema.Model.parse_raw(f.read())
