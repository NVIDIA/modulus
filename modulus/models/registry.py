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
