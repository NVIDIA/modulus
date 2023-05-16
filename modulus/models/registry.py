"""Create-read-update-delete (CRUD) operations for the FCN model registry

The location of the registry is configured using `config.MODEL_REGISTRY`. Both
s3:// and local paths are supported.

The top-level structure of the registry is like this::

    afno_26ch_v/
    baseline_afno_26/
    gfno_26ch_sc3_layers8_tt64/
    hafno_baseline_26ch_edim512_mlp2/
    modulus_afno_20/
    sfno_73ch/
    tfno_no-patching_lr5e-4_full_epochs/


The name of the model is the folder name. Each of these folders has the
following structure::

    sfno_73ch/about.txt            # optional information (e.g. source path)
    sfno_73ch/global_means.npy
    sfno_73ch/global_stds.npy
    sfno_73ch/weights.tar          # model checkpoint
    sfno_73ch/metadata.json


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

Its schema is provided by the :py:class:`fcn_mip.schema.Model`.

The checkpoint file `weights.tar` should have a dictionary of model weights and
parameters in the `model_state` key. For backwards compatibility with FCN
checkpoints produced as of March 1, 2023 the keys should include prefixed
`module.` prefix. This checkpoint format may change in the future.


Scoring FCNs under active development
-------------------------------------

One can use fcn-mip to score models not packaged in fcn-mip using a metadata
file like this::

    {
        "architecture": "pickle",
        ...
    }

This will load ``weights.tar`` using `torch.load`. This is not recommended for
long-time archival of model checkpoints but does allow scoring models under
active development. Once a reasonable skill is achieved the model's source code
can be stabilized and packaged within fcn-mip for long-term archival.

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


class ModelRegistry:

    SEPERATOR: str = "/"

    def __init__(self, path: str):
        self.path = path

    def list_models(self):
        return [os.path.basename(f) for f in filesystem.ls(self.path)]

    def get_model(self, name: str):
        return Package(self.get_path(name), seperator=self.SEPERATOR)

    def get_path(self, name, *args):
        return self.SEPERATOR.join([self.path, name, *args])

    def get_model_path(self, name: str):
        return self.get_path(name)

    def get_weight_path(self, name: str):
        return self.get_path(name, "weights.tar")

    def get_scale_path(self, name: str):
        return self.get_path(name, "global_stds.npy")

    def get_center_path(self, name: str):
        return self.get_path(name, "global_means.npy")

    def put_metadata(self, name: str, metadata: schema.Model):
        metadata_path = self.get_path(name, METADATA)
        filesystem.pipe(metadata_path, metadata.json().encode())

    def get_metadata(self, name: str) -> schema.Model:
        return self.get_model(name).metadata()
