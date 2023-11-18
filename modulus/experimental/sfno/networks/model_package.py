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

"""
Model package for easy inference/packaging. Model packages contain all the necessary data to 
perform inference and its interface is compatible with earth2mip
"""
import os
import shutil
import json
import numpy as np
import torch
from modulus.experimental.sfno.networks.models import get_model
from modulus.experimental.sfno.utils.YParams import ParamsBase
from modulus.experimental.sfno.third_party.climt.zenith_angle import cos_zenith_angle

import datetime

import logging

import warnings

try:
    import jsbeautifier
    use_jsbeautifier = True
except ImportError:
    warnings.warn('jsbeautifier is not installed. Please install it with "pip install jsbeautifier"')
    use_jsbeautifier = False

class LocalPackage:
    """
    Implements the earth2mip/modulus Package interface. 
    """
    def __init__(self, root):
        self.root = root
    
    def get(self, path):
        return os.path.join(self.root, path)


logger = logging.getLogger(__name__)

MODEL_PACKAGE_CHECKPOINT_PATH = "training_checkpoints/best_ckpt_mp0.tar"
MINS_FILE = "mins.npy"
MAXS_FILE = "maxs.npy"
MEANS_FILE = "global_means.npy"
STDS_FILE = "global_stds.npy"

class ModelWrapper(torch.nn.Module):
    """
    Model wrapper to make inference simple outside of makani.

    Attributes
    ----------
    model : torch.nn.Module
        ML model that is wrapped.
    params : ParamsBase
        parameter object containing information on how the model was initialized in makani

    Methods
    -------
    forward(x, time):
        performs a single prediction steps
    """

    def __init__(self, model, params):
        super().__init__()
        self.model = model
        self.params = params
        nlat = params.img_shape_x
        nlon = params.img_shape_y

        self.lats = 90 - 180 * np.arange(nlat) / (nlat-1)
        self.lons = 360 * np.arange(nlon) / nlon
        self.add_zenith = params.add_zenith

    def forward(self, x, time):

        if self.add_zenith:
            lon_grid, lat_grid = np.meshgrid(self.lons, self.lats)
            cosz = cos_zenith_angle(time, lon_grid, lat_grid)
            cosz = cosz.astype(np.float32)
            z = torch.from_numpy(cosz).to(device=x.device)
            while z.ndim != x.ndim:
                z = z[None]
            x = torch.cat([x, z], dim=1)
        
        return self.model(x)

def save_model_package(params):
    """
    Saves out a self-contained model-package.
    The idea is to save anything necessary for inference beyond the checkpoints in one location.
    """
    # save out the current state of the parameters, make it human readable
    config_path = os.path.join(params.experiment_dir, "config.json")

    msg = json.dumps(params.to_dict())
    if use_jsbeautifier:
        jsopts = jsbeautifier.default_options()
        jsopts.indent_size = 2

        msg = jsbeautifier.beautify(msg, jsopts)

    with open(config_path, "w") as f:
        f.write(msg)

    if hasattr(params, "add_orography") and params.add_orography:
        shutil.copy(params.orography_path, os.path.join(params.experiment_dir, "orography.nc"))
        
    if hasattr(params, "add_landmask") and params.add_landmask:
        shutil.copy(params.landmask_path, os.path.join(params.experiment_dir, "land_mask.nc"))

    # a bit hacky - we should change this to get the normalization from the dataloader.
    if hasattr(params, "global_means_path") and params.global_means_path is not None:
        shutil.copy(params.global_means_path, os.path.join(params.experiment_dir, MEANS_FILE))
    if hasattr(params, "global_stds_path") and params.global_stds_path is not None:
        shutil.copy(params.global_stds_path, os.path.join(params.experiment_dir, STDS_FILE))

    if params.normalization == 'minmax':
        if hasattr(params, "min_path") and params.min_path is not None:
            shutil.copy(params.min_path, os.path.join(params.experiment_dir, MINS_FILE))
        if hasattr(params, "max_path") and params.max_path is not None:
            shutil.copy(params.max_path, os.path.join(params.experiment_dir, MAXS_FILE))    

    # write out earth2mip metadata.json
    fcn_mip_data = {
        "entrypoint": {"name": "networks.model_package:load_time_loop"},
    }
    with open(os.path.join(params.experiment_dir, "metadata.json"), "w") as f:
        msg = json.dumps(fcn_mip_data)
        if use_jsbeautifier:
            msg = jsbeautifier.beautify(msg, jsopts)
        f.write(msg)


def _load_static_data(package, params):
    if hasattr(params, "add_orography") and params.add_orography:
        params.orography_path = package.get("orography.nc")
        
    if hasattr(params, "add_landmask") and params.add_landmask:
        params.landmask_path = package.get("land_mask.nc")

    # a bit hacky - we should change this to correctly
    if params.normalization == "zscore":
        if hasattr(params, "global_means_path") and params.global_means_path is not None:
            params.global_means_path = package.get(MEANS_FILE)
        if hasattr(params, "global_stds_path") and params.global_stds_path is not None:
            params.global_stds_path = package.get(STDS_FILE)
    elif params.normalization == "minmax":
        if hasattr(params, "min_path") and params.min_path is not None:
            params.min_path = package.get(MINS_FILE)
        if hasattr(params, "max_path") and params.max_path is not None:
            params.max_path = package.get(MAXS_FILE)
    else:
        raise ValueError("Unknown normalization mode.")


def load_model_package(package, pretrained=True, device='cpu'):
    """
    Loads model package and return the wrapper which can be used for inference.
    """
    path = package.get("config.json")
    params = ParamsBase.from_json(path)
    logger.info(str(params.to_dict()))
    _load_static_data(package, params)


    # assume we are not distributed
    # distributed checkpoints might be saved with different params values
    params.img_local_offset_x = 0
    params.img_local_offset_y = 0
    params.img_local_shape_x = params.img_shape_x
    params.img_local_shape_y = params.img_shape_y

    # get the model and 
    model = get_model(params).to(device)

    if pretrained:
        best_checkpoint_path = package.get(MODEL_PACKAGE_CHECKPOINT_PATH)
        # critical that this map_location be cpu, rather than the device to
        # avoid out of memory errors.
        checkpoint = torch.load(best_checkpoint_path, map_location=device)
        state_dict = checkpoint['model_state']
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, "module.")
        model.load_state_dict(state_dict, strict=True)

    model = ModelWrapper(model, params=params)

    # by default we want to do evaluation so setting it to eval here
    # 1-channel difference in training/eval mode
    model.eval()

    return model


def load_time_loop(package, device=None, time_step_hours=None):
    """This function loads an earth2mip TimeLoop object that 
    can be used for inference.

    A TimeLoop encapsulates normalization, regridding, and other logic, so is a
    very minimal interface to expose to a framework like earth2mip.

    See https://github.com/NVIDIA/earth2mip/blob/main/docs/concepts.rst
    for more info on this interface.
    """
    # put import here to make dependency on earth2mip optional
    # earth2mip can be installed following these instructions: 
    # https://gitlab-master.nvidia.com/modulus/earth-2/earth2-mip
    from earth2mip.networks import Inference
    from earth2mip.schema import Grid
    config = package.get("config.json")
    params = ParamsBase.from_json(config)

    if params.in_channels != params.out_channels:
        raise NotImplementedError(
            "Non-equal input and output channels are not implemented yet."
        )

    names = [params.channel_names[i] for i in params.in_channels]



    if params.normalization == 'minmax':
        min_path = package.get(MINS_FILE)
        max_path = package.get(MAXS_FILE)

        a = np.load(min_path)
        a = np.squeeze(a)[params.in_channels]

        b = np.load(max_path)
        b = np.squeeze(b)[params.in_channels]

        # work around to implement minmax scaling based with the earth2mip
        # Inference class below
        center = (a + b) / 2
        scale = (b - a) / 2
    else:
        center_path = package.get(MEANS_FILE)
        scale_path = package.get(STDS_FILE)

        center = np.load(center_path)
        center = np.squeeze(center)[params.in_channels]

        scale = np.load(scale_path)
        scale = np.squeeze(scale)[params.in_channels]

    model = load_model_package(package, pretrained=True, device=device)
    shape = (params.img_shape_x, params.img_shape_y)

    grid = None
    if  shape == (721, 1440):
        grid = Grid.grid_721x1440
    elif shape == (720, 1440):
        grid = Grid.grid_720x1440

    if time_step_hours is None:
        time_step_data = datetime.timedelta(hours=6)
        time_step = time_step_data * params.get("dt", 1)
    else:
        time_step  = datetime.timedelta(hours=time_step_hours)

    # Here we use the built-in class earth2mip.networks.Inference
    # will later be extended to use the makani inferencer 
    inference = Inference(
        model=model,
        channel_names=names,
        center=center,
        scale=scale,
        grid=grid,
        n_history=params.n_history,
        time_step=time_step,
    )
    inference.to(device)
    return inference

