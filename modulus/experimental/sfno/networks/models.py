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

from functools import partial

import torch
import torch.nn as nn

# preprocessor we need too
from modulus.experimental.sfno.networks.preprocessor import Preprocessor2D

_supported_models = ['fno', 'sfno', 'afno', 'afno:v1', 'debug']

def list_models():
    return _supported_models

class SingleStepWrapper(nn.Module):
    def __init__(self, params, model_handle):
        super(SingleStepWrapper, self).__init__()
        self.preprocessor = Preprocessor2D(params)
        self.model = model_handle()

    def forward(self, inp):
        
        # first append unpredicted features
        inpa = self.preprocessor.append_unpredicted_features(inp)

        # now normalize
        self.preprocessor.history_compute_stats(inpa)
        inpan = self.preprocessor.history_normalize(inpa, target=False)
        
        # now add static features if requested
        inpans = self.preprocessor.add_static_features(inpan)

        # forward pass
        yn = self.model(inpans)

        # undo normalization
        y = self.preprocessor.history_denormalize(yn, target=True)

        return y
        
    
class MultiStepWrapper(nn.Module):
    def __init__(self, params, model_handle):
        super(MultiStepWrapper, self).__init__()
        self.preprocessor = Preprocessor2D(params)
        self.model = model_handle()

        # collect parameters for history
        self.n_future = params.n_future

    def _forward_train(self, inp):

        result = []
        inpt = inp
        for step in range(self.n_future + 1):

            # add unpredicted features
            inpa = self.preprocessor.append_unpredicted_features(inpt)

            # do history normalization
            self.preprocessor.history_compute_stats(inpa)
            inpan = self.preprocessor.history_normalize(inpa, target=False)

            # add static features
            inpans = self.preprocessor.add_static_features(inpan)

            # prediction
            predn = self.model(inpans)

            # append the denormalized result to output list
            # important to do that here, otherwise normalization stats
            # will have been updated later:
            pred = self.preprocessor.history_denormalize(predn, target=True)
            result.append(pred)

            if (step == self.n_future):
                break
            
            # append history
            inpt = self.preprocessor.append_history(inpt, pred, step)
            
        # concat the tensors along channel dim to be compatible with flattened target
        result = torch.cat(result, dim=1)
        
        return result

    def _forward_eval(self, inp):
        # first append unpredicted features
        inpa = self.preprocessor.append_unpredicted_features(inp)

        # do history normalization
        self.preprocessor.history_compute_stats(inpa)
        inpan = self.preprocessor.history_normalize(inpa, target=False)

        # add static features
        inpans = self.preprocessor.add_static_features(inpan)
        
        # important, remove normalization here,
        # because otherwise normalization stats are already outdated
        yn = self.model(inpans)

        # important, remove normalization here,
        # because otherwise normalization stats are already outdated 
        y = self.preprocessor.history_denormalize(yn, target=True)
        
        return y

    def forward(self, inp):
        # decide which routine to call
        if self.training:
            y = self._forward_train(inp)
        else:
            y = self._forward_eval(inp)

        return y

    
def get_model(params):
    """
    Convenience routine that returns a model handle to construct the model.
    Unloads all the parameters in the params datastructure as a dict.
    This is to keep models as modular as possible, by not having them depend
    on the params datastructure.
    """

    model_handle = None

    # makani requires that these entries are set in params for now
    inp_shape = (params.img_crop_shape_x, params.img_crop_shape_y)
    out_shape = (params.out_shape_x, params.out_shape_y) if hasattr(params, "out_shape_x") and hasattr(params, "out_shape_y") else inp_shape
    inp_chans = params.N_in_channels
    out_chans = params.N_out_channels

    # choose the right model handle depending on specified architecture
    if params.nettype == 'fno' or params.nettype == 'sfno':
        if params.nettype == 'fno':
            params.spectral_transform = 'fft'
        else:
            params.spectral_transform = 'sht'

        from networks.sfnonet import SphericalFourierNeuralOperatorNet
        model_handle = partial(SphericalFourierNeuralOperatorNet,
                               inp_shape = inp_shape,
                               out_shape = out_shape,
                               inp_chans = inp_chans,
                               out_chans = out_chans,
                               **params.to_dict())

    elif params.nettype == 'afno' or params.nettype == "afno:v1":
        if params.nettype == 'afno':
            from networks.afnonet_v2 import AdaptiveFourierNeuralOperatorNet
        else:
            from networks.afnonet import AdaptiveFourierNeuralOperatorNet
        
        model_handle = partial(AdaptiveFourierNeuralOperatorNet,
                               inp_shape = inp_shape,
                               inp_chans = inp_chans,
                               out_chans = out_chans,
                               use_complex_kernels=True,
                               **params.to_dict())

    elif params.nettype == "debug":
        from networks.debug import DebugNet
        model_handle = DebugNet

    else:
         raise NotImplementedError(f"Error, net type {params.nettype} not implemented")

    # wrap into Multi-Step if requested
    if params.n_future > 0:
        model = MultiStepWrapper(params, model_handle)
    else:
        model = SingleStepWrapper(params, model_handle)

    return model
         
