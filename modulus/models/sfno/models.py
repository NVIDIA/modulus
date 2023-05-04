import torch
import torch.nn as nn

from functools import partial
from torch.utils.checkpoint import checkpoint

from modulus.models.sfno.preprocessor import Preprocessor2D
from modulus.models.sfno.sfnonet import SphericalFourierNeuralOperatorNet

class MultiStepWrapper(nn.Module):
    def __init__(self, params, model_handle):
        super(MultiStepWrapper, self).__init__()
        self.preprocessor = Preprocessor2D(params)
        self.model = model_handle(params)

        # collect parameters for history
        self.n_future = params.n_future

    def _forward_train(self, inp):

        result = []
        inpt = inp
        for _ in range(self.n_future + 1):
            
            pred = self.model(inpt)

            result.append(pred)
            
            # postprocess: this steps removes the grid
            inpt = self.preprocessor.append_history(inpt, pred)
            
            # add back the grid
            inpt, _ = self.preprocessor(inpt)
            
        # concat the tensors along channel dim to be compatible with flattened target
        result = torch.cat(result, dim=1)
        
        return result

    def _forward_eval(self, inp):
        return self.model(inp)

    def forward(self, inp):

        if self.training:
            return self._forward_train(inp)
        else:
            return self._forward_eval(inp)

    
def get_model(params):

    model_handle = None

    if params.nettype == 'sfno':
        assert params.spectral_transform == 'sht'

        # use the Helmholtz decomposition
        
        model_handle = partial(SphericalFourierNeuralOperatorNet, use_complex_kernels=True)
    else:
         raise NotImplementedError(f"Error, net type {params.nettype} not implemented")

    # wrap into Multi-Step if requested
    if params.n_future > 0:
        model = MultiStepWrapper(params, model_handle)
    else:
        model = model_handle(params)


    return model
