import torch
import torch.nn as nn


class BaseModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.device_indicator_param = nn.Parameter(torch.empty(0))

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.device_indicator_param.device


class BaseModel(BaseModule):
    def data_dict_to_input(self, data_dict, **kwargs):
        """Convert data dictionary to appropriate input for the model."""
        raise NotImplementedError

    def loss_dict(self, data_dict, **kwargs):
        """Compute the loss dictionary for the model."""
        raise NotImplementedError

    @torch.no_grad()
    def eval_dict(self, data_dict, **kwargs):
        """Compute the evaluation dictionary for the model."""
        raise NotImplementedError
