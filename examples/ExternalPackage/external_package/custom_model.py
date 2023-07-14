import torch

from modulus.models import Module

class CustomModel(Module):
    def __init__(self, layer_size=16):
        super().__init__()
        self.layer_size = layer_size
        self.layer = torch.nn.Linear(layer_size, layer_size)

    def forward(self, x):
        return self.layer(x)
