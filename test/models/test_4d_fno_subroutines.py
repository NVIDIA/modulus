import torch
import torch.nn as nn
import pytest
import random

from modulus.models.mlp import FullyConnected
from modulus.models.fno import FNO
import modulus.models.layers as layers
# import common


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_conv_nd(device, dimension):
    device = "cuda:0"
    bsize = 8
    in_channels = 4
    out_channels = 2
    tens_size = 16

    conv_nd = layers.ConvNdKernel1Layer(in_channels, out_channels).to(device)

    ini_w, ini_b = random.uniform(0, 1), random.uniform(0, 1)
    if dimension == 1:
        invar = torch.randn(bsize, in_channels, tens_size).to(device)
        comp_nn = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True).to(device)
    elif dimension == 2:
        invar = torch.randn(bsize, in_channels, tens_size, tens_size).to(device)
        comp_nn = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True).to(device)
    elif dimension == 3:
        invar = torch.randn(bsize, in_channels, tens_size, tens_size, tens_size).to(device)
        comp_nn = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=True).to(device)

    nn.init.constant_(conv_nd.conv.bias, ini_b)
    nn.init.constant_(conv_nd.conv.weight, ini_w)
    nn.init.constant_(comp_nn.bias, ini_b)
    nn.init.constant_(comp_nn.weight, ini_w)
    with torch.no_grad():
        assert torch.allclose(conv_nd(invar), comp_nn(invar), rtol=1e-06, atol=1e-06), \
            f'failed for {dim}d case :('


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_conv_4dfc(device, dimension):
    device = "cuda:0"
    bsize = 8
    in_channels = 4
    out_channels = 2
    tens_size = 16

    conv_nd = layers.ConvNdFCLayer(in_channels, out_channels).to(device)

    if dimension == 1:
        invar = torch.randn(bsize, in_channels, tens_size).to(device)
        comp_nn = layers.Conv1dFCLayer(in_channels, out_channels).to(device)
    elif dimension == 2:
        invar = torch.randn(bsize, in_channels, tens_size, tens_size).to(device)
        comp_nn = layers.Conv2dFCLayer(in_channels, out_channels).to(device)
    elif dimension == 3:
        invar = torch.randn(bsize, in_channels, tens_size, tens_size, tens_size).to(device)
        comp_nn = layers.Conv3dFCLayer(in_channels, out_channels).to(device)

    # initialise weights, biases are already set to 0
    torch.manual_seed(0)
    conv_nd.reset_parameters()
    torch.manual_seed(0)
    comp_nn.reset_parameters()
    with torch.no_grad():
        assert torch.allclose(conv_nd(invar), comp_nn(invar), rtol=1e-06, atol=1e-06), \
            f'failed for {dim}d case :('


if __name__ == "__main__":
    for dim in [1,2,3]:
        test_conv_nd(device="cuda:0", dimension=dim)
        test_conv_4dfc(device="cuda:0", dimension=dim)
