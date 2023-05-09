import torch
import torch.nn.functional as F
from torch import nn

from modulus.models.sfno.activations import ComplexReLU, ComplexActivation

def test_ComplexReLU_cartesian():
    relu = ComplexReLU(mode="cartesian")
    z = torch.randn(2, 2, dtype=torch.cfloat)
    output = relu(z)
    assert torch.allclose(output.real, F.relu(z.real)) and torch.allclose(output.imag, F.relu(z.imag))

def test_ComplexReLU_real():
    relu = ComplexReLU(mode="real")
    z = torch.randn(2, 2, dtype=torch.cfloat)
    output = relu(z)
    assert torch.allclose(output.real, F.relu(z.real)) and torch.allclose(output.imag, z.imag)

def test_ComplexActivation_cartesian():
    activation = ComplexActivation(nn.ReLU(), mode="cartesian")
    z = torch.randn(2, 2, dtype=torch.cfloat)
    output = activation(z)
    assert torch.allclose(output.real, F.relu(z.real)) and torch.allclose(output.imag, F.relu(z.imag))

def test_ComplexActivation_modulus():
    activation = ComplexActivation(nn.ReLU(), mode="modulus")
    z = torch.randn(2, 2, dtype=torch.cfloat)
    output = activation(z)
    assert torch.allclose(output.abs(), F.relu(z.abs()))

