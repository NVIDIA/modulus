import torch
import torch.nn as nn
# import torch.nn.functional as F
import pytest
import random

# from modulus.models.mlp import FullyConnected
# from modulus.models.fno import FNO
import modulus.models.layers as layers

class SpectralConv4d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3, modes4):
        super(SpectralConv4d, self).__init__()

        """
        4D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3
        self.modes4 = modes4

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))
        self.weights3 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))
        self.weights4 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))
        self.weights5 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))
        self.weights6 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))
        self.weights7 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))
        self.weights8 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3, self.modes4, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul4d(self, input, weights):
        # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
        return torch.einsum("bixyzt,ioxyzt->boxyzt", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-4,-3,-2,-1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-4), x.size(-3), x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)

        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3, :self.modes4] = self.compl_mul4d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3, :self.modes4], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3, :self.modes4] = self.compl_mul4d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3, :self.modes4], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3, :self.modes4] = self.compl_mul4d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3, :self.modes4], self.weights3)
        out_ft[:, :, :self.modes1, :self.modes2, -self.modes3:, :self.modes4] = self.compl_mul4d(x_ft[:, :, :self.modes1, :self.modes2, -self.modes3:, :self.modes4], self.weights4)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3, :self.modes4] = self.compl_mul4d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3, :self.modes4], self.weights5)
        out_ft[:, :, -self.modes1:, :self.modes2, -self.modes3:, :self.modes4] = self.compl_mul4d(x_ft[:, :, -self.modes1:, :self.modes2, -self.modes3:, :self.modes4], self.weights6)
        out_ft[:, :, :self.modes1, -self.modes2:, -self.modes3:, :self.modes4] = self.compl_mul4d(x_ft[:, :, :self.modes1, -self.modes2:, -self.modes3:, :self.modes4], self.weights7)
        out_ft[:, :, -self.modes1:, -self.modes2:, -self.modes3:, :self.modes4] = self.compl_mul4d(x_ft[:, :, -self.modes1:, -self.modes2:, -self.modes3:, :self.modes4], self.weights8)

        #Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-4), x.size(-3), x.size(-2), x.size(-1)))
        return x

# class Block4d(nn.Module):
#     def __init__(self, width, width2, modes1, modes2, modes3, modes4, out_dim):
#         super(Block4d, self).__init__()
#         self.modes1 = modes1
#         self.modes2 = modes2
#         self.modes3 = modes3
#         self.modes4 = modes4

#         self.width = width
#         self.width2 = width2
#         self.out_dim = out_dim
#         self.padding = 8

#         # channel
#         self.conv0 = SpectralConv4d(self.width, self.width, self.modes1, self.modes2, self.modes3, self.modes4)
#         self.conv1 = SpectralConv4d(self.width, self.width, self.modes1, self.modes2, self.modes3, self.modes4)
#         self.conv2 = SpectralConv4d(self.width, self.width, self.modes1, self.modes2, self.modes3, self.modes4)
#         self.conv3 = SpectralConv4d(self.width, self.width, self.modes1, self.modes2, self.modes3, self.modes4)
#         self.w0 = nn.Conv1d(self.width, self.width, 1)
#         self.w1 = nn.Conv1d(self.width, self.width, 1)
#         self.w2 = nn.Conv1d(self.width, self.width, 1)
#         self.w3 = nn.Conv1d(self.width, self.width, 1)
#         # self.fc1 = nn.Linear(self.width, self.width2)
#         # self.fc2 = nn.Linear(self.width2, self.out_dim)
#         self.decoder = FullyConnected(in_features=self.width,
#                                       out_features=self.out_dim,
#                                       num_layers=1,
#                                       layer_size=self.width2,
#                                     )

#     def forward(self, x):
#         batchsize = x.shape[0]
#         size_x, size_y, size_z, size_t = x.shape[2], x.shape[3], x.shape[4], x.shape[5]
# #         print(size_x, size_y, size_z, size_t)
#         # channel
# #         print(x.shape)
#         x1 = self.conv0(x)
# #         print(x1.shape)
#         x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z, size_t)
#         x = x1 + x2
#         x = F.gelu(x)

#         x1 = self.conv1(x)
#         x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z, size_t)
#         x = x1 + x2
#         x = F.gelu(x)

#         x1 = self.conv2(x)
#         x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z, size_t)
#         x = x1 + x2
#         x = F.gelu(x)

#         x1 = self.conv3(x)
#         x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y, size_z, size_t)
#         x = x1 + x2

#         x = x[:, :, self.padding:-self.padding, self.padding*2:-self.padding*2,
#               self.padding*2:-self.padding*2, self.padding:-self.padding]

#         x = x.permute(0, 2, 3, 4, 5, 1)  # pad the domain if input is non-periodic
#         x = self.decoder(x)
#         # x1 = self.fc1(x)
#         # x = F.gelu(x1)
#         # x = self.fc2(x)

#         return x

# class FNO4d(nn.Module):
#     def __init__(self, modes1, modes2, modes3, modes4, width, in_dim):
#         super(FNO4d, self).__init__()

#         self.modes1 = modes1
#         self.modes2 = modes2
#         self.modes3 = modes3
#         self.modes4 = modes4
#         self.width = width
#         self.width2 = width*4
#         self.in_dim = in_dim
#         self.out_dim = 1
#         self.padding = 8  # pad the domain if input is non-periodic

#         # self.fc0 = nn.Linear(self.in_dim, self.width)
#         # Initial lift network
#         self.lift_network = torch.nn.Sequential()
#         self.lift_network.append(
#             layers.ConvNdFCLayer(self.in_dim, int(self.width / 2))
#         )
#         self.lift_network.append(nn.GELU())
#         self.lift_network.append(
#             layers.ConvNdFCLayer(int(self.width / 2), self.width)
#         )
#         self.conv = Block4d(self.width, self.width2,
#                                self.modes1, self.modes2, self.modes3, self.modes4, self.out_dim)

#     def forward(self, x, gradient=False):
#         # x = self.fc0(x)
#         x = x.permute(0, 5, 1, 2, 3, 4)
#         x = self.lift_network(x)
#         x = F.pad(x, [self.padding, self.padding, self.padding*2, self.padding*2, self.padding*2,
#                       self.padding*2, self.padding, self.padding])

#         x = self.conv(x)

#         return x

@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_conv_nd(device, dimension):
    """compare output of ConvNdKernel1Layer with that of layer for specfic n_dim"""

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
            f'ConvNdKernel1Layer output not identical to that of layer specific for {dim}d fields :('


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_conv_ndfc(device, dimension):
    """compare output of ConvNdFCLayer with that of layer for specfic n_dim"""
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

    # initialise weights, biases
    torch.manual_seed(0)
    conv_nd.reset_parameters()
    torch.manual_seed(0)
    comp_nn.reset_parameters()
    with torch.no_grad():
        assert torch.allclose(conv_nd(invar), comp_nn(invar), rtol=1e-06, atol=1e-06), \
            f'ConvNdFCLayer output not identical to that of layer specific for {dim}d fields :('


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_spec_conv_4d(device):
    """compare output of SpectralConv4d with that of layer used in literature."""
    bsize = 8
    in_channels = 8
    out_channels = 4
    tens_size = 16
    fno_modes = 6

    torch.manual_seed(0)
    spec_conv_orig = SpectralConv4d(in_channels, out_channels,
                                    fno_modes, fno_modes, fno_modes, fno_modes).to(device)
    torch.manual_seed(0)
    spec_conv_modulus = layers.SpectralConv4d(in_channels, out_channels,
                                        fno_modes, fno_modes, fno_modes, fno_modes).to(device)

    invar = torch.randn(bsize, in_channels, tens_size, tens_size, tens_size, tens_size).to(device)
    with torch.no_grad():
        assert torch.allclose(spec_conv_orig(invar), spec_conv_modulus(invar), rtol=1e-06, atol=1e-06), \
            f'SpectralConv4d output not identical to that of refrence layer'


# # def initialise_parameters(model):
# #     """Reset layer weights"""
# #     if hasattr(model, 'bias'):
# #         nn.init.constant_(model.bias, 0)
# #     if hasattr(model, 'weight'):
# #         nn.init.constant_(model.weight, .25)
# #         # nn.init.xavier_uniform_(model.weight)

# # def reset_parameters(model):
# #     model.apply(initialise_parameters) # recursively apply initialisations


# @pytest.mark.parametrize("device", ["cuda:0", "cpu"])
# def test_4d_fno(device):
#     bsize = 4
#     in_channels = 8
#     out_channels = 4     # CAUTION is hard coded to 1 in in paper example
#     tens_size = 16
#     fno_modes = 4
#     latent_channels = 16

#     """Test FNO forward pass"""
#     torch.manual_seed(0)
#     # Construct FNO model
#     decoder = FullyConnected(
#         in_features=latent_channels,
#         out_features=out_channels,
#         num_layers=1,
#         layer_size=latent_channels*4,
#     )
#     mod_model = FNO(
#         decoder_net=decoder,
#         in_channels=in_channels,
#         dimension=4,
#         latent_channels=latent_channels,
#         num_fno_layers=4,
#         num_fno_modes=fno_modes,
#         padding=0,
#         coord_features=False
#     ).to(device)

#     ori_model = FNO4d(modes1=4,
#                       modes2=4,
#                       modes3=4,
#                       modes4=4,
#                       width=latent_channels, # number of latent features
#                       in_dim=in_channels
#     ).to(device)

#     invar = torch.randn(bsize, in_channels, tens_size, tens_size, tens_size, tens_size).to(device)

#     # torch.manual_seed(0)
#     # reset_parameters(mod_model)
#     # torch.manual_seed(0)
#     # reset_parameters(ori_model)

#     mod_res = mod_model(invar)
#     print(f'mod_res.size(): {mod_res.size()}')
#     ori_res = ori_model(invar.permute(0, 2, 3, 4, 5, 1)).permute(0, 5, 1, 2, 3, 4)
#     print(f'ori_res.size(): {ori_res.size()}')

#     print(torch.allclose(ori_res, mod_res, rtol=1e-06, atol=1e-06))

# if __name__ == "__main__":
#     # for dim in [1,2,3]:
#     #     test_conv_nd(device="cuda:0", dimension=dim)
#     #     test_conv_ndfc(device="cuda:0", dimension=dim)
#     # test_spec_conv_4d()
#     test_4d_fno("cuda:0")
