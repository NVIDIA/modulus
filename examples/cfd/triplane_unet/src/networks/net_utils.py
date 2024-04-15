import numpy as np
import torch
import torch.nn as nn


# A simple feedforward neural network
class MLP(torch.nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super().__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j + 1]))

            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j + 1]))

                self.layers.append(nonlinearity())

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)
        return x


class MLPBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        activation=nn.GELU,
    ):
        super().__init__()
        if hidden_channels is None:
            hidden_channels = in_channels
        if out_channels is None:
            out_channels = in_channels
        self.in_channels = in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.norm1 = nn.LayerNorm(hidden_channels)
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.shortcut = nn.Linear(in_channels, out_channels)
        self.activation = activation()

    def forward(self, x):
        out = self.activation(self.norm1(self.fc1(x)))
        out = self.norm2(self.fc2(out))
        # add skip connection
        out = self.activation(out + self.shortcut(x))
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, num_channels: int, data_range: float = 2):
        super().__init__()
        assert (
            num_channels % 2 == 0
        ), f"num_channels must be even for sin/cos, got {num_channels}"
        self.num_channels = num_channels
        self.data_range = data_range

    def forward(self, x):
        freqs = 2 ** torch.arange(
            start=0, end=self.num_channels // 2, device=x.device
        ).to(x.dtype)
        freqs = (2 * np.pi / self.data_range) * freqs
        x = x.unsqueeze(-1)
        # Make freq to have the same dimensions as x. X can be of any shape
        freqs = freqs.reshape((1,) * (len(x.shape) - 1) + freqs.shape)
        x = x * freqs
        x = torch.cat([x.cos(), x.sin()], dim=-1).flatten(start_dim=-2)
        return x


class AdaIN(nn.Module):
    def __init__(self, embed_dim, in_channels, mlp=None, eps=1e-5):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.eps = eps

        if mlp is None:
            mlp = nn.Sequential(
                nn.Linear(embed_dim, 512), nn.GELU(), nn.Linear(512, 2 * in_channels)
            )
        self.mlp = mlp

        self.embedding = None
        # to keep track of which device the nn.Module is on
        self.device_indicator_param = nn.Parameter(torch.empty(0))

    @property
    def device(self):
        """Returns the device that the model is on."""
        return self.device_indicator_param.device

    def update_embeddding(self, x):
        self.embedding = x.reshape(
            self.embed_dim,
        ).to(self.device)

    def forward(self, x):
        assert (
            self.embedding is not None
        ), "AdaIN: update embeddding before running forward"

        weight, bias = torch.split(
            self.mlp(self.embedding.to(self.device)), self.in_channels, dim=0
        )

        return nn.functional.group_norm(x, self.in_channels, weight, bias, eps=self.eps)
