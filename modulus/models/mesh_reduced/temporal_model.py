import torch
from modulus.models.layers.transformer_decoder import (
    DecoderOnlyLayer,
    TransformerDecoder,
)
from modulus.models.meshgraphnet.meshgraphnet import MeshGraphNet
from modulus.models.gnn_layers.mesh_graph_mlp import MeshGraphMLP
from torch import nn
from torch.nn import LayerNorm
from torch import Tensor


class Sequence_Model(torch.nn.Module):
    """Decoder-only multi-head attention architecture
    Parameters
    ----------
    input_dim : int
        Number of latent features for the graph (#povital_position x output_decode_dim)
    input_context_dim: int
        Number of physical context features
    dropout_rate: float
        Dropout value for attention decoder, by default 2
    num_layers_decoder: int
        Number of sub-decoder-layers in the attention decoderm by default 3
    num_heads: int
        Number of heads in the attention decoder, by default 8
    dim_feedforward_scale: int
        The ration between the dimension of the feedforward network model and input_dim
    num_layers_context_encoder: int
        Number of MLP layers for the physical context feature encoder, by default 2
    num_layers_input_encoder: int
        Number of MLP layers for the input feature encoder, by default 2
    num_layers_output_encoder: int
        Number of MLP layers for the output feature encoder, by default 2
    activation: str
        Activation function of the attention decoder, can be 'relu' or 'gelu', by default 'gelu'
    Note
    ----
    Reference: Han, Xu, et al. "Predicting physics in mesh-reduced space with temporal attention."
    arXiv preprint arXiv:2201.09113 (2022).
    """

    def __init__(
        self,
        input_dim: int,
        input_context_dim: int,
        dist,
        dropout_rate: float = 0.0000,
        num_layers_decoder: int = 3,
        num_heads: int = 8,
        dim_feedforward_scale: int = 4,
        num_layers_context_encoder: int = 2,
        num_layers_input_encoder: int = 2,
        num_layers_output_encoder: int = 2,
        activation: str = "gelu",
    ):
        super().__init__()
        self.dist = dist
        decoder_layer = DecoderOnlyLayer(
            input_dim,
            num_heads,
            dim_feedforward_scale * input_dim,
            dropout_rate,
            activation,
            layer_norm_eps=1e-5,
            batch_first=True,
            norm_first=False,
            bias=True,
        )
        decoder_norm = LayerNorm(input_dim, eps=1e-5, bias=True)
        self.decoder = TransformerDecoder(
            decoder_layer, num_layers_decoder, decoder_norm
        )
        self.input_encoder = MeshGraphMLP(
            input_dim,
            output_dim=input_dim,
            hidden_dim=input_dim * 2,
            hidden_layers=num_layers_input_encoder,
            activation_fn=nn.ReLU(),
            norm_type="LayerNorm",
            recompute_activation=False,
        )
        self.output_encoder = MeshGraphMLP(
            input_dim,
            output_dim=input_dim,
            hidden_dim=input_dim * 2,
            hidden_layers=num_layers_output_encoder,
            activation_fn=nn.ReLU(),
            norm_type=None,
            recompute_activation=False,
        )
        self.context_encoder = MeshGraphMLP(
            input_context_dim,
            output_dim=input_dim,
            hidden_dim=input_dim * 2,
            hidden_layers=num_layers_context_encoder,
            activation_fn=nn.ReLU(),
            norm_type="LayerNorm",
            recompute_activation=False,
        )

    def forward(self, x, context=None):
        if context != None:
            context = self.context_encoder(context)
            x = torch.cat([context, x], dim=1)
        x = self.input_encoder(x)
        tgt_mask = self.generate_square_subsequent_mask(
            x.size()[1], device=self.dist.device
        )
        output = self.decoder(x, tgt_mask=tgt_mask)

        output = self.output_encoder(output)

        return output[:, 1:]

    @torch.no_grad()
    def sample(self, z0, step_size, context=None):
        z = z0  # .unsqueeze(1)

        for i in range(step_size):
            prediction = self.forward(z, context)[:, -1].unsqueeze(1)
            z = torch.concat([z, prediction], dim=1)
        return z

    @staticmethod
    def generate_square_subsequent_mask(
        sz: int,
        device: torch.device = torch.device(torch._C._get_default_device()),
        dtype: torch.dtype = torch.get_default_dtype(),
    ) -> Tensor:

        return torch.triu(
            torch.full((sz, sz), float("-inf"), dtype=dtype, device=device),
            diagonal=1,
        )
