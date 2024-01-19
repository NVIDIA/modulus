import torch
from modulus.models.layers.transformer_decoder import DecoderOnlyLayer, TransformerDecoder
from modulus.models.meshgraphnet.meshgraphnet import MeshGraphNet
from modulus.models.gnn_layers.mesh_graph_mlp import MeshGraphMLP
from torch import nn
from torch.nn import LayerNorm
from torch import Tensor


class Sequence_Model(torch.nn.Module):
    def __init__(
			    self, 
				input_dim: int,
                input_content_dim: int,
                dist,
                context_length: int = 1,
                hidden_dim: int = 1024,
                dropout_rate: float = 0.0000,#0.1,
                n_blocks: int = 3,
                num_layers_decoder: int = 3,
                num_heads: int = 8,
                mask_length: int = 400,
                dim_feedforward_scale: int = 4,
                num_layers_content_encoder: int = 2,
                activation: str = 'gelu'):
        super().__init__()
        self.dist = dist
        decoder_layer = DecoderOnlyLayer(input_dim, num_heads, dim_feedforward_scale*input_dim, dropout_rate,
                                                    activation, layer_norm_eps = 1e-5, batch_first = True, norm_first = False,
                                                    bias = True)
        decoder_norm = LayerNorm(input_dim, eps = 1e-5, bias=True)
        self.decoder = TransformerDecoder(decoder_layer, num_layers_decoder, decoder_norm)
        self.input_encoder = MeshGraphMLP(
            input_dim,
            output_dim=input_dim,
            hidden_dim=input_dim*2,
            hidden_layers=num_layers_content_encoder,
            activation_fn=nn.ReLU(),
            norm_type="LayerNorm",
            recompute_activation=False,
        )
        self.output_encoder = MeshGraphMLP(
            input_dim,
            output_dim=input_dim,
            hidden_dim=input_dim*2,
            hidden_layers=num_layers_content_encoder,
            activation_fn=nn.ReLU(),
            norm_type=None,
            recompute_activation=False,
        )
        self.context_encoder = MeshGraphMLP(
            input_content_dim,
            output_dim=input_dim,
            hidden_dim=input_dim*2,
            hidden_layers=num_layers_content_encoder,
            activation_fn=nn.ReLU(),
            norm_type="LayerNorm",
            recompute_activation=False,
        )

    def forward(self, x, context = None):
        if context != None:
            context = self.context_encoder(context)
            x = torch.cat([context,x], dim=1)
        x= self.input_encoder(x)
        tgt_mask = self.generate_square_subsequent_mask(x.size()[1], device=self.dist.device)
        output = self.decoder(
			x,
			tgt_mask = tgt_mask
		)

        output = self.output_encoder(output)


        return output[:,1:]
    
    @torch.no_grad()
    def sample(self, z0, step_size, context = None):
        z = z0#.unsqueeze(1)
       
        

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
        torch.full((sz, sz), float('-inf'), dtype=dtype, device=device),
        diagonal=1,
        )


        


        
        
        
