import torch
from torch import Tensor
import torch.nn as nn
import dgl

try:
    from dgl import DGLGraph
except:
    raise ImportError(
        "Mesh Graph Net requires the DGL library. Install the "
        + "desired CUDA version at: \n https://www.dgl.ai/pages/start.html"
    )
from typing import Callable, Tuple, List, Union
from dataclasses import dataclass

import modulus
from modulus.models.meshgraphnet.meshgraphnet import MeshGraphNet
import torch_cluster
import torch_scatter






class Mesh_Reduced(torch.nn.Module):
    def __init__(
			    self, 
				input_dim_nodes: int, 
			    input_dim_edges: int,
                output_decode_dim: int,
			    output_encode_dim: int = 3,
			    num_layers_node_processor: int = 2,
                num_layers_edge_processor: int = 2,
				hidden_dim_processor: int = 128,
                hidden_dim_node_encoder: int = 128,
                num_layers_node_encoder: int = 2,
                hidden_dim_edge_encoder: int = 128,
                num_layers_edge_encoder: int = 2,
                hidden_dim_node_decoder: int = 128,
                num_layers_node_decoder: int = 2,
                k: int = 3,
				aggregation='mean'):
        super(Mesh_Reduced, self).__init__()
        self.encoder_processor = MeshGraphNet(input_dim_nodes, input_dim_edges, output_encode_dim, num_layers_node_processor, num_layers_edge_processor,
										hidden_dim_processor, hidden_dim_node_encoder, num_layers_node_encoder, hidden_dim_edge_encoder,
										num_layers_edge_encoder, hidden_dim_node_decoder,num_layers_node_decoder,aggregation)
        self.decoder_processor = MeshGraphNet(output_encode_dim, input_dim_edges, output_decode_dim, num_layers_node_processor, num_layers_edge_processor,
										hidden_dim_processor, hidden_dim_node_encoder, num_layers_node_encoder, hidden_dim_edge_encoder,
										num_layers_edge_encoder, hidden_dim_node_decoder,num_layers_node_decoder,aggregation)
        self.k = k
        self.PivotalNorm=torch.nn.LayerNorm(output_encode_dim)
		
	
    def knn_interpolate(self, x: torch.Tensor, pos_x: torch.Tensor, pos_y:torch.Tensor, 
                        batch_x: torch.Tensor = None, batch_y: torch.Tensor = None, k: int=3, num_workers: int=1):
        with torch.no_grad():
            assign_index = torch_cluster.knn(pos_x, pos_y, k, batch_x=batch_x, batch_y=batch_y,
                           num_workers=num_workers)
            y_idx, x_idx = assign_index[0], assign_index[1]
            diff = pos_x[x_idx] - pos_y[y_idx]
            squared_distance = (diff * diff).sum(dim=-1, keepdim=True)
            weights = 1.0 / torch.clamp(squared_distance, min=1e-16)

        y = torch_scatter.scatter(x[x_idx] * weights, y_idx, 0, pos_y.size(0), reduce='sum')
        y = y / torch_scatter.scatter(weights, y_idx, 0, pos_y.size(0), reduce='sum')

        return y
	
	
		
    def encode(self, x, graph, edge_features, Proj, position_mesh, position_pivotal):
        x = self.encoder_processor(x, edge_features, graph)
        x = self.PivotalNorm(x)
        
        if Proj is not None:
            x = torch.matmul(Proj, x)
        else:
            x = self.knn_interpolate(x=x,pos_x=position_mesh,pos_y=position_pivotal)
        
		
        return x

    def decode(self, x, graph, edge_features, Proj, position_mesh, position_pivotal):
        if Proj is not None:
            x = torch.matmul(Proj, x)
        else:
            x = self.knn_interpolate(x=x,pos_x=position_pivotal,pos_y=position_mesh)

     
        x = self.decoder_processor(x, edge_features, graph)
        return x