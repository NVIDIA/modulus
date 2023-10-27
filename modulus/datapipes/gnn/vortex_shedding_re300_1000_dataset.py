from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
import numpy as np
import os
import torch
import dgl
import pdb
from utils import read_vtp_file, save_json, load_json



class VortexSheddingRe300To1000Dataset(DGLDataset):
	def __init__(
		self,
		name="dataset",
		data_dir='./',
		split = 'train',
		sequence_ids=[2],
		verbose=False
	):
		
		super().__init__(
				name=name,
				verbose=verbose,
			)
		self.data_dir = data_dir
		self.sequence_ids = sequence_ids
		self.split = split
		self.rawData = np.load(os.path.join(self.data_dir,'rawData.npy'),
							   allow_pickle = True)
		
		# solution states are velocity and pressure
		self.solution_states = torch.from_numpy(self.rawData['x'][self.sequence_ids,:,:,:])

		# cell volume
		#self.M = torch.from_numpy(self.rawData['mass'])

		# edge information
		self.E = torch.from_numpy(self.rawData['edge_attr'])

		# edge connection
		self.A = torch.from_numpy(self.rawData['edge_index']).type(torch.long)

		# sequence length
		self.sequence_len = self.solution_states.shape[1]
		self.sequence_num = self.solution_states.shape[0]
		self.num_nodes    = self.solution_states.shape[2]

		if self.split == "train":
			self.edge_stats = self._get_edge_stats()
		else:
			self.edge_stats = load_json("edge_stats.json")
			
		if self.split == "train":
			self.node_stats = self._get_node_stats()
		else:
			self.node_stats = load_json("node_stats.json")

		# handle the normalization
		for i in range(self.sequence_num):
			for j in range(self.sequence_len):	
				self.solution_states[i,j] = self.normalize(self.solution_states[i,j], 
					                                       self.node_stats["node_mean"], 
														   self.node_stats["node_std"])
		self.E = self.normalize(self.E,
			                    self.edge_stats["edge_mean"], 
		 						self.edge_stats["edge_std"])
	
	def __len__(self):
		return self.sequence_len * self.sequence_num
	
	def __getitem__(self, idx):
		sidx = idx // self.sequence_len
		tidx = idx % self.sequence_len
		
		# node_features = torch.cat([self.solution_states[sidx,tidx],
		# 						   self.M],dim=1)
		node_features = self.solution_states[sidx,tidx]
		node_targets = self.solution_states[sidx,tidx]
		graph = dgl.graph((self.A[0], self.A[1]), num_nodes=self.num_nodes)
		graph.ndata["x"] = node_features
		graph.ndata["y"] = node_targets
		graph.edata["a"] = self.E
		return graph
	

	def _get_edge_stats(self):
		stats = {
			"edge_mean": self.E.mean(dim=0),
			"edge_std": self.E.std(dim=0),
		}
		save_json(stats, "edge_stats.json")
		return stats



	def _get_node_stats(self):
		stats = {
			"node_mean": self.solution_states.mean(dim=[0,1,2]),
			"node_std": self.solution_states.std(dim=[0,1,2]),
		}
		save_json(stats, "edge_stats.json")
		return stats




	@staticmethod
	def normalize(invar, mu, std):
		"""normalizes a tensor"""
		assert invar.size()[-1] == mu.size()[-1]
		assert invar.size()[-1] == std.size()[-1]
		return (invar - mu.expand(invar.size())) / std.expand(invar.size())
	
	@staticmethod
	def denormalize(invar, mu, std):
		"""denormalizes a tensor"""
		# assert invar.size()[-1] == mu.size()[-1]
		# assert invar.size()[-1] == std.size()[-1]
		denormalized_invar = invar * std + mu
		return denormalized_invar
	
if __name__ == '__main__':
	data_set = VortexSheddingRe300To1000Dataset(split = 'train')
	data_loader = GraphDataLoader(data_set,
								  batch_size=5,
								  shuffle=True)
	for batched_graph in data_loader:
		pdb.set_trace()
