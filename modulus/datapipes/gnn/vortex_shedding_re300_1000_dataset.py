from dgl.data import DGLDataset
import numpy as np
import os
import torch
import dgl
import pdb




class VortexSheddingRe300To1000Dataset(DGLDataset):
	def __init__(
		self,
		name="dataset",
		data_dir='./',
		sequence_ids=[2],
		verbose=False,
	):
		
		super().__init__(
				name=name,
				verbose=verbose,
			)
		self.data_dir = data_dir
		self.sequence_ids = sequence_ids
		self.rawData = np.load(os.path.join(self.data_dir,'rawData.npy'),
							   allow_pickle = True)
		
		# solution states are velocity and pressure
		self.solution_states = torch.from_numpy(self.rawData['x'][self.sequence_ids,:,:,:])

		# cell volume
		self.M = torch.from_numpy(self.rawData['mass'])

		# edge information
		self.E = torch.from_numpy(self.rawData['edge_attr'])

		# edge connection
		self.A = torch.from_numpy(self.rawData['edge_index']).type(torch.long)

		# sequence length
		self.sequence_len = self.solution_states.shape[1]
		self.sequence_num = self.solution_states.shape[0]
		self.num_nodes    = self.solution_states.shape[2]

		
	
	def __len__(self):
		return self.sequence_len * self.sequence_num
	
	def __getitem__(self, idx):
		sidx = idx // self.sequence_len
		tidx = idx % self.sequence_len
		
		node_features = torch.cat([self.solution_states[sidx,tidx],
								   self.M],dim=1)
		node_targets = self.solution_states[sidx,tidx]
		graph = dgl.graph((self.A[0], self.A[1]), num_nodes=self.num_nodes)
		graph.ndata["x"] = node_features
		graph.ndata["y"] = node_targets
		graph.edata["a"] = self.E
		return graph
	
if __name__ == '__main__':
	dloader = VortexSheddingRe300To1000Dataset()
	pdb.set_trace()