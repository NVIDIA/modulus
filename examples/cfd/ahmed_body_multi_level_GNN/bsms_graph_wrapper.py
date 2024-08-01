import numpy as np
from sparse_dot_mkl import dot_product_mkl
from graph_wrapper import Graph, GraphType

_INF = 1 + 1e10


class BistrideMultiLayerGraph:
    def __init__(self, flat_edge, num_layers, num_nodes, pos_mesh):
        """
        Initialize the BistrideMultiLayerGraph object.

        Parameters:
        flat_edge (np.ndarray): The flat edge list of shape [2, num_edges].
        num_layers (int): The number of layers to generate.
        num_nodes (int): The number of nodes in the graph.
        pos_mesh (np.ndarray): The positions of the nodes in the mesh.
        """
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        self.pos_mesh = pos_mesh

        # Initialize the first layer graph
        self.m_gs = [Graph(flat_edge, GraphType.FLAT_EDGE, num_nodes)]
        self.m_flat_es = [self.m_gs[0].get_flat_edge()]
        self.m_ids = []

        self.generate_multi_layer_graphs()

    def generate_multi_layer_graphs(self):
        """
        Generate multiple layers of graphs with pooling.
        """
        g_l = self.m_gs[0]
        pos_l = self.pos_mesh

        for l in range(self.num_layers):
            n_l = self.num_nodes if l == 0 else len(index_to_keep)
            index_to_keep, g_l = self.bstride_selection(g_l, pos_l, n_l)
            pos_l = pos_l[index_to_keep]
            self.m_gs.append(g_l)
            self.m_flat_es.append(g_l.get_flat_edge())
            self.m_ids.append(index_to_keep)

    def get_multi_layer_graphs(self):
        """
        Get the multi-layer graph structures.

        Returns:
        tuple: A tuple containing three lists:
            - m_gs (list): List of graph wrappers for each layer.
            - m_flat_es (list): List of flat edges for each layer.
            - m_ids (list): List of node indices to be pooled at each layer.
        """
        return self.m_gs, self.m_flat_es, self.m_ids

    @staticmethod
    def bstride_selection(g, pos_mesh, n):
        """
        Perform bstride selection to pool nodes and edges.

        Parameters:
        g (Graph): The graph wrapper object.
        pos_mesh (np.ndarray): The positions of the nodes in the mesh.
        n (int): The number of nodes.

        Returns:
        tuple: A tuple containing:
            - combined_idx_kept (list): List of node indices to be pooled.
            - new_g (Graph): The new graph wrapper object after pooling.
        """
        combined_idx_kept = set()
        adj_mat = g.get_sparse_adj_mat()
        adj_mat.setdiag(1)
        clusters = g.clusters

        seeds = BistrideMultiLayerGraph.nearest_center_seed(pos_mesh, clusters)

        for seed, c in zip(seeds, clusters):
            even, odd = set(), set()
            dist_from_central_node = g.bfs_dist(seed)

            for i, dist in enumerate(dist_from_central_node):
                if dist % 2 == 0 and dist != _INF:
                    even.add(i)
                elif dist % 2 == 1 and dist != _INF:
                    odd.add(i)

            if len(even) <= len(odd) or not odd:
                index_kept, index_rmvd = even, odd
            else:
                index_kept, index_rmvd = odd, even

            combined_idx_kept = combined_idx_kept.union(index_kept)

        combined_idx_kept = list(combined_idx_kept)
        combined_idx_kept.sort()
        adj_mat = adj_mat.tocsr().astype(float)
        adj_mat = dot_product_mkl(adj_mat, adj_mat)
        adj_mat.setdiag(0)
        new_g = BistrideMultiLayerGraph.pool_edge(adj_mat, n, combined_idx_kept)

        return combined_idx_kept, new_g

    @staticmethod
    def nearest_center_seed(pos_mesh, clusters):
        """
        Find the nearest center seed for each cluster.

        Parameters:
        pos_mesh (np.ndarray): The positions of the nodes in the mesh.
        clusters (list): List of clusters, each cluster is a list of node indices.

        Returns:
        list: List of seeds per cluster.
        """
        seeds = []
        for c in clusters:
            center = np.mean(pos_mesh[c], axis=0)
            delta_to_center = pos_mesh[c] - center[None, :]
            dist_to_center = np.linalg.norm(delta_to_center, 2, axis=-1)
            min_node = c[np.argmin(dist_to_center)]
            seeds.append(min_node)

        return seeds

    @staticmethod
    def pool_edge(adj_mat, num_nodes, idx):
        """
        Pool the edges based on the provided node indices.

        Parameters:
        adj_mat (scipy.sparse.csr_matrix): The adjacency matrix in CSR format.
        num_nodes (int): The number of nodes in the input graph.
        idx (list): List of node indices to be kept.

        Returns:
        Graph: The new graph wrapper object after pooling.
        """
        flat_e = Graph.adj_mat_to_flat_edge(adj_mat)
        idx = np.array(idx, dtype=np.int64)
        idx_new_valid = np.arange(len(idx)).astype(np.int64)
        idx_new_all = -1 * np.ones(num_nodes).astype(np.int64)
        idx_new_all[idx] = idx_new_valid
        new_flat_e = -1 * np.ones_like(flat_e).astype(np.int64)
        new_flat_e[0] = idx_new_all[flat_e[0]]
        new_flat_e[1] = idx_new_all[flat_e[1]]
        both_valid = np.logical_and(new_flat_e[0] >= 0, new_flat_e[1] >= 0)
        e_idx = np.where(both_valid)[0]
        new_flat_e = new_flat_e[:, e_idx]
        new_g = Graph(new_flat_e, GraphType.FLAT_EDGE, len(idx))

        return new_g





if __name__ == "__main__":
    # Example flat edge list
    flat_edges = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    flat_edges = np.concatenate((flat_edges, flat_edges[::-1]), axis=1)
    num_nodes = 11
    num_layers = 2
    pos_mesh_x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    pos_mesh_y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    pos_mesh_z = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    pos_mesh = np.vstack((pos_mesh_x, pos_mesh_y, pos_mesh_z)).T

    # Initialize BistrideMultiLayerGraph
    multi_layer_graph = BistrideMultiLayerGraph(flat_edges, num_layers, num_nodes, pos_mesh)

    # Get multi-layer graphs
    m_gs, m_flat_es, m_ids = multi_layer_graph.get_multi_layer_graphs()
    print("Multi-layer Graphs (graph wrappers):", m_gs)
    print("Multi-layer Graphs (flat edges):", m_flat_es)
    print("Multi-layer Graphs (node indices):", m_ids)
