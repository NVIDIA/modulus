from enum import Enum
import numpy as np
import scipy.sparse


class GraphType(Enum):
    """
    Enumeration to define the types of graph representations.
    """

    FLAT_EDGE = 1
    ADJ_LIST = 2
    ADJ_MAT = 3


class Graph:
    def __init__(self, g, g_type, num_nodes):
        """
        Initialize the Graph object.

        Parameters:
        g (np.ndarray, list, or scipy.sparse.coo_matrix): The graph data in the specified format.
        g_type (GraphType): The type of the input graph representation.
        num_nodes (int): The number of nodes in the graph.
        """
        self.num_nodes = num_nodes

        if g_type == GraphType.FLAT_EDGE:
            self.flat_edges = g
        elif g_type == GraphType.ADJ_LIST:
            self.flat_edges = self.adj_list_to_flat_edge(g)
        elif g_type == GraphType.ADJ_MAT:
            self.flat_edges = self.adj_mat_to_flat_edge(g)
        else:
            raise ValueError(f"Unknown graph type: {g_type}")

        self.adj_list = self.get_adj_list()
        self.clusters = self.find_clusters()

    def get_flat_edge(self):
        """
        Get the flat edge representation of the graph.

        Returns:
        np.ndarray: The flat edge representation of the graph.
        """
        return self.flat_edges

    def get_adj_list(self):
        """
        Get the adjacency list representation of the graph.

        Returns:
        list: The adjacency list representation of the graph.
        """
        return self.flat_edge_to_adj_list(self.flat_edges, self.num_nodes)

    def get_sparse_adj_mat(self):
        """
        Get the sparse adjacency matrix representation of the graph.

        Returns:
        scipy.sparse.coo_matrix: The sparse adjacency matrix representation of the graph.
        """
        return self.flat_edge_to_adj_mat(self.flat_edges, self.num_nodes)

    def bfs_dist(self, seed):
        """
        Perform a Breadth-First Search (BFS) to find the shortest distance from the seed to all other nodes.

        Parameters:
        seed (int or list): The starting node(s) for BFS.

        Returns:
        np.ndarray: The shortest distance from the seed to all other nodes.
        """
        _INF = 1 + 1e10
        res = np.ones(self.num_nodes) * _INF
        visited = [False for _ in range(self.num_nodes)]
        if isinstance(seed, list):
            for s in seed:
                res[s] = 0
                visited[s] = True
            frontier = seed
        else:
            res[seed] = 0
            visited[seed] = True
            frontier = [seed]

        depth = 0
        track = [frontier]
        while frontier:
            this_level = frontier
            depth += 1
            frontier = []
            while this_level:
                f = this_level.pop(0)
                for n in self.adj_list[f]:
                    if not visited[n]:
                        visited[n] = True
                        frontier.append(n)
                        res[n] = depth
            track.append(frontier)

        return res

    def find_clusters(self):
        """
        Find connected clusters in the graph using BFS.

        Returns:
        list: A list of clusters, each cluster is a list of node indices.
        """
        _INF = 1 + 1e10
        remaining_nodes = list(range(self.num_nodes))
        clusters = []
        while remaining_nodes:
            if len(remaining_nodes) > 1:
                seed = remaining_nodes[0]
                dist = self.bfs_dist(seed)
                tmp = []
                new_remaining = []
                for n in remaining_nodes:
                    if dist[n] != _INF:
                        tmp.append(n)
                    else:
                        new_remaining.append(n)
                clusters.append(tmp)
                remaining_nodes = new_remaining
            else:
                clusters.append([remaining_nodes[0]])
                break

        return clusters

    @staticmethod
    def flat_edge_to_adj_mat(edge_list, n):
        """
        Convert a flat edge list to a sparse adjacency matrix.

        Parameters:
        edge_list (np.ndarray): The flat edge list of shape [2, num_edges].
        n (int): The number of nodes.

        Returns:
        scipy.sparse.coo_matrix: The sparse adjacency matrix.
        """
        adj_mat = scipy.sparse.coo_matrix((np.ones_like(edge_list[0]), (edge_list[0], edge_list[1])), shape=(n, n))
        return adj_mat

    @staticmethod
    def flat_edge_to_adj_list(edge_list, n):
        """
        Convert a flat edge list to an adjacency list.

        Parameters:
        edge_list (np.ndarray): The flat edge list of shape [2, num_edges].
        n (int): The number of nodes.

        Returns:
        list: The adjacency list.
        """
        adj_list = [[] for _ in range(n)]
        for i in range(len(edge_list[0])):
            adj_list[edge_list[0, i]].append(edge_list[1, i])
        return adj_list

    @staticmethod
    def adj_list_to_flat_edge(adj_list):
        """
        Convert an adjacency list to a flat edge list.

        Parameters:
        adj_list (list): The adjacency list.

        Returns:
        np.ndarray: The flat edge list of shape [2, num_edges].
        """
        edge_list = []
        for i in range(len(adj_list)):
            for n in adj_list[i]:
                edge_list.append([i, n])
        return np.array(edge_list).transpose()

    @staticmethod
    def adj_mat_to_flat_edge(adj_mat):
        """
        Convert a sparse adjacency matrix to a flat edge list.

        Parameters:
        adj_mat (np.ndarray or scipy.sparse.spmatrix): The sparse adjacency matrix.

        Returns:
        np.ndarray: The flat edge list of shape [2, num_edges].
        """
        if isinstance(adj_mat, np.ndarray):
            s, r = np.where(adj_mat.astype(bool))
        elif isinstance(adj_mat, scipy.sparse.coo_matrix):
            s, r = adj_mat.row, adj_mat.col
            dat = adj_mat.data
            valid = np.where(dat.astype(bool))[0]
            s, r = s[valid], r[valid]
        elif isinstance(adj_mat, scipy.sparse.csr_matrix):
            adj_mat = scipy.sparse.coo_matrix(adj_mat)
            s, r = adj_mat.row, adj_mat.col
            dat = adj_mat.data
            valid = np.where(dat.astype(bool))[0]
            s, r = s[valid], r[valid]
        else:
            raise ValueError(
                "Unsupported adjacency matrix type in adj_mat_to_flat_edge. Now only support numpy.ndarray, scipy.sparse.coo_matrix, scipy.sparse.csr_matrix."
            )
        return np.array([s, r])


if __name__ == "__main__":
    # Example flat edge list
    flat_edges = np.array([[0, 1, 2, 3, 4, 5], [1, 2, 0, 4, 5, 3]])
    num_nodes = 6

    # Initialize graph with flat edges
    graph = Graph(flat_edges, GraphType.FLAT_EDGE, num_nodes)

    # Get adjacency list
    adj_list = graph.get_adj_list()
    print("Adjacency List:", adj_list)
    # Get sparse adjacency matrix
    adj_mat = graph.get_sparse_adj_mat()
    print("Sparse Adjacency Matrix:\n", adj_mat.toarray())
    # Check the clusters
    clusters = graph.clusters
    print("Clusters:", clusters)

    # Use adj list to init graph
    graph_from_adj_list = Graph(adj_list, GraphType.ADJ_LIST, num_nodes)
    # Use adj mat to init graph
    graph_from_adj_mat = Graph(adj_mat, GraphType.ADJ_MAT, num_nodes)

    # Assert all three instances of Graph are equivalent
    assert np.array_equal(graph.get_flat_edge(), graph_from_adj_list.get_flat_edge())
    assert np.array_equal(graph.get_flat_edge(), graph_from_adj_mat.get_flat_edge())
