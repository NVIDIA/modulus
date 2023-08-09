# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import json
import numpy as np

from torch import Tensor
from sklearn.neighbors import NearestNeighbors
import logging

from .graph_utils import (
    cell_to_adj,
    create_graph,
    create_heterograph,
    add_edge_features,
    add_node_features,
    latlon2xyz,
    get_edge_len,
)

logger = logging.getLogger(__name__)


class Graph:
    """Graph class for creating the graph2mesh, multimesh, and mesh2graph graphs.

    Parameters
    ----------
    icospheres_path : str
        Path to the icospheres json file.
        If the file does not exist, it will try to generate it using PyMesh.
    lat_lon_grid : Tensor
        Tensor with shape (lat, lon, 2) that includes the latitudes and longitudes
        meshgrid.
    dtype : torch.dtype, optional
        Data type of the graph, by default torch.float
    """

    def __init__(
        self, icospheres_path: str, lat_lon_grid: Tensor, dtype=torch.float
    ) -> None:
        self.dtype = dtype
        # Get or generate the icospheres
        try:
            with open(icospheres_path, "r") as f:
                loaded_dict = json.load(f)
                icospheres = {
                    key: (np.array(value) if isinstance(value, list) else value)
                    for key, value in loaded_dict.items()
                }
                logger.info(f"Opened pre-computed graph at {icospheres_path}.")
        except:
            from modulus.utils.graphcast.icospheres import (
                generate_and_save_icospheres,
            )  # requires PyMesh

            logger.info(
                f"Could not open {icospheres_path}...generating mesh from scratch."
            )
            generate_and_save_icospheres()

        self.icospheres = icospheres
        self.max_order = (
            len([key for key in self.icospheres.keys() if "faces" in key]) - 2
        )

        # flatten lat/lon gird
        self.lat_lon_grid_flat = lat_lon_grid.permute(2, 0, 1).view(2, -1).permute(1, 0)

    def create_mesh_graph(self, verbose: bool = True) -> Tensor:
        """Create the multimesh graph.

        Parameters
        ----------
        verbose : bool, optional
            verbosity, by default True

        Returns
        -------
        DGLGraph
            Multimesh graph.
        """
        # create the bi-directional mesh graph
        multimesh_faces = self.icospheres["order_0_faces"]
        for i in range(1, self.max_order + 1):
            multimesh_faces = np.concatenate(
                (multimesh_faces, self.icospheres["order_" + str(i) + "_faces"])
            )

        src, dst = cell_to_adj(multimesh_faces)
        mesh_graph = create_graph(
            src, dst, to_bidirected=True, add_self_loop=False, dtype=torch.int32
        )
        mesh_pos = torch.tensor(
            self.icospheres["order_" + str(self.max_order) + "_vertices"],
            dtype=torch.float32,
        )
        mesh_graph = add_edge_features(mesh_graph, mesh_pos)
        mesh_graph = add_node_features(mesh_graph, mesh_pos)
        # ensure fields set to dtype to avoid later conversions
        mesh_graph.ndata["x"] = mesh_graph.ndata["x"].to(dtype=self.dtype)
        mesh_graph.edata["x"] = mesh_graph.edata["x"].to(dtype=self.dtype)
        if verbose:
            print("mesh graph:", mesh_graph)
        return mesh_graph

    def create_g2m_graph(self, verbose: bool = True) -> Tensor:
        """Create the graph2mesh graph.

        Parameters
        ----------
        verbose : bool, optional
            verbosity, by default True

        Returns
        -------
        DGLGraph
            Graph2mesh graph.
        """
        # get the max edge length of icosphere with max order
        edge_src = self.icospheres["order_" + str(self.max_order) + "_vertices"][
            self.icospheres["order_" + str(self.max_order) + "_faces"][:, 0]
        ]
        edge_dst = self.icospheres["order_" + str(self.max_order) + "_vertices"][
            self.icospheres["order_" + str(self.max_order) + "_faces"][:, 1]
        ]
        edge_len_1 = np.max(get_edge_len(edge_src, edge_dst))
        edge_src = self.icospheres["order_" + str(self.max_order) + "_vertices"][
            self.icospheres["order_" + str(self.max_order) + "_faces"][:, 0]
        ]
        edge_dst = self.icospheres["order_" + str(self.max_order) + "_vertices"][
            self.icospheres["order_" + str(self.max_order) + "_faces"][:, 2]
        ]
        edge_len_2 = np.max(get_edge_len(edge_src, edge_dst))
        edge_src = self.icospheres["order_" + str(self.max_order) + "_vertices"][
            self.icospheres["order_" + str(self.max_order) + "_faces"][:, 1]
        ]
        edge_dst = self.icospheres["order_" + str(self.max_order) + "_vertices"][
            self.icospheres["order_" + str(self.max_order) + "_faces"][:, 2]
        ]
        edge_len_3 = np.max(get_edge_len(edge_src, edge_dst))
        edge_len = max([edge_len_1, edge_len_2, edge_len_3])

        # create the grid2mesh bipartite graph
        cartesian_grid = latlon2xyz(self.lat_lon_grid_flat)
        n_nbrs = 4
        neighbors = NearestNeighbors(n_neighbors=n_nbrs).fit(
            self.icospheres["order_" + str(self.max_order) + "_vertices"]
        )
        distances, indices = neighbors.kneighbors(cartesian_grid)

        src, dst = [], []
        for i in range(len(cartesian_grid)):
            for j in range(n_nbrs):
                if distances[i][j] <= 0.6 * edge_len:
                    src.append(i)
                    dst.append(indices[i][j])
                    # NOTE this gives 1,624,344 edges, in the paper it is 1,618,746
                    # this number is very sensitive to the chosen edge_len, not clear
                    # in the paper what they use.

        g2m_graph = create_heterograph(
            src, dst, ("grid", "g2m", "mesh"), dtype=torch.int32
        )  # number of edges is 3,114,720, exactly matches with the paper
        g2m_graph.srcdata["pos"] = cartesian_grid.to(torch.float32)
        g2m_graph.dstdata["pos"] = torch.tensor(
            self.icospheres["order_" + str(self.max_order) + "_vertices"],
            dtype=torch.float32,
        )
        g2m_graph = add_edge_features(
            g2m_graph, (g2m_graph.srcdata["pos"], g2m_graph.dstdata["pos"])
        )
        # avoid potential conversions at later points
        g2m_graph.srcdata["pos"] = g2m_graph.srcdata["pos"].to(dtype=self.dtype)
        g2m_graph.dstdata["pos"] = g2m_graph.dstdata["pos"].to(dtype=self.dtype)
        g2m_graph.ndata["pos"]["grid"] = g2m_graph.ndata["pos"]["grid"].to(
            dtype=self.dtype
        )
        g2m_graph.ndata["pos"]["mesh"] = g2m_graph.ndata["pos"]["mesh"].to(
            dtype=self.dtype
        )
        g2m_graph.edata["x"] = g2m_graph.edata["x"].to(dtype=self.dtype)
        if verbose:
            print("g2m graph:", g2m_graph)
        return g2m_graph

    def create_m2g_graph(self, verbose: bool = True) -> Tensor:
        """Create the mesh2grid graph.

        Parameters
        ----------
        verbose : bool, optional
            verbosity, by default True

        Returns
        -------
        DGLGraph
            Mesh2grid graph.
        """
        # create the mesh2grid bipartite graph
        cartesian_grid = latlon2xyz(self.lat_lon_grid_flat)
        n_nbrs = 1
        neighbors = NearestNeighbors(n_neighbors=n_nbrs).fit(
            self.icospheres["order_" + str(self.max_order) + "_face_centroid"]
        )
        _, indices = neighbors.kneighbors(cartesian_grid)
        indices = indices.flatten()

        src = [
            p
            for i in indices
            for p in self.icospheres["order_" + str(self.max_order) + "_faces"][i]
        ]
        dst = [i for i in range(len(cartesian_grid)) for _ in range(3)]
        m2g_graph = create_heterograph(
            src, dst, ("mesh", "m2g", "grid"), dtype=torch.int32
        )  # number of edges is 3,114,720, exactly matches with the paper
        m2g_graph.srcdata["pos"] = torch.tensor(
            self.icospheres["order_" + str(self.max_order) + "_vertices"],
            dtype=torch.float32,
        )
        m2g_graph.dstdata["pos"] = cartesian_grid.to(dtype=torch.float32)
        m2g_graph = add_edge_features(
            m2g_graph, (m2g_graph.srcdata["pos"], m2g_graph.dstdata["pos"])
        )
        # avoid potential conversions at later points
        m2g_graph.srcdata["pos"] = m2g_graph.srcdata["pos"].to(dtype=self.dtype)
        m2g_graph.dstdata["pos"] = m2g_graph.dstdata["pos"].to(dtype=self.dtype)
        m2g_graph.ndata["pos"]["grid"] = m2g_graph.ndata["pos"]["grid"].to(
            dtype=self.dtype
        )
        m2g_graph.ndata["pos"]["mesh"] = m2g_graph.ndata["pos"]["mesh"].to(
            dtype=self.dtype
        )
        m2g_graph.edata["x"] = m2g_graph.edata["x"].to(dtype=self.dtype)

        if verbose:
            print("m2g graph:", m2g_graph)
        return m2g_graph
