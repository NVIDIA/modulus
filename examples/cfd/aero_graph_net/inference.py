# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

from dgl import DGLGraph
from dgl.dataloading import GraphDataLoader

import hydra
from hydra.utils import to_absolute_path

import numpy as np
import pyvista as pv
import torch

from omegaconf import DictConfig

from modulus.datapipes.gnn.ahmed_body_dataset import AhmedBodyDataset
from modulus.launch.logging import PythonLogger
from modulus.launch.utils import load_checkpoint
from modulus.models.meshgraphnet import MeshGraphNet

from utils import relative_lp_error


def dgl_to_pyvista(graph: DGLGraph):
    """
    Converts a DGL graph to a PyVista graph.

    Parameters:
    -----------
    graph: DGLGraph
        The input DGL graph.

    Returns:
    --------
    pv_graph:
        The output PyVista graph.
    """

    # Convert the DGL graph to a NetworkX graph
    nx_graph = graph.to_networkx(
        node_attrs=["pos", "p_pred", "p", "s_pred", "wallShearStress"]
    ).to_undirected()

    # Initialize empty lists for storing data
    points = []
    lines = []
    p_pred = []
    s_pred = []
    p = []
    wallShearStress = []

    # Iterate over the nodes in the NetworkX graph
    for node, attributes in nx_graph.nodes(data=True):
        # Append the node and attribute data to the respective lists
        points.append(attributes["pos"].numpy())
        p_pred.append(attributes["p_pred"].numpy())
        s_pred.append(attributes["s_pred"].numpy())
        p.append(attributes["p"].numpy())
        wallShearStress.append(attributes["wallShearStress"].numpy())

    # Add edges to the lines list
    for edge in nx_graph.edges():
        lines.extend([2, edge[0], edge[1]])

    # Initialize a PyVista graph
    pv_graph = pv.PolyData()

    # Assign the points, lines, and attributes to the PyVista graph
    pv_graph.points = np.array(points)
    pv_graph.lines = np.array(lines)
    pv_graph.point_data["p_pred"] = np.array(p_pred)
    pv_graph.point_data["s_pred"] = np.array(s_pred)
    pv_graph.point_data["p"] = np.array(p)
    pv_graph.point_data["wallShearStress"] = np.array(wallShearStress)

    return pv_graph


class AhmedBodyRollout:
    """MGN inference on Ahmed Body dataset"""

    def __init__(self, cfg: DictConfig, logger: PythonLogger):
        # set device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        logger.info(f"Using {self.device} device")

        # instantiate dataset
        self.dataset = AhmedBodyDataset(
            name="ahmed_body_test",
            data_dir=to_absolute_path(cfg.data_dir),
            split="test",
            num_samples=cfg.num_test_samples,
            compute_drag=True,
        )

        # instantiate dataloader
        self.dataloader = GraphDataLoader(
            self.dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            drop_last=False,
        )

        # instantiate the model
        self.model = MeshGraphNet(
            cfg.input_dim_nodes,
            cfg.input_dim_edges,
            cfg.output_dim,
            aggregation=cfg.aggregation,
            hidden_dim_node_encoder=cfg.hidden_dim_node_encoder,
            hidden_dim_edge_encoder=cfg.hidden_dim_edge_encoder,
            hidden_dim_node_decoder=cfg.hidden_dim_node_decoder,
            mlp_activation_fn="silu" if cfg.recompute_activation else "relu",
            do_concat_trick=cfg.do_concat_trick,
            num_processor_checkpoint_segments=cfg.num_processor_checkpoint_segments,
            recompute_activation=cfg.recompute_activation,
        )
        self.model = self.model.to(self.device)

        # enable train mode
        self.model.eval()

        # load checkpoint
        load_checkpoint(
            to_absolute_path(cfg.ckpt_path),
            models=self.model,
            device=self.device,
        )

    def predict(self, save_results=False):
        """
        Run the prediction process.

        Parameters:
        -----------
        save_results: bool
            Whether to save the results in form of a .vtp file, by default False


        Returns:
        --------
        None
        """

        self.pred, self.exact, self.faces, self.graphs = [], [], [], []

        for i, (graph, sid, normals, areas, coeff) in enumerate(self.dataloader):
            graph = graph.to(self.device)
            normals = normals.to(self.device, torch.float32).squeeze()
            areas = areas.to(self.device, torch.float32).squeeze()
            coeff = coeff.to(self.device, torch.float32).squeeze()
            sid = sid.item()
            self.logger.info(f"Processing sample ID {sid}")
            pred = self.model(graph.ndata["x"], graph.edata["x"], graph).detach()

            gt = graph.ndata["y"]
            graph.ndata["p_pred"] = pred[:, 0]
            graph.ndata["s_pred"] = pred[:, 1:]
            graph.ndata["p"] = gt[:, 0]
            graph.ndata["wallShearStress"] = gt[:, 1:]

            error = relative_lp_error(pred, gt)
            self.logger.info(f"Test error (%): {error}")

            if save_results:
                # Convert DGL graph to PyVista graph and save it
                pv_graph = dgl_to_pyvista(graph.cpu())
                pv_graph.save(f"graph_{sid}.vtp")


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logger = PythonLogger("main")  # General python logger
    logger.file_logging()

    logger.info("Rollout started...")
    rollout = AhmedBodyRollout(cfg, logger)
    rollout.predict(save_results=True)


if __name__ == "__main__":
    main()
