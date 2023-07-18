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
import re
from typing import List, Tuple, Dict, Union, Optional, Any

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from dataclasses import dataclass

from .utils import read_vtp_file, save_json, load_json
from modulus.datapipes.datapipe import Datapipe
from modulus.datapipes.meta import DatapipeMetaData

try:
    import dgl
    from dgl.data import DGLDataset
except:
    raise ImportError(
        "Ahmed Body Dataset requires the DGL library. Install the "
        + "desired CUDA version at: \n https://www.dgl.ai/pages/start.html"
    )

try:
    import vtk
    import pyvista as pv
except:
    raise ImportError(
        "Ahmed Body Dataset requires the vtk and pyvista libraries. Install with "
        + "pip install vtk pyvista"
    )


@dataclass
class MetaData(DatapipeMetaData):
    name: str = "AhmedBody"
    # Optimization
    auto_device: bool = True
    cuda_graphs: bool = False
    # Parallel
    ddp_sharding: bool = True


class AhmedBodyDataset(DGLDataset, Datapipe):
    """
    In-memory Ahmed body Dataset

    Parameters
    ----------
    data_dir: str
        The directory where the data is stored.
    split: str, optional
        The dataset split. Can be 'train', 'validation', or 'test', by default 'train'.
    num_samples: int, optional
        The number of samples to use, by default 10.
    invar_keys: List[str], optional
        The input node features to consider. Default includes 'pos', 'velocity', 'reynolds_number', 'length', 'width', 'height', 'ground_clearance', 'slant_angle', and 'fillet_radius'.
    outvar_keys: List[str], optional
        The output features to consider. Default includes 'p' and 'wallShearStress'.
    normalize_keys List[str], optional
        The features to normalize. Default includes 'p', 'wallShearStress', 'velocity', 'length', 'width', 'height', 'ground_clearance', 'slant_angle', and 'fillet_radius'.
    normalization_bound: Tuple[float, float], optional
        The lower and upper bounds for normalization. Default is (-1, 1).
    force_reload: bool, optional
        If True, forces a reload of the data, by default False.
    name: str, optional
        The name of the dataset, by default 'dataset'.
    verbose: bool, optional
        If True, enables verbose mode, by default False.
    compute_drag: bool, optional
        If True, also returns the coefficient and mesh area and normals that are required for computing the drag coefficient.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        num_samples: int = 10,
        invar_keys: List[str] = [
            "pos",
            "velocity",
            "reynolds_number",
            "length",
            "width",
            "height",
            "ground_clearance",
            "slant_angle",
            "fillet_radius",
        ],
        outvar_keys: List[str] = ["p", "wallShearStress"],
        normalize_keys: List[str] = [
            "p",
            "wallShearStress",
            "velocity",
            "reynolds_number",
            "length",
            "width",
            "height",
            "ground_clearance",
            "slant_angle",
            "fillet_radius",
        ],
        normalization_bound: Tuple[float, float] = (-1.0, 1.0),
        force_reload: bool = False,
        name: str = "dataset",
        verbose: bool = False,
        compute_drag: bool = False,
    ):
        DGLDataset.__init__(
            self,
            name=name,
            force_reload=force_reload,
            verbose=verbose,
        )
        Datapipe.__init__(
            self,
            meta=MetaData(),
        )
        self.split = split
        self.num_samples = num_samples
        self.data_dir = os.path.join(data_dir, self.split)
        self.info_dir = os.path.join(data_dir, self.split + "_info")
        self.input_keys = invar_keys
        self.output_keys = outvar_keys
        self.normalization_bound = normalization_bound
        self.compute_drag = compute_drag

        # get the list of all files in the data_dir
        all_entries = os.listdir(self.data_dir)
        all_info = os.listdir(self.info_dir)

        data_list = [
            os.path.join(self.data_dir, entry)
            for entry in all_entries
            if os.path.isfile(os.path.join(self.data_dir, entry))
        ]
        info_list = [
            os.path.join(self.info_dir, entry)
            for entry in all_info
            if os.path.isfile(os.path.join(self.info_dir, entry))
        ]

        numbers = []
        for directory in data_list:
            match = re.search(r"\d+", directory)
            if match:
                numbers.append(int(match.group()))
        numbers_info = []
        for directory in info_list:
            match = re.search(r"\d+", directory)
            if match:
                numbers_info.append(int(match.group()))
        numbers = [int(n) for n in numbers]
        numbers_info = [int(n) for n in numbers_info]

        # sort the data_list and info_list according to the numbers
        args = np.argsort(numbers)
        data_list = [data_list[index] for index in args]
        numbers = [numbers[index] for index in args]
        args = np.argsort(numbers_info)
        info_list = [info_list[index] for index in args]
        numbers_info = [numbers_info[index] for index in args]

        assert sorted(numbers) == sorted(numbers_info)
        self.numbers = numbers

        # create the graphs and add the node and features
        self.length = min(len(data_list), self.num_samples)

        if self.num_samples > self.length:
            raise ValueError(
                f"Number of available {self.split} dataset entries "
                f"({self.length}) is less than the number of samples "
                f"({self.num_samples})"
            )

        self.graphs = []
        if self.compute_drag:
            self.normals = []
            self.areas = []
            self.coeff = []
        for i in range(self.length):
            file_path = data_list[i]
            info_path = info_list[i]
            polydata = read_vtp_file(file_path)
            graph = self._create_dgl_graph(polydata, outvar_keys, dtype=torch.int32)
            (
                velocity,
                reynolds_number,
                length,
                width,
                height,
                ground_clearance,
                slant_angle,
                fillet_radius,
            ) = self._read_info_file(info_path)
            if "velocity" in invar_keys:
                graph.ndata["velocity"] = velocity * torch.ones_like(
                    graph.ndata["pos"][:, [0]]
                )
            if "reynolds_number" in invar_keys:
                graph.ndata["reynolds_number"] = reynolds_number * torch.ones_like(
                    graph.ndata["pos"][:, [0]]
                )
            if "length" in invar_keys:
                graph.ndata["length"] = length * torch.ones_like(
                    graph.ndata["pos"][:, [0]]
                )
            if "width" in invar_keys:
                graph.ndata["width"] = width * torch.ones_like(
                    graph.ndata["pos"][:, [0]]
                )
            if "height" in invar_keys:
                graph.ndata["height"] = height * torch.ones_like(
                    graph.ndata["pos"][:, [0]]
                )
            if "ground_clearance" in invar_keys:
                graph.ndata["ground_clearance"] = ground_clearance * torch.ones_like(
                    graph.ndata["pos"][:, [0]]
                )
            if "slant_angle" in invar_keys:
                graph.ndata["slant_angle"] = slant_angle * torch.ones_like(
                    graph.ndata["pos"][:, [0]]
                )
            if "fillet_radius" in invar_keys:
                graph.ndata["fillet_radius"] = fillet_radius * torch.ones_like(
                    graph.ndata["pos"][:, [0]]
                )

            if "normals" in invar_keys or self.compute_drag:
                mesh = pv.read(file_path)
                mesh.compute_normals(
                    cell_normals=True, point_normals=False, inplace=True
                )
                if "normals" in invar_keys:
                    graph.ndata["normals"] = torch.from_numpy(
                        mesh.cell_data_to_point_data()["Normals"]
                    )
                if self.compute_drag:
                    mesh = mesh.compute_cell_sizes()
                    mesh = mesh.cell_data_to_point_data()
                    frontal_area = width * height / 2 * (10 ** (-6))
                    self.coeff.append(2.0 / ((velocity**2) * frontal_area))
                    self.normals.append(torch.from_numpy(mesh["Normals"]))
                    self.areas.append(torch.from_numpy(mesh["Area"]))
            self.graphs.append(graph)

        # add the edge features
        self.graphs = self.add_edge_features()

        # normalize the node and edge features
        if self.split == "train":
            self.node_stats = self._get_node_stats(keys=normalize_keys)
            self.edge_stats = self._get_edge_stats()
        else:
            self.node_stats = load_json("node_stats.json")
            self.edge_stats = load_json("edge_stats.json")

        self.graphs = self.normalize_node()
        self.graphs = self.normalize_edge()

    def __getitem__(self, idx):
        graph = self.graphs[idx]
        if self.compute_drag:
            sid = self.numbers[idx]
            return graph, sid, self.normals[idx], self.areas[idx], self.coeff[idx]
        return graph

    def __len__(self):
        return self.length

    def add_edge_features(self) -> List[dgl.DGLGraph]:
        """
        Add relative displacement and displacement norm as edge features for each graph
        in the list of graphs. The calculations are done using the 'pos' attribute in the
        node data of each graph. The resulting edge features are stored in the 'x' attribute
        in the edge data of each graph.

        This method will modify the list of graphs in-place.

        Returns
        -------
        List[dgl.DGLGraph]
            The list of graphs with updated edge features.
        """
        if not hasattr(self, "graphs") or not self.graphs:
            raise ValueError("The list 'graphs' is empty.")

        for graph in self.graphs:
            pos = graph.ndata.get("pos")
            if pos is None:
                raise ValueError(
                    "'pos' does not exist in the node data of one or more graphs."
                )

            row, col = graph.edges()
            row = row.long()
            col = col.long()

            disp = pos[row] - pos[col]
            disp_norm = torch.linalg.norm(disp, dim=-1, keepdim=True)
            graph.edata["x"] = torch.cat((disp, disp_norm), dim=-1)

        return self.graphs

    def normalize_node(self) -> List[dgl.DGLGraph]:
        """
        Normalize node data in each graph in the list of graphs using min-max normalization.
        The normalization is performed in-place. The normalization formula used is:

        normalized_data = 2.0 * normalization_bound[1] * (data - node_min) / (node_max - node_min) + normalization_bound[0]

        This will bring the node data in each graph into the range of [normalization_bound[0], normalization_bound[1]].
        After normalization, node data is concatenated according to the keys defined in 'self.input_keys'
        and 'self.output_keys', resulting in new node data 'x' and 'y', respectively.

        Returns
        -------
        List[dgl.DGLGraph]
            The list of graphs with normalized and concatenated node data.
        """
        if not hasattr(self, "graphs") or not self.graphs:
            raise ValueError("The list 'graphs' is empty.")

        if not hasattr(self, "node_stats") or not isinstance(self.node_stats, dict):
            raise ValueError(
                "The 'node_stats' attribute does not exist or is not a dictionary."
            )

        invar_keys = set(
            [
                key.replace("_min", "").replace("_max", "")
                for key in self.node_stats.keys()
            ]
        )

        for graph in self.graphs:
            for key in invar_keys:
                node_min = self.node_stats.get(key + "_min")
                node_max = self.node_stats.get(key + "_max")

                if node_min is None or node_max is None:
                    raise ValueError(
                        f"The keys '{key}_min' and/or '{key}_max' do not exist in 'node_stats'."
                    )

                if node_max.equal(node_min):
                    raise ValueError(
                        f"The values of '{key}_max' and '{key}_min' are equal, causing a division by zero."
                    )

                node_data_key = graph.ndata.get(key)
                if node_data_key is None:
                    raise ValueError(
                        f"The key '{key}' does not exist in the node data of one or more graphs."
                    )

                graph.ndata[key] = (
                    2.0
                    * self.normalization_bound[1]
                    * (node_data_key - node_min)
                    / (node_max - node_min)
                    + self.normalization_bound[0]
                )

            graph.ndata["x"] = torch.cat(
                [graph.ndata.get(key) for key in self.input_keys],
                dim=-1,
            )
            graph.ndata["y"] = torch.cat(
                [graph.ndata.get(key) for key in self.output_keys],
                dim=-1,
            )

        return self.graphs

    def normalize_edge(self) -> List[dgl.DGLGraph]:
        """
        Normalize edge data 'x' in each graph in the list of graphs using min-max normalization.
        The normalization is performed in-place. The normalization formula used is:

        normalized_x = 2.0 * normalization_bound[1] * (x - edge_min) / (edge_max - edge_min) + normalization_bound[0]

        This will bring the edge data 'x' in each graph into the range of [normalization_bound[0], normalization_bound[1]].

        Returns
        -------
        List[dgl.DGLGraph]
            The list of graphs with normalized edge data 'x'.
        """
        if not hasattr(self, "graphs") or not self.graphs:
            raise ValueError("The list 'graphs' is empty.")

        if not hasattr(self, "edge_stats") or not isinstance(self.edge_stats, dict):
            raise ValueError(
                "The 'edge_stats' attribute does not exist or is not a dictionary."
            )

        edge_min = self.edge_stats.get("edge_min")
        edge_max = self.edge_stats.get("edge_max")

        if edge_min is None or edge_max is None:
            raise ValueError(
                "The keys 'edge_min' and/or 'edge_max' do not exist in 'edge_stats'."
            )

        if edge_max.equal(edge_min):
            raise ValueError(
                "The values of 'edge_max' and 'edge_min' are equal, causing a division by zero."
            )

        for graph in self.graphs:
            edge_data_x = graph.edata.get("x")
            if edge_data_x is None:
                raise ValueError(
                    "The key 'x' does not exist in the edge data of one or more graphs."
                )

            graph.edata["x"] = (
                2.0
                * self.normalization_bound[1]
                * (edge_data_x - edge_min)
                / (edge_max - edge_min)
                + self.normalization_bound[0]
            )

        return self.graphs

    def denormalize(self, pred, gt, device) -> Tuple[Tensor, Tensor]:
        """
        Denormalize the graph node data.

        Parameters:
        -----------
        pred: Tensor
            Normalized prediction
        gt: Tensor
            Normalized ground truth
        device: Any
            The device

        Returns:
        --------
        Tuple(Tensor, Tensor)
            Denormalized prediction and ground truth
        """

        stats = self.node_stats
        stats = {key: val.to(device) for key, val in stats.items()}
        p_pred = pred
        p_pred = pred[:, [0]]
        s_pred = pred[:, 1:]
        p_gt = gt[:, [0]]
        s_gt = gt[:, 1:]
        p_pred = (p_pred - self.normalization_bound[0]) * (
            stats["p_max"] - stats["p_min"]
        ) / (2 * self.normalization_bound[1]) + stats["p_min"]
        s_pred = (s_pred - self.normalization_bound[0]) * (
            stats["wallShearStress_max"] - stats["wallShearStress_min"]
        ) / (2 * self.normalization_bound[1]) + stats["wallShearStress_min"]
        p_gt = (p_gt - self.normalization_bound[0]) * (
            stats["p_max"] - stats["p_min"]
        ) / (2 * self.normalization_bound[1]) + stats["p_min"]
        s_gt = (s_gt - self.normalization_bound[0]) * (
            stats["wallShearStress_max"] - stats["wallShearStress_min"]
        ) / (2 * self.normalization_bound[1]) + stats["wallShearStress_min"]
        pred = torch.cat((p_pred, s_pred), dim=-1)
        gt = torch.cat((p_gt, s_gt), dim=-1)
        return pred, gt

    def _get_edge_stats(self) -> Dict[str, Any]:
        """
        Compute and save the minimum and maximum values of each edge attribute 'x' in the graphs.

        Returns
        -------
        dict
            A dictionary with keys 'edge_min' and 'edge_max' and the corresponding values being
            1-D tensors containing the min or max value for each dimension of the edge attribute 'x'.
        """
        if not self.graphs:
            raise ValueError("The list 'graphs' is empty.")

        first_edge_data = self.graphs[0].edata.get("x")
        if first_edge_data is None:
            raise ValueError(
                "The key 'x' does not exist in the edge data of the first graph."
            )

        stats = {
            "edge_min": torch.full_like(first_edge_data[0, :], float("inf")),
            "edge_max": torch.full_like(first_edge_data[0, :], float("-inf")),
        }

        for i in range(len(self.graphs)):
            edge_data = self.graphs[i].edata.get("x")
            if edge_data is None:
                raise ValueError(
                    f"The key 'x' does not exist in the edge data of the {i}-th graph."
                )

            min_val, _ = edge_data.min(dim=0)
            max_val, _ = edge_data.max(dim=0)
            min_val, max_val = min_val.reshape(-1), max_val.reshape(-1)

            stats["edge_min"] = torch.minimum(stats["edge_min"], min_val)
            stats["edge_max"] = torch.maximum(stats["edge_max"], max_val)

        # Save to file
        save_json(stats, "edge_stats.json")

        return stats

    def _get_node_stats(self, keys: List[str]) -> Dict[str, Any]:
        """
        Compute and save the minimum and maximum values of each node attribute
        for the list of keys in the graphs.

        Parameters
        ----------
        keys : list of str
            List of keys for the node attributes.

        Returns
        -------
        dict
            A dictionary with each key being a string of format '[key]_min' or '[key]_max'
            and each value being a 1-D tensor containing the min or max value for each
            dimension of the node attribute.
        """
        if not self.graphs:
            raise ValueError("The list 'graphs' is empty.")

        stats = {}
        for key in keys:
            first_node_data = self.graphs[0].ndata.get(key)
            if first_node_data is None:
                raise ValueError(
                    f"The key '{key}' does not exist in the node data of the first graph."
                )

            stats[key + "_min"] = torch.full_like(first_node_data[0, :], float("inf"))
            stats[key + "_max"] = torch.full_like(first_node_data[0, :], float("-inf"))

        for i in range(len(self.graphs)):
            for key in keys:
                node_data = self.graphs[i].ndata.get(key)
                if node_data is None:
                    raise ValueError(
                        f"The key '{key}' does not exist in the node data of the {i}-th graph."
                    )

                min_val, _ = node_data.min(dim=0)
                max_val, _ = node_data.max(dim=0)
                min_val, max_val = min_val.reshape(-1), max_val.reshape(-1)

                stats[key + "_min"] = torch.minimum(stats[key + "_min"], min_val)
                stats[key + "_max"] = torch.maximum(stats[key + "_max"], max_val)

        # Save to file
        save_json(stats, "node_stats.json")

        return stats

    @staticmethod
    def _read_info_file(
        file_path: str,
    ) -> Tuple[float, float, float, float, float, float, float, float]:
        """
        Parse the values of specific parameters from a given text file.

        Parameters
        ----------
        file_path : str
            Path to the text file.

        Returns
        -------
        tuple
            A tuple containing values of velocity, reynolds number, length, width, height, ground clearance, slant angle, and fillet radius.
        """
        # Initialize variables to default value 0.0
        velocity = (
            reynolds_number
        ) = (
            length
        ) = width = height = ground_clearance = slant_angle = fillet_radius = 0.0

        with open(file_path, "r") as file:
            for line in file:
                if "Velocity" in line:
                    velocity = float(line.split(":")[1].strip())
                elif "Re" in line:
                    reynolds_number = float(line.split(":")[1].strip())
                elif "Length" in line:
                    length = float(line.split(":")[1].strip())
                elif "Width" in line:
                    width = float(line.split(":")[1].strip())
                elif "Height" in line:
                    height = float(line.split(":")[1].strip())
                elif "GroundClearance" in line:
                    ground_clearance = float(line.split(":")[1].strip())
                elif "SlantAngle" in line:
                    slant_angle = float(line.split(":")[1].strip())
                elif "FilletRadius" in line:
                    fillet_radius = float(line.split(":")[1].strip())

        return (
            velocity,
            reynolds_number,
            length,
            width,
            height,
            ground_clearance,
            slant_angle,
            fillet_radius,
        )

    @staticmethod
    def _create_dgl_graph(
        polydata: Any,
        outvar_keys: List[str],
        to_bidirected: bool = True,
        add_self_loop: bool = False,
        dtype: Union[torch.dtype, str] = torch.int32,
    ) -> dgl.DGLGraph:

        """
        Create a DGL graph from vtkPolyData.

        Parameters
        ----------
        polydata : vtkPolyData
            vtkPolyData from which the DGL graph is created.
        outvar_keys : list of str
            List of keys for the node attributes to be extracted from the vtkPolyData.
        to_bidirected : bool, optional
            Whether to make the graph bidirected. Default is True.
        add_self_loop : bool, optional
            Whether to add self-loops in the graph. Default is False.
        dtype : torch.dtype or str, optional
            Data type for the graph. Default is torch.int32.

        Returns
        -------
        dgl.DGLGraph
            The DGL graph created from the vtkPolyData.
        """
        # Extract point data and connectivity information from the vtkPolyData
        points = polydata.GetPoints()
        if points is None:
            raise ValueError("Failed to get points from the polydata.")

        vertices = np.array(
            [points.GetPoint(i) for i in range(points.GetNumberOfPoints())]
        )

        polys = polydata.GetPolys()
        if polys is None:
            raise ValueError("Failed to get polygons from the polydata.")

        polys.InitTraversal()
        edge_list = []
        for i in range(polys.GetNumberOfCells()):
            id_list = vtk.vtkIdList()
            polys.GetNextCell(id_list)
            for j in range(id_list.GetNumberOfIds() - 1):
                edge_list.append((id_list.GetId(j), id_list.GetId(j + 1)))

        # Create DGL graph using the connectivity information
        graph = dgl.graph(edge_list, idtype=dtype)
        if to_bidirected:
            graph = dgl.to_bidirected(graph)
        if add_self_loop:
            graph = dgl.add_self_loop(graph)

        # Assign node features using the vertex data
        graph.ndata["pos"] = torch.tensor(vertices, dtype=torch.float32)

        # Extract node attributes from the vtkPolyData
        point_data = polydata.GetPointData()
        if point_data is None:
            raise ValueError("Failed to get point data from the polydata.")

        for i in range(point_data.GetNumberOfArrays()):
            array = point_data.GetArray(i)
            array_name = array.GetName()
            if array_name in outvar_keys:
                array_data = np.zeros(
                    (points.GetNumberOfPoints(), array.GetNumberOfComponents())
                )
                for j in range(points.GetNumberOfPoints()):
                    array.GetTuple(j, array_data[j])

                # Assign node attributes to the DGL graph
                graph.ndata[array_name] = torch.tensor(array_data, dtype=torch.float32)

        return graph
