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


import os
import numpy as np
import torch
import vtk
import dgl

try:
    import nvidia.dali as dali
    import nvidia.dali.plugin.pytorch as dali_pth
except ImportError:
    raise ImportError(
        "DALI dataset requires NVIDIA DALI package to be installed. "
        + "The package can be installed at:\n"
        + "https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html"
    )

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple, Union, List, Any

from modulus.datapipes.datapipe import Datapipe
from modulus.datapipes.meta import DatapipeMetaData

Tensor = torch.Tensor


@dataclass
class MetaData(DatapipeMetaData):
    name: str = "VTKHDF"
    # Optimization
    auto_device: bool = True
    cuda_graphs: bool = True
    # Parallel
    ddp_sharding: bool = True


class VTKHDFDatapipe(Datapipe):
    """ DALI data pipeline for VTKHDF files

    Parameters
    ----------
    data_dir : str
        Directory where ERA5 data is stored
    stats_dir : Union[str, None], optional
        Directory to data statistic numpy files for normalization, if None, no normalization
        will be used, by default None
    batch_size : int, optional
        Batch size, by default 1
    num_steps : int, optional
        Number of timesteps are included in the output variables, by default 1
    shuffle : bool, optional
        Shuffle dataset, by default True
    num_workers : int, optional
        Number of workers, by default 1
    device: Union[str, torch.device], optional
        Device for DALI pipeline to run on, by default cuda
    process_rank : int, optional
        Rank ID of local process, by default 0
    world_size : int, optional
        Number of training processes, by default 1
    """

    def __init__(
        self,
        data_dir: str,
        data_type: str = "vtp",
        stats_dir: Union[str, None] = None,
        vars: List[str] = ["p", "wallShearStress"],
        batch_size: int = 1,
        num_samples: int = 1,
        shuffle: bool = True,
        num_workers: int = 1,
        device: Union[str, torch.device] = "cuda",
        process_rank: int = 0,
        world_size: int = 1,
    ):
        super().__init__(meta=MetaData())
        self.data_type=data_type
        self.vars = vars
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.data_dir = Path(data_dir)
        self.stats_dir = Path(stats_dir) if stats_dir is not None else None
        self.num_samples = num_samples
        self.process_rank = process_rank
        self.world_size = world_size

        if self.batch_size > 1:
            raise NotImplementedError("Batch size greater than 1 is not supported yet")

        # Set up device, needed for pipeline
        if isinstance(device, str):
            device = torch.device(device)
        # Need a index id if cuda
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda:0")
        self.device = device

        # check root directory exists
        if not self.data_dir.is_dir():
            raise IOError(f"Error, data directory {self.data_dir} does not exist")

        self.parse_dataset_files()

        self.pipe = self._create_pipeline()

    def parse_dataset_files(self) -> None:
        """Parses the data directory for valid VTKHDF files and determines training samples

        Raises
        ------
        ValueError
            In channels specified or number of samples per year is not valid
        """
        # get all input data files
        if self.data_type == "vtp":
            self.data_paths = sorted([str(path) for path in self.data_dir.glob("*.vtp")])
        elif self.data_type == "vtu":
            self.data_paths = sorted([str(path) for path in self.data_dir.glob("*.vtu")])

        for data_path in self.data_paths:
            self.logger.info(f"File found: {data_path}")
        self.all_samples = len(self.data_paths)

        if self.num_samples > self.all_samples:
            raise ValueError(
                f"Number of requested samples is greater than the total number of available samples!"
            )
        self.logger.info(f"Number of total samples: {self.all_samples}, number of requested samples: {self.num_samples}")

    def _create_pipeline(self) -> dali.Pipeline:
        """Create DALI pipeline

        Returns
        -------
        dali.Pipeline
            VTKHDF DALI pipeline
        """
        pipe = dali.Pipeline(
            batch_size=self.batch_size,
            num_threads=2,
            prefetch_queue_depth=2,
            py_num_workers=self.num_workers,
            device_id=self.device.index,
            py_start_method="spawn",
        )

        with pipe:
            source = VTKHDFDaliExternalSource(
                data_paths=self.data_paths,
                data_type=self.data_type,
                vars=self.vars,
                num_samples=self.num_samples,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                process_rank=self.process_rank,
                world_size=self.world_size,
            )
            # Update length of dataset
            self.length = len(source) // self.batch_size
            # Read current batch.
            vertices, attributes, edges = dali.fn.external_source(
                source,
                num_outputs=3,
                parallel=True,
                batch=False,
            )

            if self.device.type == "cuda":
                # Move tensors to GPU as external_source won't do that.
                vertices = vertices.gpu()
                attributes = attributes.gpu()
                edges = edges.gpu()

            # Set outputs.
            pipe.set_outputs(vertices, attributes, edges)

        return pipe

    def __iter__(self):
        # Reset the pipeline before creating an iterator to enable epochs.
        self.pipe.reset()
        # Create DALI PyTorch iterator.
        return dali_pth.DALIGenericIterator([self.pipe], ["vertices", "x", "edges"])

    def __len__(self):
        return self.length


class VTKHDFDaliExternalSource:
    """DALI Source for lazy-loading the VTKHDF files

    Parameters
    ----------
    data_paths : Iterable[str]
        Directory where data is stored
    num_samples : int
        Total number of training samples
    batch_size : int, optional
        Batch size, by default 1
    shuffle : bool, optional
        Shuffle dataset, by default True
    process_rank : int, optional
        Rank ID of local process, by default 0
    world_size : int, optional
        Number of training processes, by default 1

    Note
    ----
    For more information about DALI external source operator:
    https://docs.nvidia.com/deeplearning/dali/archives/dali_1_13_0/user-guide/docs/examples/general/data_loading/parallel_external_source.html
    """

    def __init__(
        self,
        data_paths: Iterable[str],
        data_type: str,
        vars: List[str],
        num_samples: int,
        batch_size: int = 1,
        shuffle: bool = True,
        process_rank: int = 0,
        world_size: int = 1,
    ):
        self.data_paths = list(data_paths)
        self.data_type=data_type
        self.vars = vars
        # Will be populated later once each worker starts running in its own process.
        self.poly_data = None
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.last_epoch = None

        self.indices = np.arange(num_samples)
        # Shard from indices if running in parallel
        self.indices = np.array_split(self.indices, world_size)[process_rank]

        # Get number of full batches, ignore possible last incomplete batch for now.
        # Also, DALI external source does not support incomplete batches in parallel mode.
        self.num_batches = len(self.indices) // self.batch_size

        self.vtk_reader_fn = self.vtk_reader()
        self.parse_vtk_data_fn = self.parse_vtk_data()
        

    def __call__(self, sample_info: dali.types.SampleInfo) -> Tuple[Tensor, Tensor]:
        if sample_info.iteration >= self.num_batches:
            raise StopIteration()

        # Shuffle before the next epoch starts.
        if self.shuffle and sample_info.epoch_idx != self.last_epoch:

            # All workers use the same rng seed so the resulting
            # indices are the same across workers.
            np.random.default_rng(seed=sample_info.epoch_idx).shuffle(self.indices)
            self.last_epoch = sample_info.epoch_idx

        # Get local indices from global index.
        idx = self.indices[sample_info.idx_in_epoch]

        #if self.poly_data is None:  # TODO check
        # This will be called once per worker. Workers are persistent,
        # so there is no need to explicitly close the files - this will be done
        # when corresponding pipeline/dataset is destroyed.
        data = self.vtk_reader_fn(self.data_paths[idx])
        #vertices, pressure, wss = self.parse_vtkpolydata(polydata)

        return  self.parse_vtk_data_fn(data, self.vars)

    def __len__(self):
        return len(self.indices)
    
    def vtk_reader(self):
        if self.data_type == "vtp":
            return read_vtp_file
        elif self.data_type=="vtu":
            return read_vtu_file
        else:
            raise NotImplementedError(f"Data type {self.data_type} is not supported yet")
    
    def parse_vtk_data(self):
        if self.data_type == "vtp":
            return _parse_vtk_polydata
        elif self.data_type=="vtu":
            return _parse_vtk_unstructuredgrid
        else:
            raise NotImplementedError(f"Data type {self.data_type} is not supported yet")

def _parse_vtk_polydata(polydata, vars):
    # Fetch vertices
    points = polydata.GetPoints()
    if points is None:
        raise ValueError("Failed to get points from the polydata.")
    vertices = torch.tensor(np.array(
        [points.GetPoint(i) for i in range(points.GetNumberOfPoints())]
    ), dtype=torch.float32)

    # Fetch node attributes  # TODO modularize
    attributes = []
    for array_name in vars:
        try:
            array = point_data.GetArray(array_name)
        except ValueError:
            raise ValueError(f"Failed to get array {array_name} from the unstructured grid.")
        array_data = np.zeros(
            (points.GetNumberOfPoints(), array.GetNumberOfComponents())
        )
        for j in range(points.GetNumberOfPoints()):
            array.GetTuple(j, array_data[j])
        attributes.append(torch.tensor(array_data, dtype=torch.float32))
    attributes = torch.cat(attributes, dim=-1)

    # Fetch edges
    polys = polydata.GetPolys()
    if polys is None:
        raise ValueError("Failed to get polygons from the polydata.")
    polys.InitTraversal()
    edges = []
    for i in range(polys.GetNumberOfCells()):
        id_list = vtk.vtkIdList()
        polys.GetNextCell(id_list)
        for j in range(id_list.GetNumberOfIds() - 1):
            edges.append(
                (id_list.GetId(j), id_list.GetId(j + 1))
            )
    edges = torch.tensor(edges, dtype=torch.long)

    return vertices, attributes, edges

def _parse_vtk_unstructuredgrid(grid, vars):
    # Fetch vertices
    points = grid.GetPoints()
    if points is None:
        raise ValueError("Failed to get points from the unstructured grid.")
    vertices = torch.tensor(np.array(
        [points.GetPoint(i) for i in range(points.GetNumberOfPoints())]
    ), dtype=torch.float32)

    # Fetch node attributes  # TODO modularize
    point_data = grid.GetPointData()
    if point_data is None:
        raise ValueError("Failed to get point data from the unstructured grid.")
    
    attributes = []
    for array_name in vars:
        try:
            array = point_data.GetArray(array_name)
        except ValueError:
            raise ValueError(f"Failed to get array {array_name} from the unstructured grid.")
        array_data = np.zeros(
            (points.GetNumberOfPoints(), array.GetNumberOfComponents())
        )
        for j in range(points.GetNumberOfPoints()):
            array.GetTuple(j, array_data[j])
        attributes.append(torch.tensor(array_data, dtype=torch.float32))
        print(array_data[1002,:])
        print(array_name)
    attributes = torch.cat(attributes, dim=-1)

    # Return a dummy tensor of zeros for edges since they are not directly computable
    return vertices, attributes, torch.zeros((0, 2), dtype=torch.long)  # Dummy tensor for edges
    
def read_vtp_file(file_path: str) -> Any:
    """
    Read a VTP file and return the polydata.

    Parameters
    ----------
    file_path : str
        Path to the VTP file.

    Returns
    -------
    vtkPolyData
        The polydata read from the VTP file.
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist.")

    # Check if file has .vtp extension
    if not file_path.endswith(".vtp"):
        raise ValueError(f"Expected a .vtp file, got {file_path}")

    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(file_path)
    reader.Update()

    # Get the polydata
    polydata = reader.GetOutput()

    # Check if polydata is valid
    if polydata is None:
        raise ValueError(f"Failed to read polydata from {file_path}")

    return polydata
    
def read_vtu_file(file_path: str) -> Any:
    """
    Read a VTU file and return the unstructured grid data.

    Parameters
    ----------
    file_path : str
        Path to the VTU file.

    Returns
    -------
    vtkUnstructuredGrid
        The unstructured grid data read from the VTU file.
    """
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} does not exist.")

    # Check if file has .vtu extension
    if not file_path.endswith(".vtu"):
        raise ValueError(f"Expected a .vtu file, got {file_path}")

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(file_path)
    reader.Update()

    # Get the unstructured grid data
    grid = reader.GetOutput()

    # Check if grid is valid
    if grid is None:
        raise ValueError(f"Failed to read unstructured grid data from {file_path}")

    return grid
    
def create_dgl_graph(
        vertices, attributes, edges, bidirected: bool = True, add_self_loop: bool = False, edge_idx_dtype=torch.int32
    ) -> dgl.DGLGraph:

        # Create DGL graph using the connectivity information
        # DALI gives a tensor of shape (1, num_edges, 2). Neet to convert it to a list of tuples
        edges= [(x[0], x[1]) for x in edges.squeeze(0).tolist()]
        graph = dgl.graph(edges, idtype=edge_idx_dtype) # TODO(mnabian) check if idtype is correct
        if bidirected:
            graph = dgl.to_bidirected(graph)
        if add_self_loop:
            graph = dgl.add_self_loop(graph)

        # Assign node features using the vertex data
        graph.ndata["coordinates"] = vertices.squeeze(0)

        # Assign node attributes to the DGL graph
        graph.ndata["x"] = attributes.squeeze(0)

        # Assign edge features to the DGL graph
        row, col = graph.edges()
        row = row.long()
        col = col.long()

        disp = graph.ndata["coordinates"][row] - graph.ndata["coordinates"][col]
        disp_norm = torch.linalg.norm(disp, dim=-1, keepdim=True)
        graph.edata["x"] = torch.cat((disp, disp_norm), dim=-1)

        return graph