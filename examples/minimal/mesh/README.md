# PhysicsNeMo-Mesh Tutorials

These notebooks teach PhysicsNeMo-Mesh, the PyTorch mesh library of PhysicsNeMo,
in nine steps: from a first mesh to meshes in a machine learning pipeline.

PhysicsNeMo-Mesh stores simplicial meshes (point clouds, curves, triangle
surfaces, and tetrahedral volumes) as PyTorch tensors, with data on the points,
on the cells, and on the whole mesh. Its operations are batched tensor
operations, so they run on a GPU and work with autograd. For an overview of the
module, see the [physicsnemo.mesh README](../../../physicsnemo/mesh/README.md).

## Tutorials

Work through the tutorials in order. Each one assumes the earlier ones.

<!-- markdownlint-disable MD013 -->
| # | Notebook | Topics |
| --- | -------- | ------ |
| 1 | [Getting Started](tutorial_1_getting_started.ipynb) | The `Mesh` data structure, dimensions, data at three levels, drawing, devices, autograd |
| 2 | [Loading, Converting, and Saving Meshes](tutorial_2_io_interop.ipynb) | PyVista import and export, file formats, polygon meshes, the native format |
| 3 | [Primitives and Mesh Generation](tutorial_3_primitives_generation.ipynb) | Example meshes, filling a boundary, meshing implicit shapes, isosurfaces |
| 4 | [Mesh Operations](tutorial_4_operations.ipynb) | Transformations, subdivision, slicing, merging, boundaries |
| 5 | [Discrete Calculus and Curvature](tutorial_5_calculus.ipynb) | Gradients, divergence, curl, curvature |
| 6 | [Neighbors, Adjacency, and Spatial Queries](tutorial_6_neighbors_spatial.ipynb) | Adjacency, BVH queries, sampling, interpolation |
| 7 | [Quality, Validation, and Repair](tutorial_7_quality_repair.ipynb) | Quality metrics, validation, topology checks, repair |
| 8 | [Simulation Domains with DomainMesh](tutorial_8_domain_mesh.ipynb) | An interior mesh with named boundaries, as one object |
| 9 | [Meshes in ML Pipelines](tutorial_9_ml_integration.ipynb) | Performance, batching, `torch.compile`, field schemas |
<!-- markdownlint-enable MD013 -->

### 1. Getting Started

The `Mesh` class: what its five fields hold, and how to use them.

- Build a mesh from a `points` tensor and a `cells` tensor
- Manifold dimension, spatial dimension, and codimension
- Load meshes from PyVista and from saved files
- Attach point, cell, and global data of any rank, with nested keys
- Draw a mesh colored by a field, move it to a GPU, and differentiate through
  its geometry

### 2. Loading, Converting, and Saving Meshes

How to bring meshes in and out of PhysicsNeMo-Mesh.

- Why importing a mesh means splitting its cells into simplices
- `from_pyvista` and `to_pyvista`, including the dual graph of a mesh
- Reading and writing VTK and STL files, and which data each format keeps
- Raw polygon meshes with `Adjacency`, `triangulate`, and `Mesh.from_polygons`
- Ear clipping for non-convex polygons
- The native format of `Mesh.save` and `Mesh.load` (`.pmsh`, and `.pdmsh` for a
  `DomainMesh`)

### 3. Primitives and Mesh Generation

How to get a mesh without a mesh file.

- The seven families of example meshes in `physicsnemo.mesh.primitives`
- Filling closed 2D boundary loops with quality triangles (`fill_interior`)
- Meshing shapes given by implicit functions, in 2D and 3D
  (`mesh_implicit_domain`), with sharp corners kept by `feature_points`
- Isosurfaces of a 3D grid (`marching_cubes`)
- Gradients with respect to shape parameters (`refit_mesh_to_implicit`)

### 4. Mesh Operations

How to change a mesh.

- Translation, rotation, scaling, and general linear transformations, with and
  without the attached data
- Subdivision (linear, Loop, and Butterfly)
- Slicing by cells or points, and merging meshes
- Boundary and facet extraction
- Moving data between points and cells

### 5. Discrete Calculus and Curvature

Derivatives of fields on a mesh.

- Gradients by least squares (LSQ) and by discrete exterior calculus (DEC)
- Divergence and curl
- Gaussian and mean curvature
- Intrinsic and extrinsic derivatives on surfaces
- Vector calculus identities on a mesh

### 6. Neighbors, Adjacency, and Spatial Queries

The queries that graph neural networks and point-based models need.

- Point and cell neighbors, stored as sparse adjacency, and conversion to
  PyTorch Geometric
- Bounding volume hierarchy (BVH) queries and point containment
- Random sampling of points on cells
- Interpolation of data at query points, including points outside the mesh

### 7. Quality, Validation, and Repair

How to find and fix problems in a mesh.

- Quality metrics (aspect ratio, angles, quality score) and mesh statistics
- Validation of a mesh
- Topology checks: watertight and manifold meshes
- Repair: duplicate points, degenerate cells, orientation

### 8. Simulation Domains with DomainMesh

How to represent a whole simulation case.

- A `DomainMesh` from a real CFD case: an interior mesh and named boundary
  meshes
- Transformations of a whole domain, for data augmentation
- Validation, and a check that the boundaries close the domain
- Drawing boundary patches by boundary condition type
- Spatial queries on a domain

### 9. Meshes in ML Pipelines

How to use PhysicsNeMo-Mesh in training and inference code.

- Performance compared with PyVista and VTK
- Padding and batching for `torch.compile`
- Feature extraction and boundary conditions
- An end-to-end preprocessing workflow for CAE data
- Declaring the fields of a model with `FieldSchema`

## Install

The tutorials need PhysicsNeMo with its `mesh-extras` extra (matplotlib, PyVista,
and VTK) and Jupyter. From this directory, install them with:

```bash
pip install -r requirements.txt
```

To use a clone of the PhysicsNeMo repository instead, run this from the root of
the clone:

```bash
pip install -e ".[mesh-extras]" jupyter ipykernel ipywidgets
```

PhysicsNeMo needs Python 3.11 or later. The notebooks run on a CPU, and they use
a CUDA GPU when one is present. Some cells download PyVista example meshes the
first time that they run, so they need internet access once.

## Run

Start Jupyter in this directory, because the notebooks read files from
`assets/`:

```bash
cd examples/minimal/mesh
jupyter lab
```

Then open a `tutorial_*.ipynb` file. VS Code and other editors with notebook
support open the files directly.

The notebooks render 3D figures with PyVista as static images
(`pv.set_jupyter_backend("static")`), so that the figures show on GitHub. To
rotate and zoom a 3D figure, install `trame` (`pip install "pyvista[jupyter]"`)
and change the backend to `"trame"`.

## Assets

The `assets/` directory holds meshes that the tutorials load:

- `bunny.pt`: a coarse Stanford bunny, saved with `torch.save`. Refine it with
  `.subdivide()`.

## More information

- [PhysicsNeMo-Mesh User Guide](https://docs.nvidia.com/physicsnemo/latest/user-guide/mesh.html)
- [PhysicsNeMo-Mesh API reference](https://docs.nvidia.com/physicsnemo/latest/physicsnemo/api/mesh/index.html)
- [physicsnemo.mesh README](../../../physicsnemo/mesh/README.md)
- [TensorDict](https://github.com/pytorch/tensordict) and
  [PyVista](https://pyvista.org/)
