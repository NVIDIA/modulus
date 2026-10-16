# PhysicsNeMo-Mesh Tutorials

These notebooks teach PhysicsNeMo-Mesh, the PyTorch mesh library of PhysicsNeMo,
in nine steps, from a first mesh to meshes in a machine learning (ML)
pipeline.

PhysicsNeMo-Mesh stores simplicial meshes (point clouds, curves, triangle
surfaces, and tetrahedral volumes) as PyTorch tensors, with data on the points,
on the cells, and on the whole mesh. Its operations are batched tensor
operations, so they run on a GPU and work with autograd. For an overview of the
module, refer to the
[physicsnemo.mesh README](../../../physicsnemo/mesh/README.md).

## Tutorials

Work through the tutorials in order. Each one assumes the earlier ones. The
following table lists them:

<!-- markdownlint-disable MD013 -->
| # | Notebook | Topics |
| --- | -------- | ------ |
| 1 | [Getting Started](tutorial_1_getting_started.ipynb) | The `Mesh` data structure, dimensions, data at three levels, drawing, devices, autograd |
| 2 | [Loading, Converting, and Saving Meshes](tutorial_2_io_interop.ipynb) | PyVista import and export, file formats, polygon meshes, the native format, Zarr |
| 3 | [Primitives and Mesh Generation](tutorial_3_primitives_generation.ipynb) | Example meshes, filling a boundary, meshing implicit shapes, isosurfaces |
| 4 | [Mesh Operations](tutorial_4_operations.ipynb) | Transformations, subdivision, slicing, merging, boundaries, edge graphs |
| 5 | [Discrete Calculus and Curvature](tutorial_5_calculus.ipynb) | Gradients, divergence, curl, the Laplacian, curvature, physics losses |
| 6 | [Neighbors, Adjacency, and Spatial Queries](tutorial_6_neighbors_spatial.ipynb) | Adjacency, bounding volume hierarchy (BVH) queries, sampling, interpolation, signed distance |
| 7 | [Quality, Validation, and Repair](tutorial_7_quality_repair.ipynb) | Quality metrics, validation, topology checks, repair |
| 8 | [Simulation Domains with DomainMesh](tutorial_8_domain_mesh.ipynb) | An interior mesh with named boundaries, as one object |
| 9 | [Meshes in ML Pipelines](tutorial_9_ml_integration.ipynb) | Batching, features, a trained graph model, `torch.compile`, field schemas, performance |
<!-- markdownlint-enable MD013 -->

### 1. Getting Started

This tutorial covers the `Mesh` class, what its five fields hold, and how to
use them. It includes the following topics:

- Build a mesh from a `points` tensor and a `cells` tensor
- Manifold dimension, spatial dimension, and codimension
- Load meshes from PyVista and from saved files
- Attach point, cell, and global data of any rank, with nested keys
- Draw a mesh colored by a field, move it to a GPU, and differentiate through
  its geometry

### 2. Loading, Converting, and Saving Meshes

This tutorial shows how to bring meshes in and out of PhysicsNeMo-Mesh. It
includes the following topics:

- Why importing a mesh means splitting its cells into simplices
- `from_pyvista` and `to_pyvista`, including the dual graph of a mesh
- Reading and writing VTK and STL files, and which data each format keeps
- Raw polygon meshes with `Adjacency`, `triangulate`, and `Mesh.from_polygons`
- Ear clipping for non-convex polygons
- The native format of `Mesh.save` and `Mesh.load` (`.pmsh`, and `.pdmsh` for a
  `DomainMesh`)
- Chunked, compressed Zarr stores for training data, with `to_zarr` and
  `from_zarr`

### 3. Primitives and Mesh Generation

This tutorial shows how to get a mesh without a mesh file. It includes the
following topics:

- The seven families of example meshes in `physicsnemo.mesh.primitives`
- Filling closed 2D boundary loops with quality triangles (`fill_interior`)
- Meshing shapes given by implicit functions, in 2D and 3D
  (`mesh_implicit_domain`), with sharp corners kept by `feature_points`
- Isosurfaces of a 3D grid (`marching_cubes`)
- Gradients with respect to shape parameters (`refit_mesh_to_implicit`)

### 4. Mesh Operations

This tutorial shows how to change a mesh before it reaches a model. It
includes the following topics:

- Translation, scaling, rotation, and general linear maps, with vector fields
  rotated together with the geometry
- Changing the spatial dimension of a mesh
- Subdivision with the linear, Loop, and Butterfly schemes
- Selecting parts of a mesh with `slice_cells` and `slice_points`
- Combining meshes with `Mesh.merge`
- Boundary meshes, facet meshes, and edge graphs (`to_edge_graph`)
- Converting fields between point data and cell data

### 5. Discrete Calculus and Curvature

This tutorial covers derivatives of fields on a mesh, and how accurate they
are. It includes the following topics:

- Gradients of scalar and vector fields with `compute_point_derivatives`
- Choosing between least squares (LSQ) and discrete exterior calculus (DEC),
  including on irregular cells
- Divergence and curl, and why vector identities such as curl(grad f) = 0 hold
  only approximately for LSQ
- The Laplacian
- Intrinsic and extrinsic gradients on a surface
- Gaussian and mean curvature, and the Gauss-Bonnet theorem
- Backpropagating through the operators in a physics loss

### 6. Neighbors, Adjacency, and Spatial Queries

This tutorial covers the lookups that graph neural networks and training
pipelines need. It includes the following topics:

- The four adjacency queries: point to points, point to cells, cell to cells,
  and cell to points
- The `Adjacency` class for ragged neighbor lists, and the `edge_index` tensor
  of graph libraries, built with `expand_to_pairs`
- Finding the cells that contain a point with a BVH
- Sampling random points on cells, and interpolating fields at any point
- Signed distance to a closed surface with `signed_distance_field`

### 7. Quality, Validation, and Repair

This tutorial shows how to find and fix defects in a mesh. It includes the
following topics:

- Per-cell quality metrics and summary statistics
- What `validate()` checks, and what it does not check
- Topology checks: watertight and manifold meshes, including volume meshes and
  the Stanford bunny
- Repairs with `clean()`, `repair_mesh()`, `fix_orientation()`, and
  `fill_holes()`

### 8. Simulation Domains with DomainMesh

This tutorial shows how to keep the parts of one simulation sample together.
It includes the following topics:

- Building a `DomainMesh` from a computational fluid dynamics (CFD) result:
  an interior mesh, named boundary meshes, and case parameters
- Inspecting and iterating over the interior and the boundaries
- Checking a domain with `validate()` and `is_boundary_watertight()`
- Drawing the interior colored by a field, with the boundary curves over it
- Rotating a domain together with its vector fields, for data augmentation
- Spatial queries across the meshes of a domain, such as wall distance
- Saving and loading a domain

### 9. Meshes in ML Pipelines

This tutorial shows how to turn meshes into model inputs. It includes the
following topics:

- Padding meshes of different sizes for batching and `torch.compile`
- Geometric features (normals and curvature) and boundary conditions stored as
  mesh data
- End to end: from a mesh to a trained graph model
- Compiling mesh code with `torch.compile`
- Declaring and checking the fields of a model with `RankSpec` and `FieldSchema`
- Performance of PhysicsNeMo-Mesh on a GPU compared with PyVista and VTK

## Install

The tutorials need PhysicsNeMo with its `mesh-extras` extra (matplotlib, PyVista,
and VTK) and Jupyter. From this directory, install them with:

```bash
pip install -r requirements.txt
```

The tutorials use APIs from PhysicsNeMo 2.3, such as `FieldSchema` in
Tutorial 9. Until version 2.3.0 is released, install PhysicsNeMo from its main
branch instead:

<!-- markdownlint-disable MD013 -->
```bash
pip install "nvidia-physicsnemo[mesh-extras] @ git+https://github.com/NVIDIA/physicsnemo.git" \
    jupyter ipykernel ipywidgets
```
<!-- markdownlint-enable MD013 -->

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
`assets/`. From the repository root, run:

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

## More Information

The following pages have more information:

- [PhysicsNeMo-Mesh User Guide](https://docs.nvidia.com/physicsnemo/latest/user-guide/mesh.html)
- [PhysicsNeMo-Mesh API Reference](https://docs.nvidia.com/physicsnemo/latest/physicsnemo/api/mesh/index.html)
- [physicsnemo.mesh README](../../../physicsnemo/mesh/README.md)
- [TensorDict](https://github.com/pytorch/tensordict) and
  [PyVista](https://pyvista.org/)
