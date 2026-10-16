PhysicsNeMo Mesh
================

.. currentmodule:: physicsnemo.mesh

PhysicsNeMo-Mesh (``physicsnemo.mesh``) is a PyTorch library for simplicial
meshes in any dimension. A simplex is a point, edge, triangle, tetrahedron, or
its higher-dimensional analog. One :class:`~physicsnemo.mesh.mesh.Mesh` type
holds point clouds, curves, surfaces, and volumes, together with tensor fields
of any rank on its points, its cells, and the mesh as a whole. Operations run
on CPU or GPU, and most of them are differentiable with
`PyTorch autograd <https://pytorch.org/docs/stable/autograd.html>`_.

Where to Start
--------------

The following resources document the library:

- The `PhysicsNeMo-Mesh User Guide
  <https://docs.nvidia.com/physicsnemo/latest/user-guide/mesh.html>`_
  explains the data model (manifold dimension, spatial dimension, and
  codimension), helps you choose between methods, and shows how meshes fit
  into an ML pipeline.
- The `tutorial notebooks
  <https://github.com/NVIDIA/physicsnemo/tree/main/examples/minimal/mesh>`_
  teach the library step by step, from a first mesh to an ML pipeline.
- This API reference has one page for each submodule. Each page gives the
  purpose of the submodule, a short example, a table of its public API, and
  the full reference.

Quick Start
-----------

This example runs on a CPU. It builds two triangles in the plane, attaches a
scalar field to the points, and computes the gradient of the field:

.. code:: python

    import torch
    from physicsnemo.mesh import Mesh

    points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    cells = torch.tensor([[0, 1, 2], [0, 2, 3]])
    mesh = Mesh(points=points, cells=cells)

    print(mesh.n_manifold_dims, mesh.n_spatial_dims)  # 2 2
    print(mesh.cell_areas)  # tensor([0.5000, 0.5000])

    # A linear field T = x + 2y has the gradient (1, 2) everywhere.
    mesh.point_data["T"] = mesh.points[:, 0] + 2 * mesh.points[:, 1]
    mesh = mesh.compute_point_derivatives(keys="T", method="lsq")
    print(mesh.point_data["T_gradient"][0])  # tensor([1., 2.])

To run the same code on a GPU, move the mesh and all of its data with
``mesh = mesh.to("cuda")``.

Submodules
----------

Each submodule has its own page:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Page
     - Contents
   * - :doc:`core`
     - The ``Mesh`` and ``DomainMesh`` data structures.
   * - :doc:`fields`
     - Declarations of named fields and their tensor ranks (``RankSpec``,
       ``FieldSchema``).
   * - :doc:`primitives`
     - A catalog of ready-made meshes, from single triangles to tetrahedral
       spheres.
   * - :doc:`io`
     - Conversion to and from PyVista, and native Zarr storage.
   * - :doc:`tessellation`
     - Triangulation of polygon meshes and quality filling of closed
       boundaries.
   * - :doc:`generate`
     - Volume meshes of implicit domains, and marching cubes.
   * - :doc:`geometry`
     - Low-level geometric quantities: angles, circumcenters, dual volumes,
       and cotangent weights.
   * - :doc:`curvature`
     - Gaussian and mean curvature.
   * - :doc:`calculus`
     - Gradient, divergence, curl, and Laplacian, by discrete exterior
       calculus (DEC) and least squares (LSQ), and integration.
   * - :doc:`boundaries`
     - Boundary detection, facet extraction, and watertight and manifold
       checks.
   * - :doc:`neighbors`
     - Point and cell adjacency in a ragged ``Adjacency`` format.
   * - :doc:`spatial`
     - Bounding volume hierarchies, cluster trees, and signed distance.
   * - :doc:`sampling`
     - Random points on cells and interpolation of data at query points.
   * - :doc:`transformations`
     - Rigid and linear transformations, deformations, and projections.
   * - :doc:`deformation`
     - Differentiable penalty energies for deformations.
   * - :doc:`subdivision`
     - Linear, Loop, and Butterfly subdivision.
   * - :doc:`smoothing`
     - Laplacian smoothing.
   * - :doc:`remeshing`
     - Surface remeshing to a target vertex count, and cell partitioning.
   * - :doc:`repair`
     - Removal of duplicate and degenerate elements, orientation, and hole
       filling.
   * - :doc:`validation`
     - Structural validation, per-cell quality metrics, and mesh statistics.
   * - :doc:`visualization`
     - Plots of meshes and their data with matplotlib or PyVista.

Tutorials
---------

The notebooks are in
`examples/minimal/mesh/ <https://github.com/NVIDIA/physicsnemo/tree/main/examples/minimal/mesh>`_.
Work through them in order:

1. `Getting Started <https://github.com/NVIDIA/physicsnemo/blob/main/examples/minimal/mesh/tutorial_1_getting_started.ipynb>`__
2. `Loading, Converting, and Saving Meshes <https://github.com/NVIDIA/physicsnemo/blob/main/examples/minimal/mesh/tutorial_2_io_interop.ipynb>`__
3. `Primitives and Mesh Generation <https://github.com/NVIDIA/physicsnemo/blob/main/examples/minimal/mesh/tutorial_3_primitives_generation.ipynb>`__
4. `Mesh Operations <https://github.com/NVIDIA/physicsnemo/blob/main/examples/minimal/mesh/tutorial_4_operations.ipynb>`__
5. `Discrete Calculus and Curvature <https://github.com/NVIDIA/physicsnemo/blob/main/examples/minimal/mesh/tutorial_5_calculus.ipynb>`__
6. `Neighbors, Adjacency, and Spatial Queries <https://github.com/NVIDIA/physicsnemo/blob/main/examples/minimal/mesh/tutorial_6_neighbors_spatial.ipynb>`__
7. `Quality, Validation, and Repair <https://github.com/NVIDIA/physicsnemo/blob/main/examples/minimal/mesh/tutorial_7_quality_repair.ipynb>`__
8. `Simulation Domains with DomainMesh <https://github.com/NVIDIA/physicsnemo/blob/main/examples/minimal/mesh/tutorial_8_domain_mesh.ipynb>`__
9. `Meshes in ML Pipelines <https://github.com/NVIDIA/physicsnemo/blob/main/examples/minimal/mesh/tutorial_9_ml_integration.ipynb>`__

.. toctree::
   :maxdepth: 1
   :caption: Core

   core
   fields

.. toctree::
   :maxdepth: 1
   :caption: Building Meshes

   primitives
   io
   tessellation
   generate

.. toctree::
   :maxdepth: 1
   :caption: Geometry and Calculus

   geometry
   curvature
   calculus

.. toctree::
   :maxdepth: 1
   :caption: Topology and Queries

   boundaries
   neighbors
   spatial
   sampling

.. toctree::
   :maxdepth: 1
   :caption: Editing

   transformations
   deformation
   subdivision
   smoothing
   remeshing
   repair

.. toctree::
   :maxdepth: 1
   :caption: Checking and Viewing

   validation
   visualization
