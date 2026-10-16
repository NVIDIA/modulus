Mesh
====

.. currentmodule:: physicsnemo.mesh.mesh

:class:`Mesh` is the central data structure of PhysicsNeMo-Mesh. It holds an
n-dimensional simplicial mesh in m-dimensional space, with
:math:`n \le m`. The manifold dimension :math:`n` (``n_manifold_dims``) is the
dimension of the cells: 0 for points, 1 for edges, 2 for triangles, 3 for
tetrahedra. The spatial dimension :math:`m` (``n_spatial_dims``) is the number
of coordinates of each point. :class:`~physicsnemo.mesh.domain_mesh.DomainMesh`
groups an interior mesh with named boundary meshes.

A ``Mesh`` has five fields:

- ``points``: the coordinates, with shape ``(n_points, n_spatial_dims)``.
- ``cells``: the point indices of each cell, with shape
  ``(n_cells, n_manifold_dims + 1)`` and an integer dtype.
- ``point_data``, ``cell_data``, and ``global_data``: ``TensorDict``
  containers for fields of any rank and with nested keys, one value for each
  point, for each cell, or for the whole mesh. Refer to :doc:`fields`.

``Mesh`` and ``DomainMesh`` are
`tensorclasses <https://pytorch.org/tensordict/stable/reference/tensorclass.html>`_,
so they also support the TensorDict API, for example ``.to(device)``,
``.clone()``, ``.detach()``, and ``torch.stack``. One ``.to("cuda")`` call
moves all points, cells, and data. ``Mesh`` computes geometric quantities,
such as centroids, areas, normals, and curvature, on first access and caches
them.

Most mesh operations are available both as ``Mesh`` methods and as functions
in the submodules. For many of them, the method is the function itself, and
Python supplies the mesh as its first argument. For example,
``mesh.rotate(...)`` is ``rotate(mesh, ...)`` from :doc:`transformations`.

Example
-------

.. code:: python

    import torch
    from physicsnemo.mesh import Mesh

    points = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
    cells = torch.tensor([[0, 1, 2]])
    mesh = Mesh(points=points, cells=cells)

    # Geometric properties (computed on first access, then cached)
    print(mesh.cell_centroids)   # shape (1, 2)
    print(mesh.cell_areas)       # tensor([0.5000])

    # Fields of any rank, at three levels
    mesh.point_data["T"] = torch.tensor([1.0, 2.0, 3.0])              # scalar
    mesh.point_data["velocity"] = torch.zeros(3, 2)                    # vector
    mesh.cell_data["stress"] = torch.eye(2).unsqueeze(0)               # rank-2 tensor
    mesh.global_data["reynolds_number"] = torch.tensor(1.0e6)          # one value

    # Derivatives are stored as new point fields
    mesh = mesh.compute_point_derivatives(keys="T", method="lsq")
    print(mesh.point_data["T_gradient"].shape)  # torch.Size([3, 2])

To build a triangle mesh from a surface whose cells are arbitrary polygons,
use :meth:`Mesh.from_polygons`. Refer to :doc:`tessellation`.

Cache-Aware Functional Updates
------------------------------

Use the cache-aware replacement methods to make a mesh that keeps the same
point or cell indexing. Each method returns a new ``Mesh`` with independent
``TensorDict`` containers that share their tensor leaves. The following table
lists the methods:

.. list-table::
   :header-rows: 1

   * - Method
     - Intended Change
     - Default Cache Policy
   * - :meth:`Mesh.with_data`
     - Replace point, cell, or global field data
     - Retain every geometry and topology cache
   * - :meth:`Mesh.with_points`
     - Replace coordinates without changing point indexing or connectivity
     - Retain topology and clear geometry caches
   * - :meth:`Mesh.with_cells`
     - Replace connectivity without changing cell indexing or simplex type
     - Clear every cache
   * - :meth:`Mesh.strip_caches`
     - Remove cached values without changing the mesh
     - Clear every cache

The ``keep`` override accepts any of the following:

- A top-level string such as ``"topology"``
- A single nested-key tuple such as ``("cell", "areas")``
- A sequence of keys

Use a list to retain several keys, for example
``["topology", ("cell", "areas")]``. Retaining a cache that depends on
replaced geometry or connectivity is an expert operation. You are responsible
for the validity of the retained cache.

Summary
-------

.. rubric:: Classes

.. autosummary::
   :nosignatures:

   Mesh
   ~physicsnemo.mesh.domain_mesh.DomainMesh

.. rubric:: Construction and Data

.. autosummary::
   :nosignatures:

   Mesh.from_polygons
   Mesh.merge
   Mesh.slice_points
   Mesh.slice_cells
   Mesh.pad
   Mesh.pad_to_next_power
   Mesh.with_points
   Mesh.with_cells
   Mesh.with_data
   Mesh.strip_caches
   Mesh.cell_data_to_point_data
   Mesh.point_data_to_cell_data

.. rubric:: Dimensions and Geometry

.. autosummary::
   :nosignatures:

   Mesh.n_points
   Mesh.n_cells
   Mesh.n_spatial_dims
   Mesh.n_manifold_dims
   Mesh.codimension
   Mesh.cell_centroids
   Mesh.cell_areas
   Mesh.cell_normals
   Mesh.point_normals
   Mesh.compute_point_normals
   Mesh.gaussian_curvature_vertices
   Mesh.gaussian_curvature_cells
   Mesh.mean_curvature_vertices

.. rubric:: Topology and Conversion

.. autosummary::
   :nosignatures:

   Mesh.get_facet_mesh
   Mesh.get_boundary_mesh
   Mesh.is_manifold
   Mesh.is_watertight
   Mesh.get_point_to_points_adjacency
   Mesh.get_point_to_cells_adjacency
   Mesh.get_cell_to_cells_adjacency
   Mesh.get_cell_to_points_adjacency
   Mesh.to_edge_graph
   Mesh.to_dual_graph
   Mesh.to_point_cloud

.. rubric:: Calculus, Integration, and Sampling

.. autosummary::
   :nosignatures:

   Mesh.compute_point_derivatives
   Mesh.compute_cell_derivatives
   Mesh.gradient
   Mesh.divergence
   Mesh.curl
   Mesh.laplacian
   Mesh.integrate
   Mesh.integrate_samples
   Mesh.integrate_flux
   Mesh.integrate_moment
   Mesh.sample_random_points_on_cells
   Mesh.sample_data_at_points

.. rubric:: Transformations and Deformations

.. autosummary::
   :nosignatures:

   Mesh.translate
   Mesh.rotate
   Mesh.scale
   Mesh.transform
   Mesh.displace
   Mesh.sobolev_deform
   Mesh.morph
   Mesh.radial_basis_function_deform
   Mesh.free_form_deform
   Mesh.shrinkwrap

.. rubric:: Editing, Checking, and Viewing

.. autosummary::
   :nosignatures:

   Mesh.subdivide
   Mesh.remesh
   Mesh.clean
   Mesh.validate
   Mesh.quality_metrics
   Mesh.statistics
   Mesh.draw

Related Topics
--------------

For more information, refer to the following resources:

- User Guide: `Concepts and Data Model
  <https://docs.nvidia.com/physicsnemo/latest/user-guide/mesh/concepts.html>`_
  and `Simulation Domains with DomainMesh
  <https://docs.nvidia.com/physicsnemo/latest/user-guide/mesh/domain_mesh.html>`_
- Tutorials: `Getting Started
  <https://github.com/NVIDIA/physicsnemo/blob/main/examples/minimal/mesh/tutorial_1_getting_started.ipynb>`_
  and `Simulation Domains with DomainMesh
  <https://github.com/NVIDIA/physicsnemo/blob/main/examples/minimal/mesh/tutorial_8_domain_mesh.ipynb>`_

API Reference
-------------

.. autoclass:: Mesh
   :members:
   :show-inheritance:

DomainMesh
^^^^^^^^^^

.. currentmodule:: physicsnemo.mesh.domain_mesh

:class:`DomainMesh` groups an interior mesh with named boundary meshes and
domain-level data. Operations such as :meth:`DomainMesh.morph` and
:meth:`DomainMesh.radial_basis_function_deform` apply one geometry change to
every component and return a new domain. The following example builds a
domain from a mesh and its boundary:

.. code:: python

    from physicsnemo.mesh import DomainMesh

    wall = mesh.get_boundary_mesh()
    domain = DomainMesh(interior=mesh, boundaries={"wall": wall})
    print(domain.boundary_names)  # ['wall']

.. autoclass:: DomainMesh
   :members:
   :show-inheritance:
