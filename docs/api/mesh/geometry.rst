Geometry
========

.. currentmodule:: physicsnemo.mesh.geometry

``physicsnemo.mesh.geometry`` computes the geometric quantities that the
:doc:`calculus` and :doc:`curvature` modules build on:

- Cell areas (the n-dimensional volume of each simplex) and cell normals
- Interior angles of each cell at each of its vertices
- Circumcenters of each cell, which define the circumcentric dual mesh that
  DEC (discrete exterior calculus) uses
- Dual volumes: the Voronoi region around each vertex and the dual cell of
  each edge
- Cotangent weights, the standard edge weights of the FEM and DEC Laplacian

Most users do not call these functions directly. The ``Mesh`` properties
(``mesh.cell_areas``, ``mesh.cell_normals``) and the calculus and curvature
functions call them and cache the results. Use them to write your own DEC
operators or finite element (FEM) assembly.

Some functions take a mesh. Others take tensors, so that you can call them
on cell geometry without a ``Mesh``:

- :func:`compute_circumcenters` takes the vertex coordinates of each cell,
  ``mesh.points[mesh.cells]``, with shape
  ``(n_cells, n_manifold_dims + 1, n_spatial_dims)``.
- :func:`compute_cell_areas` and :func:`compute_cell_normals` take the edge
  vectors from the first vertex of each cell, with shape
  ``(n_cells, n_manifold_dims, n_spatial_dims)``.

Example
-------

.. code:: python

    import math

    from physicsnemo.mesh.geometry import (
        compute_cell_areas,
        compute_circumcenters,
        compute_cotan_weights_fem,
        compute_dual_volumes_0,
        compute_vertex_angle_sums,
        compute_vertex_angles,
    )
    from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

    mesh = sphere_icosahedral.load(subdivisions=2)  # 162 points, 320 triangles

    angles = compute_vertex_angles(mesh)  # (n_cells, n_manifold_dims + 1)
    circumcenters = compute_circumcenters(mesh.points[mesh.cells])  # (n_cells, n_spatial_dims)
    dual_areas = compute_dual_volumes_0(mesh)  # (n_points,)
    weights, edges = compute_cotan_weights_fem(mesh)  # (n_edges,) and (n_edges, 2)
    print(angles.shape, circumcenters.shape, dual_areas.shape)
    # torch.Size([320, 3]) torch.Size([320, 3]) torch.Size([162])
    print(weights.shape, edges.shape)  # torch.Size([480]) torch.Size([480, 2])

    # The dual areas tile the surface.
    print(dual_areas.sum().item(), mesh.cell_areas.sum().item())  # equal

    # Gauss-Bonnet: the angle defects sum to 2 pi times the Euler characteristic,
    # which is 4 pi for a sphere.
    print((2 * math.pi - compute_vertex_angle_sums(mesh)).sum().item(), 4 * math.pi)

    # Tensor-level helpers take edge vectors from the first vertex of each cell.
    edge_vectors = mesh.points[mesh.cells[:, 1:]] - mesh.points[mesh.cells[:, :1]]
    areas = compute_cell_areas(edge_vectors)  # equals mesh.cell_areas

Choosing a Function
-------------------

- :func:`get_or_compute_circumcenters` and
  :func:`get_or_compute_dual_volumes_0` read the mesh cache first and
  compute only on a miss. Use them in place of
  :func:`compute_circumcenters` and :func:`compute_dual_volumes_0` when you
  call them more than once on one mesh.
- :func:`compute_vertex_angles` gives one angle per cell vertex.
  :func:`compute_vertex_angle_sums` adds them up per point, which gives the
  angle defect for Gaussian curvature.
- :func:`compute_triangle_angles` and :func:`stable_angle_between_vectors`
  take point coordinates or vectors as tensors, not a mesh.

Summary
-------

.. autosummary::
   :nosignatures:

   compute_cell_areas
   compute_cell_normals
   compute_vertex_angles
   compute_vertex_angle_sums
   compute_triangle_angles
   stable_angle_between_vectors
   compute_circumcenters
   get_or_compute_circumcenters
   compute_dual_volumes_0
   get_or_compute_dual_volumes_0
   compute_dual_volumes_1
   compute_cotan_weights_fem

See Also
--------

- :doc:`calculus` and :doc:`curvature`, which use these quantities.
- Tutorial: `Discrete Calculus and Curvature
  <https://github.com/NVIDIA/physicsnemo/blob/main/examples/minimal/mesh/tutorial_5_calculus.ipynb>`_

API Reference
-------------

.. autofunction:: compute_cell_areas

.. autofunction:: compute_cell_normals

.. autofunction:: compute_vertex_angles

.. autofunction:: compute_vertex_angle_sums

.. autofunction:: compute_triangle_angles

.. autofunction:: stable_angle_between_vectors

.. autofunction:: compute_circumcenters

.. autofunction:: get_or_compute_circumcenters

.. autofunction:: compute_dual_volumes_0

.. autofunction:: get_or_compute_dual_volumes_0

.. autofunction:: compute_dual_volumes_1

.. autofunction:: compute_cotan_weights_fem
