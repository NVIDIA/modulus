I/O
===

.. currentmodule:: physicsnemo.mesh.io

``physicsnemo.mesh.io`` converts between :class:`~physicsnemo.mesh.mesh.Mesh`
and `PyVista <https://pyvista.org/>`_ meshes, and saves meshes in the native
Zarr format. PyVista reads and writes many file formats, such as VTK, STL,
PLY, and OBJ, so it is the main way to load and save files. The module has
the following functions:

- :func:`from_pyvista` converts a ``pyvista.PolyData``,
  ``pyvista.UnstructuredGrid``, or ``pyvista.PointSet`` to a ``Mesh``, with
  its point data and cell data. It detects the manifold dimension, or you set
  it with ``manifold_dim``. Polygons and polyhedra become triangles and
  tetrahedra. With ``point_source="cell_centroids"``, the cell centroids
  become a point cloud or a dual graph, which avoids tetrahedralization of a
  large polyhedral mesh.
- :func:`to_pyvista` converts a ``Mesh`` to a ``pyvista.PolyData`` (curves
  and surfaces), a ``pyvista.UnstructuredGrid`` (volumes), or a
  ``pyvista.PointSet`` (a point cloud without cells).
- :func:`to_zarr` and :func:`from_zarr` save and load a ``Mesh`` or a
  :class:`~physicsnemo.mesh.domain_mesh.DomainMesh`, with all of its data, in
  a Zarr store.

Example
-------

This example makes a PyVista sphere. To load a file, use
``pv.read("geometry.stl")`` in its place.

.. code:: python

    import pyvista as pv
    from physicsnemo.mesh.io import from_pyvista, from_zarr, to_pyvista, to_zarr

    pv_mesh = pv.Sphere(theta_resolution=16, phi_resolution=16)
    mesh = from_pyvista(pv_mesh)
    print(mesh.n_manifold_dims, mesh.n_spatial_dims)  # 2 3

    mesh = mesh.subdivide(levels=1, filter="loop")

    # Back to PyVista, to save in any format it supports.
    pv_out = to_pyvista(mesh)
    pv_out.save("refined.vtk")

    # Native storage keeps every field and nested key.
    to_zarr(mesh, "refined.zarr")
    restored = from_zarr("refined.zarr")
    print(restored.n_cells == mesh.n_cells)  # True

.. tip::

   PyVista conversions preserve ``float32`` and ``float64`` point coordinates.
   Integer, complex, and reduced-precision coordinates are converted to
   ``float32`` as part of PhysicsNeMo's PyVista compatibility policy. To
   normalize imported geometry and its floating data explicitly, use
   ``mesh = from_pyvista(pv_mesh).to(torch.float32)``. Likewise, use
   ``to_pyvista(mesh.to(torch.float32))`` before export. Retaining ``float64``
   coordinates doubles their storage relative to ``float32`` and may keep
   downstream PyVista computations in double precision.

Summary
-------

.. autosummary::
   :nosignatures:

   from_pyvista
   to_pyvista
   from_zarr
   to_zarr

Related Topics
--------------

Refer to the following pages for related topics:

- :doc:`tessellation` for polygon meshes.
- Tutorial: `Loading, Converting, and Saving Meshes
  <https://github.com/NVIDIA/physicsnemo/blob/main/examples/minimal/mesh/tutorial_2_io_interop.ipynb>`_

API Reference
-------------

.. autofunction:: from_pyvista

.. autofunction:: to_pyvista

.. autofunction:: from_zarr

.. autofunction:: to_zarr
