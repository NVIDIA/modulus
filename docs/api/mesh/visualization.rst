Visualization
=============

.. currentmodule:: physicsnemo.mesh.visualization

``physicsnemo.mesh.visualization`` draws a mesh and its scalar data. The
entry point is :func:`draw`, which is also the ``mesh.draw()`` method. It has
two backends:

- Matplotlib draws static 2D plots, and 3D plots through ``mplot3d``. It
  draws a volume mesh as its boundary surface.
- PyVista draws interactive 3D scenes with the Visualization Toolkit (VTK).
  It supports meshes with at most three spatial dimensions.

With ``backend="auto"``, the default, :func:`draw` selects the backend from
the spatial dimension of the mesh only. It uses Matplotlib for meshes with
two or fewer spatial dimensions and PyVista for meshes in 3D. If the
preferred backend is not installed, it uses the other one. To choose a
backend, pass ``backend="matplotlib"`` or ``backend="pyvista"``.

Pass ``point_scalars`` or ``cell_scalars``, but not both, to color the mesh by
a field. Each one accepts a tensor or a key of ``point_data`` or
``cell_data``. For a vector or tensor field, :func:`draw` colors by the L2
norm of the trailing dimensions.

Example
-------

.. code:: python

    from physicsnemo.mesh.primitives.planar import unit_square
    from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

    # A planar mesh selects matplotlib; the call returns its Axes.
    square = unit_square.load(subdivisions=3)
    square.point_data["x"] = square.points[:, 0]
    ax = square.draw(point_scalars="x", show=False)

    # A surface in 3D selects PyVista; the call returns its Plotter.
    sphere = sphere_icosahedral.load(subdivisions=2)
    sphere.point_data["height"] = sphere.points[:, 2]
    plotter = sphere.draw(point_scalars="height", show_edges=True, show=False)

    # The standalone function is the same operation.
    from physicsnemo.mesh.visualization import draw

    ax = draw(sphere, backend="matplotlib", point_scalars="height", show=False)

With ``show=False``, :func:`draw` returns the Matplotlib ``Axes`` or the
PyVista ``Plotter`` without showing it. Pass it back as ``ax`` to draw
several meshes in one scene. ``backend_options`` passes keyword arguments to
PyVista's ``Plotter.add_mesh``. Matplotlib ignores them with a warning.

``draw_mesh`` is a compatibility name for :func:`draw`. It is pending
deprecation.

Summary
-------

.. autosummary::
   :nosignatures:

   draw
   draw_mesh

Related Topics
--------------

For more information, refer to the following resources:

- Tutorial: `Getting Started
  <https://github.com/NVIDIA/physicsnemo/blob/main/examples/minimal/mesh/tutorial_1_getting_started.ipynb>`_
- :doc:`io` to export a mesh to PyVista for custom rendering.

API Reference
-------------

.. autofunction:: draw

.. autofunction:: draw_mesh
