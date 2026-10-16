Repair
======

.. currentmodule:: physicsnemo.mesh.repair

``physicsnemo.mesh.repair`` fixes common mesh problems. Each repair is a
separate function, and two functions combine them:

- :func:`clean_mesh`, also ``mesh.clean()``, merges duplicate points,
  removes duplicate cells, and removes unused points.
- :func:`repair_mesh` runs a configurable sequence: remove degenerate cells,
  merge duplicate points, remove isolated points, and, if you enable them,
  fix the orientation and fill holes.

The single repairs are:

- :func:`remove_degenerate_cells`: remove cells with zero area or volume, or
  with a repeated point.
- :func:`remove_isolated_points`: remove points that no cell uses.
- :func:`fix_orientation`: give all cells a consistent winding.
- :func:`fill_holes`: close boundary loops of a triangle mesh.
- :func:`merge_duplicate_points`, :func:`remove_duplicate_cells`, and
  :func:`remove_unused_points`: tensor-level steps that take ``points``,
  ``cells``, and data, not a ``Mesh``.

Example
-------

The following example cleans and repairs a mesh made of two copies of one
sphere:

.. code:: python

    import torch
    from physicsnemo.mesh import Mesh
    from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
    from physicsnemo.mesh.repair import clean_mesh, fill_holes, fix_orientation, repair_mesh

    # Two copies of one sphere: every point and every cell appears twice.
    sphere = sphere_icosahedral.load(subdivisions=1)  # 42 points, 80 triangles
    mesh = Mesh(
        points=torch.cat([sphere.points, sphere.points]),
        cells=torch.cat([sphere.cells, sphere.cells + sphere.n_points]),
    )

    # The Mesh method returns only the mesh.
    cleaned = mesh.clean()
    print(cleaned.n_points, cleaned.n_cells)  # 42 80

    # The functions also return statistics.
    cleaned, stats = clean_mesh(mesh)
    print(stats["n_cells_before_dedup"], stats["n_cells_after_dedup"])  # 160 80

    repaired, repair_stats = repair_mesh(cleaned, fix_orientation=True)
    print(list(repair_stats))  # ['degenerates', 'merge_points', 'clean', 'isolated', 'orientation']

    oriented, orientation_stats = fix_orientation(cleaned)
    filled, hole_stats = fill_holes(cleaned)

Choosing a Function
-------------------

Use the following guidelines to choose a function:

- To remove duplicates after a merge or a file import, use ``mesh.clean()``.
  It merges points closer than ``tolerance`` (default ``1e-12``).
- To repair a mesh from an outside source, use :func:`repair_mesh`. Its
  ``tolerance`` default is ``1e-6``. It does not remove duplicate cells, so
  run :func:`clean_mesh` first if the mesh can have them.
- :func:`fill_holes` works only on meshes with a manifold dimension of two,
  and only closes loops of at most ``max_hole_edges`` edges (default 10).

Summary
-------

.. autosummary::
   :nosignatures:

   clean_mesh
   repair_mesh
   remove_degenerate_cells
   remove_isolated_points
   fix_orientation
   fill_holes
   merge_duplicate_points
   remove_duplicate_cells
   remove_unused_points

Related Topics
--------------

Refer to the following pages for related topics:

- :doc:`validation` to find problems before you repair them.
- Tutorial: `Quality, Validation, and Repair
  <https://github.com/NVIDIA/physicsnemo/blob/main/examples/minimal/mesh/tutorial_7_quality_repair.ipynb>`_

API Reference
-------------

.. autofunction:: clean_mesh

.. autofunction:: repair_mesh

.. autofunction:: remove_degenerate_cells

.. autofunction:: remove_isolated_points

.. autofunction:: fix_orientation

.. autofunction:: fill_holes

.. autofunction:: merge_duplicate_points

.. autofunction:: remove_duplicate_cells

.. autofunction:: remove_unused_points
