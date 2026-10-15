Smoothing
=========

.. currentmodule:: physicsnemo.mesh.smoothing

``physicsnemo.mesh.smoothing`` makes a mesh more regular by moving its
points. It has one function, :func:`smooth_laplacian`, for
`Laplacian smoothing <https://en.wikipedia.org/wiki/Laplacian_smoothing>`_:
each iteration moves each point toward a weighted average of its neighbors.
Meshes of codimension 1 with a manifold dimension of 2 or more use cotangent
weights. Other meshes use uniform weights. The connectivity does not change.

By default, boundary points do not move (``preserve_boundaries=True``). On a
mesh of codimension 1, set ``preserve_features=True`` to also hold the points
of edges with a dihedral angle above ``feature_angle`` degrees.

Example
-------

.. code:: python

    from physicsnemo.mesh.primitives.procedural import lumpy_sphere
    from physicsnemo.mesh.smoothing import smooth_laplacian

    mesh = lumpy_sphere.load(subdivisions=3, seed=0)
    smoothed = smooth_laplacian(mesh, n_iter=50, relaxation_factor=0.1)
    print(smoothed.n_points == mesh.n_points)  # True

``relaxation_factor`` sets how far each point moves toward the average in one
iteration. ``n_iter`` sets the number of iterations. Many iterations shrink a
closed surface: 500 iterations with ``relaxation_factor=0.1`` reduce the mean
radius of a unit icosahedral sphere to about 0.57. To change the connectivity,
use :doc:`remeshing` or :doc:`subdivision`.

Summary
-------

.. autosummary::
   :nosignatures:

   smooth_laplacian

See Also
--------

- Tutorial: `Quality, Validation, and Repair
  <https://github.com/NVIDIA/physicsnemo/blob/main/examples/minimal/mesh/tutorial_7_quality_repair.ipynb>`_

API Reference
-------------

.. autofunction:: smooth_laplacian
