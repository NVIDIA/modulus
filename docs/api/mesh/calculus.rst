Discrete Calculus
=================

.. currentmodule:: physicsnemo.mesh.calculus

``physicsnemo.mesh.calculus`` computes derivatives of fields on a mesh and
integrates fields over it. It has four differential operators:

- Gradient :math:`\nabla\varphi`: scalar to vector
- Divergence :math:`\nabla\cdot\mathbf{v}`: vector to scalar
- Curl :math:`\nabla\times\mathbf{v}`: vector to vector, for meshes in 3D
  space only
- Laplacian :math:`\Delta\varphi = \nabla\cdot\nabla\varphi`: scalar to
  scalar (the Laplace-Beltrami operator on a curved surface)

Two methods compute them:

- Weighted least squares (LSQ) fits a linear function to each point's
  neighborhood, as in finite-volume CFD. LSQ gives the gradient,
  divergence, and curl, at points and at cells.
- Discrete exterior calculus (DEC) uses the circumcentric dual mesh and the
  Hodge star, after Desbrun, Hirani, Leok, and Marsden
  (`arXiv:math/0508341 <https://arxiv.org/abs/math/0508341>`_). DEC gives the
  gradient, divergence, and Laplacian at points.

On a curved surface or curve, an intrinsic gradient lies in the tangent space
of the mesh, and an extrinsic gradient lies in the ambient space.

.. figure:: ../../img/mesh/calculus_lsq_vs_dec.png
   :alt: Two square maps of the gradient error, one for LSQ and one for DEC,
         dark inside with a bright rim on the boundary. Beside them, a log-log
         plot of the error against the mesh spacing: four parallel lines (LSQ
         and DEC, at interior and boundary vertices) next to a slope-1
         reference line. The boundary lines lie five to eight times higher
         than the interior lines.
   :align: center
   :width: 100%

   Error of the LSQ and DEC vertex gradients of a smooth field on a jittered
   grid. (a) Error maps at :math:`h = 1/16`: the error is largest on the
   boundary. (b) RMS error against the mesh spacing: both methods converge at
   first order on this grid, and the error at boundary vertices stays five to
   eight times larger. On a regular grid, the interior error falls at second
   order.

Example
-------

.. code:: python

    from physicsnemo.mesh.calculus import (
        compute_curl_points_lsq,
        compute_divergence_points_lsq,
        compute_laplacian_points_dec,
    )
    from physicsnemo.mesh.primitives.volumes import cube_volume

    mesh = cube_volume.load(subdivisions=4)  # tetrahedra in 3D

    # The gradient of T = x + 2y is (1, 2, 0). The Mesh method stores it as "T_gradient".
    mesh.point_data["T"] = mesh.points[:, 0] + 2 * mesh.points[:, 1]
    mesh = mesh.compute_point_derivatives(keys="T", method="lsq")
    print(mesh.point_data["T_gradient"][0])  # about (1, 2, 0)

    # v = (x, y, z) has divergence 3 and curl 0.
    velocity = mesh.points.clone()
    div_v = compute_divergence_points_lsq(mesh, velocity)  # (n_points,), about 3
    curl_v = compute_curl_points_lsq(mesh, velocity)  # (n_points, 3), about 0

    # The DEC Laplacian of u = x^2 + y^2 + z^2 is 6 at interior points.
    u = (mesh.points**2).sum(dim=-1)
    lap_u = compute_laplacian_points_dec(mesh, u)  # (n_points,)

Choosing a Method
-----------------

- Use :meth:`~physicsnemo.mesh.mesh.Mesh.compute_point_derivatives` or
  :meth:`~physicsnemo.mesh.mesh.Mesh.compute_cell_derivatives` to add
  gradients of named fields to the mesh data. ``method="lsq"`` is the
  default. ``method="dec"`` selects DEC, for point derivatives only:
  ``compute_cell_derivatives`` raises ``NotImplementedError`` for it.
- Use the ``compute_*`` functions for a tensor result without changing the
  mesh.
- For the Laplacian, use :func:`compute_laplacian_points_dec`. LSQ has no
  Laplacian.
- Expect larger errors at boundary points, where the neighborhood is one-sided
  (see the figure). The User Guide compares the two methods in detail.

Summary
-------

.. rubric:: Derivatives

.. autosummary::
   :nosignatures:

   compute_point_derivatives
   compute_cell_derivatives
   compute_gradient_points_lsq
   compute_gradient_points_dec
   compute_gradient_cells_lsq
   compute_divergence_points_lsq
   compute_divergence_points_dec
   compute_divergence_cells_lsq
   compute_curl_points_lsq
   compute_curl_cells_lsq
   compute_laplacian_points_dec

.. rubric:: Integration and measures

.. autosummary::
   :nosignatures:

   integrate
   integrate_cell_data
   integrate_point_data
   integrate_samples
   integrate_flux
   integrate_moment
   cell_measures
   point_measures
   set_cell_measures
   set_point_measures
   scale_measures
   lumped_point_measures

Effective Measures and Integration
----------------------------------

The reserved ``_effective_measure`` field stores one complete integration
measure per cell or point in ``cell_data`` or ``point_data``. Read these with
``cell_measures(mesh)`` or ``point_measures(mesh)``. They already include any
geometric contribution and sampling correction.

* ``mesh.integrate(field, data_source="cells")`` integrates piecewise-constant
  cell values. ``data_source="points"`` integrates piecewise-linear vertex
  values over the same cells. Both use cell measures, which default to the
  geometric simplex measures.
* ``mesh.integrate_samples(field)`` sums independent point samples times their
  point measures, regardless of connectivity. Explicit point measures are
  required. For counting measure, use an ordinary sum.

Use ``set_cell_measures`` or ``set_point_measures`` to assign measures. Point
measures require their represented ``dimension``:

- 0 for counting
- 1 for length
- 2 for area
- 3 for volume

``scale_measures`` multiplies existing measures by a scalar or a factor per
cell or point, such as an inverse sampling probability. Raw slicing does not
apply a sampling correction.

Converting cells to centroid samples transfers their measures automatically:

.. code:: python

    from physicsnemo.mesh.primitives.planar import unit_square

    square = unit_square.load(subdivisions=3)
    queries = square.to_point_cloud(point_source="cell_centroids")
    values = queries.points[:, 0]  # Integrate f(x, y) = x over the unit square.
    print(queries.integrate_samples(values))  # tensor(0.5000)

For vertex quadrature, ``lumped_point_measures(mesh)`` distributes each cell's
measure equally among its vertices without modifying the mesh. For finite
fields, these weights reproduce piecewise-linear integration.

See :doc:`transformations` for the geometric changes that keep reference
measures.

See Also
--------

- :doc:`curvature` and :doc:`geometry`.
- User Guide: `Using Mesh
  <https://docs.nvidia.com/physicsnemo/latest/user-guide/mesh/using_mesh.html>`_
- Tutorial: `Discrete Calculus and Curvature
  <https://github.com/NVIDIA/physicsnemo/blob/main/examples/minimal/mesh/tutorial_5_calculus.ipynb>`_

API Reference
-------------

Derivatives
^^^^^^^^^^^

.. autofunction:: compute_point_derivatives

.. autofunction:: compute_cell_derivatives

.. autofunction:: compute_gradient_points_lsq

.. autofunction:: compute_gradient_points_dec

.. autofunction:: compute_gradient_cells_lsq

.. autofunction:: compute_divergence_points_lsq

.. autofunction:: compute_divergence_points_dec

.. autofunction:: compute_divergence_cells_lsq

.. autofunction:: compute_curl_points_lsq

.. autofunction:: compute_curl_cells_lsq

.. autofunction:: compute_laplacian_points_dec

Integration and Measures
^^^^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: integrate

.. autofunction:: integrate_cell_data

.. autofunction:: integrate_point_data

.. autofunction:: integrate_samples

.. autofunction:: integrate_flux

.. autofunction:: integrate_moment

.. autofunction:: cell_measures

.. autofunction:: point_measures

.. autofunction:: set_cell_measures

.. autofunction:: set_point_measures

.. autofunction:: scale_measures

.. autofunction:: lumped_point_measures

.. py:data:: EFFECTIVE_MEASURE_KEY
   :value: "_effective_measure"

   The reserved ``cell_data`` and ``point_data`` key that stores measures.

.. py:data:: POINT_MEASURE_DIMENSION_KEY
   :value: "_point_measure_dimension"

   The reserved ``global_data`` key that stores the dimension of point
   measures.
