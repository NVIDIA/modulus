Curvature
=========

.. currentmodule:: physicsnemo.mesh.curvature

``physicsnemo.mesh.curvature`` computes discrete curvature at the points and
cells of a simplicial mesh. It has two kinds:

- Gaussian curvature :math:`K` is intrinsic: it depends only on lengths
  measured in the mesh (Gauss's Theorema Egregium), so it works for any
  codimension. It uses the
  `angle defect <https://en.wikipedia.org/wiki/Angular_defect>`_:
  :math:`K_i = (\theta_{\text{full}} - \sum_j \theta_{ij}) / A_i^*`, where
  :math:`\theta_{ij}` are the angles of the cells at point :math:`i` and
  :math:`A_i^*` is the dual (Voronoi) area of the point. It is available at
  points and at cells.
- Mean curvature :math:`H` is extrinsic: it depends on how the mesh bends in
  its ambient space. It uses the
  `cotangent Laplacian <https://en.wikipedia.org/wiki/Discrete_Laplace_operator#Mesh_Laplacians>`_
  :math:`L` of the point positions :math:`\mathbf{x}`:
  :math:`H_i = \|L \mathbf{x}\|_i / (2 A_i^*)`. It needs codimension 1
  (``n_spatial_dims == n_manifold_dims + 1``), so that the mesh has normals.
  Its sign follows the normals: positive where the surface bulges toward
  them. Boundary points get ``NaN`` unless you pass
  ``include_boundary=True``.

The same quantities are cached ``Mesh`` properties:
``mesh.gaussian_curvature_vertices``, ``mesh.gaussian_curvature_cells``, and
``mesh.mean_curvature_vertices``.

.. figure:: ../../img/mesh/airplane_mean_curvature.png
   :alt: Mean curvature on the surface of an airplane model
   :align: center
   :width: 80%

   Mean curvature on the PyVista airplane surface.

Example
-------

.. code:: python

    from physicsnemo.mesh.curvature import gaussian_curvature_vertices, mean_curvature_vertices
    from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

    # A sphere of radius r has K = 1 / r**2 and H = 1 / r.
    mesh = sphere_icosahedral.load(radius=2.0, subdivisions=3)
    K = gaussian_curvature_vertices(mesh)  # (n_points,)
    H = mean_curvature_vertices(mesh)  # (n_points,)
    print(K.mean().item(), H.mean().item())  # about 0.25 and 0.5

    # The cached Mesh properties give the same values.
    K = mesh.gaussian_curvature_vertices
    H = mesh.mean_curvature_vertices

Summary
-------

.. autosummary::
   :nosignatures:

   gaussian_curvature_vertices
   gaussian_curvature_cells
   mean_curvature_vertices

See Also
--------

- :doc:`geometry` for the angles and dual areas behind these formulas.
- Tutorial: `Discrete Calculus and Curvature
  <https://github.com/NVIDIA/physicsnemo/blob/main/examples/minimal/mesh/tutorial_5_calculus.ipynb>`_

API Reference
-------------

.. autofunction:: gaussian_curvature_vertices

.. autofunction:: gaussian_curvature_cells

.. autofunction:: mean_curvature_vertices
