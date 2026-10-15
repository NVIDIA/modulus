Tessellation
============

.. currentmodule:: physicsnemo.mesh.tessellation

``physicsnemo.mesh.tessellation`` has two jobs:

- :func:`triangulate` decomposes the polygons of a *polygon soup* into the
  simplices that the :class:`~physicsnemo.mesh.mesh.Mesh` data structure
  stores. It uses a vectorized vertex-0 fan for convex polygons and `ear
  clipping <https://en.wikipedia.org/wiki/Polygon_triangulation>`_ for the
  rare non-convex ones.
- :func:`fill_interior` fills a closed boundary mesh with quality simplices
  (see `Exact-Boundary Interior Filling`_ below).

Each polygon must be simple (no self-intersections), with no repeated
consecutive vertices or zero-length edges. :func:`triangulate` does not detect
these cases, and the result is undefined. Ear clipping also assumes that a
non-convex polygon is approximately planar.

A *polygon soup* is a flat list of polygonal cells, each given only by its
vertex ring, with no shared-edge connectivity between cells -- the form a
surface mesh takes when read straight from a polygonal face list (e.g. a VTP or
STL file). Equivalently, it is a surface mesh whose cells may be arbitrary
polygons rather than triangles.

.. note::

   This is unrelated to the (now deprecated) ``Tessellation`` *geometry* class
   in PhysicsNeMo-Sym (``physicsnemo.sym.geometry.tessellation``), which
   represents a tessellated STL surface as a solid for sampling
   surface/interior points and signed-distance values. The utilities here
   instead split the polygonal cells of an existing mesh into simplices.

Handling non-convex polygons correctly matters for any unsigned-area-weighted
quantity (wall-shear / viscous force integration, or total wetted area): the
signed *vector* area of a vertex-0 fan telescopes to the polygon's total area regardless
of convexity, but the sum of the *unsigned* triangle areas does not.

Every ``k``-gon yields exactly ``k - 2`` triangles, so per-polygon data is
broadcast to the output identically in both paths using the returned
``parent_index``.

.. code:: python

    import torch
    from physicsnemo.mesh import Mesh
    from physicsnemo.mesh.neighbors import Adjacency
    from physicsnemo.mesh.tessellation import triangulate

    # A polygon soup as a cell-to-vertex Adjacency (CSR): one quad (vertices 0-3).
    points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                           [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    polygons = Adjacency(offsets=torch.tensor([0, 4]),
                         indices=torch.tensor([0, 1, 2, 3]))

    # Low-level: triangle connectivity plus the parent-polygon index.
    cells, parent_index = triangulate(points, polygons)
    cells          # tensor([[0, 1, 2], [0, 2, 3]])
    parent_index   # tensor([0, 0]); broadcast data via cell_data[parent_index]

    # High-level: build a Mesh directly, broadcasting per-polygon cell data.
    mesh = Mesh.from_polygons(
        points, polygons, cell_data={"pressure": torch.tensor([2.5])}
    )

Exact-Boundary Interior Filling
-------------------------------

Beyond decomposing existing cells, the package also *generates* quality
meshes. :func:`fill_interior` takes a closed codimension-one boundary
``Mesh`` and fills the enclosed interior with quality simplices. In 2D, the
boundary is an edge mesh forming one or more loops, in any order and
orientation. The algorithm resolves holes, multiple components, and
islands-inside-holes automatically through containment.

The algorithm uses constrained Delaunay triangulation
(Bowyer--Watson insertion with constrained-edge recovery, holes removed
topologically by even-odd parity flood fill) followed by Ruppert's
Delaunay refinement, so that every 2D output triangle satisfies the
requested minimum angle and, optionally, a maximum cell size.

The minimum-angle guarantee has these conditions:

- Adjacent input segments must meet at 60 degrees or more. A sharper input
  corner raises ``ValueError``. To fill such a boundary, pass
  ``min_angle_degrees=0.0`` for an unrefined constrained triangulation, or
  blunt the corner.
- ``min_angle_degrees`` must be in :math:`[0, 33]`. The default is 30.
- Ruppert's proof of termination covers bounds up to about 20.7 degrees.
  Larger bounds, such as the default, terminate in practice. If refinement
  exceeds its budget of inserted vertices, it raises ``RuntimeError``.

Optional optimal-Delaunay-triangulation (ODT) smoothing, controlled by
``smooth_iterations``, moves each interior vertex to the area-weighted
average of its incident triangles' circumcenters (Chen and Xu 2004). This
improves the *typical* angle while preserving both bounds. The
exact-boundary contract provides three guarantees:

- Every input vertex that an edge references appears bit-identically in
  the output, in the leading rows and in input order. Vertices that no edge
  references are dropped.
- The pipeline only ever *subdivides* boundary facets and never moves
  them.
- The entire process is deterministic.

You can attach provenance fields (``"boundary_marker"`` and
``"source_point"``) to the output's ``point_data`` by setting
``provenance=True``. By default, the pipeline claims no keys in the
user-owned namespace.

The contract is dimension-generic by design. The ``n = 3`` case (a
watertight surface ``Mesh[2, 3]`` producing tetrahedra) currently raises
:class:`NotImplementedError`: exact 3D boundary recovery is a
substantially harder problem that requires its own implementation effort.
For implicit domains, or approximate volume meshing of a surface through
its SDF, refer to :func:`physicsnemo.mesh.generate.mesh_implicit_domain`.

.. code:: python

    import math
    import torch
    from physicsnemo.mesh import Mesh
    from physicsnemo.mesh.tessellation import fill_interior

    square = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    edges = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]])
    h = 0.1  # target interior edge length
    filled = fill_interior(
        Mesh(points=square, cells=edges),
        max_cell_size=math.sqrt(3.0) / 4.0 * h * h,
        min_angle_degrees=30.0,
        smooth_iterations=3,  # optional ODT smoothing
    )

:func:`polygon_interior_point` is a companion utility returning a point
strictly inside a simple polygon.

Summary
-------

.. autosummary::
   :nosignatures:

   triangulate
   fill_interior
   polygon_interior_point

See Also
--------

- :doc:`generate` for meshes of implicit domains, in any dimension.
- :doc:`io` to load polygon meshes from files.
- Tutorials: `Loading, Converting, and Saving Meshes
  <https://github.com/NVIDIA/physicsnemo/blob/main/examples/minimal/mesh/tutorial_2_io_interop.ipynb>`_
  and `Primitives and Mesh Generation
  <https://github.com/NVIDIA/physicsnemo/blob/main/examples/minimal/mesh/tutorial_3_primitives_generation.ipynb>`_

API Reference
-------------

.. autofunction:: triangulate

.. autofunction:: fill_interior

.. autofunction:: polygon_interior_point
