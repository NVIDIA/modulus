Generate
========

.. currentmodule:: physicsnemo.mesh.generate

``physicsnemo.mesh.generate`` generates meshes from implicit
representations. It provides isosurface extraction and dimension-generic
volume meshing.

Choosing a Mesher
-----------------

PhysicsNeMo-Mesh has three generation tools with distinct contracts:

.. list-table::
   :header-rows: 1

   * - Tool
     - Input
     - Output
     - Use When
   * - ``physicsnemo.mesh.tessellation.fill_interior``
     - closed boundary ``Mesh`` (2D today)
     - volume ``Mesh``, exact boundary, deterministic, a guaranteed
       minimum angle of up to 33 degrees for input corners of 60 degrees or
       more (refer to :doc:`tessellation`)
     - the boundary discretization is the contract (coupling, boundary
       conditions on each boundary, guaranteed worst-element quality)
   * - :func:`mesh_implicit_domain`
     - implicit function :math:`\varphi(x)`, such as a signed distance
       function (SDF), a level set, or a neural field
     - volume ``Mesh`` in any dimension, boundary approximated to
       :math:`O(h^2)` where it is smooth (sharp corners are rounded at the
       ``h`` scale unless passed as ``feature_points``), GPU-native,
       differentiable refit
     - geometry is implicit, you need 3D or ND volumes, GPU scale, or
       gradients with regard to shape
   * - :func:`marching_cubes`
     - 3D scalar field on a grid
     - surface ``Mesh`` (the isosurface only, no interior)
     - the boundary surface itself is the deliverable, for visualization or
       boundary representation (B-rep) export

Volume Meshing of Implicit Domains
----------------------------------

:func:`mesh_implicit_domain` generates a simplex volume mesh of
``{x : phi(x) < 0}`` for an arbitrary implicit function ``phi`` (a signed
distance function, or any level set with a usable gradient, including
neural implicit fields), in *any spatial dimension*, entirely in PyTorch
tensor operations on CPU or CUDA. The meshed set is ``{phi < 0}`` intersected
with the bounding box. Where the domain reaches the box, the generator
treats its faces as a boundary, so external-flow "box minus obstacle"
domains work directly. Every optimization step is validity-gated, so when
the function returns, the mesh is positively oriented and has a
closed-manifold boundary. Difficult inputs lower element quality, which
``full_output=True`` reports in a diagnostics dictionary, but do not by
themselves stop generation. The function raises ``ValueError`` when no
lattice cell fits inside the domain at this ``h``, or when the coverage guard
trips. The
coverage guard raises the error, rather than dropping geometry, when the
domain has features below the target edge length ``h`` or when coverage
cannot be certified. The latter can happen, for example, if a neural
field returns NaN inside the box when queried outside its training range.
Without ``feature_points``, sharp convex corners are rounded off at the
``h`` scale. Pass the corners as ``feature_points`` to interpolate them
exactly.

:func:`refit_mesh_to_implicit` is the differentiable companion. It
re-projects a mesh's boundary onto ``phi = 0`` with graph-preserving
Newton steps at fixed topology, so gradients flow from mesh coordinates to
shape parameters inside ``phi``. This enables meshing as a
differentiable layer for shape optimization. For a box-minus-obstacle mesh,
pass the same box as ``bounds=``, so that the vertices on the box faces stay
on them.

The following example meshes a spherical shell and differentiates its volume
with respect to the outer radius:

.. code:: python

    import torch
    from physicsnemo.mesh.generate import (
        mesh_implicit_domain,
        refit_mesh_to_implicit,
        sdf_difference,
        sdf_sphere,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # A spherical shell meshed with tetrahedra. On a GPU, a finer h such as 0.05 is fast.
    shell = sdf_difference(sdf_sphere([0.0] * 3, 0.8), sdf_sphere([0.0] * 3, 0.35))
    mesh = mesh_implicit_domain(shell, ([-1] * 3, [1] * 3), h=0.1, device=device)

    # Differentiable geometry at fixed topology: the shell volume as a function of
    # the outer radius r. Its derivative is close to the area 4 pi r**2, about 8.
    r = torch.tensor(0.8, requires_grad=True)
    shell_r = sdf_difference(sdf_sphere([0.0] * 3, r), sdf_sphere([0.0] * 3, 0.35))
    refit = refit_mesh_to_implicit(mesh, shell_r)
    refit.cell_areas.sum().backward()
    print(r.grad)  # about 8

The refit does not gate its Newton steps, so that it stays differentiable. It
warns when the projection inverts cells. On this coarse mesh, it warns that
about two percent of the cells inverted, most of them near the outer sphere.

The module also provides the following signed-distance building blocks for
convenience:

- :func:`sdf_sphere`
- :func:`sdf_box`
- :func:`sdf_polygon_2d`
- :func:`sdf_union`, a constructive solid geometry (CSG) combinator
- :func:`sdf_intersection`, a CSG combinator
- :func:`sdf_difference`, a CSG combinator
- :func:`project_to_zero_set`, a Newton projection onto ``phi = 0``

Any callable with the signature ``phi(x: (..., d)) -> (...)`` also works,
if it meets the following conditions:

- It is negative inside the domain and positive outside.
- It is differentiable, almost everywhere, under autograd.
- It has a nonzero gradient near ``phi = 0``.
- It does not return NaN inside the bounds.

Isosurface Extraction
---------------------

:func:`marching_cubes` extracts a triangle isosurface from a 3D scalar
field.

Summary
-------

.. autosummary::
   :nosignatures:

   mesh_implicit_domain
   refit_mesh_to_implicit
   marching_cubes
   sdf_sphere
   sdf_box
   sdf_polygon_2d
   sdf_union
   sdf_intersection
   sdf_difference
   project_to_zero_set

Related Topics
--------------

For more information, refer to the following resources:

- :doc:`tessellation` for exact-boundary filling of a closed boundary mesh.
- Tutorial: `Primitives and Mesh Generation
  <https://github.com/NVIDIA/physicsnemo/blob/main/examples/minimal/mesh/tutorial_3_primitives_generation.ipynb>`_

API Reference
-------------

.. autofunction:: mesh_implicit_domain

.. autofunction:: refit_mesh_to_implicit

.. autofunction:: marching_cubes

.. autofunction:: sdf_sphere

.. autofunction:: sdf_box

.. autofunction:: sdf_polygon_2d

.. autofunction:: sdf_union

.. autofunction:: sdf_intersection

.. autofunction:: sdf_difference

.. autofunction:: project_to_zero_set
