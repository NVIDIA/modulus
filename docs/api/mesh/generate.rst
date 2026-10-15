Generate
========

.. currentmodule:: physicsnemo.mesh.generate

Mesh generation from implicit representations: isosurface extraction and
dimension-generic volume meshing.

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
     - volume ``Mesh``, exact boundary, guaranteed angles,
       deterministic
     - the boundary discretization is the contract (coupling, per-boundary
       BCs, guaranteed worst-element quality)
   * - :func:`mesh_implicit_domain`
     - implicit function :math:`\varphi(x)` (SDF, level set, neural field)
     - volume ``Mesh`` in any dimension, boundary approximated to
       :math:`O(h^2)`, GPU-native, differentiable refit
     - geometry is implicit, you need 3D/ND volumes, GPU scale, or
       gradients with regard to shape
   * - :func:`marching_cubes`
     - 3D scalar field on a grid
     - surface ``Mesh`` (the isosurface only, no interior)
     - the boundary surface itself is the deliverable (visualization,
       B-rep export)

Volume Meshing of Implicit Domains
----------------------------------

:func:`mesh_implicit_domain` generates a simplex volume mesh of
``{x : phi(x) < 0}`` for an arbitrary implicit function ``phi`` (a signed
distance function, or any level set with a usable gradient, including
neural implicit fields), in *any spatial dimension*, entirely in PyTorch
tensor ops on CPU or CUDA. The meshed set is ``{phi < 0}`` intersected
with the bounding box. Where the domain reaches the box, the generator
treats its faces as a boundary, so external-flow "box minus obstacle"
domains work directly. Every
optimization step is validity-gated, so it always returns a positively
oriented mesh with a closed-manifold boundary. Difficult inputs degrade
element quality (reported in diagnostics), but never prevent the mesh
from being generated. A coverage guard raises an error, rather than
silently dropping geometry, when the domain
has features below the target edge length ``h`` or when coverage cannot
be certified at all. The latter can happen, for example, if a neural field
returns NaN inside the box when queried outside its training range. You can
interpolate sharp corners exactly through ``feature_points``.

:func:`refit_mesh_to_implicit` is the differentiable companion: it
re-projects a mesh's boundary onto ``phi = 0`` with graph-preserving
Newton steps at fixed topology, so gradients flow from mesh coordinates to
shape parameters inside ``phi``. This enables meshing as a
differentiable layer for shape optimization.

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
warns when the projection inverts cells. On this coarse mesh, it inverts a
few cells near the inner sphere.

The following signed-distance building blocks are provided for
convenience:

- :func:`sdf_sphere`
- :func:`sdf_box`
- :func:`sdf_polygon_2d`
- :func:`sdf_union` (CSG combinator)
- :func:`sdf_intersection` (CSG combinator)
- :func:`sdf_difference` (CSG combinator)
- :func:`project_to_zero_set` (Newton projection onto ``phi = 0``)

Any callable with the signature ``phi(x: (..., d)) -> (...)`` also works.

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

See Also
--------

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
