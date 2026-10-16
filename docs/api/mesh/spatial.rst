Spatial Queries
===============

.. currentmodule:: physicsnemo.mesh.spatial

``physicsnemo.mesh.spatial`` has data structures and functions for geometric
queries on large meshes: a bounding volume hierarchy, a cluster tree, and a
signed distance field. Unlike :doc:`neighbors`, these queries use distances,
not connectivity.

The :class:`BVH` (bounding volume hierarchy) is a tree of axis-aligned
bounding boxes over the cells of a mesh. It speeds up two queries:

- Point containment: the cell, if any, that contains each query point
- Nearest-cell search: the closest cell to each query point

The sampling module uses the BVH internally, in
:func:`~physicsnemo.mesh.sampling.sample_data_at_points` and
:func:`~physicsnemo.mesh.sampling.find_containing_cells`, to avoid brute-force
search over all cells. The following example builds a BVH over a sphere and
finds the candidate cells of random query points:

.. code:: python

    import torch
    from physicsnemo.mesh.spatial import BVH
    from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

    mesh = sphere_icosahedral.load(subdivisions=3)
    bvh = BVH.from_mesh(mesh)

    query_points = torch.randn(1000, 3)
    candidate_cells = bvh.find_candidate_cells(query_points)  # an Adjacency

:class:`ClusterTree` is a hierarchy of point clusters for Barnes-Hut
acceleration of kernel and attention operators. Far-apart clusters interact
through aggregates instead of point by point. :class:`DualInteractionPlan`
and :class:`SourceAggregates` hold the plan and the cluster aggregates of
that computation.

Signed Distance Field
---------------------

:func:`signed_distance_field` computes the signed distance from a set of
query points to a triangle surface mesh, together with the closest point on the
surface for each query. It is a :class:`~physicsnemo.mesh.Mesh`-typed wrapper
around the Warp-backed :func:`physicsnemo.nn.functional.signed_distance_field`
function, which runs NVIDIA Warp mesh queries on CPU and CUDA.

``use_sign_winding_number`` selects one of two methods for the sign:

- ``False`` (default): the angle-weighted pseudo-normal of the closest mesh
  feature (``wp.mesh_query_point_sign_normal``). This method is fast and
  suits watertight meshes.
- ``True``: the generalized winding number
  (``wp.mesh_query_point_sign_winding_number``). This method suits open or
  self-intersecting geometry, such as a triangle soup.

The following example computes the signed distance from two points to a unit
sphere:

.. code:: python

    import torch
    from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral
    from physicsnemo.mesh.spatial import signed_distance_field

    sphere = sphere_icosahedral.load(subdivisions=3)  # unit sphere, 1280 triangles
    query = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    sdf, hit_points, hit_faces = signed_distance_field(sphere, query)
    print(sdf)  # about (-1, 1): negative inside, positive outside
    # hit_points: (2, 3) closest surface points
    # hit_faces:  (2,)   index of the nearest face in sphere.cells

Summary
-------

.. autosummary::
   :nosignatures:

   BVH
   ClusterTree
   DualInteractionPlan
   SourceAggregates
   signed_distance_field

Related Topics
--------------

Refer to the following pages for related topics:

- :doc:`sampling`, which uses the BVH to interpolate data at query points.
- Tutorial: `Neighbors, Adjacency, and Spatial Queries
  <https://github.com/NVIDIA/physicsnemo/blob/main/examples/minimal/mesh/tutorial_6_neighbors_spatial.ipynb>`_

API Reference
-------------

.. automodule:: physicsnemo.mesh.spatial
   :members:
   :show-inheritance:
