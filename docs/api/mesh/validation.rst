Validation and Quality
======================

.. currentmodule:: physicsnemo.mesh.validation

``physicsnemo.mesh.validation`` checks the integrity of a mesh and measures
the quality of its cells. It has three functions:

- :func:`validate`, also ``mesh.validate()``, checks the structure. By
  default, it checks index ranges, duplicate points, and degenerate cells.
  Options add checks for inverted cells and for manifold topology. It
  returns a ``dict`` with the key ``"valid"`` and a count for each problem,
  or raises ``ValueError`` if you pass ``raise_on_error=True``.
- :func:`compute_quality_metrics`, also the ``mesh.quality_metrics``
  property, returns a ``TensorDict`` with one value per cell: aspect ratio,
  edge length ratio, minimum and maximum angle, minimum and maximum edge
  length, and an overall ``quality_score``.
- :func:`compute_mesh_statistics`, also the ``mesh.statistics`` property,
  returns a ``dict`` of counts and of summaries (minimum, maximum, mean, and
  standard deviation) of cell areas, edge lengths, aspect ratios, and quality
  scores.

``validate_mesh`` is a compatibility name for :func:`validate`. It is pending
deprecation.

Example
-------

.. code:: python

    from physicsnemo.mesh.primitives.surfaces import sphere_icosahedral

    mesh = sphere_icosahedral.load(subdivisions=2)

    report = mesh.validate()
    print(report["valid"])  # True

    quality = mesh.quality_metrics  # TensorDict, batch size (n_cells,)
    print(list(quality.keys()))
    # ['aspect_ratio', 'edge_length_ratio', 'min_angle', 'max_angle',
    #  'min_edge_length', 'max_edge_length', 'quality_score']

    stats = mesh.statistics
    print(stats["n_cells"], stats["n_degenerate_cells"])  # 320 0

Summary
-------

.. autosummary::
   :nosignatures:

   validate
   compute_quality_metrics
   compute_mesh_statistics
   validate_mesh

See Also
--------

- :doc:`repair` to fix the problems that validation finds.
- :doc:`boundaries` for watertight and manifold checks.
- Tutorial: `Quality, Validation, and Repair
  <https://github.com/NVIDIA/physicsnemo/blob/main/examples/minimal/mesh/tutorial_7_quality_repair.ipynb>`_

API Reference
-------------

.. autofunction:: validate

.. autofunction:: compute_quality_metrics

.. autofunction:: compute_mesh_statistics

.. autofunction:: validate_mesh
