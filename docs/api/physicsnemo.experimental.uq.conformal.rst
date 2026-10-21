PhysicsNeMo Conformal Prediction
================================

.. currentmodule:: physicsnemo.experimental.uq.conformal

Calibrated prediction intervals for the field outputs of a trained model,
without retraining it.

.. warning::

    This module is experimental. Its API may change without a deprecation
    period.

Overview
--------

A surrogate model returns a field, for example surface pressure on a car,
but not how far that field can be from the truth. Conformal prediction
answers that question after training. It measures the model's errors on a
held-out *calibration set*, picks a threshold from those errors, and turns
every new prediction into a lower and an upper bound, ``(lo, hi)``, with a
stated probability of containing the true field.

The guarantee holds for any model and any data distribution. It needs only
one assumption: the calibration samples and the new samples are
*exchangeable*, for example drawn independently from the same distribution.
If the deployment data drifts away from the calibration data, the
guarantee no longer holds.

The workflow has three steps:

1. Pass each calibration sample, the prediction and the target, to a
   calibrator's ``update``.
2. Call ``finalize`` to fit the thresholds and get a
   :class:`ConformalPredictor`.
3. Call ``predictor.predict_interval`` on each new model output to get
   ``(lo, hi)``.

The predictor can be saved, loaded elsewhere, and checked on held-out data
with a :class:`CoverageAccumulator`.


Quick Start
-----------

.. code:: python

    from physicsnemo.experimental.uq.conformal import (
        AbsoluteErrorScore,
        RiskControlCalibrator,
    )

    calibrator = RiskControlCalibrator(AbsoluteErrorScore(), alpha=0.1)
    for prediction, target in calibration_set:  # meshes may differ per sample
        calibrator.update(prediction, target)
    predictor = calibrator.finalize()

    lo, hi = predictor.predict_interval(model(new_input))

With ``alpha=0.1``, a new field has on average at most 10% of its points
outside ``[lo, hi]``. ``prediction`` and ``target`` have shape
:math:`(n_{\text{points}}, *\text{dims})`, for example
``(n_points, 4)`` for pressure and wall shear stress.


Choosing a Calibrator
---------------------

The calibrator sets what the guarantee covers. Pick it by the guarantee
your application needs and by whether every sample uses the same mesh.

.. list-table::
   :header-rows: 1
   :widths: 24 40 18 18

   * - Calibrator
     - Guarantee
     - Meshes
     - Band width
   * - :class:`CellwiseCalibrator`
     - Each output element is covered with probability at least
       :math:`1 - \alpha`, one element at a time.
     - Same mesh, same point order
     - A separate width per element
   * - :class:`FunctionalBandCalibrator`
     - A new field lies inside the band at *every* point with probability
       at least :math:`1 - \alpha`.
     - May differ
     - Widest
   * - :class:`RiskControlCalibrator`
     - The expected fraction of points outside the band is at most
       :math:`\alpha`.
     - May differ
     - Narrowest of the two mesh-free calibrators

For fields on meshes that change between samples, such as CFD results on
different geometries, start with :class:`RiskControlCalibrator`. Use
:class:`FunctionalBandCalibrator` when a single miss anywhere on the field
matters, for example to bound the worst-case error over a surface. Use
:class:`CellwiseCalibrator` when every sample shares one grid or mesh and
you need an interval for each output element. It takes the mesh
coordinates as ``points=`` on every call and rejects any other mesh.


Choosing a Score
----------------

The score sets the shape of the interval. Pick it by what the model
outputs. The score reads any extra model outputs from the ``aux``
argument.

.. list-table::
   :header-rows: 1
   :widths: 28 30 22 20

   * - Score
     - Model output
     - ``aux`` entries
     - Interval
   * - :class:`AbsoluteErrorScore`
     - A point prediction :math:`\hat y`
     - None
     - :math:`[\hat y - t,\ \hat y + t]`
   * - :class:`NormalizedErrorScore`
     - A mean :math:`\mu` and a standard deviation :math:`\sigma`
     - ``"sigma"``
     - :math:`[\mu - t\sigma,\ \mu + t\sigma]`
   * - :class:`QuantileRegressionScore`
     - Lower and upper quantile heads
     - ``"lo"``, ``"hi"``
     - :math:`[q_{lo} - t,\ q_{hi} + t]`

:math:`t` is the threshold the calibrator fits. :class:`NormalizedErrorScore`
gives narrow intervals where the model is confident and wide intervals
where it is not, so it is the better choice when the model has a
:math:`\sigma` head, an MC-dropout spread, or an ensemble spread.

Pass ``sigma`` the same way at calibration and at prediction:

.. code:: python

    from physicsnemo.experimental.uq.conformal import (
        NormalizedErrorScore,
        RiskControlCalibrator,
    )

    calibrator = RiskControlCalibrator(NormalizedErrorScore(), alpha=0.1)
    for prediction, target, std in calibration_set:
        calibrator.update(prediction, target, aux={"sigma": std})
    predictor = calibrator.finalize()

    lo, hi = predictor.predict_interval(new_prediction, aux={"sigma": new_std})

:class:`FunctionalBandCalibrator` and :class:`RiskControlCalibrator` also
accept an :class:`AuxDifficulty`, which scales the threshold by a per-point
uncertainty read from ``aux``. Pair it with :class:`AbsoluteErrorScore`.
Pairing ``AuxDifficulty("sigma")`` with :class:`NormalizedErrorScore` would
scale by :math:`\sigma` twice, so the calibrators reject that pair.


Custom Scores
-------------

When none of the built-in scores fits the model, define your own. Subclass
:class:`NonconformityScore`, implement two methods, and register the class
with :func:`register_score`:

- ``score(prediction, target, *, aux=None)`` returns the per-element score;
  larger is worse.
- ``bounds(prediction, threshold, *, aux=None)`` returns ``(lo, hi)``, the
  targets whose score is at most ``threshold``.

For example, a band proportional to the prediction:

.. code:: python

    from physicsnemo.experimental.uq.conformal import (
        NonconformityScore,
        RiskControlCalibrator,
        check_score,
        register_score,
    )

    @register_score("relative_error")
    class RelativeErrorScore(NonconformityScore):
        config_fields = ("floor",)  # saved with the predictor

        def __init__(self, floor=1e-6):
            self.floor = floor

        def score(self, prediction, target, *, aux=None):
            scale = prediction.abs().clamp_min(self.floor)
            return (target - prediction).abs() / scale

        def bounds(self, prediction, threshold, *, aux=None):
            half = threshold * prediction.abs().clamp_min(self.floor)
            return prediction - half, prediction + half

    check_score(RelativeErrorScore(), prediction, target)
    calibrator = RiskControlCalibrator(RelativeErrorScore(floor=1e-3), alpha=0.1)

The coverage guarantee holds only if the two methods agree: for every
element, the targets with ``score <= t`` are exactly the values in
``bounds(t)``, and that interval grows with ``t``. :func:`check_score` tests
this on sample data, and every calibrator runs the same test on its first
sample. You do not handle low precision yourself: ``bounds`` receives the
prediction and threshold in float64, and the base class rounds the
interval outward to the prediction's dtype.

Set ``aux_keys`` to the ``aux`` entries the score reads, so calibrators
require and check them. Set ``config_fields`` to the constructor arguments
a saved predictor should store; their values must be ``bool``, ``int``,
``float``, or ``str``. To load a predictor saved with a custom score,
import the module that registers the score first.


Multiple Fields
---------------

Pass a ``TensorDict`` to calibrate several fields at once. Each field gets
its own threshold and its own guarantee. Nest ``aux`` by field name:

.. code:: python

    from tensordict import TensorDict

    from physicsnemo.experimental.uq.conformal import (
        NormalizedErrorScore,
        RiskControlCalibrator,
    )

    calibrator = RiskControlCalibrator(NormalizedErrorScore(), alpha=0.1)
    for sample in calibration_set:
        n_points = sample["pressure"].shape[0]
        prediction = TensorDict(
            {"pressure": sample["pressure"], "wss": sample["wss"]},
            batch_size=[n_points],
        )
        target = TensorDict(
            {"pressure": sample["pressure_true"], "wss": sample["wss_true"]},
            batch_size=[n_points],
        )
        aux = {
            "pressure": {"sigma": sample["pressure_std"]},
            "wss": {"sigma": sample["wss_std"]},
        }
        calibrator.update(prediction, target, aux=aux)
    predictor = calibrator.finalize()

``predict_interval`` then returns ``lo`` and ``hi`` as ``TensorDict``
containers with the same fields. To calibrate only some fields, pass
``keys=["pressure"]`` to the calibrator.


Saving and Checking a Predictor
-------------------------------

A fitted predictor saves to a single file that loads with
``torch.load(..., weights_only=True)``. Check its coverage on held-out data
that was not used for calibration. The coverage report measures the same
quantity the calibrator guarantees:

.. code:: python

    from physicsnemo.experimental.uq.conformal import ConformalPredictor

    predictor.save("predictor.pt", provenance={"dataset": "holdout-v1"})
    predictor = ConformalPredictor.load("predictor.pt", map_location="cuda")

    accumulator = predictor.coverage_accumulator()
    for prediction, target in test_set:
        lo, hi = predictor.predict_interval(prediction)
        accumulator.update(lo, hi, target)
    report = accumulator.finalize()

For a risk-control predictor, ``report["fields"]["tensor"]["empirical_mean_risk"]``
is the mean fraction of points per sample that fell outside the band. It
should be close to or below ``alpha``. A plain-tensor field is reported
under the key ``"tensor"``; ``TensorDict`` fields under their own names.


Practical Notes
---------------

**Calibration set size.**
    ``finalize`` needs at least :math:`(1 - \alpha) / \alpha` samples, for
    example 9 for ``alpha=0.1``. With more samples the fitted threshold
    varies less from one calibration set to the next.

**Single process.**
    Calibration does not communicate across ranks. Predictions may come
    from many GPUs, but send every calibration sample to the process that
    calls ``finalize``. A threshold fit on part of the samples has no
    guarantee for the full set.

**Memory.**
    :class:`CellwiseCalibrator` and :class:`RiskControlCalibrator` keep the
    calibration scores on the CPU until ``finalize``. Cellwise memory grows
    as :math:`n_{cal}` times the field size; risk control stores one
    float64 per calibration point. :class:`FunctionalBandCalibrator` stores
    one number per sample.

**Devices and dtypes.**
    Inputs may live on any device and use any floating-point dtype,
    including ``float16`` and ``bfloat16``. Interval endpoints are computed
    in float64 and rounded outward to the input dtype, so low-precision
    inputs do not break the guarantee.

**Separate data for the uncertainty estimate.**
    Whatever produces ``sigma`` or the ``AuxDifficulty`` scale must not see
    the calibration samples: fit it on training data or a separate split.


References
----------

- J. Lei, M. G'Sell, A. Rinaldo, R. J. Tibshirani, and L. Wasserman,
  `Distribution-Free Predictive Inference for Regression
  <https://arxiv.org/abs/1604.04173>`_, 2018. Split conformal prediction.
- Y. Romano, E. Patterson, and E. Candes, `Conformalized Quantile
  Regression <https://arxiv.org/abs/1905.03222>`_, 2019.
- J. Diquigiovanni, M. Fontana, and S. Vantini, `Conformal Prediction Bands
  for Multivariate Functional Data <https://arxiv.org/abs/2106.01792>`_,
  2021. Functional bands.
- A. N. Angelopoulos, S. Bates, A. Fisch, L. Lei, and T. Schuster,
  `Conformal Risk Control <https://arxiv.org/abs/2208.02814>`_, 2022.
- V. Gopakumar et al., `Uncertainty Quantification of
  Surrogate Models using Conformal Prediction
  <https://arxiv.org/abs/2408.09881>`_, 2024. Cellwise intervals on
  fields.


API Reference
-------------

Calibrators
^^^^^^^^^^^

.. autoclass:: physicsnemo.experimental.uq.conformal.CellwiseCalibrator
    :members:
    :inherited-members:

.. autoclass:: physicsnemo.experimental.uq.conformal.FunctionalBandCalibrator
    :members:
    :inherited-members:

.. autoclass:: physicsnemo.experimental.uq.conformal.RiskControlCalibrator
    :members:
    :inherited-members:

Scores
^^^^^^

.. autoclass:: physicsnemo.experimental.uq.conformal.AbsoluteErrorScore
    :members: score, interval

.. autoclass:: physicsnemo.experimental.uq.conformal.NormalizedErrorScore
    :members: score, interval

.. autoclass:: physicsnemo.experimental.uq.conformal.QuantileRegressionScore
    :members: score, interval

.. autoclass:: physicsnemo.experimental.uq.conformal.AuxDifficulty
    :members: __call__

Custom Score API
^^^^^^^^^^^^^^^^

.. autoclass:: physicsnemo.experimental.uq.conformal.NonconformityScore
    :members: score, bounds, interval

.. autofunction:: physicsnemo.experimental.uq.conformal.register_score

.. autofunction:: physicsnemo.experimental.uq.conformal.check_score

Predictor
^^^^^^^^^

.. autoclass:: physicsnemo.experimental.uq.conformal.ConformalPredictor
    :members:

Coverage Diagnostics
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: physicsnemo.experimental.uq.conformal.CoverageAccumulator
    :members:
