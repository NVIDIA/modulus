Modulus Metrics
===============

.. automodule:: modulus.metrics
.. currentmodule:: modulus.metrics

Basics
-------

Modulus provides several general and domain-specific metric calculations you can
leverage in your custom training and inference workflows. These metrics are optimized to
oprate on PyTorch tensors. 

A summary of all the available metrics can be found below:

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Metric
     - Description
   * - `modulus.metrics.general.mse.mse <#modulus.metrics.general.mse.mse>`_
     - Mean Squared error between two tensors
   * - `modulus.metrics.general.mse.rmse <#modulus.metrics.general.mse.rmse>`_
     - Root Mean Squared error between two tensors
   * - `modulus.metrics.general.histogram.Histogram <#modulus.metrics.general.histogram.Histogram>`_
     - Convenience class for computing histograms, possibly as a part of a distributed or iterative environment
   * - `modulus.metrics.general.histogram.cdf <#modulus.metrics.general.histogram.cdf>`_
     - Cumulative density function of a set of tensors over the leading dimension
   * - `modulus.metrics.general.histogram.histogram <#modulus.metrics.general.histogram.histogram>`_
     - Histogram of a set of tensors over the leading dimension
   * - `modulus.metrics.general.histogram.normal_cdf <#modulus.metrics.general.histogram.normal_cdf>`_
     - Cumulative density function of a normal variable with given mean and standard deviation
   * - `modulus.metrics.general.histogram.normal_pdf <#modulus.metrics.general.histogram.normal_pdf>`_
     - Probability density function of a normal variable with given mean and standard deviation
   * - `modulus.metrics.general.calibration.find_rank <#modulus.metrics.general.calibration.find_rank>`_
     - Find the rank of the observation with respect to the given counts and bins
   * - `modulus.metrics.general.calibration.rank_probability_score <#modulus.metrics.general.calibration.rank_probability_score>`_
     - Rank Probability Score for the passed ranks
   * - `modulus.metrics.general.crps.crps <#modulus.metrics.general.crps.crps>`_
     - Local Continuous Ranked Probability Score (CRPS) by computing a histogram and CDF of the predictions
   * - `modulus.metrics.general.ensemble_metrics.EnsembleMetrics <#modulus.metrics.general.ensemble_metrics.EnsembleMetrics>`_
     - Abstract class for ensemble performance related metrics, useful for distributed and sequential computations
   * - `modulus.metrics.general.ensemble_metrics.Mean <#modulus.metrics.general.ensemble_metrics.Mean>`_
     - Abstract class for computing mean over batched or ensemble dimension, useful for distributed and sequential computation
   * - `modulus.metrics.general.ensemble_metrics.Variance <#modulus.metrics.general.ensemble_metrics.Variance>`_
     - Utility class that computes the variance over a batched or ensemble dimension, useful for distributed and sequential computation
   * - `modulus.metrics.general.reduction.WeightedMean <#modulus.metrics.general.reduction.WeightedMean>`_
     - Weighted Mean
   * - `modulus.metrics.general.reduction.WeightedStatistic <#modulus.metrics.general.reduction.WeightedStatistic>`_
     - Weighted Statistic
   * - `modulus.metrics.general.reduction.WeightedVariance <#modulus.metrics.general.reduction.WeightedVariance>`_
     - Weighted Variance
   * - `modulus.metrics.general.wasserstein.wasserstein <#modulus.metrics.general.wasserstein.wasserstein>`_
     - 1-Wasserstein distance between two discrete CDF functions
   * - `modulus.metrics.climate.acc.acc <#modulus.metrics.climate.acc.acc>`_
     - Anomaly Correlation Coefficient
   * - `modulus.metrics.climate.efi.efi <#modulus.metrics.climate.efi.efi>`_
     - Extreme Forecast Index (EFI) for an ensemble forecast against a climatological distribution
   * - `modulus.metrics.climate.efi.normalized_entropy <#modulus.metrics.climate.efi.normalized_entropy>`_
     - Relative entropy, or surprise, of using the prediction distribution as opposed to the climatology distribution.
   * - `modulus.metrics.climate.reduction.global_mean <#modulus.metrics.climate.reduction.global_mean>`_
     - Global mean
   * - `modulus.metrics.climate.reduction.global_var <#modulus.metrics.climate.reduction.global_var>`_
     - Global variance
   * - `modulus.metrics.climate.reduction.zonal_mean <#modulus.metrics.climate.reduction.zonal_mean>`_
     - Zonal mean, weighting over the latitude direction
   * - `modulus.metrics.climate.reduction.zonal_var <#modulus.metrics.climate.reduction.zonal_var>`_
     - Zonal variance, weighting over the latitude direction

Below shows some examples of how to use these metrics in your own workflows. 

To compute RMSE metric:

.. code:: python

    >>> import torch
    >>> from modulus.metrics.general.mse import rmse
    >>> pred_tensor = torch.randn(16, 32)
    >>> targ_tensor = torch.randn(16, 32)
    >>> rmse(pred_tensor, targ_tensor)
    tensor(1.4781)
    
To compute the Anomaly Correlation Coefficient, a metric widely used in weather and
climate sciences:

.. code:: python

    >>> import torch
    >>> import numpy as np
    >>> from modulus.metrics.climate.acc import acc
    >>> channels = 1
    >>> img_shape = (32, 64)
    >>> time_means = np.pi / 2 * np.ones((channels, img_shape[0], img_shape[1]), dtype=np.float32)
    >>> x = np.linspace(-180, 180, img_shape[1], dtype=np.float32)
    >>> y = np.linspace(-90, 90, img_shape[0], dtype=np.float32)
    >>> xv, yv = np.meshgrid(x, y)
    >>> pred_tensor_np = np.cos(2 * np.pi * yv / (180))
    >>> targ_tensor_np = np.cos(np.pi * yv / (180))
    >>> pred_tensor = torch.from_numpy(pred_tensor_np).expand(channels, -1, -1)
    >>> targ_tensor = torch.from_numpy(targ_tensor_np).expand(channels, -1, -1)
    >>> means_tensor = torch.from_numpy(time_means)
    >>> lat = torch.from_numpy(y)
    >>> acc(pred_tensor, targ_tensor, means_tensor, lat)
    tensor([0.9841])


.. autosummary::
   :toctree: generated

General
---------

.. automodule:: modulus.metrics.general.mse
    :members:
    :show-inheritance:

.. _histogram:

.. automodule:: modulus.metrics.general.histogram
    :members:
    :show-inheritance:

.. automodule:: modulus.metrics.general.entropy
    :members:
    :show-inheritance:

.. automodule:: modulus.metrics.general.calibration
    :members:
    :show-inheritance:

.. automodule:: modulus.metrics.general.crps
    :members:
    :show-inheritance:

.. automodule:: modulus.metrics.general.ensemble_metrics
    :members:
    :show-inheritance:

.. automodule:: modulus.metrics.general.reduction
    :members:
    :show-inheritance:

.. automodule:: modulus.metrics.general.wasserstein
    :members:
    :show-inheritance:

Weather and climate metrics
---------------------------

.. automodule:: modulus.metrics.climate.acc
    :members:
    :show-inheritance:

.. automodule:: modulus.metrics.climate.efi
    :members:
    :show-inheritance:

.. automodule:: modulus.metrics.climate.reduction
    :members:
    :show-inheritance:


