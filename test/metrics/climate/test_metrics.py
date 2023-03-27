# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch
import torch.distributed as dist
import numpy as np
import os
from modulus.metrics.general.mse import mse, rmse
import modulus.metrics.general.histogram as hist
import modulus.metrics.general.ensemble_metrics as em
import modulus.metrics.general.crps as crps
import modulus.metrics.general.wasserstein as w
from modulus.metrics.climate.acc import acc
import modulus.metrics.climate.reduction as clim_red
import modulus.metrics.general.reduction as gen_red

from modulus.distributed.manager import DistributedManager

Tensor = torch.Tensor


@pytest.fixture
def test_data(channels=2, img_shape=(721, 1440)):
    # create dummy data
    time_means = (
        np.pi / 2 * np.ones((channels, img_shape[0], img_shape[1]), dtype=np.float32)
    )

    # Set lat/lon in terms of degrees (for use with _compute_lat_weights)
    x = np.linspace(-180, 180, img_shape[1], dtype=np.float32)
    y = np.linspace(-90, 90, img_shape[0], dtype=np.float32)
    xv, yv = np.meshgrid(x, y)

    pred_tensor_np = np.cos(2 * np.pi * yv / (180))
    targ_tensor_np = np.cos(np.pi * yv / (180))

    return channels, x, y, pred_tensor_np, targ_tensor_np, time_means


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_climate_acc_mse(test_data, device, rtol: float = 1e-3, atol: float = 1e-3):
    channels, lon, lat, pred_tensor_np, targ_tensor_np, time_means = test_data
    lat = torch.from_numpy(lat).to(device)
    lon = torch.from_numpy(lon).to(device)

    pred_tensor = torch.from_numpy(pred_tensor_np).expand(channels, -1, -1).to(device)
    targ_tensor = torch.from_numpy(targ_tensor_np).expand(channels, -1, -1).to(device)
    means_tensor = torch.from_numpy(time_means).to(device)

    # Independent of the time means, the ACC score for cos(2*x) and cos(x) is 1/8 π sqrt(15/(32 - 3 π^2))
    # or about 0.98355. For derivation, note that the lat weight gives an extra factor of cos(x)/2 and
    # p1 = int[ (cos(x) -y - E[cos(x)-y]) * (cos(2x) - y - E[cos(2x)-y])] = pi/24
    # p2 = int[ (cos(2x) - y - E[cos(x) - y])^2 cos(x)/2 ] = 16/45
    # p3 = int[ (cos(x) - y - E[cos(x) - y])^2 cos(x)/2 ] = 2/3 - pi^2/16 (here E[.] denotes mean)
    # and acc = p / sqrt(p2 * p3) = 1/8 π sqrt(15/(32 - 3 π^2))

    acc_ = acc(pred_tensor, targ_tensor, means_tensor, lat)
    assert torch.allclose(
        acc_,
        0.9836 * torch.ones(channels).to(device),
        rtol=rtol,
        atol=atol,
    )

    # int( cos(x)^2 - cos(2x)^2 )dx, x = 0...2*pi = pi/4
    # So MSE should be pi/4 / (pi) = 0.25
    error = mse(pred_tensor**2, targ_tensor**2, dim=(1, 2))
    rerror = rmse(pred_tensor**2, targ_tensor**2, dim=(1, 2))
    assert torch.allclose(
        error,
        0.25 * torch.ones([1], dtype=torch.float32, device=device),
        rtol=rtol,
        atol=atol,
    )
    assert torch.allclose(
        rerror,
        np.sqrt(0.25) * torch.ones([1], dtype=torch.float32, device=device),
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_climate_reductions(test_data, device, rtol: float = 1e-3, atol: float = 1e-3):
    channels, lon, lat, pred_tensor_np, targ_tensor_np, time_means = test_data
    pred_tensor = torch.from_numpy(pred_tensor_np).expand(channels, -1, -1).to(device)
    lat = torch.from_numpy(lat).to(device)
    weights = clim_red._compute_lat_weights(lat)
    # Check main class
    ws = gen_red.WeightedStatistic(weights)
    # Check that it normalizes
    assert torch.allclose(
        torch.sum(ws.weights),
        torch.ones([1], dtype=torch.float32, device=device),
        rtol=rtol,
        atol=atol,
    )

    # Check when weights are 1 dimensional
    weights = weights.flatten()
    wm = gen_red.WeightedMean(weights)
    our_weighted_mean = wm(pred_tensor, dim=1)
    np_weighted_mean = np.average(pred_tensor.cpu(), weights=weights.cpu(), axis=1)
    assert torch.allclose(
        our_weighted_mean,
        torch.from_numpy(np_weighted_mean).to(device),
        rtol=rtol,
        atol=atol,
    )

    # Check when weights are same shape as pred_tensor
    weights = weights.unsqueeze(0)
    weights = weights.unsqueeze(-1)
    wm = gen_red.WeightedMean(weights)
    our_weighted_mean = wm(pred_tensor, dim=1)

    np_weighted_mean = np.average(
        pred_tensor.cpu(), weights=weights.flatten().cpu(), axis=1
    )
    assert torch.allclose(
        our_weighted_mean,
        torch.from_numpy(np_weighted_mean).to(device),
        rtol=rtol,
        atol=atol,
    )

    # Check zonal mean == our_weighted_mean
    zonal_mean = clim_red.zonal_mean(pred_tensor, lat, dim=1)
    assert torch.allclose(
        our_weighted_mean,
        zonal_mean,
        rtol=rtol,
        atol=atol,
    )

    # Test variance
    # Check when weights are 1 dimensional
    weights = weights.flatten()
    wv = clim_red.WeightedVariance(weights)
    our_weighted_var = wv(pred_tensor, dim=1)

    np_weighted_mean = np.average(
        pred_tensor.cpu(),
        weights=weights.cpu(),
        axis=1,
    )
    np_weighted_var = np.average(
        (pred_tensor.cpu() - np_weighted_mean[:, None]) ** 2,
        weights=weights.cpu(),
        axis=1,
    )
    assert torch.allclose(
        our_weighted_var,
        torch.from_numpy(np_weighted_var).to(device),
        rtol=rtol,
        atol=atol,
    )

    # Check zonal var == our_weighted_var
    zonal_var = clim_red.zonal_var(pred_tensor, lat, dim=1)
    zonal_std = clim_red.zonal_var(pred_tensor, lat, dim=1, std=True)
    assert torch.allclose(
        our_weighted_var,
        zonal_var,
        rtol=rtol,
        atol=atol,
    )
    assert torch.allclose(
        torch.sqrt(our_weighted_var),
        zonal_std,
        rtol=rtol,
        atol=atol,
    )

    # Check global means and vars
    global_mean = clim_red.global_mean(pred_tensor, lat)
    assert torch.allclose(
        torch.mean(our_weighted_mean, dim=-1),
        global_mean,
        rtol=rtol,
        atol=atol,
    )

    # Global variance of cos(2x) should be
    # int[ (cos(2x) - E[cos(2x)])^2 * cos(2x)/2 ] dx
    # = int[ (cos(2x) - 1/3)^2 * cos(2x)/2 ] dx
    # = 16/45
    global_var = clim_red.global_var(pred_tensor, lat)
    global_std = clim_red.global_var(pred_tensor, lat, std=True)
    assert torch.allclose(
        16 / 45 * torch.ones([1], device=device),
        global_var,
        rtol=rtol,
        atol=atol,
    )
    assert torch.allclose(
        4 / 3 / np.sqrt(5) * torch.ones([1], device=device),
        global_std,
        rtol=rtol,
        atol=atol,
    )


def get_disagreements(inputs, bins, counts, test):
    """
    Utility for testing disagreements in the bin counts.
    """
    sum_counts = torch.sum(counts, dim=0)
    disagreements = torch.nonzero(sum_counts != test, as_tuple=True)
    print("Disagreements: ", str(disagreements))

    number_of_disagree = len(disagreements[0])
    for i in range(number_of_disagree):
        ind = [disagreements[0][i], disagreements[1][i], disagreements[2][i]]
        print("Ind", ind)
        print(
            "Input ",
            inputs[:, disagreements[0][i], disagreements[1][i], disagreements[2][i]],
        )
        print(
            "Bins ",
            bins[:, disagreements[0][i], disagreements[1][i], disagreements[2][i]],
        )
        print(
            "Counts",
            counts[:, disagreements[0][i], disagreements[1][i], disagreements[2][i]],
        )

        trueh = torch.histogram(
            inputs[:, disagreements[0][i], disagreements[1][i], disagreements[2][i]],
            bins[:, disagreements[0][i], disagreements[1][i], disagreements[2][i]],
        )
        print("True counts", trueh)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("input_shape", [(1, 72, 144), (1, 720, 1440)])
def test_climate_histogram(device, input_shape, rtol: float = 1e-3, atol: float = 1e-3):
    DistributedManager._shared_state = {}
    if (device == "cuda:0") and (not dist.is_initialized()):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12345"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        DistributedManager.setup()
        manager = DistributedManager()
        dist.init_process_group(
            "nccl", rank=manager.rank, world_size=manager.world_size
        )
    x = torch.randn([10, *input_shape], device=device)
    y = torch.randn([5, *input_shape], device=device)

    # Test linspace
    start = torch.zeros(input_shape, device=device)
    end = torch.ones(input_shape, device=device)
    l = hist.linspace(start, end, 10)
    assert l.shape[0] == 11
    l_np = np.linspace(start.cpu(), end.cpu(), 11)
    assert torch.allclose(
        l,
        torch.from_numpy(l_np).to(device),
        rtol=rtol,
        atol=atol,
    )

    # Test histogram correctness
    xx = x[:, 0, 0, 0]
    xx_np = xx.cpu().numpy()
    bins, counts = hist.histogram(xx, bins=10)
    counts_np, bins_np = np.histogram(xx_np, bins=10)
    assert torch.allclose(
        bins,
        torch.from_numpy(bins_np).to(device),
        rtol=rtol,
        atol=atol,
    )
    assert torch.allclose(
        counts,
        torch.from_numpy(counts_np).to(device),
        rtol=rtol,
        atol=atol,
    )

    # Test low and high memory bin counts
    bins = l
    counts = torch.zeros([10, *input_shape], device=device)
    counts_low_counts = hist._low_memory_bin_reduction_counts(x, bins, counts, 10)
    counts_high_counts = hist._high_memory_bin_reduction_counts(x, bins, counts, 10)
    counts_low_cdf = hist._low_memory_bin_reduction_cdf(x, bins, counts, 10)
    counts_high_cdf = hist._high_memory_bin_reduction_cdf(x, bins, counts, 10)
    assert torch.allclose(
        counts_low_counts,
        counts_high_counts,
        rtol=rtol,
        atol=atol,
    )
    assert torch.allclose(
        counts_low_cdf,
        counts_high_cdf,
        rtol=rtol,
        atol=atol,
    )

    binsx, countsx = hist.histogram(x, bins=10)
    assert torch.allclose(
        torch.sum(countsx, dim=0),
        10 * torch.ones([1], dtype=torch.int64, device=device),
        rtol=rtol,
        atol=atol,
    ), get_disagreements(
        x, binsx, countsx, 10 * torch.ones([1], dtype=torch.int64, device=device)
    )

    binsxy, countsxy = hist.histogram(x, y, bins=5)
    assert torch.allclose(
        torch.sum(countsxy, dim=0),
        15 * torch.ones([1], dtype=torch.int64, device=device),
        rtol=rtol,
        atol=atol,
    ), get_disagreements(
        y,
        binsxy,
        countsxy - countsx,
        5 * torch.ones([1], dtype=torch.int64, device=device),
    )

    binsxy, countsxy = hist.histogram(x, y, bins=binsx)
    assert torch.allclose(
        torch.sum(countsxy, dim=0),
        15 * torch.ones([1], dtype=torch.int64, device=device),
        rtol=rtol,
        atol=atol,
    ), get_disagreements(
        y, binsxy, countsxy, 15 * torch.ones([1], dtype=torch.int64, device=device)
    )

    H = hist.Histogram(input_shape, bins=10, device=device)
    binsx, countsx = H(x)
    assert torch.allclose(
        torch.sum(countsx, dim=0),
        10 * torch.ones([1], dtype=torch.int64, device=device),
        rtol=rtol,
        atol=atol,
    ), get_disagreements(
        x, binsx, countsx, 10 * torch.ones([1], dtype=torch.int64, device=device)
    )

    binsxy, countsxy = H.update(y)
    if binsxy.shape[0] != binsx.shape[0]:
        dbins = binsx[1, 0, 0, 0] - binsx[0, 0, 0, 0]
        ind = torch.isclose(
            binsxy[:, 0, 0, 0], binsx[0, 0, 0, 0], rtol=0.1 * dbins, atol=1e-3
        ).nonzero(as_tuple=True)[0]
        new_counts = countsxy[ind : ind + 10] - countsx
    else:
        new_counts = countsxy - countsx
    assert torch.allclose(
        torch.sum(countsxy, dim=0),
        15 * torch.ones([1], dtype=torch.int64, device=device),
        rtol=rtol,
        atol=atol,
    ), get_disagreements(
        y, binsxy, new_counts, 5 * torch.ones([1], dtype=torch.int64, device=device)
    )

    _, pdf = H.finalize()
    _, cdf = H.finalize(cdf=True)
    assert torch.allclose(
        cdf[-1],
        torch.ones([1], dtype=torch.float32, device=device),
        rtol=rtol,
        atol=atol,
    )
    assert torch.allclose(
        torch.sum(pdf, dim=0),
        torch.ones([1], dtype=torch.float32, device=device),
        rtol=rtol,
        atol=atol,
    )
    if (device == "cuda:0") and (not dist.is_initialized()):
        DistributedManager.cleanup()


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_climate_crps(device, rtol: float = 1e-3, atol: float = 1e-3):
    # Uses eq (5) from Gneiting et al. https://doi.org/10.1175/MWR2904.1
    # crps(N(0, 1), 0.0) = 2 / sqrt(2*pi) - 1/sqrt(pi) ~= 0.23...
    x = torch.randn((1_000_000, 1), device=device, dtype=torch.float32)
    y = torch.zeros((1,), device=device, dtype=torch.float32)

    # Test pure crps
    c = crps.crps(x, y, bins=1_000)
    true_crps = (np.sqrt(2) - 1.0) / np.sqrt(np.pi)
    assert torch.allclose(
        c,
        true_crps * torch.ones([1], dtype=torch.float32, device=device),
        rtol=rtol,
        atol=atol,
    )

    # Test when input is numpy array
    c = crps.crps(x, y.cpu().numpy(), bins=1_000)
    assert torch.allclose(
        c,
        true_crps * torch.ones([1], dtype=torch.float32, device=device),
        rtol=rtol,
        atol=atol,
    )

    # Test from counts
    binsx, countsx = hist.histogram(x, bins=1_000)
    assert torch.allclose(
        torch.sum(countsx, dim=0),
        1_000_000 * torch.ones([1], dtype=torch.int64, device=device),
        rtol=rtol,
        atol=atol,
    )
    c = crps._crps_from_counts(binsx, countsx, y)
    assert torch.allclose(
        c,
        true_crps * torch.ones([1], dtype=torch.float32, device=device),
        rtol=rtol,
        atol=atol,
    )
    # Counts, numpy
    c = crps._crps_from_counts(binsx, countsx, y.cpu().numpy())
    assert torch.allclose(
        c,
        true_crps * torch.ones([1], dtype=torch.float32, device=device),
        rtol=rtol,
        atol=atol,
    )

    # Test from cdf
    binsx, cdfx = hist.cdf(x, bins=1_000)
    assert torch.allclose(
        cdfx[-1],
        torch.ones([1], dtype=torch.float32, device=device),
        rtol=rtol,
        atol=atol,
    )
    c = crps._crps_from_cdf(binsx, cdfx, y)
    assert torch.allclose(
        c,
        true_crps * torch.ones([1], dtype=torch.float32, device=device),
        rtol=rtol,
        atol=atol,
    )

    assert torch.allclose(
        w.wasserstein(binsx, cdfx, cdfx),
        torch.zeros([1], dtype=torch.float32, device=device),
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_climate_means_var(device, rtol: float = 1e-3, atol: float = 1e-3):
    DistributedManager._shared_state = {}
    if (device == "cuda:0") and (not dist.is_initialized()):
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12345"
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        DistributedManager.setup()
        manager = DistributedManager()
        # Test Raises Error, since process_group is not initiated
        with pytest.raises(RuntimeError) as e_info:
            em.EnsembleMetrics((1, 72, 144), device=device)
        dist.init_process_group(
            "nccl", rank=manager.rank, world_size=manager.world_size
        )

    ens_metric = em.EnsembleMetrics((1, 72, 144), device=device)
    with pytest.raises(NotImplementedError) as e_info:
        ens_metric.__call__()
    with pytest.raises(NotImplementedError) as e_info:
        ens_metric.update()
    with pytest.raises(NotImplementedError) as e_info:
        ens_metric.finalize()

    x = torch.randn((10, 1, 72, 144), device=device)
    y = torch.randn((5, 1, 72, 144), device=device)

    M = em.Mean((1, 72, 144))
    meanx = M(x)
    assert torch.allclose(meanx, torch.mean(x, dim=0))
    meanxy = M.update(y)
    assert torch.allclose(
        meanxy, torch.mean(torch.cat((x, y), dim=0), dim=0), rtol=rtol, atol=atol
    )
    assert torch.allclose(meanxy, M.finalize(), rtol=rtol, atol=atol)

    # Test _update_mean utility
    _sumxy, _n = em._update_mean(meanx * 10, 10, y, batch_dim=0)
    assert torch.allclose(meanxy, _sumxy / _n, rtol=rtol, atol=atol)
    # Test with flattened y
    _sumxy, _n = em._update_mean(meanx * 10, 10, y[0], batch_dim=None)
    _sumxy, _n = em._update_mean(_sumxy, _n, y[1:], batch_dim=0)
    assert torch.allclose(meanxy, _sumxy / _n, rtol=rtol, atol=atol)

    V = em.Variance((1, 72, 144))
    varx = V(x)
    assert torch.allclose(varx, torch.var(x, dim=0))
    varxy = V.update(y)
    assert torch.allclose(
        varxy, torch.var(torch.cat((x, y), dim=0), dim=0), rtol=rtol, atol=atol
    )
    varxy = V.finalize()
    assert torch.allclose(
        varxy, torch.var(torch.cat((x, y), dim=0), dim=0), rtol=rtol, atol=atol
    )
    stdxy = V.finalize(std=True)
    assert torch.allclose(
        stdxy, torch.std(torch.cat((x, y), dim=0), dim=0), rtol=rtol, atol=atol
    )

    # Test _update_var utility function
    _sumxy, _sum2xy, _n = em._update_var(10 * meanx, 9 * varx, 10, y, batch_dim=0)
    assert _n == 15
    assert torch.allclose(varxy, _sum2xy / (_n - 1.0), rtol=rtol, atol=atol)

    # Test with flattened array
    # Test with flattened y
    _sumxy, _sum2xy, _n = em._update_var(10 * meanx, 9 * varx, 10, y[0], batch_dim=None)
    assert _n == 11
    assert torch.allclose(
        _sumxy / _n,
        torch.mean(torch.cat((x, y[0][None, ...]), dim=0), dim=0),
        rtol=rtol,
        atol=atol,
    )
    assert torch.allclose(
        _sum2xy / (_n - 1.0),
        torch.var(torch.cat((x, y[0][None, ...]), dim=0), dim=0),
        rtol=rtol,
        atol=atol,
    )
    _sumxy, _sum2xy, _n = em._update_var(_sumxy, _sum2xy, _n, y[1:], batch_dim=0)
    assert torch.allclose(varxy, _sum2xy / (_n - 1.0), rtol=rtol, atol=atol)
