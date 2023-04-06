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

import modulus.metrics.general.histogram as hist
import modulus.metrics.general.ensemble_metrics as em
import modulus.metrics.general.crps as crps
import modulus.metrics.general.wasserstein as w
import modulus.metrics.general.calibration as cal
import modulus.metrics.general.entropy as ent

from modulus.distributed.manager import DistributedManager

Tensor = torch.Tensor


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
def test_histogram(device, input_shape, rtol: float = 1e-3, atol: float = 1e-3):
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

    binsx, countsx = hist.histogram(x, bins=10, verbose=True)
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
def test_crps(device, rtol: float = 1e-3, atol: float = 1e-3):
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

    # Test Gaussian CRPS
    mm = torch.zeros([1], dtype=torch.float32, device=device)
    vv = torch.ones([1], dtype=torch.float32, device=device)
    gaussian_crps = crps._crps_gaussian(mm, vv, y)
    assert torch.allclose(
        gaussian_crps,
        true_crps * torch.ones([1], dtype=torch.float32, device=device),
        rtol=rtol,
        atol=atol,
    )
    gaussian_crps = crps._crps_gaussian(mm, vv, y.cpu().numpy())
    assert torch.allclose(
        gaussian_crps,
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
def test_means_var(device, rtol: float = 1e-3, atol: float = 1e-3):
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


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_calibration(device, rtol: float = 1e-2, atol: float = 1e-2):

    x = torch.randn((10_000, 30, 30), device=device, dtype=torch.float32)
    y = torch.randn((30, 30), device=device, dtype=torch.float32)

    bin_edges, bin_counts = hist.histogram(x, bins=30)

    # Test getting rank from histogram
    ranks = cal.find_rank(bin_edges, bin_counts, y)

    assert ranks.shape == y.shape
    assert torch.all(torch.le(ranks, 1.0))
    assert torch.all(torch.ge(ranks, 0.0))

    # Test getting rank from histogram (numpy)
    y = np.random.randn(30, 30)
    ranks_np = cal.find_rank(bin_edges, bin_counts, y)

    assert ranks_np.shape == y.shape
    assert torch.all(torch.le(ranks_np, 1.0))
    assert torch.all(torch.ge(ranks_np, 0.0))

    ranks = ranks.flatten()
    rank_bin_edges = torch.linspace(0, 1, 11).to(device)
    rank_bin_edges, rank_counts = hist.histogram(ranks, bins=rank_bin_edges)
    rps = cal._rank_probability_score_from_counts(rank_bin_edges, rank_counts)

    assert rps > 0.0
    assert rps < 1.0
    assert torch.allclose(
        rps, torch.zeros([1], device=device, dtype=torch.float32), rtol=rtol, atol=atol
    )

    rps = cal.rank_probability_score(ranks)
    assert rps > 0.0
    assert rps < 1.0
    assert torch.allclose(
        rps, torch.zeros([1], device=device, dtype=torch.float32), rtol=rtol, atol=atol
    )

    num_obs = 500

    x = torch.randn((1_000, num_obs, 10, 10), device=device, dtype=torch.float32)
    bin_edges, bin_counts = hist.histogram(x, bins=30)

    obs = torch.randn((num_obs, 10, 10), device=device, dtype=torch.float32)
    ranks = cal.find_rank(bin_edges, bin_counts, obs)
    assert ranks.shape == (num_obs, 10, 10)

    rps = cal.rank_probability_score(ranks)
    assert rps.shape == (10, 10)
    assert torch.allclose(
        rps, torch.zeros([1], device=device, dtype=torch.float32), rtol=rtol, atol=atol
    )


@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_entropy(device, rtol: float = 1e-2, atol: float = 1e-2):
    one = torch.ones([1], device=device, dtype=torch.float32)

    x = torch.randn((100_000, 10, 10), device=device, dtype=torch.float32)
    bin_edges, bin_counts = hist.histogram(x, bins=30)
    entropy = ent._entropy_from_counts(bin_counts, bin_edges, normalized=False)
    assert entropy.shape == (10, 10)
    assert torch.allclose(
        entropy, (0.5 + 0.5 * np.log(2 * np.pi)) * one, atol=atol, rtol=rtol
    )
    entropy = ent._entropy_from_counts(bin_counts, bin_edges, normalized=True)
    assert torch.all(torch.le(entropy, one))
    assert torch.all(torch.ge(entropy, 0.0 * one))

    # Test Maximum Entropy
    x = torch.rand((100_000, 10, 10), device=device, dtype=torch.float32)
    bin_edges, bin_counts = hist.histogram(x, bins=30)
    entropy = ent._entropy_from_counts(bin_counts, bin_edges, normalized=True)
    assert entropy.shape == (10, 10)
    assert torch.allclose(entropy, one, rtol=rtol, atol=atol)

    # Test Relative Entropy
    x = torch.randn((100_000, 10, 10), device=device, dtype=torch.float32)
    bin_edges, x_bin_counts = hist.histogram(x, bins=30)
    x1 = torch.randn((100_000, 10, 10), device=device, dtype=torch.float32)
    _, x1_bin_counts = hist.histogram(x, bins=bin_edges)
    x2 = 0.1 * torch.randn((100_000, 10, 10), device=device, dtype=torch.float32)
    _, x2_bin_counts = hist.histogram(x, bins=bin_edges)

    rel_ent_1 = ent._relative_entropy_from_counts(
        x_bin_counts, x1_bin_counts, bin_edges
    )
    rel_ent_2 = ent._relative_entropy_from_counts(
        x_bin_counts, x2_bin_counts, bin_edges
    )

    assert torch.all(torch.le(rel_ent_1, rel_ent_2))
    assert torch.allclose(rel_ent_1, 0.0 * one)
    assert torch.all(torch.ge(rel_ent_2, 0.0 * one))
