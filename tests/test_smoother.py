#!/usr/bin/env python3
"""Tests for the Kalman smoother."""
import numpy as np
import pytest
import largekalman


def test_smoother_basic(tmp_folder, sample_observations):
    """Test basic smoother functionality."""
    observations, F, Q, H, R = sample_observations

    gen, stats = largekalman.smooth(
        tmp_folder, F, Q, H, R,
        iter(observations),
        store_observations=False
    )
    smoothed = list(gen)

    assert len(smoothed) == len(observations)
    assert stats['num_datapoints'] == len(observations)

    # Check that smoothed estimates have correct dimensions
    for mu, cov, lag1_cov in smoothed:
        assert len(mu) == 2
        assert len(cov) == 2
        assert len(cov[0]) == 2


def test_smoother_deterministic(tmp_folder, sample_observations):
    """Test that smoother produces same results on repeated runs."""
    observations, F, Q, H, R = sample_observations

    # First run
    gen1, stats1 = largekalman.smooth(
        tmp_folder, F, Q, H, R,
        iter(observations),
        store_observations=False
    )
    smoothed1 = list(gen1)

    # Second run (need fresh tmp folder)
    import shutil
    shutil.rmtree(tmp_folder)
    import os
    os.makedirs(tmp_folder)

    gen2, stats2 = largekalman.smooth(
        tmp_folder, F, Q, H, R,
        iter(observations),
        store_observations=False
    )
    smoothed2 = list(gen2)

    # Results should be identical
    assert stats1['num_datapoints'] == stats2['num_datapoints']
    np.testing.assert_array_almost_equal(
        stats1['latents_mu_sum'],
        stats2['latents_mu_sum']
    )

    for (mu1, _, _), (mu2, _, _) in zip(smoothed1, smoothed2):
        np.testing.assert_array_almost_equal(mu1, mu2)


def test_smoother_stats_shapes(tmp_folder, sample_observations):
    """Test that sufficient statistics have correct shapes."""
    observations, F, Q, H, R = sample_observations
    n_latents = len(Q)
    n_obs = len(R)

    gen, stats = largekalman.smooth(
        tmp_folder, F, Q, H, R,
        iter(observations),
        store_observations=False
    )
    list(gen)  # Consume generator

    assert len(stats['latents_mu_sum']) == n_latents
    assert len(stats['latents_cov_sum']) == n_latents * n_latents
    assert len(stats['latents_cov_lag1_sum']) == n_latents * n_latents
    assert len(stats['obs_sum']) == n_obs
    assert len(stats['obs_obs_sum']) == n_obs * n_obs
    assert len(stats['obs_latents_sum']) == n_obs * n_latents


def test_smoother_non_square_H(tmp_folder):
    """Test smoother with non-square observation matrix."""
    np.random.seed(42)

    n_latents = 3
    n_obs = 2

    F = [[0.9, 0.1, 0.0], [0.0, 0.9, 0.1], [0.0, 0.0, 0.9]]
    Q = [[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]]
    H = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]  # 2x3 matrix
    R = [[0.5, 0.0], [0.0, 0.5]]

    # Generate observations
    observations = []
    x = np.zeros(n_latents)
    for _ in range(30):
        x = np.array(F) @ x + np.random.multivariate_normal(np.zeros(n_latents), Q)
        y = np.array(H) @ x + np.random.multivariate_normal(np.zeros(n_obs), R)
        observations.append(y.tolist())

    gen, stats = largekalman.smooth(
        tmp_folder, F, Q, H, R,
        iter(observations),
        store_observations=False
    )
    smoothed = list(gen)

    assert len(smoothed) == len(observations)
    assert stats['n_latents'] == n_latents
    assert stats['n_obs'] == n_obs

    # Check dimensions
    for mu, cov, lag1_cov in smoothed:
        assert len(mu) == n_latents
        assert len(cov) == n_latents


def test_smoother_memory_basic(sample_observations):
    """Test in-memory smoother (tmp_folder=None)."""
    observations, F, Q, H, R = sample_observations

    gen, stats = largekalman.smooth(
        None, F, Q, H, R,
        iter(observations)
    )
    smoothed = list(gen)

    assert len(smoothed) == len(observations)
    assert stats['num_datapoints'] == len(observations)

    # Check dimensions
    for mu, cov, lag1_cov in smoothed:
        assert len(mu) == 2
        assert len(cov) == 2
        assert len(cov[0]) == 2


def test_smoother_memory_vs_disk(tmp_folder, sample_observations):
    """Test that in-memory and disk-based produce comparable results."""
    observations, F, Q, H, R = sample_observations

    # Disk-based
    gen_disk, stats_disk = largekalman.smooth(
        tmp_folder, F, Q, H, R,
        iter(observations),
        store_observations=False
    )
    smoothed_disk = list(gen_disk)

    # In-memory
    gen_mem, stats_mem = largekalman.smooth(
        None, F, Q, H, R,
        iter(observations)
    )
    smoothed_mem = list(gen_mem)

    # Both should have same number of datapoints
    assert stats_disk['num_datapoints'] == stats_mem['num_datapoints']
    assert len(smoothed_disk) == len(smoothed_mem)

    # Both should produce valid results (no NaN)
    for mu, cov, lag1 in smoothed_disk:
        assert not np.any(np.isnan(mu))
        assert not np.any(np.isnan(cov))

    for mu, cov, lag1 in smoothed_mem:
        assert not np.any(np.isnan(mu))
        assert not np.any(np.isnan(cov))

    # Means should be in similar range (float32 vs float64 causes some drift)
    for (mu_d, _, _), (mu_m, _, _) in zip(smoothed_disk, smoothed_mem):
        np.testing.assert_allclose(mu_d, mu_m, rtol=0.05, atol=0.1)


def test_smoother_memory_requires_observations():
    """Test that in-memory mode requires observations."""
    F = [[0.9, 0.1], [0.0, 0.9]]
    Q = [[0.1, 0.0], [0.0, 0.1]]
    H = [[1.0, 0.0], [0.0, 1.0]]
    R = [[0.5, 0.0], [0.0, 0.5]]

    with pytest.raises(ValueError, match="observations_iter required"):
        largekalman.smooth(None, F, Q, H, R, observations_iter=None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
