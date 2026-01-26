#!/usr/bin/env python3
"""Tests for EM fitting of Kalman Filter parameters."""
import numpy as np
import pytest
import largekalman
import os
import shutil


def generate_data(F, Q, H, R, T, seed=42):
    """Generate synthetic data from a linear Gaussian state space model."""
    np.random.seed(seed)
    n_latents = F.shape[0]
    n_obs = H.shape[0]

    x = np.zeros(n_latents)
    observations = []

    for t in range(T):
        x = F @ x + np.random.multivariate_normal(np.zeros(n_latents), Q)
        y = H @ x + np.random.multivariate_normal(np.zeros(n_obs), R)
        observations.append(y.tolist())

    return observations


def em_step(tmp_folder, F, Q, H, R, observations):
    """Run one EM iteration: E-step (smoothing) + M-step (parameter update)."""
    n_latents = F.shape[0]
    n_obs = H.shape[0]

    gen, stats = largekalman.smooth(
        tmp_folder, F.tolist(), Q.tolist(), H.tolist(), R.tolist(),
        iter(observations), store_observations=False
    )
    list(gen)  # Consume generator

    n = stats['num_datapoints']

    # Reshape statistics into matrices
    latents_cov_sum = np.array(stats['latents_cov_sum']).reshape(n_latents, n_latents)
    latents_cov_lag1_sum = np.array(stats['latents_cov_lag1_sum']).reshape(n_latents, n_latents)

    # M-step
    E_xx = latents_cov_sum / n
    E_xx_lag1 = latents_cov_lag1_sum / (n - 1)

    F_new = E_xx_lag1 @ np.linalg.inv(E_xx)

    # Ensure spectral radius < 1
    eigvals = np.linalg.eigvals(F_new)
    max_eig = np.max(np.abs(eigvals))
    if max_eig > 0.99:
        F_new = F_new * (0.99 / max_eig)

    Q_new = E_xx - E_xx_lag1 @ F_new.T
    Q_new = (Q_new + Q_new.T) / 2

    # Ensure positive definite
    eigvals, eigvecs = np.linalg.eigh(Q_new)
    eigvals = np.clip(eigvals, 1e-4, None)
    Q_new = eigvecs @ np.diag(eigvals) @ eigvecs.T

    return F_new, Q_new


def test_em_single_step(tmp_folder):
    """Test that a single EM step runs without error."""
    F = np.array([[0.9, 0.1], [0.0, 0.9]])
    Q = np.array([[0.1, 0.0], [0.0, 0.1]])
    H = np.eye(2)
    R = np.eye(2)

    observations = generate_data(F, Q, H, R, T=50)

    F_new, Q_new = em_step(tmp_folder, F, Q, H, R, observations)

    # Check outputs are valid
    assert F_new.shape == F.shape
    assert Q_new.shape == Q.shape
    assert not np.any(np.isnan(F_new))
    assert not np.any(np.isnan(Q_new))

    # Q should be positive definite
    eigvals = np.linalg.eigvalsh(Q_new)
    assert np.all(eigvals > 0)


def test_em_convergence(tmp_folder):
    """Test that EM converges to reasonable parameters."""
    F_true = np.array([[0.9, 0.1], [0.0, 0.9]])
    Q_true = np.array([[0.1, 0.05], [0.05, 0.1]])
    H = np.eye(2)
    R = np.eye(2)

    observations = generate_data(F_true, Q_true, H, R, T=100)

    # Start from true parameters
    F = F_true.copy()
    Q = Q_true.copy()

    # Run a few EM iterations
    for i in range(5):
        # Clean tmp folder between iterations
        shutil.rmtree(tmp_folder)
        os.makedirs(tmp_folder)

        F, Q = em_step(tmp_folder, F, Q, H, R, observations)

    # Parameters should stay close to true values
    F_error = np.linalg.norm(F - F_true)
    Q_error = np.linalg.norm(Q - Q_true)

    # Allow some deviation due to finite sample
    assert F_error < 0.5, f"F error too large: {F_error}"
    assert Q_error < 0.5, f"Q error too large: {Q_error}"


def test_sufficient_stats_consistency(tmp_folder):
    """Test that sufficient statistics are consistent across runs."""
    F = [[0.9, 0.0], [0.0, 0.9]]
    Q = [[0.1, 0.0], [0.0, 0.1]]
    H = [[1.0, 0.0], [0.0, 1.0]]
    R = [[0.5, 0.0], [0.0, 0.5]]

    np.random.seed(42)
    observations = []
    x = np.zeros(2)
    for _ in range(30):
        x = np.array(F) @ x + np.random.multivariate_normal([0, 0], Q)
        y = np.array(H) @ x + np.random.multivariate_normal([0, 0], R)
        observations.append(y.tolist())

    # Run twice
    gen1, stats1 = largekalman.smooth(
        tmp_folder, F, Q, H, R,
        iter(observations), store_observations=False
    )
    list(gen1)

    shutil.rmtree(tmp_folder)
    os.makedirs(tmp_folder)

    gen2, stats2 = largekalman.smooth(
        tmp_folder, F, Q, H, R,
        iter(observations), store_observations=False
    )
    list(gen2)

    # Stats should be identical
    np.testing.assert_array_almost_equal(
        stats1['latents_cov_sum'],
        stats2['latents_cov_sum']
    )
    np.testing.assert_array_almost_equal(
        stats1['latents_cov_lag1_sum'],
        stats2['latents_cov_lag1_sum']
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
