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


def test_em_single_step(tmp_folder):
    """Test that a single EM step runs without error."""
    F = np.array([[0.9, 0.1], [0.0, 0.9]])
    Q = np.array([[0.1, 0.0], [0.0, 0.1]])
    H = np.eye(2)
    R = np.eye(2)

    observations = generate_data(F, Q, H, R, T=50)

    F_new, Q_new, H_new, R_new, stats = largekalman.em_step(
        tmp_folder, F, Q, H, R, observations
    )

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
    H_true = np.eye(2)
    R_true = np.eye(2)

    observations = generate_data(F_true, Q_true, H_true, R_true, T=100)

    # Run EM with fixed H (default)
    params, history = largekalman.em(
        tmp_folder,
        observations,
        n_latents=2,
        n_iters=5,
        init_params={'F': F_true, 'Q': Q_true, 'H': H_true, 'R': R_true},
    )

    # Parameters should stay close to true values
    F_error = np.linalg.norm(params['F'] - F_true)
    Q_error = np.linalg.norm(params['Q'] - Q_true)

    # Allow some deviation due to finite sample
    assert F_error < 0.5, f"F error too large: {F_error}"
    assert Q_error < 0.5, f"Q error too large: {Q_error}"


def test_em_function(tmp_folder):
    """Test the main em() function."""
    F_true = np.array([[0.9, 0.0], [0.0, 0.9]])
    Q_true = np.array([[0.1, 0.0], [0.0, 0.1]])
    H_true = np.eye(2)
    R_true = np.eye(2) * 0.5

    observations = generate_data(F_true, Q_true, H_true, R_true, T=100)

    # H fixed by default for identifiability
    params, history = largekalman.em(
        tmp_folder,
        observations,
        n_latents=2,
        n_iters=10,
        init_params={
            'F': np.eye(2) * 0.8,
            'Q': np.eye(2) * 0.1,
            'H': H_true,
            'R': np.eye(2) * 0.5,
        },
    )

    # Check that we got valid parameters
    assert params['F'].shape == (2, 2)
    assert params['Q'].shape == (2, 2)
    assert params['H'].shape == (2, 2)
    assert params['R'].shape == (2, 2)

    # Check no NaN
    assert not np.any(np.isnan(params['F']))
    assert not np.any(np.isnan(params['Q']))
    assert not np.any(np.isnan(params['R']))

    # Check history
    assert len(history) == 10

    # Q and R should be positive definite
    assert np.all(np.linalg.eigvalsh(params['Q']) > 0)
    assert np.all(np.linalg.eigvalsh(params['R']) > 0)

    # H should be unchanged
    np.testing.assert_array_equal(params['H'], H_true)


def test_em_identifiability_error(tmp_folder):
    """Test that EM raises error when no parameters are fixed."""
    observations = [[[1.0, 2.0], [2.0, 3.0]]]

    with pytest.raises(ValueError, match="not identifiable"):
        largekalman.em(tmp_folder, observations, n_latents=2, fixed='')


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
