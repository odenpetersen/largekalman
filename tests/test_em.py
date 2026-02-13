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

    F_new, Q_new, H_new, R_new, c_new, d_new, stats = largekalman.em_step(
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


def generate_data_with_drift(F, Q, H, R, c, d, T, seed=42):
    """Generate synthetic data from a state space model with drift."""
    np.random.seed(seed)
    n_latents = F.shape[0]
    n_obs = H.shape[0]

    x = np.zeros(n_latents)
    observations = []

    for t in range(T):
        x = F @ x + c + np.random.multivariate_normal(np.zeros(n_latents), Q)
        y = H @ x + d + np.random.multivariate_normal(np.zeros(n_obs), R)
        observations.append(y.tolist())

    return observations


def test_em_with_drift(tmp_folder):
    """Test that EM can recover drift parameters c and d."""
    F_true = np.array([[0.8, 0.0], [0.0, 0.8]])
    Q_true = np.array([[0.1, 0.0], [0.0, 0.1]])
    H_true = np.eye(2)
    R_true = np.eye(2) * 0.5
    c_true = np.array([0.5, -0.3])
    d_true = np.array([1.0, -0.5])

    observations = generate_data_with_drift(F_true, Q_true, H_true, R_true,
                                             c_true, d_true, T=500, seed=42)

    params, history = largekalman.em(
        tmp_folder,
        observations,
        n_latents=2,
        n_iters=20,
        init_params={
            'F': F_true, 'Q': Q_true, 'H': H_true, 'R': R_true,
            'c': np.zeros(2), 'd': np.zeros(2),
        },
        fixed='H',
        verbose=False,
    )

    # Check c and d are recovered
    c_error = np.linalg.norm(params['c'] - c_true)
    d_error = np.linalg.norm(params['d'] - d_true)

    assert c_error < 0.5, f"c error too large: {c_error}, got {params['c']}, expected {c_true}"
    assert d_error < 0.5, f"d error too large: {d_error}, got {params['d']}, expected {d_true}"

    # F should still be close
    F_error = np.linalg.norm(params['F'] - F_true)
    assert F_error < 0.5, f"F error too large: {F_error}"

    # c and d should be in history
    assert 'c' in history[0]
    assert 'd' in history[0]


def test_em_drift_fixed(tmp_folder):
    """Test that fixing C and D keeps them at initial values."""
    F_true = np.array([[0.9, 0.0], [0.0, 0.9]])
    Q_true = np.array([[0.1, 0.0], [0.0, 0.1]])
    H_true = np.eye(2)
    R_true = np.eye(2) * 0.5

    observations = generate_data(F_true, Q_true, H_true, R_true, T=100)

    params, history = largekalman.em(
        tmp_folder,
        observations,
        n_latents=2,
        n_iters=5,
        init_params={'H': H_true},
        fixed='HCD',
    )

    # c and d should remain zero (default init)
    np.testing.assert_array_equal(params['c'], np.zeros(2))
    np.testing.assert_array_equal(params['d'], np.zeros(2))


def test_smoother_with_drift(tmp_folder):
    """Test that smoother works with nonzero c and d."""
    F = [[0.9, 0.0], [0.0, 0.9]]
    Q = [[0.1, 0.0], [0.0, 0.1]]
    H = [[1.0, 0.0], [0.0, 1.0]]
    R = [[0.5, 0.0], [0.0, 0.5]]
    c = [0.5, -0.3]
    d = [1.0, -0.5]

    np.random.seed(42)
    observations = []
    x = np.zeros(2)
    for _ in range(50):
        x = np.array(F) @ x + np.array(c) + np.random.multivariate_normal([0, 0], Q)
        y = np.array(H) @ x + np.array(d) + np.random.multivariate_normal([0, 0], R)
        observations.append(y.tolist())

    # Disk-based
    gen, stats = largekalman.smooth(
        tmp_folder, F, Q, H, R, iter(observations),
        c=c, d=d, store_observations=False
    )
    results = list(gen)
    assert len(results) == 50
    assert len(results[0][0]) == 2  # mu is 2-dimensional

    # Check new stats fields exist
    assert 'latents_mu_prev_sum' in stats
    assert 'latents_mu_next_sum' in stats
    assert 'latents_cov_prev_sum' in stats
    assert 'latents_cov_next_sum' in stats


def test_smoother_memory_with_drift():
    """Test in-memory smoother with drift."""
    F = [[0.9, 0.0], [0.0, 0.9]]
    Q = [[0.1, 0.0], [0.0, 0.1]]
    H = [[1.0, 0.0], [0.0, 1.0]]
    R = [[0.5, 0.0], [0.0, 0.5]]
    c = [0.5, -0.3]
    d = [1.0, -0.5]

    np.random.seed(42)
    observations = []
    x = np.zeros(2)
    for _ in range(30):
        x = np.array(F) @ x + np.array(c) + np.random.multivariate_normal([0, 0], Q)
        y = np.array(H) @ x + np.array(d) + np.random.multivariate_normal([0, 0], R)
        observations.append(y.tolist())

    gen, stats = largekalman.smooth(
        None, F, Q, H, R, iter(observations),
        c=c, d=d
    )
    results = list(gen)
    assert len(results) == 30

    # Smoothed means should roughly track the drifting latent state
    # After many steps with F=0.9 and c=0.5, the stationary mean is c/(1-F) = 0.5/0.1 = 5.0
    # Check that later smoothed means are positive (reflecting upward drift)
    last_mu = results[-1][0]
    assert last_mu[0] > 0, f"Expected positive drift in dim 0, got {last_mu[0]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
