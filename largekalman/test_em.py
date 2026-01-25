#!/usr/bin/env python3
"""Test EM fitting of Kalman Filter to synthetic panel data."""
import numpy as np
import filter
import os
import shutil

def generate_data(F, Q, H, R, T, seed=42):
    """Generate synthetic data from a linear Gaussian state space model."""
    np.random.seed(seed)
    n_latents = F.shape[0]
    n_obs = H.shape[0]

    # Initialize latent state
    x = np.zeros(n_latents)

    observations = []
    for t in range(T):
        # Transition
        x = F @ x + np.random.multivariate_normal(np.zeros(n_latents), Q)
        # Observation
        y = H @ x + np.random.multivariate_normal(np.zeros(n_obs), R)
        observations.append(y.tolist())

    return observations

def em_step(tmp_folder, F, Q, H, R, observations):
    """Run one EM iteration: E-step (smoothing) + M-step (parameter update)."""
    n_latents = F.shape[0]
    n_obs = H.shape[0]
    T = len(observations)

    # Check for NaN in input params
    if np.any(np.isnan(F)) or np.any(np.isnan(Q)) or np.any(np.isnan(H)) or np.any(np.isnan(R)):
        raise ValueError("NaN in input parameters")

    # E-step: run smoother and get sufficient statistics
    gen, stats = filter.smooth(tmp_folder, F.tolist(), Q.tolist(), H.tolist(), R.tolist(),
                                iter(observations), store_observations=False)

    # Consume the generator (we need to run through it to completion)
    smoothed = list(gen)

    # Extract sufficient statistics
    n = stats['num_datapoints']

    # Reshape statistics into matrices
    latents_mu_sum = np.array(stats['latents_mu_sum'])
    latents_cov_sum = np.array(stats['latents_cov_sum']).reshape(n_latents, n_latents)
    latents_cov_lag1_sum = np.array(stats['latents_cov_lag1_sum']).reshape(n_latents, n_latents)
    obs_sum = np.array(stats['obs_sum'])
    obs_obs_sum = np.array(stats['obs_obs_sum']).reshape(n_obs, n_obs)
    obs_latents_sum = np.array(stats['obs_latents_sum']).reshape(n_obs, n_latents)

    # M-step: update parameters
    E_xx = latents_cov_sum / n  # E[x_t x_t^T]
    E_xx_lag1 = latents_cov_lag1_sum / (n - 1)  # E[x_{t+1} x_t^T], only n-1 pairs

    # F = E[x_{t+1} x_t^T] @ inv(E[x_t x_t^T])
    F_new = E_xx_lag1 @ np.linalg.inv(E_xx)
    # Ensure spectral radius < 1 for stability
    eigvals, eigvecs = np.linalg.eig(F_new)
    max_eig = np.max(np.abs(eigvals))
    if max_eig > 0.99:
        F_new = F_new * (0.99 / max_eig)

    # Q = E[(x_{t+1} - F x_t)(x_{t+1} - F x_t)^T]
    Q_new = E_xx - E_xx_lag1 @ F_new.T
    Q_new = (Q_new + Q_new.T) / 2  # Symmetrize
    # Ensure positive definite by clipping eigenvalues
    eigvals, eigvecs = np.linalg.eigh(Q_new)
    eigvals = np.clip(eigvals, 1e-4, None)
    Q_new = eigvecs @ np.diag(eigvals) @ eigvecs.T

    # Fix H = I for identifiability (don't update H)
    H_new = H.copy()

    # R = E[(y - H x)(y - H x)^T] = E[y y^T] - 2 H E[x y^T] + H E[x x^T] H^T
    # With H = I: R = E[y y^T] - 2 E[x y^T] + E[x x^T]
    E_yy = obs_obs_sum / n
    E_yx = obs_latents_sum / n
    R_new = E_yy - E_yx - E_yx.T + E_xx
    R_new = (R_new + R_new.T) / 2  # Symmetrize
    # Ensure positive definite
    eigvals, eigvecs = np.linalg.eigh(R_new)
    eigvals = np.clip(eigvals, 1e-4, None)
    R_new = eigvecs @ np.diag(eigvals) @ eigvecs.T

    return F_new, Q_new, H_new, R_new

def run_em(observations, n_latents, n_obs, n_iters=20, verbose=True, init_params=None):
    """Run EM algorithm to fit Kalman Filter parameters."""
    tmp_folder = 'tmp_em'
    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)
    os.makedirs(tmp_folder)

    if init_params is not None:
        F, Q, H, R = init_params
        F, Q, H, R = F.copy(), Q.copy(), H.copy(), R.copy()
    else:
        # Initialize parameters randomly
        np.random.seed(123)
        F = np.eye(n_latents) * 0.5 + np.random.randn(n_latents, n_latents) * 0.1
        Q = np.eye(n_latents) * 0.5
        H = np.random.randn(n_obs, n_latents) * 0.5
        R = np.eye(n_obs) * 0.5

    if verbose:
        print("Initial parameters:")
        print(f"F =\n{F}")
        print(f"Q =\n{Q}")
        print(f"H =\n{H}")
        print(f"R =\n{R}")
        print()

    for i in range(n_iters):
        try:
            F, Q, H, R = em_step(tmp_folder, F, Q, H, R, observations)

            # Ensure Q and R are positive definite
            Q = (Q + Q.T) / 2
            R = (R + R.T) / 2
            min_eig_Q = np.min(np.linalg.eigvalsh(Q))
            min_eig_R = np.min(np.linalg.eigvalsh(R))
            if min_eig_Q < 1e-6:
                Q += (1e-6 - min_eig_Q) * np.eye(n_latents)
            if min_eig_R < 1e-6:
                R += (1e-6 - min_eig_R) * np.eye(n_obs)

            if verbose:
                print(f"Iteration {i+1}:")
                print(f"  F diag: {np.diag(F)}")
                print(f"  Q diag: {np.diag(Q)}")
                print(f"  H[0,:]: {H[0,:]}")
                print(f"  R diag: {np.diag(R)}")
        except Exception as e:
            print(f"Iteration {i+1} failed: {e}")
            break

    # Cleanup
    shutil.rmtree(tmp_folder)

    return F, Q, H, R

def test_em_recovery():
    """Test that EM can recover known parameters from synthetic data."""
    print("=" * 60)
    print("Testing EM parameter recovery on synthetic data")
    print("=" * 60)

    # True parameters - use larger R for numerical stability
    n_latents = 2
    n_obs = 2
    T = 100  # More data for better estimation

    F_true = np.array([[0.9, 0.1],
                       [0.0, 0.9]])
    Q_true = np.array([[0.1, 0.05],
                       [0.05, 0.1]])
    H_true = np.array([[1.0, 0.0],
                       [0.0, 1.0]])
    R_true = np.array([[1.0, 0.0],
                       [0.0, 1.0]])

    print("\nTrue parameters:")
    print(f"F_true =\n{F_true}")
    print(f"Q_true =\n{Q_true}")
    print(f"H_true =\n{H_true}")
    print(f"R_true =\n{R_true}")
    print()

    # Generate data
    print(f"Generating {T} observations...")
    observations = generate_data(F_true, Q_true, H_true, R_true, T)
    print(f"Generated {len(observations)} observations")
    print()

    # Run EM from true parameters (to verify M-step)
    print("Running EM from true parameters...")
    F_est, Q_est, H_est, R_est = run_em(observations, n_latents, n_obs, n_iters=10, verbose=True,
                                         init_params=(F_true, Q_true, H_true, R_true))

    print("\n" + "=" * 60)
    print("Final estimated parameters:")
    print(f"F_est =\n{F_est}")
    print(f"Q_est =\n{Q_est}")
    print(f"H_est =\n{H_est}")
    print(f"R_est =\n{R_est}")

    print("\n" + "=" * 60)
    print("Parameter errors (Frobenius norm):")
    print(f"||F_est - F_true|| = {np.linalg.norm(F_est - F_true):.4f}")
    print(f"||Q_est - Q_true|| = {np.linalg.norm(Q_est - Q_true):.4f}")
    print(f"||H_est - H_true|| = {np.linalg.norm(H_est - H_true):.4f}")
    print(f"||R_est - R_true|| = {np.linalg.norm(R_est - R_true):.4f}")

if __name__ == "__main__":
    test_em_recovery()
