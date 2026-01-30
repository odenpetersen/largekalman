"""EM algorithm for learning Kalman filter parameters."""
import numpy as np
import os
import shutil
from .filter import smooth, write_observations, write_params, write_forwards, write_backwards


def _m_step(stats, n_latents, n_obs, F, Q, H, R, fixed_params):
    """M-step: update parameters given sufficient statistics.

    Args:
        stats: Sufficient statistics dict from E-step
        n_latents: Number of latent dimensions
        n_obs: Number of observation dimensions
        F, Q, H, R: Current parameters
        fixed_params: Set of parameter names to hold fixed

    Returns:
        F, Q, H, R: Updated parameters
    """
    n = stats['num_datapoints']

    # Extract sufficient statistics
    latents_cov_sum = np.array(stats['latents_cov_sum']).reshape(n_latents, n_latents)
    latents_cov_lag1_sum = np.array(stats['latents_cov_lag1_sum']).reshape(n_latents, n_latents)
    obs_obs_sum = np.array(stats['obs_obs_sum']).reshape(n_obs, n_obs)
    obs_latents_sum = np.array(stats['obs_latents_sum']).reshape(n_obs, n_latents)

    # Compute expectations
    E_xx = latents_cov_sum / n
    E_xx_lag1 = latents_cov_lag1_sum / (n - 1)
    E_yy = obs_obs_sum / n
    E_yx = obs_latents_sum / n

    # Update F: F = E[x_{t+1} x_t^T] @ inv(E[x_t x_t^T])
    if 'F' not in fixed_params:
        try:
            F_new = E_xx_lag1 @ np.linalg.inv(E_xx)
            if not np.any(np.isnan(F_new)):
                # Ensure spectral radius < 1 for stability
                eigvals = np.linalg.eigvals(F_new)
                max_eig = np.max(np.abs(eigvals))
                if max_eig > 0.99:
                    F_new = F_new * (0.99 / max_eig)
                F = F_new
        except np.linalg.LinAlgError:
            pass  # Keep previous F

    # Update Q: Q = E[x_t x_t^T] - F @ E[x_t x_{t-1}^T]
    if 'Q' not in fixed_params:
        Q_new = E_xx - F @ E_xx_lag1.T
        Q_new = (Q_new + Q_new.T) / 2  # Symmetrize
        if not np.any(np.isnan(Q_new)):
            # Ensure positive definite
            eigvals, eigvecs = np.linalg.eigh(Q_new)
            eigvals = np.clip(eigvals, 1e-6, None)
            Q = eigvecs @ np.diag(eigvals) @ eigvecs.T

    # Update H: H = E[y x^T] @ inv(E[x x^T])
    if 'H' not in fixed_params:
        try:
            H_new = E_yx @ np.linalg.inv(E_xx)
            if not np.any(np.isnan(H_new)):
                H = H_new
        except np.linalg.LinAlgError:
            pass  # Keep previous H

    # Update R: R = E[(y - Hx)(y - Hx)^T]
    if 'R' not in fixed_params:
        R_new = E_yy - H @ E_yx.T - E_yx @ H.T + H @ E_xx @ H.T
        R_new = (R_new + R_new.T) / 2  # Symmetrize
        if not np.any(np.isnan(R_new)):
            # Ensure positive definite
            eigvals, eigvecs = np.linalg.eigh(R_new)
            eigvals = np.clip(eigvals, 1e-6, None)
            R = eigvecs @ np.diag(eigvals) @ eigvecs.T

    return F, Q, H, R


def em_step(tmp_folder, F, Q, H, R, observations=None, observations_file=None, fixed=None,
            batch_size=10000, store_observations=True):
    """Run a single EM iteration.

    Args:
        tmp_folder: Path to folder for temporary files
        F, Q, H, R: Current parameter estimates (as numpy arrays or lists)
        observations: List of observation vectors (mutually exclusive with observations_file)
        observations_file: Path to pre-written observations file (mutually exclusive with observations)
        fixed: String of parameter names to hold fixed (e.g. 'H' or 'HR'), or None to update all
        batch_size: Buffer size for file I/O operations
        store_observations: If True (default), keep observations file for reuse

    Returns:
        F_new, Q_new, H_new, R_new: Updated parameters as numpy arrays
        stats: Sufficient statistics from the E-step
    """
    if observations is None and observations_file is None:
        raise ValueError("Either observations or observations_file must be provided")
    if observations is not None and observations_file is not None:
        raise ValueError("Cannot provide both observations and observations_file")

    F = np.array(F)
    Q = np.array(Q)
    H = np.array(H)
    R = np.array(R)

    n_latents = F.shape[0]
    n_obs = H.shape[0]

    # Parse fixed params
    fixed_params = set(fixed.upper()) if fixed else set()

    # Setup folder
    if not os.path.exists(tmp_folder):
        os.makedirs(tmp_folder)

    # Write observations if provided as list
    obs_file = observations_file
    if observations is not None:
        obs_file = os.path.join(tmp_folder, 'observations.bin')
        write_observations(iter(observations), obs_file)

    # Write params and run E-step
    params_file = os.path.join(tmp_folder, 'params.bin')
    write_params(F.tolist(), Q.tolist(), H.tolist(), R.tolist(), params_file)

    forwards_file = os.path.join(tmp_folder, 'forwards.bin')
    write_forwards(obs_file, forwards_file, params_file, buffer_size=batch_size)

    backwards_file = os.path.join(tmp_folder, 'backwards.bin')
    stats = write_backwards(params_file, obs_file, forwards_file, backwards_file, buffer_size=batch_size)

    # Cleanup intermediate files
    os.remove(params_file)
    os.remove(forwards_file)
    os.remove(backwards_file)
    if not store_observations and observations is not None:
        os.remove(obs_file)

    # M-step
    F_new, Q_new, H_new, R_new = _m_step(stats, n_latents, n_obs, F, Q, H, R, fixed_params)

    return F_new, Q_new, H_new, R_new, stats


def em(tmp_folder, observations, n_latents, n_obs=None, n_iters=20,
       init_params=None, fixed='H', verbose=False, batch_size=10000, store_observations=True):
    """Fit Kalman filter parameters using Expectation-Maximization.

    Args:
        tmp_folder: Path to folder for temporary files
        observations: List of observation vectors (or iterator)
        n_latents: Number of latent dimensions
        n_obs: Number of observation dimensions (inferred from data if None)
        n_iters: Number of EM iterations
        init_params: Optional dict with initial parameters {'F', 'Q', 'H', 'R'}
        fixed: String of parameter names to hold fixed, e.g. 'H' or 'HR'.
               At least one parameter must be fixed for identifiability.
        verbose: Print progress if True
        batch_size: Buffer size for file I/O operations
        store_observations: If True (default), keep observations file for reuse

    Returns:
        params: Dict with fitted parameters {'F', 'Q', 'H', 'R'}
        history: List of parameter dicts from each iteration
    """
    # Convert to list if iterator
    if not isinstance(observations, list):
        observations = list(observations)

    if n_obs is None:
        n_obs = len(observations[0])

    # Parse fixed params string
    valid_params = {'F', 'Q', 'H', 'R'}
    fixed_params = set(fixed.upper()) if fixed else set()
    invalid = fixed_params - valid_params
    if invalid:
        raise ValueError(f"Invalid parameter names in fixed='{fixed}': {invalid}. Valid: F, Q, H, R")
    if len(fixed_params) == 0:
        raise ValueError("Model is not identifiable: at least one parameter must be fixed. "
                         "Use fixed='H' (most common) or fixed='Q' to constrain the model.")

    # Initialize parameters
    if init_params is not None:
        F = np.array(init_params.get('F', np.eye(n_latents) * 0.9))
        Q = np.array(init_params.get('Q', np.eye(n_latents) * 0.1))
        H = np.array(init_params.get('H', np.eye(n_obs, n_latents)))
        R = np.array(init_params.get('R', np.eye(n_obs) * 0.5))
    else:
        # Default initialization
        F = np.eye(n_latents) * 0.9
        Q = np.eye(n_latents) * 0.1
        if n_obs > n_latents:
            H = np.vstack([np.eye(n_latents), np.zeros((n_obs - n_latents, n_latents))])
        else:
            H = np.eye(n_obs, n_latents)
        R = np.eye(n_obs) * 0.5

    # Setup folder and write observations once
    if os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)
    os.makedirs(tmp_folder)

    observations_file = os.path.join(tmp_folder, 'observations.bin')
    write_observations(iter(observations), observations_file)

    history = []

    for iteration in range(n_iters):
        # Run EM step with pre-stored observations
        F, Q, H, R, stats = em_step(
            tmp_folder, F, Q, H, R,
            observations_file=observations_file,
            fixed=fixed,
            batch_size=batch_size
        )

        # Store history
        history.append({
            'F': F.copy(),
            'Q': Q.copy(),
            'H': H.copy(),
            'R': R.copy(),
        })

        if verbose:
            print(f"Iteration {iteration + 1}/{n_iters}:")
            print(f"  F diag: {np.diag(F)}")
            print(f"  Q diag: {np.diag(Q)}")
            print(f"  R diag: {np.diag(R)}")

    # Cleanup
    if not store_observations and os.path.exists(tmp_folder):
        shutil.rmtree(tmp_folder)

    return {'F': F, 'Q': Q, 'H': H, 'R': R}, history
