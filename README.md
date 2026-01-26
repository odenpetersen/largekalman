# largekalman

Kalman filtering and smoothing for larger-than-memory datasets.

## Features

- **Memory-efficient**: Processes data in batches, writing intermediate results to disk
- **RTS Smoother**: Full Rauch-Tung-Striebel smoothing with lag-1 covariance
- **Sufficient statistics**: Returns statistics needed for EM parameter estimation
- **Non-square observation matrices**: Supports observation dimension different from latent dimension

## Installation

```bash
pip install largekalman
```

**Requirements**: A C compiler (gcc) is needed to build the native extension.

- Ubuntu/Debian: `sudo apt install build-essential`
- macOS: `xcode-select --install`
- Fedora: `sudo dnf install gcc`

## Quick Start

```python
import largekalman

# Define state space model parameters
F = [[0.9, 0.1], [0.0, 0.9]]  # Transition matrix
Q = [[0.1, 0.0], [0.0, 0.1]]  # Process noise covariance
H = [[1.0, 0.0], [0.0, 1.0]]  # Observation matrix
R = [[0.5, 0.0], [0.0, 0.5]]  # Observation noise covariance

# Observations as an iterator (can be a generator for large datasets)
observations = [[1.2, 0.8], [1.5, 1.1], [1.8, 1.3], ...]

# Run the smoother
generator, stats = largekalman.smooth(
    'tmp_folder',      # Temporary folder for intermediate files
    F, Q, H, R,
    iter(observations),
    store_observations=False  # Don't keep observations in memory
)

# Iterate over smoothed estimates
for mu, cov, lag1_cov in generator:
    print(f"Smoothed mean: {mu}")
    print(f"Smoothed covariance: {cov}")
    print(f"Lag-1 covariance: {lag1_cov}")

# Sufficient statistics for EM
print(f"Number of datapoints: {stats['num_datapoints']}")
print(f"Sum of latent means: {stats['latents_mu_sum']}")
print(f"Sum of E[x_t x_t^T]: {stats['latents_cov_sum']}")
print(f"Sum of E[x_{t+1} x_t^T]: {stats['latents_cov_lag1_sum']}")
```

## API Reference

### `smooth(tmp_folder, F, Q, H, R, observations_iter, store_observations=True, batch_size=10000)`

Run Kalman filter forward pass followed by RTS smoother backward pass.

**Parameters:**
- `tmp_folder`: Path to folder for temporary files (created if doesn't exist)
- `F`: Transition matrix (n_latents x n_latents)
- `Q`: Process noise covariance (n_latents x n_latents)
- `H`: Observation matrix (n_obs x n_latents)
- `R`: Observation noise covariance (n_obs x n_obs)
- `observations_iter`: Iterator over observation vectors
- `store_observations`: If False, delete observations file after processing
- `batch_size`: Number of timesteps to process at once

**Returns:**
- `generator`: Yields `(mu, cov, lag1_cov)` tuples for each timestep
- `stats`: Dictionary of sufficient statistics

### Sufficient Statistics

The `stats` dictionary contains:
- `num_datapoints`: Number of observations processed
- `latents_mu_sum`: Sum of smoothed means
- `latents_cov_sum`: Sum of E[x_t x_t^T] (includes outer product of means)
- `latents_cov_lag1_sum`: Sum of E[x_{t+1} x_t^T] for consecutive pairs
- `obs_sum`: Sum of observations
- `obs_obs_sum`: Sum of E[y_t y_t^T]
- `obs_latents_sum`: Sum of E[y_t x_t^T]

## EM Parameter Estimation

The sufficient statistics enable EM updates for learning model parameters:

```python
import numpy as np

# After smoothing
n = stats['num_datapoints']
E_xx = np.array(stats['latents_cov_sum']).reshape(n_latents, n_latents) / n
E_xx_lag1 = np.array(stats['latents_cov_lag1_sum']).reshape(n_latents, n_latents) / (n - 1)

# M-step updates
F_new = E_xx_lag1 @ np.linalg.inv(E_xx)
Q_new = E_xx - E_xx_lag1 @ F_new.T
```

## License

MIT License
