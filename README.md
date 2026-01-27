# largekalman

Kalman filtering and smoothing for larger-than-memory datasets.

## Features

- **Memory-efficient**: Processes data in batches, writing intermediate results to disk
- **In-memory mode**: For small datasets, use `tmp_folder=None` to skip disk I/O
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

## In-Memory Mode

For small datasets that fit in RAM, you can skip disk I/O by passing `tmp_folder=None`:

```python
# In-memory mode (no temporary files)
generator, stats = largekalman.smooth(
    None,  # No temp folder = in-memory mode
    F, Q, H, R,
    iter(observations)
)

for mu, cov, lag1_cov in generator:
    print(f"Smoothed mean: {mu}")
```

## API Reference

### `smooth(tmp_folder, F, Q, H, R, observations_iter, store_observations=True, batch_size=10000)`

Run Kalman filter forward pass followed by RTS smoother backward pass.

**Parameters:**
- `tmp_folder`: Path to folder for temporary files, or `None` for in-memory mode
- `F`: Transition matrix (n_latents x n_latents)
- `Q`: Process noise covariance (n_latents x n_latents)
- `H`: Observation matrix (n_obs x n_latents)
- `R`: Observation noise covariance (n_obs x n_obs)
- `observations_iter`: Iterator over observation vectors
- `store_observations`: If False, delete observations file after processing (disk mode only)
- `batch_size`: Number of timesteps to process at once (disk mode only)

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

Use the built-in `em` function to learn model parameters from data:

```python
import largekalman

# Fit parameters using EM (H fixed by default for identifiability)
params, history = largekalman.em(
    'tmp_folder',
    observations,
    n_latents=2,
    n_iters=20,
    verbose=True
)

print(f"Fitted F:\n{params['F']}")
print(f"Fitted Q:\n{params['Q']}")
print(f"Fitted R:\n{params['R']}")

# Fix multiple parameters
params, _ = largekalman.em('tmp', obs, n_latents=2, fixed='HR')
```

### `em(tmp_folder, observations, n_latents, n_obs=None, n_iters=20, init_params=None, fixed='H', verbose=False)`

Fit Kalman filter parameters using Expectation-Maximization.

**Parameters:**
- `tmp_folder`: Path to folder for temporary files
- `observations`: List of observation vectors
- `n_latents`: Number of latent dimensions
- `n_obs`: Number of observation dimensions (inferred from data if None)
- `n_iters`: Number of EM iterations
- `init_params`: Optional dict with initial parameters `{'F', 'Q', 'H', 'R'}`
- `fixed`: String of parameters to hold fixed, e.g. `'H'` or `'HR'`. Required for identifiability.
- `verbose`: Print progress if True

**Returns:**
- `params`: Dict with fitted parameters `{'F', 'Q', 'H', 'R'}`
- `history`: List of parameter dicts from each iteration

### `em_step(tmp_folder, F, Q, H, R, observations)`

Run a single EM iteration for custom control over the optimization.

**Returns:**
- `F_new, Q_new, H_new, R_new`: Updated parameters as numpy arrays
- `stats`: Sufficient statistics from the E-step

## License

MIT License
