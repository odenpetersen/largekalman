"""Pytest configuration and fixtures."""
import os
import shutil
import pytest


@pytest.fixture
def tmp_folder(tmp_path):
    """Provide a temporary folder for tests."""
    folder = tmp_path / "kalman_test"
    folder.mkdir()
    yield str(folder)
    # Cleanup happens automatically with tmp_path


@pytest.fixture
def sample_observations():
    """Generate sample observations for testing."""
    import numpy as np
    np.random.seed(42)

    F = np.array([[0.9, 0.1], [0.0, 0.9]])
    Q = np.array([[0.1, 0.0], [0.0, 0.1]])
    H = np.eye(2)
    R = np.eye(2) * 0.5

    observations = []
    x = np.zeros(2)
    for _ in range(50):
        x = F @ x + np.random.multivariate_normal([0, 0], Q)
        y = H @ x + np.random.multivariate_normal([0, 0], R)
        observations.append(y.tolist())

    return observations, F.tolist(), Q.tolist(), H.tolist(), R.tolist()
