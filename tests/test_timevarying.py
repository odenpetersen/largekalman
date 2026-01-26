#!/usr/bin/env python3
"""Tests for time-varying parameter handling."""
import numpy as np
import pytest
import largekalman
import os
import shutil


def generate_timevarying_data(F_list, Q_list, H_list, R_list, segment_lengths, seed=42):
    """Generate data from a time-varying state space model."""
    np.random.seed(seed)

    n_latents = F_list[0].shape[0]

    observations = []
    true_states = []
    segment_indices = []

    x = np.zeros(n_latents)
    t = 0

    for seg_idx in range(len(F_list)):
        F = F_list[seg_idx]
        Q = Q_list[seg_idx]
        H = H_list[seg_idx]
        R = R_list[seg_idx]
        seg_len = segment_lengths[seg_idx]

        start_idx = t
        for _ in range(seg_len):
            x = F @ x + np.random.multivariate_normal(np.zeros(n_latents), Q)
            true_states.append(x.copy())
            y = H @ x + np.random.multivariate_normal(np.zeros(H.shape[0]), R)
            observations.append(y.tolist())
            t += 1

        segment_indices.append((start_idx, t))

    return observations, true_states, segment_indices


def smooth_segments(tmp_folder, observations, F_list, Q_list, H_list, R_list, segment_indices):
    """Run smoother on each segment with its own parameters."""
    all_smoothed = []

    for seg_idx, (start, end) in enumerate(segment_indices):
        F = F_list[seg_idx]
        Q = Q_list[seg_idx]
        H = H_list[seg_idx]
        R = R_list[seg_idx]

        seg_obs = observations[start:end]

        seg_folder = f"{tmp_folder}/seg_{seg_idx}"
        if os.path.exists(seg_folder):
            shutil.rmtree(seg_folder)
        os.makedirs(seg_folder)

        gen, stats = largekalman.smooth(
            seg_folder, F.tolist(), Q.tolist(), H.tolist(), R.tolist(),
            iter(seg_obs), store_observations=False
        )
        smoothed = list(gen)
        all_smoothed.extend(smoothed)

        shutil.rmtree(seg_folder)

    return all_smoothed


def test_regime_switching(tmp_folder):
    """Test filtering with regime-switching dynamics."""
    n_latents = 2
    n_obs = 2

    # Two different regimes
    F1 = np.array([[0.95, 0.0], [0.0, 0.95]])
    Q1 = np.array([[0.1, 0.0], [0.0, 0.1]])
    H1 = np.eye(n_obs)
    R1 = np.eye(n_obs) * 0.5

    F2 = np.array([[0.8, 0.2], [-0.2, 0.8]])
    Q2 = np.array([[0.2, 0.0], [0.0, 0.2]])
    H2 = np.eye(n_obs)
    R2 = np.eye(n_obs) * 0.5

    F_list = [F1, F2]
    Q_list = [Q1, Q2]
    H_list = [H1, H2]
    R_list = [R1, R2]
    segment_lengths = [25, 25]

    observations, true_states, segment_indices = generate_timevarying_data(
        F_list, Q_list, H_list, R_list, segment_lengths
    )

    # Run with correct segment-wise parameters
    smoothed_correct = smooth_segments(
        tmp_folder, observations, F_list, Q_list, H_list, R_list, segment_indices
    )

    # Run with constant (wrong) parameters
    wrong_folder = f"{tmp_folder}/wrong"
    os.makedirs(wrong_folder)

    gen_wrong, _ = largekalman.smooth(
        wrong_folder, F1.tolist(), Q1.tolist(), H1.tolist(), R1.tolist(),
        iter(observations), store_observations=False
    )
    smoothed_wrong = list(gen_wrong)

    # Compute MSE for both
    mse_correct = sum(
        np.sum((np.array(sm[0]) - true_x) ** 2)
        for sm, true_x in zip(smoothed_correct, true_states)
    ) / len(true_states)

    mse_wrong = sum(
        np.sum((np.array(sm[0]) - true_x) ** 2)
        for sm, true_x in zip(smoothed_wrong, true_states)
    ) / len(true_states)

    # Both should produce valid results
    assert not np.isnan(mse_correct)
    assert not np.isnan(mse_wrong)
    assert len(smoothed_correct) == len(observations)
    assert len(smoothed_wrong) == len(observations)


def test_observation_dimension_switching(tmp_folder):
    """Test when observation dimension changes between segments."""
    n_latents = 3

    F = np.array([[0.9, 0.1, 0.0],
                  [0.0, 0.9, 0.1],
                  [0.0, 0.0, 0.9]])
    Q = np.eye(n_latents) * 0.1

    # Segment 1: Observe 2 dimensions
    H1 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    R1 = np.eye(2) * 0.5

    # Segment 2: Observe all 3 dimensions
    H2 = np.eye(3)
    R2 = np.eye(3) * 0.5

    F_list = [F, F]
    Q_list = [Q, Q]
    H_list = [H1, H2]
    R_list = [R1, R2]
    segment_lengths = [20, 20]

    observations, true_states, segment_indices = generate_timevarying_data(
        F_list, Q_list, H_list, R_list, segment_lengths
    )

    # Run segment-wise smoother
    smoothed = smooth_segments(
        tmp_folder, observations, F_list, Q_list, H_list, R_list, segment_indices
    )

    assert len(smoothed) == sum(segment_lengths)

    # Check dimensions are correct
    for mu, cov, _ in smoothed:
        assert len(mu) == n_latents
        assert len(cov) == n_latents


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
