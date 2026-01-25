#!/usr/bin/env python3
"""Test smoother runs twice without NaN."""
import numpy as np
import filter
import os
import shutil

# Parameters - distinct eigenvalues but larger R
F = [[0.9, 0.1], [0.0, 0.85]]  # distinct eigenvalues
Q = [[0.1, 0.05], [0.05, 0.1]]
H = [[1.0, 0.0], [0.0, 1.0]]
R = [[1.0, 0.0], [0.0, 1.0]]  # larger R for stability

print(f"True F = {F}")
print(f"F eigenvalues: {np.linalg.eigvals(F)}")

# Generate simple observations
np.random.seed(42)
T = 100  # longer sequence
observations = []
x = np.zeros(2)
for t in range(T):
    x = np.array(F) @ x + np.random.multivariate_normal([0, 0], Q)
    y = np.array(H) @ x + np.random.multivariate_normal([0, 0], R)
    observations.append(y.tolist())

print(f"Generated {T} observations")
print(f"First obs: {observations[0]}")
print(f"Last obs: {observations[-1]}")

tmp_folder = 'tmp_test'

# First run
print("\n=== First smoother run ===")
if os.path.exists(tmp_folder):
    shutil.rmtree(tmp_folder)
os.makedirs(tmp_folder)

gen, stats = filter.smooth(tmp_folder, F, Q, H, R, iter(observations), store_observations=False)
smoothed = list(gen)
print(f"First mu: {smoothed[0][0]}")
print(f"Last mu: {smoothed[-1][0]}")
print(f"num_datapoints: {stats['num_datapoints']}")
print(f"latents_mu_sum: {stats['latents_mu_sum']}")

# Second run with same params
print("\n=== Second smoother run ===")
if os.path.exists(tmp_folder):
    shutil.rmtree(tmp_folder)
os.makedirs(tmp_folder)

gen2, stats2 = filter.smooth(tmp_folder, F, Q, H, R, iter(observations), store_observations=False)
smoothed2 = list(gen2)
print(f"First mu: {smoothed2[0][0]}")
print(f"Last mu: {smoothed2[-1][0]}")
print(f"num_datapoints: {stats2['num_datapoints']}")
print(f"latents_mu_sum: {stats2['latents_mu_sum']}")

# Third run - use true F (should work perfectly)
print("\n=== Third run with TRUE F ===")
if os.path.exists(tmp_folder):
    shutil.rmtree(tmp_folder)
os.makedirs(tmp_folder)

gen3, stats3 = filter.smooth(tmp_folder, F, Q, H, R, iter(observations), store_observations=False)
smoothed3 = list(gen3)
print(f"  First mu: {smoothed3[0][0]}")
print(f"  Last mu: {smoothed3[-1][0]}")
print(f"  Max |mu|: {max(max(abs(s[0][0]), abs(s[0][1])) for s in smoothed3)}")

# Cleanup
shutil.rmtree(tmp_folder)
print("\nDone!")
