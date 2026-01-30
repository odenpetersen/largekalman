import os
import itertools
import ctypes
import array
import numpy as np

_here = os.path.dirname(__file__)
lib = ctypes.CDLL(os.path.join(_here, "libfilter.so"))

# --- C function prototypes ---
lib.open_file_write.argtypes = [ctypes.c_char_p]
lib.open_file_write.restype = ctypes.c_void_p

lib.open_file_read.argtypes = [ctypes.c_char_p]
lib.open_file_read.restype = ctypes.c_void_p

lib.close_file.argtypes = [ctypes.c_void_p]
lib.close_file.restype = None

lib.write_ints.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_void_p]
lib.write_ints.restype = None

lib.write_floats.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_void_p]
lib.write_floats.restype = None

lib.write_forwards.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
lib.write_forwards.restype = None

lib.write_backwards.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]

lib.read_floats_backwards.argtypes = [ctypes.c_int, ctypes.c_void_p]
lib.read_floats_backwards.restype = ctypes.POINTER(ctypes.c_float)

class SuffStats(ctypes.Structure):
    _fields_ = [
        ("n_obs", ctypes.c_int),
        ("n_latents", ctypes.c_int),
        ("num_datapoints", ctypes.c_int),
        ("latents_mu_sum", ctypes.POINTER(ctypes.c_float)),
        ("latents_cov_sum", ctypes.POINTER(ctypes.c_float)),
        ("latents_cov_lag1_sum", ctypes.POINTER(ctypes.c_float)),
        ("obs_sum", ctypes.POINTER(ctypes.c_float)),
        ("obs_obs_sum", ctypes.POINTER(ctypes.c_float)),
        ("obs_latents_sum", ctypes.POINTER(ctypes.c_float)),
    ]

lib.write_backwards.restype = ctypes.POINTER(SuffStats)

lib.free_suffstats.argtypes = [ctypes.POINTER(SuffStats)]
lib.free_suffstats.restype = None

# --- Python wrapper functions ---

def write_observations(observations_iter, filepath, batch_size=16000):
    observations_iter = iter(observations_iter)
    first_vector = next(observations_iter)
    dim = len(first_vector)
    c_file = lib.open_file_write(filepath.encode('utf-8'))

    # Write dimension
    dim_c = ctypes.c_int(dim)
    lib.write_ints(ctypes.byref(dim_c), 1, c_file)

    # Flatten iterator
    flat_iter = (f for vector in itertools.chain([first_vector], observations_iter) for f in vector)

    while True:
        batch_list = list(itertools.islice(flat_iter, batch_size))
        if not batch_list:
            break
        arr = array.array('f', batch_list)
        ptr = ctypes.cast(arr.buffer_info()[0], ctypes.POINTER(ctypes.c_float))
        lib.write_floats(ptr, len(arr), c_file)

    lib.close_file(c_file)


def write_forwards(observations_file, forwards_file, params_file, buffer_size=10000):
    c_obs_file = lib.open_file_read(observations_file.encode('utf-8'))
    c_forw_file = lib.open_file_write(forwards_file.encode('utf-8'))
    c_params_file = lib.open_file_read(params_file.encode('utf-8'))
    lib.write_forwards(c_obs_file, c_params_file, c_forw_file, buffer_size)
    lib.close_file(c_obs_file)
    lib.close_file(c_forw_file)
    lib.close_file(c_params_file)


def write_backwards(params_file, obs_file, forwards_file, backwards_file, buffer_size=10000):
    c_params_file = lib.open_file_read(params_file.encode('utf-8'))
    c_obs_file = lib.open_file_read(obs_file.encode('utf-8'))
    c_forw_file = lib.open_file_read(forwards_file.encode('utf-8'))
    c_back_file = lib.open_file_write(backwards_file.encode('utf-8'))

    stats_ptr = lib.write_backwards(c_params_file, c_obs_file, c_forw_file, c_back_file, buffer_size)

    lib.close_file(c_params_file)
    lib.close_file(c_obs_file)
    lib.close_file(c_forw_file)
    lib.close_file(c_back_file)

    # Convert to Python dict
    stats = stats_ptr.contents
    result = {
        'n_obs': stats.n_obs,
        'n_latents': stats.n_latents,
        'num_datapoints': stats.num_datapoints,
        'latents_mu_sum': [stats.latents_mu_sum[i] for i in range(stats.n_latents)],
        'latents_cov_sum': [stats.latents_cov_sum[i] for i in range(stats.n_latents * stats.n_latents)],
        'latents_cov_lag1_sum': [stats.latents_cov_lag1_sum[i] for i in range(stats.n_latents * stats.n_latents)],
        'obs_sum': [stats.obs_sum[i] for i in range(stats.n_obs)],
        'obs_obs_sum': [stats.obs_obs_sum[i] for i in range(stats.n_obs * stats.n_obs)],
        'obs_latents_sum': [stats.obs_latents_sum[i] for i in range(stats.n_obs * stats.n_latents)],
    }

    lib.free_suffstats(stats_ptr)
    return result

def write_params(F,Q,H,R,params_file):
    c_params_file = lib.open_file_write(params_file.encode('utf-8'))

    n_latents, n_obs = len(Q), len(R)
    n_obs_c = ctypes.c_int(n_obs)
    lib.write_ints(ctypes.byref(n_obs_c), 1, c_params_file)
    n_latents_c = ctypes.c_int(n_latents)
    lib.write_ints(ctypes.byref(n_latents_c), 1, c_params_file)

    bools = array.array('i', [1,1,1,1])
    ptr = ctypes.cast(bools.buffer_info()[0], ctypes.POINTER(ctypes.c_int))
    lib.write_ints(ptr, len(bools), c_params_file)

    params_array = array.array('f',[x for matrix in (F,Q,H,R) for row in matrix for x in row])
    ptr = ctypes.cast(params_array.buffer_info()[0], ctypes.POINTER(ctypes.c_float))
    lib.write_floats(ptr, len(params_array), c_params_file)

    lib.close_file(c_params_file)

def write_files(tmp_folder_path, F,Q,H,R, observations_iter=None, store_observations=True):
    if not os.path.exists(tmp_folder_path):
        os.makedirs(tmp_folder_path)

    observations_file = f"{tmp_folder_path}/observations.bin"
    if observations_iter is not None:
        write_observations(observations_iter, observations_file)

    params_file = f"{tmp_folder_path}/params.bin"
    write_params(F,Q,H,R,params_file)

    forwards_file = f"{tmp_folder_path}/forwards.bin"
    write_forwards(observations_file, forwards_file, params_file)

    backwards_file = f"{tmp_folder_path}/backwards.bin"
    stats = write_backwards(params_file, observations_file, forwards_file, backwards_file)

    if not store_observations:
        os.remove(observations_file)

    return forwards_file, backwards_file, stats


def _smooth_memory(F, Q, H, R, observations, return_filtered=False):
    """In-memory Kalman smoother for small datasets."""
    observations = [np.array(obs) for obs in observations]
    F, Q, H, R = np.array(F), np.array(Q), np.array(H), np.array(R)
    n_latents = F.shape[0]
    n_obs = H.shape[0]
    T = len(observations)

    # Forward pass - store filtered means and covariances
    mus = []
    covs = []

    for t, y in enumerate(observations):
        if t == 0:
            # Initialize from first observation
            HHT = H @ H.T
            mu = H.T @ np.linalg.solve(HHT, y)
            cov = np.zeros((n_latents, n_latents))
        else:
            # Predict
            mu_pred = F @ mus[-1]
            cov_pred = F @ covs[-1] @ F.T + Q
            # Update
            S = H @ cov_pred @ H.T + R
            K = cov_pred @ H.T @ np.linalg.solve(S, np.eye(n_obs))
            mu = mu_pred + K @ (y - H @ mu_pred)
            cov = cov_pred - K @ H @ cov_pred
        mus.append(mu)
        covs.append(cov)

    # Backward pass - RTS smoother
    results = []

    # Initialize sufficient statistics
    stats = {
        'latents_mu_sum': np.zeros(n_latents),
        'latents_cov_sum': np.zeros((n_latents, n_latents)),
        'latents_cov_lag1_sum': np.zeros((n_latents, n_latents)),
        'obs_sum': np.zeros(n_obs),
        'obs_obs_sum': np.zeros((n_obs, n_obs)),
        'obs_latents_sum': np.zeros((n_obs, n_latents)),
    }

    # Last timestep: smoothed = filtered
    mu_smooth = mus[-1].copy()
    cov_smooth = covs[-1].copy()
    lag1_cov = np.zeros((n_latents, n_latents))
    results.append((mu_smooth.copy(), cov_smooth.copy(), lag1_cov.copy(), mus[-1].copy()))

    # Accumulate stats for last timestep
    y = observations[-1]
    stats['latents_mu_sum'] += mu_smooth
    stats['latents_cov_sum'] += cov_smooth + np.outer(mu_smooth, mu_smooth)
    stats['obs_sum'] += y
    stats['obs_obs_sum'] += np.outer(y, y)
    stats['obs_latents_sum'] += np.outer(y, mu_smooth)

    # Backward iteration
    for t in range(T - 2, -1, -1):
        mu, cov = mus[t], covs[t]
        mu_pred = F @ mu
        cov_pred = F @ cov @ F.T + Q

        # RTS gain: G = cov @ F.T @ inv(cov_pred)
        G = cov @ F.T @ np.linalg.inv(cov_pred)

        # Smoothed estimates
        mu_smooth_new = mu + G @ (mu_smooth - mu_pred)
        cov_smooth_new = cov + G @ (cov_smooth - cov_pred) @ G.T

        # Lag-1 covariance: E[x_{t+1} x_t^T]
        lag1_cov = cov_smooth @ G.T + np.outer(mu_smooth, mu_smooth_new)

        mu_smooth, cov_smooth = mu_smooth_new, cov_smooth_new
        results.append((mu_smooth.copy(), cov_smooth.copy(), lag1_cov.copy(), mus[t].copy()))

        # Accumulate stats
        y = observations[t]
        stats['latents_mu_sum'] += mu_smooth
        stats['latents_cov_sum'] += cov_smooth + np.outer(mu_smooth, mu_smooth)
        stats['latents_cov_lag1_sum'] += lag1_cov
        stats['obs_sum'] += y
        stats['obs_obs_sum'] += np.outer(y, y)
        stats['obs_latents_sum'] += np.outer(y, mu_smooth)

    # Results are in reverse order
    results = results[::-1]

    # Convert stats to lists (match disk-based format)
    stats_out = {
        'n_obs': n_obs,
        'n_latents': n_latents,
        'num_datapoints': T,
        'latents_mu_sum': stats['latents_mu_sum'].tolist(),
        'latents_cov_sum': stats['latents_cov_sum'].flatten().tolist(),
        'latents_cov_lag1_sum': stats['latents_cov_lag1_sum'].flatten().tolist(),
        'obs_sum': stats['obs_sum'].tolist(),
        'obs_obs_sum': stats['obs_obs_sum'].flatten().tolist(),
        'obs_latents_sum': stats['obs_latents_sum'].flatten().tolist(),
    }

    def gen():
        for mu_smooth, cov_smooth, lag1, mu_filt in results:
            if return_filtered:
                yield mu_smooth.tolist(), cov_smooth.tolist(), lag1.tolist(), mu_filt.tolist()
            else:
                yield mu_smooth.tolist(), cov_smooth.tolist(), lag1.tolist()

    return gen(), stats_out


def smooth(tmp_folder_path, F,Q,H,R, observations_iter=None, store_observations=True, batch_size=10000, return_filtered=False):
    # In-memory mode when tmp_folder is None
    if tmp_folder_path is None:
        if observations_iter is None:
            raise ValueError("observations_iter required for in-memory mode")
        observations = list(observations_iter)
        return _smooth_memory(F, Q, H, R, observations, return_filtered=return_filtered)

    # Disk-based mode
    forwards_file, backwards_file, stats = write_files(tmp_folder_path, F,Q,H,R, observations_iter,
                                                        store_observations=store_observations)
    n_latents = len(Q)
    smooth_record_size = n_latents + 2 * n_latents * n_latents  # mu + cov + lag1_cov
    smooth_record_bytes = smooth_record_size * 4
    filt_record_size = n_latents + n_latents * n_latents  # mu + cov
    filt_record_bytes = filt_record_size * 4

    # The backwards file contains records in reverse chronological order (T-1, T-2, ..., 0).
    # We need to reverse it to yield in chronological order (0, 1, ..., T-1).
    # Write to a new file in chronological order by reading backwards file from end to start.
    chronological_file = os.path.join(tmp_folder_path, 'chronological.bin')

    with open(backwards_file, 'rb') as f_in, open(chronological_file, 'wb') as f_out:
        f_in.seek(0, 2)
        file_size = f_in.tell()
        num_records = file_size // smooth_record_bytes

        # Read from end to beginning, write in that order (which is chronological)
        records_written = 0
        while records_written < num_records:
            batch_records = min(batch_size, num_records - records_written)
            # Seek to position: read the batch that's (records_written + batch_records) from the end
            f_in.seek(file_size - (records_written + batch_records) * smooth_record_bytes)
            data = array.array('f')
            data.fromfile(f_in, batch_records * smooth_record_size)

            # Reverse the batch before writing
            reversed_data = array.array('f')
            for i in range(batch_records - 1, -1, -1):
                offset = i * smooth_record_size
                reversed_data.extend(data[offset:offset + smooth_record_size])

            reversed_data.tofile(f_out)
            records_written += batch_records

    # Remove backwards file, keep chronological
    os.remove(backwards_file)

    def gen():
        files_to_open = [open(chronological_file, 'rb')]
        if return_filtered:
            files_to_open.append(open(forwards_file, 'rb'))

        try:
            f_smooth = files_to_open[0]
            f_filt = files_to_open[1] if return_filtered else None

            file_size = os.path.getsize(chronological_file)
            num_records = file_size // smooth_record_bytes
            records_read = 0

            while records_read < num_records:
                batch_records = min(batch_size, num_records - records_read)

                smooth_data = array.array('f')
                smooth_data.fromfile(f_smooth, batch_records * smooth_record_size)

                if return_filtered:
                    filt_data = array.array('f')
                    filt_data.fromfile(f_filt, batch_records * filt_record_size)

                for i in range(batch_records):
                    s_offset = i * smooth_record_size
                    mu = smooth_data[s_offset:s_offset+n_latents].tolist()
                    cov = [smooth_data[s_offset+n_latents+j*n_latents:s_offset+n_latents+(j+1)*n_latents].tolist() for j in range(n_latents)]
                    lag1_cov = [smooth_data[s_offset+n_latents+n_latents*n_latents+j*n_latents:s_offset+n_latents+n_latents*n_latents+(j+1)*n_latents].tolist() for j in range(n_latents)]

                    if return_filtered:
                        f_offset = i * filt_record_size
                        filt_mu = filt_data[f_offset:f_offset+n_latents].tolist()
                        yield mu, cov, lag1_cov, filt_mu
                    else:
                        yield mu, cov, lag1_cov

                records_read += batch_records
        finally:
            for f in files_to_open:
                f.close()

        os.remove(forwards_file)
        os.remove(chronological_file)

    return gen(), stats
