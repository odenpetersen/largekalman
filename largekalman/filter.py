import os
import itertools
import ctypes
import array

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
    print(f'wrote to {params_file=}')

def write_files(tmp_folder_path, F,Q,H,R, observations_iter=None, store_observations=True):
    if not os.path.exists(tmp_folder_path):
        os.makedirs(tmp_folder_path)

    observations_file = f"{tmp_folder_path}/observations.bin"
    if observations_iter is not None:
        write_observations(observations_iter, observations_file)

    print('writing params')
    params_file = f"{tmp_folder_path}/params.bin"
    write_params(F,Q,H,R,params_file)

    forwards_file = f"{tmp_folder_path}/forwards.bin"
    write_forwards(observations_file, forwards_file, params_file)

    print('write backwards')
    backwards_file = f"{tmp_folder_path}/backwards.bin"
    stats = write_backwards(params_file, observations_file, forwards_file, backwards_file)

    if not store_observations:
        os.remove(observations_file)

    return forwards_file, backwards_file, stats


def smooth(tmp_folder_path, F,Q,H,R, observations_iter=None, store_observations=True, batch_size=10000):
    forwards_file, backwards_file, stats = write_files(tmp_folder_path, F,Q,H,R, observations_iter,
                                                        store_observations=store_observations)
    n_latents = len(Q)
    record_size = n_latents + 2 * n_latents * n_latents  # mu + cov + lag1_cov
    record_bytes = record_size * 4

    def gen():
        with open(backwards_file, 'rb') as f:
            f.seek(0, 2)
            file_size = f.tell()
            num_records = file_size // record_bytes

            records_read = 0
            while records_read < num_records:
                batch_records = min(batch_size, num_records - records_read)
                f.seek(file_size - (records_read + batch_records) * record_bytes)
                data = array.array('f')
                data.fromfile(f, batch_records * record_size)

                for i in range(batch_records - 1, -1, -1):
                    offset = i * record_size
                    mu = data[offset:offset+n_latents].tolist()
                    cov = [data[offset+n_latents+j*n_latents:offset+n_latents+(j+1)*n_latents].tolist() for j in range(n_latents)]
                    lag1_cov = [data[offset+n_latents+n_latents*n_latents+j*n_latents:offset+n_latents+n_latents*n_latents+(j+1)*n_latents].tolist() for j in range(n_latents)]
                    yield mu, cov, lag1_cov
                records_read += batch_records

        os.remove(forwards_file)
        os.remove(backwards_file)

    return gen(), stats
