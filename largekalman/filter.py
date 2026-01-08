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
lib.write_backwards.restype = None

lib.read_floats_backwards.argtypes = [ctypes.c_int, ctypes.c_void_p]
lib.read_floats_backwards.restype = ctypes.POINTER(ctypes.c_float)

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
    lib.write_backwards(c_params_file, c_obs_file, c_forw_file, c_back_file, buffer_size)
    lib.close_file(c_params_file)
    lib.close_file(c_forw_file)
    lib.close_file(c_back_file)

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
    write_backwards(params_file, observations_file, forwards_file, backwards_file)

    if not store_observations:
        os.remove(observations_file)

    return forwards_file, backwards_file


def smooth(tmp_folder_path, F,Q,H,R, observations_iter=None, store_observations=True, batch_size=256000):
    print('writing files')
    forwards_file, backwards_file = write_files(tmp_folder_path, F,Q,H,R, observations_iter,
                                                store_observations=store_observations)
    print("wrote files")

    print('opening files')
    c_forw_file = lib.open_file_read(forwards_file.encode('utf-8'))
    c_back_file = lib.open_file_read(backwards_file.encode('utf-8'))

    print('main logic')
    try:
        while True:
            batch_ptr = lib.smooth()
            if not batch_ptr:
                break

            # Convert C buffer to Python array
            buf = array.array('f', [0.0] * (dim * batch_size))  # allocated buffer
            ctypes.memmove(buf.buffer_info()[0], batch_ptr, dim * batch_size * ctypes.sizeof(ctypes.c_float))
            yield buf.tolist()
    finally:
        print('closing files')
        lib.close_file(c_forw_file)
        lib.close_file(c_back_file)
        input("removing (press return)")
        os.remove(forwards_file)
        os.remove(backwards_file)

"""
def compute_suffstats(obs_file, backwards_file):
    c_forw_file = lib.open_file_read(forwards_file.encode('utf-8'))
    c_back_file = lib.open_file_read(backwards_file.encode('utf-8'))

    stats = lib.compute_suffstats(c_forw_file, c_back_file)

    lib.close_file(c_forw_file)
    lib.close_file(c_back_file)

    return stats
"""
