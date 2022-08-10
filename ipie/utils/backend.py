import importlib
import sys

from ipie.config import config

_use_gpu = config.get_option('use_gpu')
print("config : ", config)

try:
    import cupy as _cp
    _have_cupy = True
except ImportError:
    _have_cupy = False

import numpy as _np
print(_have_cupy, _use_gpu)



def to_host_cpu(array):
    return _np.array(array)

def to_host_gpu(array):
    return _cp.asnumpy(array)

def get_cpu_free_memory():
    try:
        return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1024**3.0
    except:
        return 0.0

def get_gpu_free_memory():
    free_bytes, total_bytes = _cp.cuda.Device().mem_info
    used_bytes = total_bytes - free_bytes
    return used_bytes, total_bytes

def synchronize_cpu():
    pass

def synchronize_gpu():
    _cp.cuda.stream.get_current_stream().synchronize()

if _use_gpu and _have_cupy:
    print("this:")
    arraylib = _cp
    to_host = to_host_gpu
    synchronize = synchronize_gpu
    qr_mode = 'reduced'
    qr = _cp.linalg.qr
    get_device_memory = get_gpu_free_memory
    get_host_memory = get_cpu_free_memory
else:
    arraylib = _np
    to_host = to_host_cpu
    synchronize = synchronize_cpu
    qr_mode = 'economic'
    import scipy.linalg
    qr = scipy.linalg.qr
    get_device_memory = get_cpu_free_memory
    get_host_memory = get_cpu_free_memory

def cast_to_device(self):
    for k, v in self.__dict__.items():
        if isinstance(v, np.ndarray):
            self.__dict__[k] = arraylib.toarray(v)
