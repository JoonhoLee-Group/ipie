try:
    import cupy as _cp
    _have_cupy = True
except ImportError:
    _have_cupy = False

import numpy as _np

from ipie.config import config

_use_gpu = config.get_option('use_gpu')

def to_host_cpu(array):
    return array

def to_host_gpu(array):
    return _cp.asnumpy(array)

def synchronize_cpu():
    pass

def synchronize_gpu():
    cupy.cuda.stream.get_current_stream().synchronize()

if _use_gpu and _have_cupy:
    arraylib = _cp
    to_host = to_host_gpu
    synchronize = synchronize_gpu
    qr_mode = 'reduced'
    qr = cupy.linalg.qr
else:
    arraylib = _np
    to_host = to_host_cpu
    synchronize = synchronize_cpu
    qr_mode = 'reduced'
    import scipy.linalg
    qr = scipy.linalg.qr

def cast_to_device(self):
    for k, v in self.__dict__.items():
        if isinstance(v, np.ndarray):
            self.__dict__[k] = arraylib.toarray(v)
