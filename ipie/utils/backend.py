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

if _use_gpu and _have_cupy:
    arraylib = _cp
    to_host = to_host_gpu
else:
    arraylib = _np
    to_host = to_host_cpu
