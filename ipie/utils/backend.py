
# Copyright 2022 The ipie Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Fionn Malone <fmalone@google.com>
#

import sys

from ipie.config import config

_use_gpu = config.get_option('use_gpu')

try:
    import cupy as _cp
    _have_cupy = True
except ImportError:
    _have_cupy = False

import numpy as _np

def to_host_cpu(array):
    return array

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

def cast_to_device(self, verbose=False):
    if not (_use_gpu and _have_cupy):
        return
    size = 0
    for k, v in self.__dict__.items():
        if isinstance(v, _np.ndarray):
            size += v.size
        elif isinstance(v, list) and isinstance(v[0], _np.ndarray):
            size += sum(vi.size for vi in v)
    if verbose:
        expected_bytes = size * 16.0
        expected_gb = expected_bytes / 1024.0**3.0
        print(
            f"# {self.__class__.__name__}: expected to allocate {expected_gb} GB"
            )

    for k, v in self.__dict__.items():
        if isinstance(v, _np.ndarray):
            self.__dict__[k] = arraylib.array(v)
        elif isinstance(v, list) and isinstance(v[0], _np.ndarray):
            self.__dict__[k] = [arraylib.array(vi) for vi in v]

    used_bytes, total_bytes = get_device_memory()
    used_gb, total_gb = used_bytes / 1024**3.0, total_bytes / 1024**3.0
    if verbose:
        print(
            f"# {self.__class__.__name__}: using {used_gb} GB out of {total_gb} GB memory on GPU"
            )
