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
# Author: Fionn Malone <fionn.malone@gmail.com>
#

import ctypes
import os

import numpy as np
from numpy.ctypeslib import ndpointer

_path = os.path.dirname(__file__)
try:
    _wicks_helper = np.ctypeslib.load_library("libwicks", _path)
except OSError:
    raise ImportError


def build_one_rdm(ci_coeffs, dets, norbs, nelec):
    """Compute one-particle reduced density matrix.

    Parameters
    ----------
    ci_coeffs : np.ndarray
        CI coefficients.
    dets : np.ndarray
        List of determinants making up wavefunction.
    norbs : int
        Number of orbitals
    nelec : int
        Total number of electrons
    Returns
    -------
    opdm : np.ndarray
        One-particle reduced density matrix.
    """
    ndets = len(ci_coeffs)
    if ci_coeffs.dtype == np.complex128:
        fun = _wicks_helper.compute_density_matrix_cmplx
        fun.restype = None
        fun.argtypes = [
            ndpointer(np.complex128, flags="C_CONTIGUOUS"),
            ndpointer(shape=(ndets, DET_LEN), dtype=ctypes.c_ulonglong, flags="C_CONTIGUOUS"),
            ndpointer(np.complex128, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
    elif ci_coeffs.dtype == np.float64:
        fun = _wicks_helper.compute_density_matrix_cmplx
        fun.restype = None
        fun.argtypes = [
            ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
            ndpointer(shape=(ndets, DET_LEN), dtype=ctypes.c_ulonglong, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
        ]
    else:
        raise TypeError("Unknown type for ci_coeffs")
    opdm = np.zeros((2, norbs, norbs), dtype=ci_coeffs.dtype)
    occs = np.zeros((nelec), dtype=np.int32)
    fun(ci_coeffs, dets, opdm, occs, ci_coeffs.size, norbs, nelec)
    return opdm

def compute_variational_energy(ci_coeffs, occa, occb):