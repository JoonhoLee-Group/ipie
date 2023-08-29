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
import numpy.typing as npt
from numpy.ctypeslib import ndpointer

_path = os.path.dirname(__file__)
try:
    _libci = np.ctypeslib.load_library("libci", _path)
except OSError:
    raise ImportError


def one_rdm(ci_coeffs: npt.NDArray, occa: npt.NDArray, occb: npt.NDArray, num_spatial: int):
    """Compute one-particle reduced density matrix.

    Parameters
    ----------
    ci_coeffs : np.ndarray
        CI coefficients.
    occa : np.ndarray
        Iterable of alpha occupation strings which make up wavefunction.
    occb : np.ndarray
        Iterable of beta occupation strings which make up wavefunction.
    num_spatial: int
        Number of spatial orbitals
    Returns
    -------
    opdm : np.ndarray
        One-particle reduced density matrix.
    """
    if ci_coeffs.dtype == np.complex128:
        _ci_coeffs = ci_coeffs
    elif ci_coeffs.dtype == np.float64:
        _ci_coeffs = np.array(ci_coeffs, dtype=np.complex128)
    else:
        raise TypeError(f"Unknown type for ci_coeffs {ci_coeffs.dtype}")
    fun = _libci.one_rdm_wrapper
    fun.argtypes = [
        ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
        ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
        ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
        ctypes.c_size_t,
    ]
    opdm = fun(_ci_coeffs, occa, occb, num_spatial)
    return opdm
