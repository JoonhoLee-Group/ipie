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

from typing import Tuple

import numpy as np
import numpy.typing as npt


def one_rdm(
    ci_coeffs: npt.NDArray,
    occa: npt.NDArray,
    occb: npt.NDArray,
    n_act: int,
    n_elec: Tuple[int, int],
    n_spatial: int,
    n_melting: int,
) -> npt.NDArray:
    """Compute one-particle reduced density matrix.

    Parameters
    ----------
    ci_coeffs : np.ndarray
        CI coefficients.
    occa : np.ndarray
        Array of alpha occupation strings which make up wavefunction.
    occb : np.ndarray
        Array of beta occupation strings which make up wavefunction.
    n_act : int
        Number of active spatial orbitals
    n_elec : tuple(int, int)
        Number of electrons (including melting cores)
    n_spatial: int
        Number of spatial orbitals
    n_melting : int
        Number of melting core orbitals.
    Returns
    -------
    opdm : np.ndarray
        One-particle reduced density matrix.
    """
    n_alpha, _ = n_elec
    n_alpha_act = len(occa[0])
    n_melting = n_alpha - n_alpha_act
    one_rdm_act = one_rdm_active_space(ci_coeffs, occa, occb, n_act)
    one_rdm_full = np.zeros((2, n_spatial, n_spatial), dtype=one_rdm_act.dtype)
    diag = np.diag_indices(n_melting)
    one_rdm_full[0, diag, diag] = 1.0
    one_rdm_full[1, diag, diag] = 1.0
    one_rdm_full[0, n_melting : n_melting + n_act, n_melting : n_melting + n_act] = one_rdm_act[0]
    one_rdm_full[1, n_melting : n_melting + n_act, n_melting : n_melting + n_act] = one_rdm_act[1]
    return one_rdm_full


def one_rdm_active_space(
    ci_coeffs: npt.NDArray, occa: npt.NDArray, occb: npt.NDArray, num_spatial: int
) -> npt.NDArray:
    """Compute one-particle reduced density matrix.

    Parameters
    ----------
    ci_coeffs : np.ndarray
        CI coefficients.
    occa : np.ndarray
        Array of alpha occupation strings which make up wavefunction.
    occb : np.ndarray
        Array of beta occupation strings which make up wavefunction.
    num_spatial: int
        Number of spatial orbitals
    Returns
    -------
    opdm : np.ndarray
        One-particle reduced density matrix.
    """
    try:
        import ipie.lib.libci.libci as libci
    except ImportError:
        raise ImportError("libci not found. Did you forget to compile?")

    wfn = libci.Wavefunction(ci_coeffs, occa, occb, num_spatial)
    return np.array(wfn.one_rdm()).reshape((2, num_spatial, num_spatial))


def variational_energy(
    ci_coeffs: npt.NDArray,
    occa: npt.NDArray,
    occb: npt.NDArray,
    h1e: npt.NDArray,
    h2e: npt.NDArray,
    e0: float,
) -> npt.NDArray:
    """Compute one-particle reduced density matrix.

    Parameters
    ----------
    ci_coeffs : np.ndarray
        CI coefficients.
    occa : np.ndarray
        Array of alpha occupation strings which make up wavefunction.
    occb : np.ndarray
        Array of beta occupation strings which make up wavefunction.
    num_spatial: int
        Number of spatial orbitals

    Returns
    -------
    energy : np.ndarray
        An array of length three containing (e_tot, e_1b, e_2b)
    """
    try:
        import ipie.lib.libci.libci as libci
    except ImportError:
        raise ImportError("libci not found. Did you forget to compile?")
    num_spat = h1e.shape[-1]
    ham = libci.Hamiltonian(h1e, h2e, e0)
    wfn = libci.Wavefunction(ci_coeffs, occa, occb, num_spat)
    return wfn.energy(ham)
