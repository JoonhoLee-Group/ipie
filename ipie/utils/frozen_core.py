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
# Authors: Fionn Malone <fmalone@google.com>
#
from typing import Optional, Tuple

import numpy as np
import scipy.linalg

from ipie.estimators.generic import core_contribution_cholesky
from ipie.hamiltonians.generic import GenericRealChol
from ipie.legacy.estimators.generic import local_energy_generic_cholesky
from ipie.legacy.estimators.greens_function import gab
from ipie.utils.misc import dotdict


def freeze_core(h1e, chol, ecore, X, nfrozen, verbose=False):
    # 1. Construct one-body hamiltonian
    nbasis = h1e.shape[-1]
    nchol = chol.shape[0]
    chol = chol.reshape((nchol, nbasis, nbasis))
    ham = dotdict(
        {
            "H1": np.array([h1e, h1e]),
            "chol_vecs": chol.T.copy().reshape((nbasis * nbasis, nchol)),
            "nchol": nchol,
            "ecore": ecore,
            "nbasis": nbasis,
        }
    )
    system = dotdict({"nup": 0, "ndown": 0})
    if len(X.shape) == 2:
        psi_a = np.identity(nbasis)[:, :nfrozen]
        psi_b = np.identity(nbasis)[:, :nfrozen]
    elif len(X.shape) == 3:
        C = X
        psi_a = np.identity(nbasis)[:, :nfrozen]
        Xinv = scipy.linalg.inv(X[0])
        psi_b = np.dot(Xinv, C[1])[:, :nfrozen]

    Gcore_a = gab(psi_a, psi_a)
    Gcore_b = gab(psi_b, psi_b)
    ecore = local_energy_generic_cholesky(system, ham, [Gcore_a, Gcore_b])[0]

    (hc_a, hc_b) = core_contribution_cholesky(chol, [Gcore_a, Gcore_b])
    h1e = np.array([h1e, h1e])
    h1e[0] = h1e[0] + 2 * hc_a
    h1e[1] = h1e[1] + 2 * hc_b
    h1e = h1e[:, nfrozen:, nfrozen:]
    nchol = chol.shape[0]
    nact = h1e.shape[-1]
    chol = chol[:, nfrozen:, nfrozen:].reshape((nchol, nact, nact))
    # 4. Subtract one-body term from writing H2 as sum of squares.
    if verbose:
        print(f"# Number of active orbitals: {nact}")
        print(f"# Freezing {nfrozen} core orbitals.")
        print(f"# Frozen core energy : {ecore.real:15.12e}")
    return h1e, chol, ecore.real


def active_space_hamiltonian(
    nact: int,
    nfrozen: int,
    ham: GenericRealChol,
    basis_change_mat: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Generate Active space hamiltonian from Cholesky."""
    if basis_change_mat is None:
        nbasis = ham.nbasis
        basis_change_mat = np.eye(nbasis)
    LXpq = ham.chol.T.copy()
    if nfrozen > 0:
        h1e, chol, e0 = freeze_core(ham.H1[0], LXpq, ham.ecore, basis_change_mat, nfrozen)
        h1e = h1e[0]
    else:
        h1e = ham.H1[0]
        chol = ham.chol.T.reshape((-1, nbasis, nbasis))
        e0 = ham.ecore
    h1e_act = h1e[:nact, :nact]
    chol = chol[:, :nact, :nact]
    eri_act = np.einsum("Xpq,Xrs->pqrs", chol, chol, optimize=True)
    return h1e_act, eri_act, e0
