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

from ipie.estimators.local_energy_sd import (
    ecoul_kernel_batch_real_rchol_uhf,
    exx_kernel_batch_real_rchol,
)
from ipie.hamiltonians.generic import GenericRealChol
from ipie.trial_wavefunction.noci import NOCI
from ipie.utils.backend import arraylib as xp
from ipie.walkers.uhf_walkers import UHFWalkersNOCI


def local_energy_noci(
    hamiltonian: GenericRealChol, walkers: UHFWalkersNOCI, trial: NOCI
):
    """Compute local energy for walker batch (all walkers at once).

    Multi determinant NOCI trial case.

    Parameters
    ----------
    hamiltonian : hamiltonian object
        Hamiltonian being studied.
    walkers : WalkerBatch
        Walkers object.
    trial : trial object
        Trial wavefunctioni.

    Returns
    -------
    local_energy : np.ndarray
        Total, one-body and two-body energies.
    """
    ndets = walkers.Ghalfa.shape[0]
    nwalkers = walkers.Ghalfa.shape[1]
    nalpha = walkers.Ghalfa.shape[2]
    nbeta = walkers.Ghalfb.shape[2]
    nbasis = hamiltonian.nbasis

    energy = xp.zeros((nwalkers, 3), dtype=xp.complex128)
    ovlp_fac = xp.einsum(
        "wi,wi,i->iw",
        walkers.det_ovlpas,
        walkers.det_ovlpbs,
        trial.coeffs.conj(),
        optimize=True,
    )
    for idet in range(ndets):
        Ghalfa = walkers.Ghalfa[idet].reshape(nwalkers, nalpha * nbasis)
        Ghalfb = walkers.Ghalfb[idet].reshape(nwalkers, nbeta * nbasis)
        e1b = Ghalfa.dot(trial._rH1a[idet].ravel())
        e1b += Ghalfb.dot(trial._rH1b[idet].ravel())
        energy[:, 1] += e1b * ovlp_fac[idet]

        ecoul = ecoul_kernel_batch_real_rchol_uhf(
            trial._rchola[idet], trial._rcholb[idet], Ghalfa, Ghalfb
        )
        energy[:, 2] += ecoul * ovlp_fac[idet]
        exx = exx_kernel_batch_real_rchol(
            trial._rchola[idet], Ghalfa.reshape((nwalkers, nalpha, nbasis))
        ) + exx_kernel_batch_real_rchol(
            trial._rcholb[idet], Ghalfb.reshape((nwalkers, nbeta, nbasis))
        )
        energy[:, 2] -= exx * ovlp_fac[idet]

    energy[:, 0] = energy[:, 1] + energy[:, 2]
    energy /= xp.einsum("iw->w", ovlp_fac)[:, None]  # broadcasting
    energy[0] += hamiltonian.ecore
    energy[1] += hamiltonian.ecore
    return energy
