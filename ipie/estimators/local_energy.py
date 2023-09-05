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
# Authors: Joonho Lee
#          Fionn Malone <fionn.malone@gmail.com>
#

from ipie.estimators.generic import local_energy_cholesky_opt, local_energy_generic_cholesky
from ipie.estimators.greens_function_single_det import gab_mod_ovlp
from ipie.legacy.estimators.ci import get_hmatel
from ipie.utils.backend import arraylib as xp


def local_energy_G(system, hamiltonian, trial, G, Ghalf):
    """Compute local energy from a given Green's function G.

    Parameters
    ----------
    system : system object
        System being studied.
    system : hamiltonian object
        Hamiltonian being studied.
    trial : trial wavefunction object
        Trial wavefunction.
    G : np.ndarray
        Walker Green's function.
    Ghalf : np.ndarray
        Walker half-rotated Green's function.

    Returns
    -------
    local_energy : tuple / array
        Total, one-body and two-body energies.
    """
    assert len(G) == 2

    # unfortunate interfacial problem for the HH model
    # if type(hamiltonian) == Generic[hamiltonian.chol.dtype]:
    if Ghalf is not None:
        return local_energy_cholesky_opt(
            system,
            hamiltonian.ecore,
            Ghalfa=Ghalf[0],
            Ghalfb=Ghalf[1],
            trial=trial,
        )
    else:
        return local_energy_generic_cholesky(system, hamiltonian, G)


def local_energy(system, hamiltonian, walker, trial):
    return local_energy_G(system, hamiltonian, trial, walker.G, walker.Ghalf)


def variational_energy(system, hamiltonian, trial):
    assert len(trial.psi.shape) == 2 or len(trial.psi) == 1
    return local_energy(system, hamiltonian, trial, trial)


def variational_energy_ortho_det(system, ham, occs, coeffs):
    """Compute variational energy for CI-like multi-determinant expansion.

    Parameters
    ----------
    system : :class:`ipie.system` object
        System object.
    occs : list of lists
        list of determinants.
    coeffs : :class:`numpy.ndarray`
        Expansion coefficients.

    Returns
    -------
    energy : tuple of float / complex
        Total energies: (etot,e1b,e2b).
    """
    evar = 0.0
    denom = 0.0
    one_body = 0.0
    two_body = 0.0
    nel = system.nup + system.ndown
    for i, (occi, ci) in enumerate(zip(occs, coeffs)):
        denom += ci.conj() * ci
        for j in range(0, i + 1):
            cj = coeffs[j]
            occj = occs[j]
            etot, e1b, e2b = ci.conj() * cj * get_hmatel(ham, nel, occi, occj)
            evar += etot
            one_body += e1b
            two_body += e2b
            if j < i:
                # Use Hermiticity
                evar += etot.conj()
                one_body += e1b.conj()
                two_body += e2b.conj()
    return evar / denom, one_body / denom, two_body / denom


def variational_energy_noci(system, hamiltonian, trial):
    weight = 0
    energies = 0
    denom = 0
    for i, (Ba, Bb) in enumerate(zip(trial.psia, trial.psib)):
        for j, (Aa, Ab) in enumerate(zip(trial.psia, trial.psib)):
            # construct "local" green's functions for each component of A
            Gup, _, inv_O_up = gab_mod_ovlp(Ba, Aa)
            Gdn, _, inv_O_dn = gab_mod_ovlp(Bb, Ab)
            ovlp = 1.0 / (xp.linalg.det(inv_O_up) * xp.linalg.det(inv_O_dn))
            weight = (trial.coeffs[i].conj() * trial.coeffs[j]) * ovlp
            G = xp.array([Gup, Gdn])
            # Ghalf = [Ghalfa, Ghalfb]
            e = xp.array(local_energy_G(system, hamiltonian, trial, G, Ghalf=None))
            energies += weight * e
            denom += weight
    return tuple(energies / denom)
