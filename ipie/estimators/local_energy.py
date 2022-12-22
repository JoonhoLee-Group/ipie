
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

import numpy

from ipie.estimators.generic import (local_energy_cholesky_opt,
                                     local_energy_cholesky_opt_dG,
                                     local_energy_generic_cholesky,
                                     local_energy_generic_opt)
from ipie.legacy.estimators.ci import get_hmatel
from ipie.legacy.estimators.local_energy import local_energy_G as legacy_local_energy_G


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
    ghf = G[0].shape[-1] == 2 * hamiltonian.nbasis
    # unfortunate interfacial problem for the HH model

    if hamiltonian.name == "Generic":
        if Ghalf is not None:
            if hamiltonian.exact_eri:
                return local_energy_generic_opt(system, G, Ghalf=Ghalf, eri=trial._eri)
            else:
                if hamiltonian.density_diff:
                    return local_energy_cholesky_opt_dG(
                        system,
                        hamiltonian.ecore,
                        Ghalfa=Ghalf[0],
                        Ghalfb=Ghalf[1],
                        trial=trial,
                    )
                else:
                    return local_energy_cholesky_opt(
                        system,
                        hamiltonian.ecore,
                        Ghalfa=Ghalf[0],
                        Ghalfb=Ghalf[1],
                        trial=trial,
                    )
        else:
            return local_energy_generic_cholesky(system, hamiltonian, G)
    else:
        return legacy_local_energy_G(system, hamiltonian, trial, G, Ghalf)


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
