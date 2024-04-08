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
#          Joonho Lee
#

import numpy
import pytest

try:
    import ipie.legacy.estimators.ueg_kernels
    _no_cython = False

except ModuleNotFoundError:
    _no_cython = True

from ipie.systems.generic import Generic
from ipie.utils.testing import generate_hamiltonian
from ipie.hamiltonians.generic import Generic as HamGeneric

from ipie.addons.thermal.trial.mean_field import MeanField

from ipie.legacy.hamiltonians._generic import Generic as LegacyHamGeneric
from ipie.legacy.trial_density_matrices.mean_field import MeanField as LegacyMeanField


@pytest.mark.skipif(_no_cython, reason="Need to build cython modules.")
@pytest.mark.unit
def test_mean_field():
    nup = 5
    ndown = 5
    nelec = (nup, ndown)
    nbasis = 10

    mu = -10.
    beta = 0.1
    timestep = 0.01
    
    alt_convention = False
    sparse = False
    complex_integrals = True
    verbose = True

    sym = 8
    if complex_integrals: sym = 4
    
    # Test.
    print('\n----------------------------')
    print('Constructing test objects...')
    print('----------------------------')
    system = Generic(nelec)
    h1e, chol, _, eri = generate_hamiltonian(nbasis, nelec, cplx=complex_integrals, 
                                             sym=sym, tol=1e-10)
    hamiltonian = HamGeneric(h1e=numpy.array([h1e, h1e]),
                             chol=chol.reshape((-1, nbasis**2)).T.copy(),
                             ecore=0)
    trial = MeanField(hamiltonian, nelec, beta, timestep, verbose=verbose)

    # Lgeacy.
    print('\n------------------------------')
    print('Constructing legacy objects...')
    print('------------------------------')
    legacy_system = Generic(nelec, verbose=verbose)
    legacy_system.mu = mu
    legacy_hamiltonian = LegacyHamGeneric(
                            h1e=hamiltonian.H1,
                            chol=hamiltonian.chol,
                            ecore=hamiltonian.ecore, verbose=verbose)
    legacy_hamiltonian.hs_pot = numpy.copy(hamiltonian.chol)
    legacy_hamiltonian.hs_pot = legacy_hamiltonian.hs_pot.T.reshape(
            (hamiltonian.nchol, hamiltonian.nbasis, hamiltonian.nbasis))
    legacy_hamiltonian.mu = mu
    legacy_hamiltonian._alt_convention = alt_convention
    legacy_hamiltonian.sparse = sparse
    legacy_trial = LegacyMeanField(legacy_system, legacy_hamiltonian, beta, 
                                   timestep, verbose=verbose)

    assert trial.nelec == nelec
    numpy.testing.assert_almost_equal(trial.nav, numpy.sum(nelec), decimal=5)
    assert trial.rho.shape == (2, nbasis, nbasis)
    assert trial.dmat.shape == (2, nbasis, nbasis)
    assert trial.P.shape == (2, nbasis, nbasis)
    assert trial.G.shape == (2, nbasis, nbasis)

    numpy.testing.assert_allclose(trial.mu, legacy_trial.mu)
    numpy.testing.assert_allclose(trial.nav, legacy_trial.nav)
    numpy.testing.assert_allclose(trial.P, legacy_trial.P)
    numpy.testing.assert_allclose(trial.G, legacy_trial.G)
    numpy.testing.assert_allclose(trial.dmat, legacy_trial.dmat)
    numpy.testing.assert_allclose(trial.dmat_inv, legacy_trial.dmat_inv)


if __name__ == '__main__':
    test_mean_field()
    
   

