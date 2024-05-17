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

from ipie.systems.generic import Generic
from ipie.utils.testing import generate_hamiltonian
from ipie.hamiltonians.generic import Generic as HamGeneric
from ipie.addons.thermal.trial.one_body import OneBody


@pytest.mark.unit
def test_one_body():
    nup = 5
    ndown = 5
    nelec = (nup, ndown)
    nbasis = 10

    mu = -1.
    beta = 0.1
    timestep = 0.01

    complex_integrals = True
    verbose = True

    sym = 8
    if complex_integrals: sym = 4
    
    # Test.
    system = Generic(nelec)
    h1e, chol, _, eri = generate_hamiltonian(nbasis, nelec, cplx=complex_integrals, 
                                             sym=sym, tol=1e-10)
    hamiltonian = HamGeneric(h1e=numpy.array([h1e, h1e]),
                             chol=chol.reshape((-1, nbasis**2)).T.copy(),
                             ecore=0)
    trial = OneBody(hamiltonian, nelec, beta, timestep, verbose=verbose)

    assert trial.nelec == nelec
    numpy.testing.assert_almost_equal(trial.nav, numpy.sum(nelec), decimal=6)
    assert trial.rho.shape == (2, nbasis, nbasis)
    assert trial.dmat.shape == (2, nbasis, nbasis)
    assert trial.P.shape == (2, nbasis, nbasis)
    assert trial.G.shape == (2, nbasis, nbasis)


if __name__ == '__main__':
    test_one_body()
    
   

