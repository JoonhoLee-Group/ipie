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
import scipy.linalg
import pytest

from ipie.addons.thermal.trial.chem_pot import find_chemical_potential
from ipie.legacy.trial_density_matrices.chem_pot import find_chemical_potential as legacy_find_chemical_potential


@pytest.mark.unit
def test_find_chemical_potential():
    dt = 0.01
    beta = 1
    stack_size = 3
    nstack = 20
    nav = 7
    nbsf = 14
    alt_convention = False

    dtau = dt * stack_size
    h1e = numpy.random.random((nbsf, nbsf))
    rho = numpy.array([scipy.linalg.expm(-dtau * h1e),
                       scipy.linalg.expm(-dtau * h1e)])

    mu = find_chemical_potential(alt_convention, rho, dt, nstack, nav)
    legacy_mu = legacy_find_chemical_potential(alt_convention, rho, dt, nstack, nav)

    numpy.testing.assert_allclose(mu, legacy_mu)


if __name__ == '__main__':
    test_find_chemical_potential()
    
   

