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
import pytest

from ipie.estimators.energy import local_energy

from ipie.utils.testing import build_test_case_handlers
from ipie.systems.generic import Generic 
from ipie.hamiltonians.generic import Generic as HamGeneric
from ipie.utils.misc import dotdict

@pytest.mark.unit
def test_local_energy_single_det():
    numpy.random.seed(7)
    nmo = 10
    nelec = (6, 5)
    nwalkers = 8
    nsteps = 25
    qmc = dotdict(
        {
            "dt": 0.005,
            "nstblz": 5,
            "nwalkers": nwalkers,
            "hybrid": True,
            "num_steps": nsteps,
        }
    )
    
    test_handler = build_test_case_handlers(nelec, nmo, num_dets=1, options=qmc, seed=7, complex_integrals=False, complex_trial = False, trial_type="single_det")

    ham = test_handler.hamiltonian
    walkers = test_handler.walkers
    system = Generic(nelec)
    trial = test_handler.trial
    
    chol = ham.chol

    cx_chol = numpy.array(chol, dtype=numpy.complex128)
    cx_ham = HamGeneric(numpy.array(ham.H1,dtype=numpy.complex128), cx_chol, ham.ecore, verbose=False)

    energy = local_energy(system, ham, walkers, trial)

    trial.half_rotate(cx_ham)
    cx_energy = local_energy(system, cx_ham, walkers, trial)

    numpy.testing.assert_allclose(energy, cx_energy, atol=1e-10)


if __name__ == "__main__":
    test_local_energy_single_det()
