# Copyright 2026 The ipie Developers. All Rights Reserved.
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
# Authors: Jinghong Zhang <jinghongzhang@fas.harvard.edu>
#

import numpy
import pytest

from ipie.estimators.energy import EnergyEstimator
from ipie.hamiltonians.hubbard import Hubbard
from ipie.qmc.afqmc import AFQMC
from ipie.trial_wavefunction.single_det import SingleDet


@pytest.mark.unit
def test_hubbard_energy_estimator_builds_full_greens_function():
    nbasis = 4
    nelec = (1, 1)
    h1 = numpy.zeros((nbasis, nbasis))
    for i in range(nbasis - 1):
        h1[i, i + 1] = h1[i + 1, i] = -1.0
    hamiltonian = Hubbard(numpy.array([h1, h1]), U=4.0)
    _, coeff = numpy.linalg.eigh(h1)
    trial = SingleDet(numpy.hstack([coeff[:, :1], coeff[:, :1]]), nelec, nbasis, verbose=False)
    trial.half_rotate(hamiltonian)

    afqmc = AFQMC.build(
        nelec,
        hamiltonian,
        trial,
        num_walkers=4,
        num_steps_per_block=2,
        num_blocks=1,
        timestep=0.005,
        seed=7,
        verbose=False,
    )
    estimator = EnergyEstimator(system=afqmc.system, ham=hamiltonian, trial=trial)
    data = estimator.compute_estimator(afqmc.system, afqmc.walkers, hamiltonian, trial)

    assert numpy.trace(afqmc.walkers.Ga[0]).real == pytest.approx(nelec[0])
    assert numpy.trace(afqmc.walkers.Gb[0]).real == pytest.approx(nelec[1])
    assert data[estimator.get_index("EDenom")].real == pytest.approx(afqmc.walkers.nwalkers)
    assert abs(data[estimator.get_index("ENumer")]) > 1.0e-12
