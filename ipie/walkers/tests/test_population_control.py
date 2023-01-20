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
#          Ankit Mahajan <ankitmahajan76@gmail.com>
#

import numpy
import pytest

from ipie.hamiltonians.generic import Generic as HamGeneric
from ipie.legacy.propagation.continuous import Continuous as LegacyContinuous
from ipie.legacy.walkers.handler import Walkers
from ipie.propagation.continuous import Continuous
from ipie.systems.generic import Generic
from ipie.trial_wavefunction.particle_hole import ParticleHoleNaive
from ipie.utils.misc import dotdict
from ipie.utils.mpi import MPIHandler
from ipie.utils.testing import (
    generate_hamiltonian,
    get_random_phmsd,
    build_test_case_handlers_mpi,
)
from ipie.walkers.walker_batch_handler import WalkerBatchHandler
from ipie.legacy.trial_wavefunction.multi_slater import MultiSlater
from ipie.utils.legacy_testing import build_legacy_test_case_handlers_mpi


@pytest.mark.unit
def test_pair_branch_batch():
    import mpi4py

    mpi4py.rc.recv_mprobe = False
    from mpi4py import MPI

    comm = MPI.COMM_WORLD

    mpi_handler = MPIHandler(comm)

    nelec = (5, 5)
    nwalkers = 10
    nsteps = 1
    nmo = 10

    numpy.random.seed(7)
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    sys = Generic(nelec=nelec)
    ham = HamGeneric(
        h1e=numpy.array([h1e, h1e]),
        chol=chol.reshape((-1, nmo * nmo)).T.copy(),
        ecore=0,
        options={"symmetry": False},
    )
    print("h1e: ", numpy.sum(h1e))
    ham.control_variate = False
    # Test PH type wavefunction.
    wfn, init = get_random_phmsd(sys.nup, sys.ndown, ham.nbasis, ndet=1, init=True)
    print("this: ", numpy.sum(init))
    # trial = ParticleHoleNaive(wfn, nelec, nmo)
    trial = MultiSlater(sys, ham, wfn, init=init)
    trial.half_rotate(sys, ham)
    trial.calculate_energy(sys, ham)

    # numpy.random.seed(7)
    options = {"hybrid": True, "population_control": "pair_branch"}
    qmc = dotdict({"dt": 0.005, "nstblz": 5, "nwalkers": nwalkers, "batched": True})
    qmc.ntot_walkers = qmc.nwalkers * comm.size
    numpy.random.seed(7)
    qmc = dotdict({"dt": 0.005, "nstblz": 5, "nwalkers": nwalkers, "batched": False})
    qmc.ntot_walkers = qmc.nwalkers * comm.size
    prop = LegacyContinuous(sys, ham, trial, qmc, options=options)
    handler = Walkers(sys, ham, trial, qmc, options, verbose=True, comm=comm)
    print("init old: ", numpy.sum(handler.walkers[0].phi), numpy.sum(init))

    for i in range(nsteps):
        for walker in handler.walkers:
            prop.propagate_walker(walker, sys, ham, trial, trial.energy)
            detR = walker.reortho(trial)  # reorthogonalizing to stablize
        handler.pop_control(comm)

    options = {}
    qmc = dotdict(
        {
            "dt": 0.005,
            "nstblz": 5,
            "nwalkers": nwalkers,
            "batched": False,
            "hybrid": True,
            "num_steps": nsteps,
            "population_control": "pair_branch",
        }
    )
    legacy_walkers = build_legacy_test_case_handlers_mpi(
        nelec, nmo, mpi_handler, num_dets=1, complex_trial=True, options=qmc, seed=7
    )
    qmc.batched = True
    handler_batch = build_test_case_handlers_mpi(
        nelec, nmo, mpi_handler, num_dets=1, complex_trial=True, options=qmc, seed=7
    )
    print(
        legacy_walkers.walkers[0].weight,
        handler.walkers[0].weight,
        handler_batch.walkers_batch.weight[0],
    )
    for iw in range(nwalkers):
        assert (
            pytest.approx(handler_batch.walkers_batch.weight[0]) == 0.2571750688329709
        )
        assert (
            pytest.approx(handler_batch.walkers_batch.weight[1]) == 1.0843219322894988
        )
        assert (
            pytest.approx(handler_batch.walkers_batch.weight[2]) == 0.8338283613093604
        )
        assert (
            pytest.approx(handler_batch.walkers_batch.phia[iw][0, 0])
            == -0.0005573508035052743 + 0.12432250308987346j
        )


if __name__ == "__main__":
    test_pair_branch_batch()
    test_comb_batch()
