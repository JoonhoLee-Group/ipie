
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

from ipie.estimators.greens_function_batch import compute_greens_function
from ipie.hamiltonians.generic import Generic as HamGeneric
from ipie.legacy.hamiltonians.generic import Generic as LegacyHamGeneric
from ipie.legacy.propagation.continuous import Continuous as LegacyContinuous
from ipie.legacy.trial_wavefunction.multi_slater import \
    MultiSlater as LegacyMultiSlater
from ipie.legacy.walkers.multi_det import MultiDetWalker
from ipie.legacy.walkers.single_det import SingleDetWalker
from ipie.propagation.continuous import Continuous
from ipie.propagation.force_bias import construct_force_bias_batch
from ipie.propagation.operations import kinetic_real, kinetic_spin_real_batch
from ipie.systems.generic import Generic
from ipie.trial_wavefunction.multi_slater import MultiSlater
from ipie.utils.misc import dotdict
from ipie.utils.pack import pack_cholesky
from ipie.utils.testing import (generate_hamiltonian, get_random_nomsd,
                                get_random_phmsd)
from ipie.walkers.multi_det_batch import MultiDetTrialWalkerBatch
from ipie.walkers.single_det_batch import SingleDetWalkerBatch

@pytest.mark.gpu
def test_hybrid_batch():
    numpy.random.seed(7)
    nmo = 10
    nelec = (5, 5)
    nwalkers = 10
    nsteps = 25
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    system = Generic(nelec=nelec)
    ham = LegacyHamGeneric(
        h1e=numpy.array([h1e, h1e]),
        chol=chol.reshape((-1, nmo * nmo)).T.copy(),
        ecore=0,
    )
    # Test PH type wavefunction.
    wfn, init = get_random_phmsd(
        system.nup, system.ndown, ham.nbasis, ndet=1, init=True
    )
    trial = LegacyMultiSlater(system, ham, wfn, init=init)
    trial.half_rotate(system, ham)
    trial.psi = trial.psi[0]
    trial.psia = trial.psia[0]
    trial.psib = trial.psib[0]
    trial.calculate_energy(system, ham)

    numpy.random.seed(7)
    import cupy
    cupy.random.seed(7)
    options = {"hybrid": True}
    qmc = dotdict({"dt": 0.005, "nstblz": 5})
    prop = LegacyContinuous(system, ham, trial, qmc, options=options)

    walkers = [SingleDetWalker(system, ham, trial) for iw in range(nwalkers)]
    ovlps = []
    for i in range(nsteps):
        for walker in walkers:
            ovlps += [walker.greens_function(trial)]
            kinetic_real(walker.phi, system, prop.propagator.BH1)
            detR = walker.reortho(trial)  # reorthogonalizing to stablize

    numpy.random.seed(7)
    cupy.random.seed(7)

    options = {"hybrid": True}
    qmc = dotdict(
        {
            "dt": 0.005,
            "nstblz": 5,
            "batched": True,
            "nwalkers": nwalkers,
            "batched": True,
        }
    )

    chol = chol.reshape((-1, nmo * nmo)).T.copy()

    nchol = chol.shape[-1]
    chol = chol.reshape((nmo, nmo, nchol))

    idx = numpy.triu_indices(nmo)
    cp_shape = (nmo * (nmo + 1) // 2, chol.shape[-1])
    chol_packed = numpy.zeros(cp_shape, dtype=chol.dtype)
    pack_cholesky(idx[0], idx[1], chol_packed, chol)
    chol = chol.reshape((nmo * nmo, nchol))

    ham = HamGeneric(
        h1e=numpy.array([h1e, h1e]), chol=chol, chol_packed=chol_packed, ecore=0
    )
    trial = MultiSlater(system, ham, wfn, init=init)
    trial.half_rotate(system, ham)
    trial.psi = trial.psi[0]
    trial.psia = trial.psia[0]
    trial.psib = trial.psib[0]
    prop = Continuous(system, ham, trial, qmc, options=options)
    walker_batch = SingleDetWalkerBatch(system, ham, trial, nwalkers)

    prop.cast_to_cupy()
    ham.cast_to_cupy()
    trial.cast_to_cupy()
    walker_batch.cast_to_cupy()

    numpy.random.seed(7)
    cupy.random.seed(7)
    ovlps_batch = []
    for i in range(nsteps):
        ovlps_batch += [compute_greens_function(walker_batch, trial)]
        walker_batch.phia = kinetic_spin_real_batch(
            walker_batch.phia, prop.propagator.BH1[0]
        )
        walker_batch.phib = kinetic_spin_real_batch(
            walker_batch.phib, prop.propagator.BH1[1]
        )
        walker_batch.reortho()

    phi_batch = cupy.array(walker_batch.phia)
    phi_batch = cupy.asnumpy(phi_batch)

    # assert numpy.allclose(ovlps, cupy.asnumpy(ovlps_batch))

    # Using abs following batched qr implementation on gpu which does not
    # preserve previous gauge fixing of sequential algorithm.
    for iw in range(nwalkers):
        assert numpy.allclose(
                abs(phi_batch[iw]),
                abs(walkers[iw].phi[:, : system.nup])
                )

    phi_batch = cupy.array(walker_batch.phib)
    phi_batch = cupy.asnumpy(phi_batch)
    for iw in range(nwalkers):
        assert numpy.allclose(
                abs(phi_batch[iw]),
                abs(walkers[iw].phi[:, system.nup :])
                )

if __name__ == "__main__":
    test_hybrid_batch()
