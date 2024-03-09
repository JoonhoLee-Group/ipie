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

from ipie.estimators.greens_function import greens_function_single_det_batch
from ipie.estimators.local_energy_sd import (
    local_energy_single_det_batch,
    local_energy_single_det_rhf_batch,
    local_energy_single_det_uhf,
)
from ipie.hamiltonians.generic import Generic as HamGeneric
from ipie.legacy.hamiltonians._generic import Generic as HamGenericRef
from ipie.propagation.phaseless_generic import PhaselessGeneric
from ipie.systems.generic import Generic
from ipie.trial_wavefunction.particle_hole import ParticleHole
from ipie.trial_wavefunction.single_det import SingleDet
from ipie.utils.legacy_testing import build_legacy_test_case, get_legacy_walker_energies
from ipie.utils.misc import dotdict
from ipie.utils.mpi import MPIHandler
from ipie.utils.pack_numba import pack_cholesky
from ipie.utils.testing import generate_hamiltonian, get_random_phmsd
from ipie.walkers.walkers_dispatch import UHFWalkersTrial


@pytest.mark.unit
def test_greens_function_batch():
    numpy.random.seed(7)
    nmo = 10
    nelec = (6, 5)
    nwalkers = 12
    nsteps = 25
    dt = 0.005
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    system = Generic(nelec=nelec)
    hamiltonian = HamGeneric(
        h1e=numpy.array([h1e, h1e]),
        chol=chol.reshape((-1, nmo * nmo)).T.copy(),
        ecore=0,
    )
    # Test PH type wavefunction.
    wfn, init = get_random_phmsd(system.nup, system.ndown, hamiltonian.nbasis, ndet=1, init=True)
    trial = ParticleHole(wfn, nelec, nmo)
    trial.build()
    trial.half_rotate(hamiltonian)

    from ipie.legacy.hamiltonians.generic import Generic as LegacyHamGeneric

    legacy_hamiltonian = LegacyHamGeneric(
        h1e=numpy.array([h1e, h1e]),
        chol=chol.reshape((-1, nmo * nmo)).T.copy(),
        ecore=0,
    )

    legacy_walkers = build_legacy_test_case(wfn, init, system, legacy_hamiltonian, nsteps, nwalkers, dt)
    numpy.random.seed(7)
    walkers = UHFWalkersTrial(
        trial, init, system.nup, system.ndown, hamiltonian.nbasis, nwalkers, MPIHandler()
    )
    walkers.build(trial)
    for iw in range(nwalkers):
        walkers.phia[iw] = legacy_walkers[iw].phi[:, : nelec[0]].copy()
        walkers.phib[iw] = legacy_walkers[iw].phi[:, nelec[0] :].copy()
    ovlp = greens_function_single_det_batch(walkers, trial)

    ot = [legacy_walkers[iw].ot for iw in range(walkers.nwalkers)]
    assert numpy.allclose(ovlp, ot)

    for iw in range(nwalkers):
        # assert numpy.allclose(walkers.Ga[iw], walkers[iw].G[0])
        # assert numpy.allclose(walkers.Gb[iw], walkers[iw].G[1])
        assert numpy.allclose(walkers.Ghalfa[iw], legacy_walkers[iw].Ghalf[0])
        assert numpy.allclose(walkers.Ghalfb[iw], legacy_walkers[iw].Ghalf[1])


@pytest.mark.unit
def test_local_energy_single_det_batch():
    numpy.random.seed(7)
    nmo = 10
    nelec = (5, 5)
    nwalkers = 10
    nsteps = 25
    dt = 0.005
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    system = Generic(nelec=nelec)
    hamiltonian = HamGeneric(
        h1e=numpy.array([h1e, h1e]),
        chol=chol.reshape((-1, nmo * nmo)).T.copy(),
        ecore=0,
    )
    # Test PH type wavefunction.
    ci_wfn, init = get_random_phmsd(system.nup, system.ndown, hamiltonian.nbasis, ndet=1, init=True)
    I = numpy.eye(nmo)
    wfn = numpy.zeros((nmo, sum(nelec)), dtype=numpy.complex128)
    occa0 = ci_wfn[1][0]
    occb0 = ci_wfn[2][0]
    wfn[:, : nelec[0]] = I[:, occa0]
    wfn[:, nelec[0] :] = I[:, occb0]
    trial = SingleDet(wfn, nelec, nmo)
    trial.build()
    trial.half_rotate(hamiltonian)

    from ipie.legacy.hamiltonians.generic import Generic as LegacyHamGeneric

    legacy_hamiltonian = LegacyHamGeneric(
        h1e=numpy.array([h1e, h1e]),
        chol=chol.reshape((-1, nmo * nmo)).T.copy(),
        ecore=0,
        options={"symmetry": False},
    )
    numpy.random.seed(7)
    legacy_walkers = build_legacy_test_case(ci_wfn, init, system, legacy_hamiltonian, nsteps, nwalkers, dt)
    etots, e1s, e2s = get_legacy_walker_energies(system, legacy_hamiltonian, trial, legacy_walkers)

    numpy.random.seed(7)
    qmc = dotdict({"dt": 0.005, "nstblz": 5, "batched": True, "nwalkers": 10})
    prop = PhaselessGeneric(time_step=qmc["dt"])
    prop.build(hamiltonian, trial)
    walkers = UHFWalkersTrial(
        trial, init, system.nup, system.ndown, hamiltonian.nbasis, nwalkers, MPIHandler()
    )
    walkers.build(trial)
    for i in range(nsteps):
        prop.propagate_walkers(walkers, hamiltonian, trial, 0)
        walkers.reortho()

    ovlp = greens_function_single_det_batch(walkers, trial)
    energies = local_energy_single_det_batch(system, hamiltonian, walkers, trial)
    energies_uhf = local_energy_single_det_uhf(system, hamiltonian, walkers, trial)

    assert numpy.allclose(energies, energies_uhf)

    for iw in range(nwalkers):
        assert numpy.allclose(walkers.phia[iw], legacy_walkers[iw].phi[:, : nelec[0]])
        assert numpy.allclose(walkers.phib[iw], legacy_walkers[iw].phi[:, nelec[0] :])
        assert numpy.allclose(etots[iw], energies[iw, 0])
        assert numpy.allclose(e1s[iw], energies[iw, 1])
        assert numpy.allclose(e2s[iw], energies[iw, 2])


@pytest.mark.unit
def test_local_energy_single_det_batch_packed():
    numpy.random.seed(7)
    nmo = 10
    nelec = (5, 5)
    nwalkers = 10
    nsteps = 25
    dt = 0.005
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    chol = chol.reshape((-1, nmo * nmo)).T.copy()

    nchol = chol.shape[-1]
    chol = chol.reshape((nmo, nmo, nchol))
    idx = numpy.triu_indices(nmo)
    cp_shape = (nmo * (nmo + 1) // 2, chol.shape[-1])
    chol_packed = numpy.zeros(cp_shape, dtype=chol.dtype)
    pack_cholesky(idx[0], idx[1], chol_packed, chol)
    chol = chol.reshape((nmo * nmo, nchol))

    system = Generic(nelec=nelec)

    hamiltonian = HamGeneric(
        h1e=numpy.array([h1e, h1e]),
        chol=chol,
        ecore=0,
        # chol_packed=chol_packed,
        # options={"symmetry": True},
    )

    legacy_hamiltonian = HamGenericRef(
        h1e=numpy.array([h1e, h1e]),
        chol=chol,
        ecore=0,
        chol_packed=chol_packed,
        options={"symmetry": True},
    )

    # Test PH type wavefunction.
    ci_wfn, init = get_random_phmsd(system.nup, system.ndown, hamiltonian.nbasis, ndet=1, init=True)
    I = numpy.eye(nmo)
    wfn = numpy.zeros((nmo, sum(nelec)), dtype=numpy.complex128)
    occa0 = ci_wfn[1][0]
    occb0 = ci_wfn[2][0]
    wfn[:, : nelec[0]] = I[:, occa0]
    wfn[:, nelec[0] :] = I[:, occb0]
    trial = SingleDet(wfn, nelec, nmo)
    trial.build()
    trial.half_rotate(hamiltonian)

    numpy.random.seed(7)
    legacy_walkers = build_legacy_test_case(ci_wfn, init, system, legacy_hamiltonian, nsteps, nwalkers, dt)
    etots, e1s, e2s = get_legacy_walker_energies(system, legacy_hamiltonian, trial, legacy_walkers)

    numpy.random.seed(7)
    qmc = dotdict({"dt": 0.005, "nstblz": 5, "batched": True, "nwalkers": nwalkers})
    prop = PhaselessGeneric(time_step=qmc["dt"])
    prop.build(hamiltonian, trial)
    walkers = UHFWalkersTrial(
        trial, init, system.nup, system.ndown, hamiltonian.nbasis, nwalkers, MPIHandler()
    )
    walkers.build(trial)
    for i in range(nsteps):
        prop.propagate_walkers(walkers, hamiltonian, trial, 0.0)
        walkers.reortho()

    ovlp = greens_function_single_det_batch(walkers, trial)
    energies = local_energy_single_det_batch(system, hamiltonian, walkers, trial)

    for iw in range(nwalkers):
        assert numpy.allclose(walkers.phia[iw], legacy_walkers[iw].phi[:, : nelec[0]])
        # assert numpy.allclose(energy, energies[iw])
        assert numpy.allclose(etots[iw], energies[iw, 0])
        assert numpy.allclose(e1s[iw], energies[iw, 1])
        assert numpy.allclose(e2s[iw], energies[iw, 2])


@pytest.mark.unit
def test_local_energy_single_det_batch_rhf():
    numpy.random.seed(7)
    nmo = 10
    nelec = (5, 5)
    nwalkers = 10
    nsteps = 25
    dt = 0.005
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    system = Generic(nelec=nelec)
    ham = HamGeneric(
        h1e=numpy.array([h1e, h1e]), chol=chol.reshape((-1, nmo * nmo)).T.copy(), ecore=0
    )
    ci_wfn, init = get_random_phmsd(system.nup, system.ndown, ham.nbasis, ndet=1, init=True)
    ci_wfn, init = get_random_phmsd(system.nup, system.ndown, ham.nbasis, ndet=1, init=True)
    I = numpy.eye(nmo)
    wfn = numpy.zeros((nmo, sum(nelec)), dtype=numpy.complex128)
    occa0 = ci_wfn[1][0]
    occb0 = ci_wfn[2][0]
    wfn[:, : nelec[0]] = I[:, occa0]
    wfn[:, nelec[0] :] = I[:, occb0]
    trial = SingleDet(wfn, nelec, nmo)
    trial.build()
    trial.half_rotate(ham)
    init[:, : nelec[0]] = init[:, nelec[0] :].copy()

    numpy.random.seed(7)
    legacy_ham = HamGenericRef(
        h1e=numpy.array([h1e, h1e]),
        chol=chol.reshape((-1, nmo * nmo)).T.copy(),
        ecore=0,
        options={"symmetry": False},
    )

    legacy_walkers = build_legacy_test_case(ci_wfn, init, system, legacy_ham, nsteps, nwalkers, dt)
    etots, e1s, e2s = get_legacy_walker_energies(system, legacy_ham, trial, legacy_walkers)

    numpy.random.seed(7)
    qmc = dotdict({"dt": 0.005, "nstblz": 5, "batched": True, "nwalkers": 10})
    prop = PhaselessGeneric(time_step=qmc["dt"])
    prop.build(ham, trial)
    walker_opts = dotdict({"rhf": True})
    # walkers = SingleDetWalkerBatch(
    #     system, ham, trial, nwalkers, init, walker_opts={"rhf": True}
    # )
    walkers = UHFWalkersTrial(
        trial, init, system.nup, system.ndown, ham.nbasis, nwalkers, MPIHandler()
    )
    walkers.build(trial)
    walkers.rhf = True
    for i in range(nsteps):
        prop.propagate_walkers(walkers, ham, trial, 0.0)
        walkers.reortho()

    ovlp = greens_function_single_det_batch(walkers, trial)
    energies = local_energy_single_det_rhf_batch(system, ham, walkers, trial)

    for iw in range(nwalkers):
        assert numpy.allclose(walkers.phia[iw], legacy_walkers[iw].phi[:, : nelec[0]])
        # assert numpy.allclose(energy, energies[iw])
        assert numpy.allclose(etots[iw], energies[iw, 0])
        assert numpy.allclose(e1s[iw], energies[iw, 1])
        assert numpy.allclose(e2s[iw], energies[iw, 2])


@pytest.mark.unit
def test_local_energy_single_det_batch_rhf_packed():
    numpy.random.seed(7)
    nmo = 10
    nelec = (5, 5)
    nwalkers = 10
    nsteps = 25
    dt = 0.005
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)

    chol = chol.reshape((-1, nmo * nmo)).T.copy()

    nchol = chol.shape[-1]
    chol = chol.reshape((nmo, nmo, nchol))

    idx = numpy.triu_indices(nmo)
    cp_shape = (nmo * (nmo + 1) // 2, chol.shape[-1])
    chol_packed = numpy.zeros(cp_shape, dtype=chol.dtype)
    pack_cholesky(idx[0], idx[1], chol_packed, chol)
    chol = chol.reshape((nmo * nmo, nchol))

    system = Generic(nelec=nelec)
    hamiltonian = HamGeneric(
        h1e=numpy.array([h1e, h1e]),
        chol=chol,
        ecore=0,
    )

    legacy_hamiltonian = HamGenericRef(
        h1e=numpy.array([h1e, h1e]),
        chol=chol,
        ecore=0,
        chol_packed=chol_packed,
        options={"symmetry": True},
    )

    ci_wfn, init = get_random_phmsd(system.nup, system.ndown, hamiltonian.nbasis, ndet=1, init=True)
    I = numpy.eye(nmo)
    wfn = numpy.zeros((nmo, sum(nelec)), dtype=numpy.complex128)
    occa0 = ci_wfn[1][0]
    occb0 = ci_wfn[2][0]
    wfn[:, : nelec[0]] = I[:, occa0]
    wfn[:, nelec[0] :] = I[:, occb0]
    trial = SingleDet(wfn, nelec, nmo)
    trial.build()
    trial.half_rotate(hamiltonian)
    init[:, : nelec[0]] = init[:, nelec[0] :].copy()

    numpy.random.seed(7)
    legacy_walkers = build_legacy_test_case(ci_wfn, init, system, legacy_hamiltonian, nsteps, nwalkers, dt)
    etots, e1s, e2s = get_legacy_walker_energies(system, legacy_hamiltonian, trial, legacy_walkers)

    numpy.random.seed(7)
    qmc = dotdict({"dt": 0.005, "nstblz": 5, "batched": True, "nwalkers": 10})
    prop = PhaselessGeneric(time_step=qmc["dt"])
    prop.build(hamiltonian, trial)
    walker_opts = dotdict({"rhf": True})
    # walkers = SingleDetWalkerBatch(
    #     system, hamiltonian, trial, nwalkers, init, walker_opts=walker_opts
    # )
    walkers = UHFWalkersTrial(
        trial, init, system.nup, system.ndown, hamiltonian.nbasis, nwalkers, MPIHandler()
    )
    walkers.build(trial)
    walkers.rhf = True

    for i in range(nsteps):
        prop.propagate_walkers(walkers, hamiltonian, trial, 0.0)
        walkers.reortho()

    ovlp = greens_function_single_det_batch(walkers, trial)
    energies = local_energy_single_det_rhf_batch(system, hamiltonian, walkers, trial)

    for iw in range(nwalkers):
        assert numpy.allclose(walkers.phia[iw], legacy_walkers[iw].phi[:, : nelec[0]])
        # assert numpy.allclose(energy, energies[iw])
        assert numpy.allclose(etots[iw], energies[iw, 0])
        assert numpy.allclose(e1s[iw], energies[iw, 1])
        assert numpy.allclose(e2s[iw], energies[iw, 2])


if __name__ == "__main__":
    test_greens_function_batch()
    test_local_energy_single_det_batch()
    test_local_energy_single_det_batch_packed()
    test_local_energy_single_det_batch_rhf()
    test_local_energy_single_det_batch_rhf_packed()
