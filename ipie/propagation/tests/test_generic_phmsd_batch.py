
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
# Authors: Fionn Malone <fionn.malone@gmail.com>
#          Joonho Lee
#

import numpy
import pytest

from ipie.estimators.greens_function_batch import (
    greens_function_multi_det, greens_function_multi_det_wicks)
from ipie.hamiltonians.generic import Generic as HamGeneric
from ipie.legacy.hamiltonians.generic import Generic as LegacyHamGeneric
from ipie.legacy.propagation.continuous import Continuous as LegacyContinuous
from ipie.legacy.trial_wavefunction.multi_slater import (
    MultiSlater as LegacyMultiSlater
    )
from ipie.legacy.walkers.multi_det import MultiDetWalker
from ipie.legacy.walkers.single_det import SingleDetWalker
from ipie.propagation.continuous import Continuous
from ipie.propagation.force_bias import (
    construct_force_bias_batch, construct_force_bias_batch_multi_det_trial,
    construct_force_bias_batch_single_det)
from ipie.propagation.overlap import (calc_overlap_multi_det,
                                      calc_overlap_multi_det_wicks,
                                      calc_overlap_multi_det_wicks_opt)
from ipie.systems.generic import Generic
from ipie.trial_wavefunction.multi_slater import MultiSlater
from ipie.utils.misc import dotdict
from ipie.utils.testing import (generate_hamiltonian, get_random_nomsd,
                                get_random_phmsd)
from ipie.walkers.multi_det_batch import MultiDetTrialWalkerBatch
from ipie.walkers.single_det_batch import SingleDetWalkerBatch


@pytest.mark.unit
def test_phmsd_force_bias_batch():
    numpy.random.seed(7)
    nmo = 10
    nelec = (5, 4)
    nwalkers = 10
    ndets = 2
    nsteps = 100
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    nchols = chol.shape[0]
    system = Generic(nelec=nelec)
    ham = HamGeneric(
        h1e=numpy.array([h1e, h1e]),
        chol=chol.reshape((-1, nmo * nmo)).T.copy(),
        ecore=0,
        options={"symmetry": False},
    )
    legacyham = LegacyHamGeneric(
        h1e=numpy.array([h1e, h1e]),
        chol=chol.reshape((-1, nmo * nmo)).T.copy(),
        ecore=0,
    )
    # Test PH type wavefunction.
    wfn, init = get_random_phmsd(
        system.nup, system.ndown, ham.nbasis, ndet=ndets, init=True
    )
    trial = MultiSlater(system, ham, wfn, init=init)
    trial.half_rotate(system, ham)

    legacytrial = LegacyMultiSlater(
        system, legacyham, wfn, init=init,
    )
    legacytrial.half_rotate(system, legacyham)

    numpy.random.seed(7)

    trial.calculate_energy(system, ham)
    options = {"hybrid": True}
    qmc = dotdict({"dt": 0.005, "nstblz": 5})
    prop = LegacyContinuous(system, ham, trial, qmc, options=options)

    walkers = [MultiDetWalker(system, legacyham, legacytrial) for iw in range(nwalkers)]

    # fb_slow = prop.construct_force_bias_slow(ham, walker, trial)
    # fb_multi_det = prop.construct_force_bias_multi_det(ham, walker, trial)
    fb_ref_slow = numpy.zeros((nwalkers, nchols), dtype=numpy.complex128)
    fb_ref_multi_det = numpy.zeros((nwalkers, nchols), dtype=numpy.complex128)
    for iw in range(nwalkers):
        fb_ref_slow[iw, :] = prop.propagator.construct_force_bias_slow(
            ham, walkers[iw], trial
        )
        fb_ref_multi_det[iw, :] = prop.propagator.construct_force_bias_multi_det(
            ham, walkers[iw], trial
        )

    numpy.random.seed(7)
    qmc = dotdict({"dt": 0.005, "nstblz": 5, "batched": True, "nwalkers": nwalkers})
    prop = Continuous(system, ham, trial, qmc, options=options)

    walker_batch = MultiDetTrialWalkerBatch(system, ham, trial, nwalkers)
    prop.propagator.vbias_batch = construct_force_bias_batch_multi_det_trial(
        ham, walker_batch, trial
    )  # construct_force_bias_batch(ham, walker_batch, trial)
    fb = -prop.propagator.sqrt_dt * (
        1j * prop.propagator.vbias_batch - prop.propagator.mf_shift
    )

    # prop.propagator.vbias_batch = construct_force_bias_batch_single_det(ham, walker_batch, trial) #construct_force_bias_batch(ham, walker_batch, trial)
    # fb_half = - prop.propagator.sqrt_dt * (1j*prop.propagator.vbias_batch-prop.propagator.mf_shift)

    for iw in range(nwalkers):
        assert numpy.allclose(fb_ref_slow[iw], fb_ref_multi_det[iw])
        assert numpy.allclose(fb_ref_slow[iw], fb[iw])
        assert numpy.allclose(fb_ref_multi_det[iw], fb[iw])
        # assert numpy.allclose(fb_ref_multi_det[iw], fb_half[iw])


@pytest.mark.unit
def test_phmsd_greens_function_batch():
    numpy.random.seed(7)
    nmo = 10
    nelec = (5, 4)
    nwalkers = 1
    ndets = 5
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    nchols = chol.shape[0]
    system = Generic(nelec=nelec)
    ham = HamGeneric(
        h1e=numpy.array([h1e, h1e]),
        chol=chol.reshape((-1, nmo * nmo)).T.copy(),
        ecore=0,
        options={"symmetry": False},
    )
    # Test PH type wavefunction.
    wfn, init = get_random_phmsd(
        system.nup, system.ndown, ham.nbasis, ndet=ndets, init=True
    )
    trial = MultiSlater(system, ham, wfn, init=init,
            options={"wicks": True, 'optimized': False})

    numpy.random.seed(7)

    trial.calculate_energy(system, ham)
    trial_slow = MultiSlater(system, ham, wfn, init=init, options={"wicks":
        False, 'optimized': False})
    trial_slow.calculate_energy(system, ham)

    options = {"hybrid": True}
    qmc = dotdict({"dt": 0.005, "nstblz": 5})
    prop = LegacyContinuous(system, ham, trial, qmc, options=options)

    legacyham = LegacyHamGeneric(
        h1e=numpy.array([h1e, h1e]),
        chol=chol.reshape((-1, nmo * nmo)).T.copy(),
        ecore=0,
    )
    legacytrial = LegacyMultiSlater(
            system, legacyham, wfn, init=init, options={"wicks": True,
                'optimized': False}
    )

    walkers = [MultiDetWalker(system, legacyham, legacytrial) for iw in range(nwalkers)]
    walker_batch = MultiDetTrialWalkerBatch(system, ham, trial_slow, nwalkers)

    greens_function_multi_det(walker_batch, trial_slow)
    for iw in range(nwalkers):
        assert numpy.allclose(walker_batch.Gia[iw], walkers[iw].Gi[:, 0, :, :])
        assert numpy.allclose(walker_batch.Gib[iw], walkers[iw].Gi[:, 1, :, :])
        assert numpy.allclose(walker_batch.Ga[iw], walkers[iw].G[0, :, :])
        assert numpy.allclose(walker_batch.Gb[iw], walkers[iw].G[1, :, :])

    greens_function_multi_det_wicks(walker_batch, trial)
    for iw in range(nwalkers):
        assert numpy.allclose(walker_batch.Gia[iw], walkers[iw].Gi[:, 0, :, :])
        assert numpy.allclose(walker_batch.Gib[iw], walkers[iw].Gi[:, 1, :, :])
        assert numpy.allclose(walker_batch.Ga[iw], walkers[iw].G[0, :, :])
        assert numpy.allclose(walker_batch.Gb[iw], walkers[iw].G[1, :, :])


@pytest.mark.unit
def test_phmsd_overlap_batch():
    numpy.random.seed(70)
    nmo = 10
    nelec = (5, 4)
    nwalkers = 1
    ndets = 5
    nsteps = 100
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    nchols = chol.shape[0]
    system = Generic(nelec=nelec)
    ham = HamGeneric(
        h1e=numpy.array([h1e, h1e]),
        chol=chol.reshape((-1, nmo * nmo)).T.copy(),
        ecore=0,
        options={"symmetry": False},
    )
    # Test PH type wavefunction.
    wfn, init = get_random_phmsd(
        system.nup, system.ndown, ham.nbasis, ndet=ndets, init=True
    )
    trial = MultiSlater(system, ham, wfn, init=init,
            options={'wicks': False, 'optimized': False}
            )
    trial_wicks = MultiSlater(system, ham, wfn, init=init)

    numpy.random.seed(70)

    trial.calculate_energy(system, ham)

    legacyham = LegacyHamGeneric(
        h1e=numpy.array([h1e, h1e]),
        chol=chol.reshape((-1, nmo * nmo)).T.copy(),
        ecore=0,
    )
    legacytrial = LegacyMultiSlater(
        system, legacyham, wfn, init=init, options={"wicks": True}
    )

    walkers = [MultiDetWalker(system, legacyham, legacytrial) for iw in range(nwalkers)]
    walker_batch = MultiDetTrialWalkerBatch(system, ham, trial, nwalkers)

    ovlps0 = calc_overlap_multi_det(walker_batch, trial)
    ovlps = calc_overlap_multi_det_wicks(walker_batch, trial_wicks)
    ovlps_opt = calc_overlap_multi_det_wicks_opt(walker_batch, trial_wicks)
    for iw in range(nwalkers):
        assert numpy.allclose(walkers[iw].ovlp, walker_batch.ovlp[iw])
        assert numpy.allclose(ovlps[iw], walker_batch.ovlp[iw])
        assert numpy.allclose(ovlps0[iw], walker_batch.ovlp[iw])
        assert numpy.allclose(ovlps[iw], ovlps_opt[iw])


@pytest.mark.unit
def test_get_dets_single_excitation_batched():
    nwalker = 10
    ndets = 10
    nmo = 10
    G0 = numpy.random.random((nwalker, nmo, nmo)).astype(numpy.complex128)
    G0 += 1j * numpy.random.random((nwalker, nmo, nmo))
    cre_a = [3]
    anh_a = [0]
    p = cre_a[0]
    q = anh_a[0]
    ref = numpy.zeros((nwalker, ndets), dtype=numpy.complex128)
    for iw in range(nwalker):
        for idet in range(ndets):
            ref[iw, idet] = G0[iw, p, q]

    from ipie.utils.misc import dotdict

    trial = dotdict(
        {
            "cre_ex_a": [[0], numpy.array([[p]] * ndets, dtype=int)],
            "anh_ex_a": [[0], numpy.array([[q]] * ndets, dtype=int)],
            "cre_ex_b": [[0], numpy.array([[p]] * ndets, dtype=int)],
            "anh_ex_b": [[0], numpy.array([[q]] * ndets, dtype=int)],
            "occ_map_a": numpy.arange(10, dtype=numpy.int32),
            "occ_map_b": numpy.arange(10, dtype=numpy.int32),
            "nfrozen": 0,
        }
    )
    from ipie.propagation.overlap import (
        get_dets_single_excitation_batched,
        get_dets_single_excitation_batched_opt)

    test = get_dets_single_excitation_batched(G0, G0, trial)
    assert numpy.allclose(ref, test)
    test = get_dets_single_excitation_batched_opt(G0, G0, trial)
    assert numpy.allclose(ref, test)


@pytest.mark.unit
def test_get_dets_double_excitation_batched():
    nwalker = 10
    ndets = 10
    nmo = 10
    G0 = numpy.random.random((nwalker, nmo, nmo)).astype(numpy.complex128)
    G0 += 1j * numpy.random.random((nwalker, nmo, nmo))
    cre_a = [3, 7]
    anh_a = [0, 2]
    p = cre_a[0]
    q = anh_a[0]
    r = cre_a[1]
    s = anh_a[1]
    ref = numpy.zeros((nwalker, ndets), dtype=numpy.complex128)
    for iw in range(nwalker):
        for idet in range(ndets):
            ref[iw, idet] = G0[iw, p, q] * G0[iw, r, s] - G0[iw, p, s] * G0[iw, r, q]

    from ipie.utils.misc import dotdict

    trial = dotdict(
        {
            "cre_ex_a": [[0], [0], numpy.array([cre_a] * ndets, dtype=int)],
            "anh_ex_a": [[0], [0], numpy.array([anh_a] * ndets, dtype=int)],
            "cre_ex_b": [[0], [0], numpy.array([cre_a] * ndets, dtype=int)],
            "anh_ex_b": [[0], [0], numpy.array([anh_a] * ndets, dtype=int)],
            "occ_map_a": numpy.arange(10, dtype=numpy.int32),
            "occ_map_b": numpy.arange(10, dtype=numpy.int32),
            "nfrozen": 0,
        }
    )
    from ipie.propagation.overlap import (
        get_dets_double_excitation_batched,
        get_dets_double_excitation_batched_opt)

    test = get_dets_double_excitation_batched(G0, G0, trial)
    assert numpy.allclose(ref, test)
    test = get_dets_double_excitation_batched_opt(G0, G0, trial)
    assert numpy.allclose(ref, test)


@pytest.mark.unit
def test_get_dets_triple_excitation_batched():
    nwalker = 10
    ndets = 10
    nmo = 10
    G0 = numpy.random.random((nwalker, nmo, nmo)).astype(numpy.complex128)
    G0 += 1j * numpy.random.random((nwalker, nmo, nmo))
    cre_a = [3, 7, 9]
    anh_a = [0, 1, 2]
    p = cre_a[0]
    q = anh_a[0]
    r = cre_a[1]
    s = anh_a[1]
    t = cre_a[2]
    u = anh_a[2]
    ref = numpy.zeros((nwalker, ndets), dtype=numpy.complex128)
    for iw in range(nwalker):
        for idet in range(ndets):
            G0a = G0[iw]
            ovlp_a = G0a[p, q] * (G0a[r, s] * G0a[t, u] - G0a[r, u] * G0a[t, s])
            ovlp_a -= G0a[p, s] * (G0a[r, q] * G0a[t, u] - G0a[r, u] * G0a[t, q])
            ovlp_a += G0a[p, u] * (G0a[r, q] * G0a[t, s] - G0a[r, s] * G0a[t, q])
            ref[iw, idet] = ovlp_a

    from ipie.utils.misc import dotdict

    trial = dotdict(
        {
            "cre_ex_a": [[0], [0], [0], numpy.array([cre_a] * ndets, dtype=int)],
            "anh_ex_a": [[0], [0], [0], numpy.array([anh_a] * ndets, dtype=int)],
            "cre_ex_b": [[0], [0], [0], numpy.array([cre_a] * ndets, dtype=int)],
            "anh_ex_b": [[0], [0], [0], numpy.array([anh_a] * ndets, dtype=int)],
            "occ_map_b": numpy.arange(10, dtype=numpy.int32),
            "occ_map_a": numpy.arange(10, dtype=numpy.int32),
            "nfrozen": 0,
        }
    )
    from ipie.propagation.overlap import (
        get_dets_triple_excitation_batched,
        get_dets_triple_excitation_batched_opt)

    test = get_dets_triple_excitation_batched(G0, G0, trial)
    assert numpy.allclose(ref, test)
    test = get_dets_triple_excitation_batched_opt(G0, G0, trial)
    assert numpy.allclose(ref, test)


@pytest.mark.unit
def test_get_dets_nfold_excitation_batched():
    nwalker = 10
    ndets = 10
    nmo = 18
    G0 = numpy.random.random((nwalker, nmo, nmo)).astype(numpy.complex128)
    G0 += 1j * numpy.random.random((nwalker, nmo, nmo))
    cre_a = [9, 10, 11, 13, 15]
    anh_a = [0, 1, 2, 4, 8]
    ref = numpy.zeros((nwalker, ndets), dtype=numpy.complex128)
    nex_a = 5
    for iw in range(nwalker):
        G0a = G0[iw]
        for idet in range(ndets):
            det_a = numpy.zeros((nex_a, nex_a), dtype=numpy.complex128)
            for iex in range(nex_a):
                p = cre_a[iex]
                q = anh_a[iex]
                det_a[iex, iex] = G0a[p, q]
                for jex in range(iex + 1, nex_a):
                    r = cre_a[jex]
                    s = anh_a[jex]
                    det_a[iex, jex] = G0a[p, s]
                    det_a[jex, iex] = G0a[r, q]
            ref[iw, idet] = numpy.linalg.det(det_a)
    from ipie.utils.misc import dotdict

    empty = [[0], [0], [0], [0], [0]]
    trial = dotdict(
        {
            "cre_ex_a": empty + [numpy.array([cre_a] * ndets, dtype=int)],
            "anh_ex_a": empty + [numpy.array([anh_a] * ndets, dtype=int)],
            "cre_ex_b": empty + [numpy.array([cre_a] * ndets, dtype=int)],
            "anh_ex_b": empty + [numpy.array([anh_a] * ndets, dtype=int)],
            "occ_map_b": numpy.arange(18, dtype=numpy.int32),
            "occ_map_a": numpy.arange(18, dtype=numpy.int32),
            "nfrozen": 0,
        }
    )
    from ipie.propagation.overlap import (
        get_dets_nfold_excitation_batched,
        get_dets_nfold_excitation_batched_opt)

    test = get_dets_nfold_excitation_batched(5, G0, G0, trial)
    assert numpy.allclose(ref, test)
    test = get_dets_nfold_excitation_batched_opt(5, G0, G0, trial)
    assert numpy.allclose(ref, test)


@pytest.mark.unit
def test_phmsd_propagation_batch():
    numpy.random.seed(7)
    nmo = 10
    nelec = (5, 4)
    nwalkers = 10
    ndets = 5
    nsteps = 20
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    nchols = chol.shape[0]
    system = Generic(nelec=nelec)
    ham = HamGeneric(
        h1e=numpy.array([h1e, h1e]),
        chol=chol.reshape((-1, nmo * nmo)).T.copy(),
        ecore=0,
        options={"symmetry": False},
    )
    # Test PH type wavefunction.
    wfn, init = get_random_phmsd(
        system.nup, system.ndown, ham.nbasis, ndet=ndets, init=True
    )
    trial = MultiSlater(system, ham, wfn, init=init, options={"wicks": False})

    numpy.random.seed(7)

    trial.calculate_energy(system, ham)
    options = {"hybrid": True}
    qmc = dotdict({"dt": 0.005, "nstblz": 5})
    prop = LegacyContinuous(system, ham, trial, qmc, options=options)

    legacyham = LegacyHamGeneric(
        h1e=numpy.array([h1e, h1e]),
        chol=chol.reshape((-1, nmo * nmo)).T.copy(),
        ecore=0,
    )
    legacytrial = LegacyMultiSlater(
        system, legacyham, wfn, init=init, options={"wicks": True}
    )

    walkers = [MultiDetWalker(system, legacyham, legacytrial) for iw in range(nwalkers)]
    fb_ref = numpy.zeros((nwalkers, nchols), dtype=numpy.complex128)
    for iw in range(nwalkers):
        fb_ref[iw, :] = prop.propagator.construct_force_bias(ham, walkers[iw], trial)

    numpy.random.seed(7)
    qmc = dotdict({"dt": 0.005, "nstblz": 5, "batched": True, "nwalkers": nwalkers})
    prop_batch = Continuous(system, ham, trial, qmc, options=options)

    walker_batch = MultiDetTrialWalkerBatch(system, ham, trial, nwalkers)
    prop_batch.propagator.vbias_batch = construct_force_bias_batch(
        ham, walker_batch, trial
    )

    for istep in range(nsteps):
        for iw in range(nwalkers):
            prop.propagate_walker(walkers[iw], system, ham, trial, trial.energy)

    numpy.random.seed(7)

    for istep in range(nsteps):
        prop_batch.propagate_walker_batch(
            walker_batch, system, ham, trial, trial.energy
        )

    for iw in range(nwalkers):
        assert numpy.allclose(walkers[iw].phi[:, : nelec[0]], walker_batch.phia[iw])
        assert numpy.allclose(walkers[iw].phi[:, nelec[0] :], walker_batch.phib[iw])
        assert numpy.allclose(walkers[iw].weight, walker_batch.weight[iw])


if __name__ == "__main__":
    test_phmsd_greens_function_batch()
    test_phmsd_force_bias_batch()
    test_phmsd_overlap_batch()
    test_phmsd_propagation_batch()
