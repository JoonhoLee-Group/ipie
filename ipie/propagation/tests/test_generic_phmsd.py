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

from ipie.propagation.force_bias import (
    construct_force_bias_batch_multi_det_trial,
)
from ipie.propagation.overlap import (
    calc_overlap_multi_det_wicks,
    calc_overlap_multi_det_wicks_opt,
)
from ipie.systems.generic import Generic
from ipie.utils.misc import dotdict
from ipie.utils.testing import (
    build_test_case_handlers,
)
from ipie.utils.legacy_testing import build_legacy_test_case_handlers


@pytest.mark.unit
def test_phmsd_force_bias_batch():
    numpy.random.seed(7)
    nmo = 10
    nelec = (5, 4)
    nwalkers = 10
    ndets = 2
    nsteps = 100
    qmc = dotdict(
        {
            "dt": 0.005,
            "nstblz": 5,
            "nwalkers": nwalkers,
            "batched": False,
            "hybrid": True,
            "num_steps": nsteps,
        }
    )
    legacy_data = build_legacy_test_case_handlers(
        nelec,
        nmo,
        num_dets=ndets,
        options=qmc,
        seed=7,
        trial_type="phmsd",
    )
    qmc.batched = True
    batched_data = build_test_case_handlers(
        nelec,
        nmo,
        num_dets=ndets,
        options=qmc,
        seed=7,
        trial_type="phmsd",
    )
    walkers = legacy_data.walker_handler.walkers
    walker_batch = batched_data.walkers

    nchols = batched_data.hamiltonian.nchol
    # fb_slow = prop.construct_force_bias_slow(batched_data.hamiltonian, walker, trial)
    # fb_multi_det = prop.construct_force_bias_multi_det(batched_data.hamiltonian, walker, trial)
    fb_ref_slow = numpy.zeros((nwalkers, nchols), dtype=numpy.complex128)
    fb_ref_multi_det = numpy.zeros((nwalkers, nchols), dtype=numpy.complex128)
    prop = legacy_data.propagator
    for iw in range(nwalkers):
        fb_ref_slow[iw, :] = prop.propagator.construct_force_bias_slow(
            legacy_data.hamiltonian, walkers[iw], legacy_data.trial
        )
        fb_ref_multi_det[iw, :] = prop.propagator.construct_force_bias_multi_det(
            legacy_data.hamiltonian, walkers[iw], legacy_data.trial
        )

    prop = batched_data.propagator
    prop.vbias = construct_force_bias_batch_multi_det_trial(
        batched_data.hamiltonian,
        walker_batch,
        batched_data.trial,
    )  # construct_force_bias_batch(batched_data.hamiltonian, walker_batch, trial)
    fb = -prop.sqrt_dt * (1j * prop.vbias - prop.mf_shift)

    # prop.propagator.vbias = construct_force_bias_batch_single_det(batched_data.hamiltonian, walker_batch, trial) #construct_force_bias_batch(batched_data.hamiltonian, walker_batch, trial)
    # fb_half = - prop.propagator.sqrt_dt * (1j*prop.propagator.vbias-prop.propagator.mf_shift)

    for iw in range(nwalkers):
        assert numpy.allclose(fb_ref_slow[iw], fb_ref_multi_det[iw])
        assert numpy.allclose(fb_ref_slow[iw], fb[iw])
        assert numpy.allclose(fb_ref_multi_det[iw], fb[iw])
        # assert numpy.allclose(fb_ref_multi_det[iw], fb_half[iw])


@pytest.mark.unit
def test_phmsd_greens_function_batch():
    nmo = 10
    nelec = (5, 4)
    nwalkers = 1
    ndets = 5
    nsteps = 5
    qmc = dotdict(
        {
            "dt": 0.005,
            "nstblz": 5,
            "nwalkers": nwalkers,
            "batched": False,
            "hybrid": True,
            "num_steps": nsteps,
        }
    )
    legacy_data = build_legacy_test_case_handlers(
        nelec,
        nmo,
        num_dets=ndets,
        options=qmc,
        seed=7,
        trial_type="phmsd",
    )
    qmc.batched = True
    batched_data = build_test_case_handlers(
        nelec,
        nmo,
        num_dets=ndets,
        options=qmc,
        seed=7,
        trial_type="phmsd",
        wfn_type="naive",
    )
    walkers = legacy_data.walker_handler.walkers
    walker_batch = batched_data.walkers
    for iw in range(nwalkers):
        assert numpy.allclose(walker_batch.Gia[iw], walkers[iw].Gi[:, 0, :, :])
        assert numpy.allclose(walker_batch.Gib[iw], walkers[iw].Gi[:, 1, :, :])
        assert numpy.allclose(walker_batch.Ga[iw], walkers[iw].G[0, :, :])
        assert numpy.allclose(walker_batch.Gb[iw], walkers[iw].G[1, :, :])

    batched_data = build_test_case_handlers(
        nelec,
        nmo,
        num_dets=ndets,
        options=qmc,
        seed=7,
        trial_type="phmsd",
        wfn_type="slow",
    )
    walker_batch = batched_data.walkers
    for iw in range(nwalkers):
        assert numpy.allclose(walker_batch.Ga[iw], walkers[iw].G[0, :, :])
        assert numpy.allclose(walker_batch.Gb[iw], walkers[iw].G[1, :, :])


@pytest.mark.unit
def test_phmsd_overlap_batch():
    nmo = 10
    nelec = (5, 4)
    nwalkers = 1
    ndets = 5
    nsteps = 100
    qmc = dotdict(
        {
            "dt": 0.005,
            "nstblz": 5,
            "nwalkers": nwalkers,
            "batched": False,
            "hybrid": True,
            "num_steps": nsteps,
        }
    )
    legacy_data = build_legacy_test_case_handlers(
        nelec,
        nmo,
        num_dets=ndets,
        options=qmc,
        seed=70,
        trial_type="phmsd",
    )
    qmc.batched = True
    batched_data = build_test_case_handlers(
        nelec,
        nmo,
        num_dets=ndets,
        options=qmc,
        seed=70,
        trial_type="phmsd",
        wfn_type="slow",
    )
    batched_data_opt = build_test_case_handlers(
        nelec,
        nmo,
        num_dets=ndets,
        options=qmc,
        seed=70,
        trial_type="phmsd",
        wfn_type="opt",
    )
    walkers = legacy_data.walker_handler.walkers
    walker_batch = batched_data.walkers
    walker_batch_opt = batched_data_opt.walkers

    ovlps0 = [w.calc_overlap(legacy_data.trial) for w in walkers]
    ovlps = calc_overlap_multi_det_wicks(walker_batch, batched_data.trial)
    ovlps_opt = calc_overlap_multi_det_wicks_opt(walker_batch_opt, batched_data_opt.trial)
    for iw in range(nwalkers):
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
        get_dets_single_excitation_batched_opt,
    )

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
        get_dets_double_excitation_batched_opt,
    )

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
        get_dets_triple_excitation_batched_opt,
    )

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
        get_dets_nfold_excitation_batched_opt,
    )

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
    qmc = dotdict(
        {
            "dt": 0.005,
            "nstblz": 5,
            "nwalkers": nwalkers,
            "batched": False,
            "hybrid": True,
            "num_steps": nsteps,
        }
    )
    legacy_data = build_legacy_test_case_handlers(
        nelec,
        nmo,
        num_dets=ndets,
        options=qmc,
        seed=7,
        trial_type="phmsd",
    )
    qmc.batched = True
    batched_data = build_test_case_handlers(
        nelec,
        nmo,
        num_dets=ndets,
        options=qmc,
        seed=7,
        trial_type="phmsd",
        wfn_type="opt",
    )
    legacy_walkers = legacy_data.walker_handler.walkers
    walkers = batched_data.walkers
    for iw in range(nwalkers):
        assert numpy.allclose(legacy_walkers[iw].phi[:, : nelec[0]], walkers.phia[iw])
        assert numpy.allclose(legacy_walkers[iw].phi[:, nelec[0] :], walkers.phib[iw])
        assert numpy.allclose(legacy_walkers[iw].weight, walkers.weight[iw])


if __name__ == "__main__":
    test_phmsd_greens_function_batch()
    test_phmsd_force_bias_batch()
    test_phmsd_overlap_batch()
    test_phmsd_propagation_batch()
