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

import os

import numpy
import pytest

from ipie.estimators.greens_function_multi_det import (
    greens_function_multi_det,
    greens_function_multi_det_wicks,
    greens_function_multi_det_wicks_opt,
)
from ipie.estimators.local_energy_batch import local_energy_multi_det_trial_batch
from ipie.estimators.local_energy_wicks import (
    local_energy_multi_det_trial_wicks_batch,
    local_energy_multi_det_trial_wicks_batch_opt,
    local_energy_multi_det_trial_wicks_batch_opt_chunked,
)
from ipie.propagation.overlap import (
    get_cofactor_matrix_4_batched,
    get_cofactor_matrix_batched,
)
from ipie.hamiltonians.generic import Generic as HamGeneric
from ipie.propagation.phaseless_generic import PhaselessGeneric
from ipie.propagation.overlap import get_det_matrix_batched
from ipie.systems.generic import Generic
from ipie.utils.misc import dotdict
from ipie.utils.testing import generate_hamiltonian, get_random_phmsd
from ipie.walkers.uhf_walkers import UHFWalkersTrial
from ipie.trial_wavefunction.particle_hole import (
    ParticleHoleNaive,
    ParticleHoleWicks,
    ParticleHoleWicksNonChunked,
    ParticleHoleWicksSlow,
    ParticleHoleNaive,
)

@pytest.mark.unit
def test_greens_function_wicks_opt():
    numpy.random.seed(7)
    nmo = 12
    nelec = (7, 7)
    nwalkers = 10
    nsteps = 100
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    system = Generic(nelec=nelec)
    ham = HamGeneric[chol.dtype](
        h1e=numpy.array([h1e, h1e]),
        chol=chol.reshape((-1, nmo * nmo)).T.copy(),
    )
    # Test PH type wavefunction.
    wfn, init = get_random_phmsd(
        system.nup, system.ndown, ham.nbasis, ndet=3000, init=True
    )
    ci, oa, ob = wfn
    wfn_2 = (ci[::50], oa[::50], ob[::50])
    # wfn_2 = (ci[:100], oa[:100], ob[:100])
    trial = ParticleHoleWicksSlow(wfn_2, nelec, nmo, verbose=True)
    trial.build()
    trial_slow = ParticleHoleNaive(
        wfn_2,
        nelec,
        nmo,
    )
    trial_slow.build()
    trial_opt = ParticleHoleWicksNonChunked(
        wfn_2,
        nelec,
        nmo,
    )
    trial_opt.build()
    numpy.random.seed(7)

    walkers_wick = UHFWalkersTrial[type(trial)](init,system.nup,system.ndown,ham.nbasis,nwalkers)
    walkers_wick.build(trial)

    walkers_slow = UHFWalkersTrial[type(trial_slow)](init,system.nup,system.ndown,ham.nbasis,nwalkers)
    walkers_slow.build(trial_slow)
    
    walkers_opt = UHFWalkersTrial[type(trial_opt)](init,system.nup,system.ndown,ham.nbasis,nwalkers)
    walkers_opt.build(trial_opt)

    options = {"hybrid": True}
    qmc = dotdict({"dt": 0.005, "nstblz": 5, "batched": True, "nwalkers": nwalkers})
    numpy.random.seed(7)
    prop = PhaselessGeneric(qmc["dt"])
    prop.build(ham,trial)
    for i in range(nsteps):
        prop.propagate_walkers(walkers_wick, ham, trial, 0)
        walkers_wick.reortho()
    numpy.random.seed(7)
    prop = PhaselessGeneric(qmc["dt"])
    prop.build(ham,trial_opt)
    for i in range(nsteps):
        prop.propagate_walkers(walkers_opt, ham, trial_opt, 0)
        walkers_opt.reortho()
    numpy.random.seed(7)
    walkers_slow.phia = walkers_wick.phia
    walkers_slow.phib = walkers_wick.phib
    nbasis = walkers_wick.Ga.shape[-1]
    from ipie.propagation.overlap import calc_overlap_multi_det_wicks_opt

    ovlps_ref_wick = greens_function_multi_det_wicks(walkers_wick, trial)
    ovlps_ref_slow = greens_function_multi_det(walkers_slow, trial_slow)
    ovlps_ref_opt = greens_function_multi_det_wicks_opt(walkers_opt, trial_opt)
    assert numpy.allclose(ovlps_ref_wick, ovlps_ref_slow)
    assert numpy.allclose(ovlps_ref_opt, ovlps_ref_slow)
    assert numpy.allclose(walkers_wick.Ga, walkers_slow.Ga)
    assert numpy.allclose(walkers_wick.Gb, walkers_slow.Gb)
    assert numpy.allclose(walkers_wick.Ghalfa, walkers_slow.Gihalfa[:, 0])
    assert numpy.allclose(walkers_wick.Ghalfb, walkers_slow.Gihalfb[:, 0])
    assert numpy.allclose(walkers_opt.Ga, walkers_slow.Ga)
    assert numpy.allclose(walkers_opt.Gb, walkers_slow.Gb)
    assert numpy.allclose(walkers_opt.det_ovlpas, walkers_wick.det_ovlpas)
    assert numpy.allclose(walkers_opt.det_ovlpbs, walkers_wick.det_ovlpbs)
    CIa_full = numpy.zeros_like(walkers_wick.CIa)
    CIb_full = numpy.zeros_like(walkers_wick.CIb)
    CIa_full[:, trial_opt.act_orb_alpha, trial_opt.occ_orb_alpha] = walkers_opt.CIa
    CIb_full[:, trial_opt.act_orb_beta, trial_opt.occ_orb_beta] = walkers_opt.CIb
    assert numpy.allclose(CIa_full, walkers_wick.CIa)
    assert numpy.allclose(CIb_full, walkers_wick.CIb)


# Move to propagator tests
@pytest.mark.unit
def test_cofactor_matrix():
    numpy.random.seed(7)
    nexcit = 7
    nwalkers = 10
    ndets_b = 11
    det_mat = numpy.random.random((nwalkers, ndets_b, nexcit, nexcit)).astype(
        numpy.complex128
    )
    cofactor_matrix = numpy.zeros(
        (nwalkers, ndets_b, nexcit - 1, nexcit - 1), dtype=numpy.complex128
    )
    iex = 6
    jex = 6
    from ipie.utils.linalg import minor_mask

    get_cofactor_matrix_batched(
        nwalkers, ndets_b, nexcit, iex, jex, det_mat, cofactor_matrix
    )
    for iw in range(nwalkers):
        for jdet in range(ndets_b):
            assert numpy.allclose(
                cofactor_matrix[iw, jdet], minor_mask(det_mat[iw, jdet], iex, jex)
            )

    from ipie.estimators.kernels.cpu import wicks as wk

    cofactor_matrix = numpy.zeros(
        (nwalkers, ndets_b, nexcit - 1, nexcit - 1), dtype=numpy.complex128
    )
    wk.build_cofactor_matrix(iex, jex, det_mat, cofactor_matrix)
    for iw in range(nwalkers):
        for jdet in range(ndets_b):
            assert numpy.allclose(
                cofactor_matrix[iw, jdet], minor_mask(det_mat[iw, jdet], iex, jex)
            )


# Move to propagator tests
@pytest.mark.unit
def test_cofactor_matrix_4():
    numpy.random.seed(7)
    nexcit = 3
    nwalkers = 1
    ndets_b = 1
    det_mat = numpy.random.random((nwalkers, ndets_b, nexcit, nexcit)).astype(
        numpy.complex128
    )
    cofactor_matrix = numpy.zeros(
        (nwalkers, ndets_b, nexcit - 2, nexcit - 2), dtype=numpy.complex128
    )
    import itertools

    from ipie.utils.linalg import minor_mask4

    for iex, jex, kex, lex in itertools.product(range(nexcit), repeat=4):
        if kex > iex and lex > jex:
            get_cofactor_matrix_4_batched(
                nwalkers, ndets_b, nexcit, iex, jex, kex, lex, det_mat, cofactor_matrix
            )
            for iw in range(nwalkers):
                for jdet in range(ndets_b):
                    ref = minor_mask4(det_mat[iw, jdet], iex, jex, kex, lex)
                    assert numpy.allclose(cofactor_matrix[iw, jdet], ref)
    nexcit = 6
    nwalkers = 4
    ndets_b = 10
    det_mat = numpy.random.random((nwalkers, ndets_b, nexcit, nexcit)).astype(
        numpy.complex128
    )
    cofactor_matrix = numpy.zeros(
        (nwalkers, ndets_b, nexcit - 2, nexcit - 2), dtype=numpy.complex128
    )
    from ipie.estimators.kernels.cpu import wicks as wk

    for iex, jex, kex, lex in itertools.product(range(nexcit), repeat=4):
        if kex > iex and lex > jex:
            get_cofactor_matrix_4_batched(
                nwalkers, ndets_b, nexcit, iex, jex, kex, lex, det_mat, cofactor_matrix
            )
            for iw in range(nwalkers):
                for jdet in range(ndets_b):
                    ref = minor_mask4(det_mat[iw, jdet], iex, jex, kex, lex)
                    assert numpy.allclose(cofactor_matrix[iw, jdet], ref)

    # test numba
    det_mat = numpy.random.random((nwalkers, ndets_b, nexcit, nexcit)).astype(
        numpy.complex128
    )
    cofactor_matrix = numpy.zeros(
        (nwalkers, ndets_b, nexcit - 2, nexcit - 2), dtype=numpy.complex128
    )
    for iex, jex, kex, lex in itertools.product(range(nexcit), repeat=4):
        if kex > iex and lex > jex:
            wk.build_cofactor_matrix_4(iex, jex, kex, lex, det_mat, cofactor_matrix)
            for iw in range(nwalkers):
                for jdet in range(ndets_b):
                    ref = minor_mask4(det_mat[iw, jdet], iex, jex, kex, lex)
                    assert numpy.allclose(cofactor_matrix[iw, jdet], ref)


@pytest.mark.unit
def test_det_matrix():
    numpy.random.seed(7)
    nmo = 12
    nelec = (7, 7)
    nwalkers = 10
    nsteps = 100
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    system = Generic(nelec=nelec)
    ham = HamGeneric[chol.dtype](
        h1e=numpy.array([h1e, h1e]),
        chol=chol.reshape((-1, nmo * nmo)).T.copy(),
        ecore=0,
    )
    # Test PH type wavefunction.
    wfn, init = get_random_phmsd(
        system.nup, system.ndown, ham.nbasis, ndet=3000, init=True
    )
    ci, oa, ob = wfn
    wfn_2 = (ci[::50], oa[::50], ob[::50])  # Get high excitation determinants too
    trial = ParticleHoleWicksNonChunked(
        wfn_2,
        nelec,
        nmo,
    )
    trial.build()
    trial.half_rotate(system, ham)

    numpy.random.seed(7)
    qmc = dotdict({"dt": 0.005, "nstblz": 5, "batched": True, "nwalkers": nwalkers})
    prop = PhaselessGeneric(qmc["dt"])
    prop.build(ham,trial)

    walker_batch = UHFWalkersTrial[type(trial)](init,system.nup,system.ndown,ham.nbasis,nwalkers)
    walker_batch.build(trial)

    numpy.random.seed(7)
    for i in range(nsteps):
        prop.propagate_walkers(walker_batch, ham, trial, 0)
        walker_batch.reortho()

    nexcit = 4

    ndets_b = len(trial.cre_ex_b[nexcit])
    det_mat = numpy.zeros((nwalkers, ndets_b, nexcit, nexcit), dtype=numpy.complex128)
    get_det_matrix_batched(
        nexcit,
        trial.cre_ex_b[nexcit],
        trial.anh_ex_b[nexcit],
        walker_batch.G0b,
        det_mat,
    )
    from ipie.estimators.kernels.cpu import wicks as wk

    det_mat_2 = numpy.zeros((nwalkers, ndets_b, nexcit, nexcit), dtype=numpy.complex128)
    wk.build_det_matrix(
        trial.cre_ex_b[nexcit],
        trial.anh_ex_b[nexcit],
        trial.occ_map_b,
        trial.nfrozen,
        walker_batch.G0b,
        det_mat_2,
    )

    assert numpy.allclose(det_mat, det_mat_2)


@pytest.mark.unit
def test_phmsd_local_energy():
    numpy.random.seed(7)
    nmo = 12
    nelec = (7, 7)
    nwalkers = 10
    nsteps = 100
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    system = Generic(nelec=nelec)
    ham = HamGeneric[chol.dtype](
        h1e=numpy.array([h1e, h1e]),
        chol=chol.reshape((-1, nmo * nmo)).T.copy(),
        ecore=0,
    )
    # Test PH type wavefunction.
    # wfn, init = get_random_phmsd(system.nup, system.ndown, ham.nbasis, ndet=5, init=True)
    wfn, init = get_random_phmsd(
        system.nup, system.ndown, ham.nbasis, ndet=3000, init=True, cmplx=False
    )
    ci, oa, ob = wfn
    wfn_2 = (ci[::50], oa[::50], ob[::50])  # Get high excitation determinants too
    trial = ParticleHoleWicksSlow(
        wfn_2,
        nelec,
        nmo,
    )
    trial.build()
    trial_slow = ParticleHoleNaive(
        wfn_2,
        nelec,
        nmo,
    )
    trial_slow.build()
    trial_slow.half_rotate(system, ham)
    trial_test = ParticleHoleWicksNonChunked(
        wfn_2,
        nelec,
        nmo,
    )
    trial_test.build()
    trial_test.half_rotate(system, ham)
    numpy.random.seed(7)
    walkers_wick = UHFWalkersTrial[type(trial)](init,system.nup,system.ndown,ham.nbasis,nwalkers)
    walkers_wick.build(trial)

    numpy.random.seed(7)
    qmc = dotdict({"dt": 0.005, "nstblz": 5, "batched": True, "nwalkers": nwalkers})
    options = {"hybrid": True}
    
    walker_batch = UHFWalkersTrial[type(trial_slow)](init,system.nup,system.ndown,ham.nbasis,nwalkers)
    walker_batch.build(trial_slow)
    walker_batch_test = UHFWalkersTrial[type(trial_test)](init,system.nup,system.ndown,ham.nbasis,nwalkers)
    walker_batch_test.build(trial_test)
    walker_batch_test2 = UHFWalkersTrial[type(trial)](init,system.nup,system.ndown,ham.nbasis,nwalkers)
    walker_batch_test2.build(trial)

    numpy.random.seed(7)
    prop = PhaselessGeneric(qmc["dt"])
    prop.build(ham,trial_slow)

    for i in range(nsteps):
        prop.propagate_walkers(walker_batch, ham, trial_slow, 0)
        walker_batch.reortho()

    import copy

    walker_batch_test.phia = walker_batch.phia.copy()
    walker_batch_test.phib = walker_batch.phib.copy()
    walker_batch_test2.phia = walker_batch.phia.copy()
    walker_batch_test2.phib = walker_batch.phib.copy()
    walker_batch_test.ovlp = walker_batch.ovlp
    walker_batch_test2.ovlp = walker_batch.ovlp
    greens_function_multi_det(walker_batch, trial_slow)
    greens_function_multi_det_wicks(walker_batch_test2, trial)
    greens_function_multi_det_wicks_opt(walker_batch_test, trial_test)
    assert numpy.allclose(walker_batch_test.Ghalfa, walker_batch.Gihalfa[:, 0])
    assert numpy.allclose(walker_batch_test.Ghalfb, walker_batch.Gihalfb[:, 0])
    assert numpy.allclose(walker_batch_test.Ga, walker_batch.Ga)
    assert numpy.allclose(walker_batch_test.Gb, walker_batch.Gb)
    assert numpy.allclose(walker_batch_test2.Ga, walker_batch.Ga)
    assert numpy.allclose(walker_batch_test2.Gb, walker_batch.Gb)
    assert numpy.allclose(walker_batch_test.det_ovlpas, walker_batch_test2.det_ovlpas)
    assert numpy.allclose(walker_batch_test.det_ovlpbs, walker_batch_test2.det_ovlpbs)
    e_simple = local_energy_multi_det_trial_batch(system, ham, walker_batch, trial)
    e_wicks = local_energy_multi_det_trial_wicks_batch(
        system, ham, walker_batch_test2, trial
    )
    e_wicks_opt = local_energy_multi_det_trial_wicks_batch_opt(
        system, ham, walker_batch_test, trial_test
    )

    assert numpy.allclose(e_simple, e_wicks)
    assert numpy.allclose(e_simple, e_wicks_opt)
    assert numpy.allclose(e_wicks, e_wicks_opt)


@pytest.mark.unit
def test_kernels_energy():
    numpy.random.seed(7)
    nmo = 12
    nelec = (7, 7)
    nwalkers = 10
    nsteps = 100
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    system = Generic(nelec=nelec)
    ham = HamGeneric[chol.dtype](
        h1e=numpy.array([h1e, h1e]),
        chol=chol.reshape((-1, nmo * nmo)).T.copy(),
        ecore=0,
    )
    wfn, init = get_random_phmsd(
        system.nup, system.ndown, ham.nbasis, ndet=5000, init=True
    )
    ci, oa, ob = wfn
    wfn_2 = (ci[::50], oa[::50], ob[::50])  # Get high excitation determinants too
    trial = ParticleHoleWicksNonChunked(
        wfn_2,
        nelec,
        nmo,
    )
    trial.build()

    numpy.random.seed(7)
    qmc = dotdict({"dt": 0.005, "nstblz": 5, "batched": True, "nwalkers": nwalkers})
    prop = PhaselessGeneric(qmc["dt"])
    prop.build(ham,trial)

    walker_batch = UHFWalkersTrial[type(trial)](init,system.nup,system.ndown,ham.nbasis,nwalkers)
    walker_batch.build(trial)
    numpy.random.seed(7)
    for i in range(nsteps):
        prop.propagate_walkers(walker_batch, ham, trial, 0)
        walker_batch.reortho()

    greens_function_multi_det_wicks_opt(walker_batch, trial)
    from ipie.estimators.kernels.cpu import wicks as wk
    from ipie.estimators.local_energy_wicks import (
        fill_opp_spin_factors_batched_doubles_chol,
        fill_opp_spin_factors_batched_singles,
        fill_opp_spin_factors_batched_triples_chol,
        fill_same_spin_contribution_batched_contr,
        get_same_spin_double_contribution_batched_contr,
    )

    ndets = trial.num_dets
    nchol = ham.nchol
    ref = numpy.zeros((nwalkers, ndets, nchol), dtype=numpy.complex128)
    test = numpy.zeros((nwalkers, ndets, nchol), dtype=numpy.complex128)
    slices_alpha, slices_beta = trial.slices_alpha, trial.slices_beta
    nbasis = ham.nbasis
    from ipie.utils.testing import shaped_normal

    Laa = shaped_normal((nwalkers, nbasis, system.nup, nchol))
    Lbb = shaped_normal((nwalkers, nbasis, system.ndown, nchol))
    # 1.
    fill_opp_spin_factors_batched_singles(
        trial.cre_ex_b[1],
        trial.anh_ex_b[1],
        trial.excit_map_b[1],
        Lbb,
        ref,
        slices_beta[1],
    )
    wk.fill_os_singles(
        trial.cre_ex_b[1],
        trial.anh_ex_b[1],
        trial.occ_map_b,
        trial.nfrozen,
        Lbb,
        test,
        slices_beta[1],
    )
    assert numpy.allclose(ref, test)
    # 2.
    G0 = shaped_normal((nwalkers, nbasis, nbasis), cmplx=True)
    ref = numpy.zeros((nwalkers, ndets, nchol), dtype=numpy.complex128)
    test = numpy.zeros((nwalkers, ndets, nchol), dtype=numpy.complex128)
    fill_opp_spin_factors_batched_doubles_chol(
        trial.cre_ex_b[2],
        trial.anh_ex_b[2],
        trial.excit_map_b[2],
        G0,
        Lbb,
        ref,
        slices_beta[2],
    )
    wk.fill_os_doubles(
        trial.cre_ex_b[2],
        trial.anh_ex_b[2],
        trial.occ_map_b,
        trial.nfrozen,
        G0,
        Lbb,
        test,
        slices_beta[2],
    )
    assert numpy.allclose(ref, test)
    ref = numpy.zeros((nwalkers, ndets, nchol), dtype=numpy.complex128)
    test = numpy.zeros((nwalkers, ndets, nchol), dtype=numpy.complex128)
    iexcit = 3
    fill_opp_spin_factors_batched_triples_chol(
        trial.cre_ex_b[iexcit],
        trial.anh_ex_b[iexcit],
        trial.excit_map_b[iexcit],
        G0,
        Lbb,
        ref,
        slices_beta[iexcit],
    )
    wk.fill_os_triples(
        trial.cre_ex_b[iexcit],
        trial.anh_ex_b[iexcit],
        trial.occ_map_b,
        trial.nfrozen,
        G0,
        Lbb,
        test,
        slices_beta[iexcit],
    )
    assert numpy.allclose(ref, test)
    ref = numpy.zeros((nwalkers, ndets), dtype=numpy.complex128)
    test = numpy.zeros((nwalkers, ndets), dtype=numpy.complex128)
    iexcit = 2
    get_same_spin_double_contribution_batched_contr(
        trial.cre_ex_b[iexcit],
        trial.anh_ex_b[iexcit],
        ref,
        trial.excit_map_b[iexcit],
        Lbb,
        slices_beta[iexcit],
    )
    wk.get_ss_doubles(
        trial.cre_ex_b[iexcit],
        trial.anh_ex_b[iexcit],
        trial.occ_map_b,
        Lbb,
        test,
        slices_beta[iexcit],
    )
    assert numpy.allclose(ref, test)
    iexcit = 4
    ndets_level = len(trial.cre_ex_b[iexcit])
    det_mat = numpy.zeros(
        (nwalkers, ndets_level, iexcit, iexcit), dtype=numpy.complex128
    )
    cof_mat = numpy.zeros(
        (nwalkers, ndets_level, iexcit - 1, iexcit - 1), dtype=numpy.complex128
    )
    wk.build_det_matrix(
        trial.cre_ex_b[iexcit],
        trial.anh_ex_b[iexcit],
        trial.occ_map_b,
        trial.nfrozen,
        walker_batch.G0b,
        det_mat,
    )
    ref = numpy.zeros((nwalkers, ndets, nchol), dtype=numpy.complex128)
    test = numpy.zeros((nwalkers, ndets, nchol), dtype=numpy.complex128)
    from ipie.estimators.local_energy_wicks import fill_opp_spin_factors_batched_chol

    fill_opp_spin_factors_batched_chol(
        iexcit,
        trial.cre_ex_b[iexcit],
        trial.anh_ex_b[iexcit],
        trial.excit_map_b[iexcit],
        det_mat,
        cof_mat,
        Lbb,
        ref,
        slices_beta[iexcit],
    )
    wk.fill_os_nfold(
        trial.cre_ex_b[iexcit],
        trial.anh_ex_b[iexcit],
        trial.occ_map_b,
        det_mat,
        cof_mat,
        Lbb,
        test,
        slices_beta[iexcit],
    )
    assert numpy.allclose(ref, test)
    det_mat = numpy.zeros(
        (nwalkers, ndets_level, iexcit, iexcit), dtype=numpy.complex128
    )
    cof_mat = numpy.zeros(
        (nwalkers, ndets_level, iexcit - 2, iexcit - 2), dtype=numpy.complex128
    )
    get_det_matrix_batched(
        iexcit,
        trial.cre_ex_b[iexcit],
        trial.anh_ex_b[iexcit],
        walker_batch.G0b,
        det_mat,
    )
    ref = numpy.zeros((nwalkers, ndets), dtype=numpy.complex128)
    test = numpy.zeros((nwalkers, ndets), dtype=numpy.complex128)
    from ipie.estimators.local_energy_wicks import (
        fill_same_spin_contribution_batched_contr,
    )

    fill_same_spin_contribution_batched_contr(
        iexcit,
        trial.cre_ex_b[iexcit],
        trial.anh_ex_b[iexcit],
        det_mat,
        cof_mat,
        Lbb,
        ref,
        trial.excit_map_b[iexcit],
        slices_beta[iexcit],
    )
    wk.get_ss_nfold(
        trial.cre_ex_b[iexcit],
        trial.anh_ex_b[iexcit],
        trial.occ_map_b,
        det_mat,
        cof_mat,
        Lbb,
        test,
        slices_beta[iexcit],
    )
    assert numpy.allclose(ref, test)


@pytest.mark.unit
def test_kernels_gf():
    numpy.random.seed(7)
    nmo = 12
    nelec = (7, 7)
    nwalkers = 10
    nsteps = 100
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    system = Generic(nelec=nelec)
    ham = HamGeneric[chol.dtype](
        h1e=numpy.array([h1e, h1e]),
        chol=chol.reshape((-1, nmo * nmo)).T.copy(),
        ecore=0,
        # options={"symmetry": False},
    )
    wfn, init = get_random_phmsd(
        system.nup, system.ndown, ham.nbasis, ndet=5000, init=True
    )
    ci, oa, ob = wfn
    wfn_2 = (ci[::50], oa[::50], ob[::50])  # Get high excitation determinants too
    trial = ParticleHoleWicksNonChunked(
        wfn_2,
        nelec,
        nmo,
    )
    trial.build()

    numpy.random.seed(7)
    qmc = dotdict({"dt": 0.005, "nstblz": 5, "batched": True, "nwalkers": nwalkers})
    prop = PhaselessGeneric(qmc["dt"])
    prop.build(ham,trial)

    walker_batch = UHFWalkersTrial[type(trial)](init,system.nup,system.ndown,ham.nbasis,nwalkers)
    walker_batch.build(trial)

    numpy.random.seed(7)
    for i in range(nsteps):
        prop.propagate_walkers(walker_batch, ham, trial, 0)
        walker_batch.reortho()

    trial.calc_greens_function(walker_batch)

    from ipie.estimators.kernels.cpu import wicks as wk

    ndets = trial.num_dets
    nchol = ham.nchol
    ref = numpy.zeros((nwalkers, ndets, nchol), dtype=numpy.complex128)
    test = numpy.zeros((nwalkers, ndets, nchol), dtype=numpy.complex128)
    ovlpa = walker_batch.det_ovlpas
    ovlpb = walker_batch.det_ovlpbs

    c_phasea_ovlpb = numpy.einsum(
        "wJ,J->wJ", ovlpb, trial.phase_a * trial.coeffs.conj(), optimize=True
    )
    c_phaseb_ovlpa = numpy.einsum(
        "wJ,J->wJ", ovlpa, trial.phase_b * trial.coeffs.conj(), optimize=True
    )
    walker_batch.CIa.fill(0.0 + 0.0j)
    walker_batch.CIb.fill(0.0 + 0.0j)
    from ipie.estimators.greens_function_multi_det import (
        build_CI_single_excitation,
        build_CI_single_excitation_opt,
    )

    build_CI_single_excitation(walker_batch, trial, c_phasea_ovlpb, c_phaseb_ovlpa)
    refa = walker_batch.CIa.copy()
    refb = walker_batch.CIb.copy()
    walker_batch.CIa.fill(0.0 + 0.0j)
    walker_batch.CIb.fill(0.0 + 0.0j)
    build_CI_single_excitation_opt(walker_batch, trial, c_phasea_ovlpb, c_phaseb_ovlpa)
    assert numpy.allclose(refa, walker_batch.CIa)
    assert numpy.allclose(refb, walker_batch.CIb)
    from ipie.estimators.greens_function_multi_det import (
        build_CI_double_excitation,
        build_CI_double_excitation_opt,
    )

    walker_batch.CIa.fill(0.0 + 0.0j)
    walker_batch.CIb.fill(0.0 + 0.0j)
    build_CI_double_excitation(walker_batch, trial, c_phasea_ovlpb, c_phaseb_ovlpa)
    refa = walker_batch.CIa.copy()
    refb = walker_batch.CIb.copy()
    walker_batch.CIa.fill(0.0 + 0.0j)
    walker_batch.CIb.fill(0.0 + 0.0j)
    build_CI_double_excitation_opt(walker_batch, trial, c_phasea_ovlpb, c_phaseb_ovlpa)
    assert numpy.allclose(refa, walker_batch.CIa)
    assert numpy.allclose(refb, walker_batch.CIb)
    from ipie.estimators.greens_function_multi_det import (
        build_CI_triple_excitation,
        build_CI_triple_excitation_opt,
    )

    walker_batch.CIa.fill(0.0 + 0.0j)
    walker_batch.CIb.fill(0.0 + 0.0j)
    build_CI_triple_excitation(walker_batch, trial, c_phasea_ovlpb, c_phaseb_ovlpa)
    refa = walker_batch.CIa.copy()
    refb = walker_batch.CIb.copy()
    walker_batch.CIa.fill(0.0 + 0.0j)
    walker_batch.CIb.fill(0.0 + 0.0j)
    build_CI_triple_excitation_opt(walker_batch, trial, c_phasea_ovlpb, c_phaseb_ovlpa)
    assert numpy.allclose(refa, walker_batch.CIa)
    assert numpy.allclose(refb, walker_batch.CIb)
    from ipie.estimators.greens_function_multi_det import (
        build_CI_nfold_excitation,
        build_CI_nfold_excitation_opt,
    )

    walker_batch.CIa.fill(0.0 + 0.0j)
    walker_batch.CIb.fill(0.0 + 0.0j)
    build_CI_nfold_excitation(4, walker_batch, trial, c_phasea_ovlpb, c_phaseb_ovlpa)
    refa = walker_batch.CIa.copy()
    refb = walker_batch.CIb.copy()
    walker_batch.CIa.fill(0.0 + 0.0j)
    walker_batch.CIb.fill(0.0 + 0.0j)
    build_CI_nfold_excitation_opt(
        4, walker_batch, trial, c_phasea_ovlpb, c_phaseb_ovlpa
    )
    assert numpy.allclose(refa, walker_batch.CIa)
    assert numpy.allclose(refb, walker_batch.CIb)


@pytest.mark.unit
def test_kernels_gf_active_space():
    numpy.random.seed(7)
    nmo = 30
    nelec = (9, 9)
    nwalkers = 1
    nsteps = 100
    nact = 12
    ncore = 2
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    system = Generic(nelec=nelec)
    ham = HamGeneric[chol.dtype](
        h1e=numpy.array([h1e, h1e]),
        chol=chol.reshape((-1, nmo * nmo)).T.copy(),
        ecore=0,
        # options={"symmetry": False},
    )
    wfn, init_act = get_random_phmsd(7, 7, nact, ndet=5000, init=True)
    ci, occa, occb = wfn
    core = [0, 1]
    with_core_a = [
        numpy.array(core + [orb + 2 for orb in oa], dtype=numpy.int32) for oa in occa
    ]
    with_core_b = [
        numpy.array(core + [orb + 2 for orb in ob], dtype=numpy.int32) for ob in occb
    ]
    wfn_no_act = (
        ci[::50],
        with_core_a[::50],
        with_core_b[::50],
    )  # Get high excitation determinants too
    wfn_act = (
        ci[::50],
        occa[::50],
        occb[::50],
    )  # Get high excitation determinants too

    # Big hack due to new wavefunction separation, old unit test assumed
    # MultiSlater wavefunction built all the data structures but now they are
    # separate so need to copy cre_ex_a arrays over to old unoptimized wicks class.
    trial_ref = ParticleHoleWicksSlow(wfn_no_act, nelec, nmo)
    trial_ref.optimized = False
    trial_ref.build()
    trial_tmp = ParticleHoleWicksNonChunked(
        wfn_no_act, nelec, nmo, use_active_space=False
    )
    trial_tmp.build()
    # need cre_ex_a arrays for non-active space variant
    trial_ref.__dict__.update(trial_tmp.__dict__)
    trial_ref.optimized = False

    # Trial with active space optimization
    trial_test = ParticleHoleWicksNonChunked(
        wfn_act,
        nelec,
        nmo,
    )
    trial_test.build()
    # Copy some data over

    numpy.random.seed(7)
    qmc = dotdict({"dt": 0.005, "nstblz": 5, "batched": True, "nwalkers": nwalkers})

    prop = PhaselessGeneric(qmc["dt"])
    prop.build(ham,trial_ref)

    I = numpy.eye(nmo)
    init = numpy.hstack([I[:, : nelec[0]], I[:, : nelec[1]]])

    walker_batch_ref = UHFWalkersTrial[type(trial_ref)](init,system.nup,system.ndown,ham.nbasis,nwalkers)
    walker_batch_ref.build(trial_ref)

    walker_batch_test = UHFWalkersTrial[type(trial_test)](init,system.nup,system.ndown,ham.nbasis,nwalkers)
    walker_batch_test.build(trial_test)

    numpy.random.seed(7)
    for i in range(nsteps):
        prop.propagate_walkers(walker_batch_ref, ham, trial_ref, 0)
        walker_batch_ref.reortho()

    walker_batch_test.phia = walker_batch_ref.phia.copy()
    walker_batch_test.phib = walker_batch_ref.phib.copy()
    trial_ref.calc_greens_function(walker_batch_ref)
    walker_batch_test.G0a = walker_batch_ref.G0a.copy()
    walker_batch_test.G0b = walker_batch_ref.G0b.copy()
    walker_batch_test.Ga = walker_batch_ref.Ga.copy()
    walker_batch_test.Gb = walker_batch_ref.Gb.copy()
    walker_batch_test.Ghalfa = walker_batch_ref.Ghalfa.copy()
    walker_batch_test.Ghalfb = walker_batch_ref.Ghalfb.copy()

    from ipie.estimators.kernels.cpu import wicks as wk

    ndets = trial_ref.num_dets
    nchol = ham.nchol
    ref = numpy.zeros((nwalkers, ndets, nchol), dtype=numpy.complex128)
    test = numpy.zeros((nwalkers, ndets, nchol), dtype=numpy.complex128)
    ovlpa = walker_batch_ref.det_ovlpas
    ovlpb = walker_batch_ref.det_ovlpbs

    c_phasea_ovlpb = numpy.einsum(
        "wJ,J->wJ", ovlpb, trial_ref.phase_a * trial_ref.coeffs.conj(), optimize=True
    )
    c_phaseb_ovlpa = numpy.einsum(
        "wJ,J->wJ", ovlpa, trial_ref.phase_b * trial_ref.coeffs.conj(), optimize=True
    )
    walker_batch_ref.CIa.fill(0.0 + 0.0j)
    walker_batch_ref.CIb.fill(0.0 + 0.0j)
    from ipie.estimators.greens_function_multi_det import (
        build_CI_double_excitation_opt,
        build_CI_nfold_excitation_opt,
        build_CI_single_excitation_opt,
        build_CI_triple_excitation_opt,
    )

    methods = [
        build_CI_single_excitation_opt,
        build_CI_double_excitation_opt,
        build_CI_triple_excitation_opt,
    ]
    for method in methods:
        walker_batch_ref.CIa.fill(0.0 + 0.0j)
        walker_batch_ref.CIb.fill(0.0 + 0.0j)
        method(walker_batch_ref, trial_ref, c_phasea_ovlpb, c_phaseb_ovlpa)
        refa = walker_batch_ref.CIa.copy()
        refb = walker_batch_ref.CIb.copy()
        walker_batch_test.CIa.fill(0.0 + 0.0j)
        walker_batch_test.CIb.fill(0.0 + 0.0j)
        method(walker_batch_test, trial_test, c_phasea_ovlpb, c_phaseb_ovlpa)
        CIa_full = numpy.zeros_like(refa)
        CIb_full = numpy.zeros_like(refb)
        CIa_full[
            :, trial_test.act_orb_alpha, trial_test.occ_orb_alpha
        ] = walker_batch_test.CIa
        CIb_full[
            :, trial_test.act_orb_beta, trial_test.occ_orb_beta
        ] = walker_batch_test.CIb
        assert numpy.allclose(refa, CIa_full)
        assert numpy.allclose(refb, CIb_full)
    walker_batch_test.Q0a = numpy.eye(nmo)[None, :] - walker_batch_test.G0a.copy()
    walker_batch_test.Q0b = numpy.eye(nmo)[None, :] - walker_batch_test.G0b.copy()
    from ipie.estimators.greens_function_multi_det import contract_CI

    act_orb = trial_ref.act_orb_alpha
    occ_orb = trial_ref.occ_orb_alpha
    contract_CI(
        walker_batch_ref.Q0a[:, :, act_orb].copy(),
        walker_batch_ref.CIa,
        walker_batch_ref.G0a,
        walker_batch_ref.Ga,
    )
    act_orb = trial_test.act_orb_alpha
    occ_orb = trial_test.occ_orb_alpha
    contract_CI(
        walker_batch_test.Q0a[:, :, act_orb].copy(),
        walker_batch_test.CIa,
        walker_batch_test.Ghalfa[:, act_orb].copy(),
        walker_batch_test.Ga,
    )
    assert numpy.allclose(walker_batch_ref.Ga, walker_batch_test.Ga)
    assert numpy.allclose(walker_batch_ref.Ga, walker_batch_test.Ga)
    for nexcit in range(4, trial_ref.max_excite + 1):
        walker_batch_ref.CIa.fill(0.0 + 0.0j)
        walker_batch_ref.CIb.fill(0.0 + 0.0j)
        build_CI_nfold_excitation_opt(
            nexcit, walker_batch_ref, trial_ref, c_phasea_ovlpb, c_phaseb_ovlpa
        )
        refa = walker_batch_ref.CIa.copy()
        refb = walker_batch_ref.CIb.copy()
        walker_batch_test.CIa.fill(0.0 + 0.0j)
        walker_batch_test.CIb.fill(0.0 + 0.0j)
        build_CI_nfold_excitation_opt(
            nexcit, walker_batch_test, trial_test, c_phasea_ovlpb, c_phaseb_ovlpa
        )
        CIa_full = numpy.zeros_like(refa)
        CIb_full = numpy.zeros_like(refb)
        CIa_full[
            :, trial_test.act_orb_alpha, trial_test.occ_orb_alpha
        ] = walker_batch_test.CIa
        CIb_full[
            :, trial_test.act_orb_beta, trial_test.occ_orb_beta
        ] = walker_batch_test.CIb
        assert numpy.allclose(refa, CIa_full)
        assert numpy.allclose(refb, CIb_full)
    contract_CI(
        walker_batch_test.Q0b[:, :, act_orb].copy(),
        walker_batch_test.CIb,
        walker_batch_test.Ghalfb[:, act_orb].copy(),
        walker_batch_test.Gb,
    )
    contract_CI(
        walker_batch_ref.Q0b,
        walker_batch_ref.CIb,
        walker_batch_ref.G0b,
        walker_batch_ref.Gb,
    )
    assert numpy.allclose(walker_batch_ref.Ga, walker_batch_test.Ga)


@pytest.mark.unit
def test_kernels_energy_active_space():
    numpy.random.seed(7)
    nmo = 30
    nelec = (9, 9)
    nwalkers = 1
    nsteps = 100
    nact = 12
    ncore = 2
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    system = Generic(nelec=nelec)
    ham = HamGeneric[chol.dtype](
        h1e=numpy.array([h1e, h1e]),
        chol=chol.reshape((-1, nmo * nmo)).T.copy(),
        ecore=0,
        # options={"symmetry": False},
    )
    wfn, init = get_random_phmsd(7, 7, nact, ndet=5000, init=True)
    ci, occa, occb = wfn
    core = [0, 1]
    with_core_a = [
        numpy.array(core + [orb + 2 for orb in oa], dtype=numpy.int32) for oa in occa
    ]
    with_core_b = [
        numpy.array(core + [orb + 2 for orb in ob], dtype=numpy.int32) for ob in occb
    ]
    wfn_2 = (
        ci[::50],
        occa[::50],
        occb[::50],
    )  # Get high excitation determinants too
    wfn_no_act = (
        ci[::50],
        with_core_a[::50],
        with_core_b[::50],
    )  # Get high excitation determinants too
    wfn_act = (
        ci[::50],
        occa[::50],
        occb[::50],
    )  # Get high excitation determinants too

    # Big hack due to new wavefunction separation, old unit test assumed
    # MultiSlater wavefunction built all the data structures but now they are
    # separate so need to copy cre_ex_a arrays over to old unoptimized wicks class.
    trial_ref = ParticleHoleWicksSlow(wfn_no_act, nelec, nmo)
    trial_ref.optimized = False
    trial_ref.build()
    trial_tmp = ParticleHoleWicksNonChunked(
        wfn_no_act, nelec, nmo, use_active_space=False
    )
    trial_tmp.build()
    # need cre_ex_a arrays for non-active space variant
    trial_ref.__dict__.update(trial_tmp.__dict__)
    trial_ref.optimized = False

    # Trial with active space optimization
    trial_test = ParticleHoleWicksNonChunked(
        wfn_act,
        nelec,
        nmo,
    )
    trial_test.build()

    I = numpy.eye(nmo)
    init = numpy.hstack([I[:, : nelec[0]], I[:, : nelec[1]]])

    numpy.random.seed(7)
    qmc = dotdict({"dt": 0.005, "nstblz": 5, "batched": True, "nwalkers": nwalkers})

    prop = PhaselessGeneric(qmc["dt"])
    prop.build(ham,trial_ref)

    walker_batch_ref = UHFWalkersTrial[type(trial_ref)](init,system.nup,system.ndown,ham.nbasis,nwalkers)
    walker_batch_ref.build(trial_ref)

    walker_batch_test = UHFWalkersTrial[type(trial_test)](init,system.nup,system.ndown,ham.nbasis,nwalkers)
    walker_batch_test.build(trial_test)

    numpy.random.seed(7)
    for i in range(nsteps):
        prop.propagate_walkers(walker_batch_ref, ham, trial_ref, 0)
        walker_batch_ref.reortho()

    walker_batch_test.phia = walker_batch_ref.phia.copy()
    walker_batch_test.phib = walker_batch_ref.phib.copy()
    greens_function_multi_det_wicks(walker_batch_ref, trial_ref)
    walker_batch_test.G0a = walker_batch_ref.G0a.copy()
    walker_batch_test.G0b = walker_batch_ref.G0b.copy()
    walker_batch_test.Ga = walker_batch_ref.Ga.copy()
    walker_batch_test.Gb = walker_batch_ref.Gb.copy()
    walker_batch_test.Ghalfa = walker_batch_ref.Ghalfa.copy()
    walker_batch_test.Ghalfb = walker_batch_ref.Ghalfb.copy()

    from ipie.estimators.kernels.cpu import wicks as wk

    ndets = trial_ref.num_dets
    nchol = ham.nchol
    ref = numpy.zeros((nwalkers, ndets, nchol), dtype=numpy.complex128)
    test = numpy.zeros((nwalkers, ndets, nchol), dtype=numpy.complex128)
    ovlpa = walker_batch_ref.det_ovlpas
    ovlpb = walker_batch_ref.det_ovlpbs

    c_phasea_ovlpb = numpy.einsum(
        "wJ,J->wJ", ovlpb, trial_ref.phase_a * trial_ref.coeffs.conj(), optimize=True
    )
    c_phaseb_ovlpa = numpy.einsum(
        "wJ,J->wJ", ovlpa, trial_ref.phase_b * trial_ref.coeffs.conj(), optimize=True
    )
    walker_batch_ref.CIa.fill(0.0 + 0.0j)
    walker_batch_ref.CIb.fill(0.0 + 0.0j)
    from ipie.utils.testing import shaped_normal

    Lbb = shaped_normal((nwalkers, nmo, system.ndown, nchol))
    slices_alpha, slices_beta = trial_test.slices_alpha, trial_test.slices_beta
    assert trial_ref.nfrozen != trial_test.nfrozen
    # 1.
    wk.fill_os_singles(
        trial_ref.cre_ex_b[1],
        trial_ref.anh_ex_b[1],
        trial_ref.occ_map_b,
        trial_ref.nfrozen,
        Lbb,
        ref,
        slices_beta[1],
    )
    act_orb = trial_test.act_orb_beta
    occ_orb = trial_test.occ_orb_beta
    wk.fill_os_singles(
        trial_test.cre_ex_b[1],
        trial_test.anh_ex_b[1],
        trial_test.occ_map_b,
        trial_test.nfrozen,
        Lbb[:, act_orb, occ_orb, :].copy(),
        test,
        slices_beta[1],
    )
    assert numpy.allclose(ref, test)
    # 2.
    G0 = shaped_normal((nwalkers, nmo, nmo), cmplx=True)
    ref = numpy.zeros((nwalkers, ndets, nchol), dtype=numpy.complex128)
    test = numpy.zeros((nwalkers, ndets, nchol), dtype=numpy.complex128)
    wk.fill_os_doubles(
        trial_ref.cre_ex_b[2],
        trial_ref.anh_ex_b[2],
        trial_ref.occ_map_b,
        trial_ref.nfrozen,
        G0,
        Lbb,
        ref,
        slices_beta[2],
    )
    wk.fill_os_doubles(
        trial_test.cre_ex_b[2],
        trial_test.anh_ex_b[2],
        trial_test.occ_map_b,
        trial_test.nfrozen,
        G0,
        Lbb[:, act_orb, occ_orb, :].copy(),
        test,
        slices_beta[2],
    )
    assert numpy.allclose(ref, test)
    ref = numpy.zeros((nwalkers, ndets, nchol), dtype=numpy.complex128)
    test = numpy.zeros((nwalkers, ndets, nchol), dtype=numpy.complex128)
    iexcit = 3
    wk.fill_os_triples(
        trial_ref.cre_ex_b[iexcit],
        trial_ref.anh_ex_b[iexcit],
        trial_ref.occ_map_b,
        trial_ref.nfrozen,
        G0,
        Lbb,
        ref,
        slices_beta[iexcit],
    )
    wk.fill_os_triples(
        trial_test.cre_ex_b[iexcit],
        trial_test.anh_ex_b[iexcit],
        trial_test.occ_map_b,
        trial_test.nfrozen,
        G0,
        Lbb[:, act_orb, occ_orb, :].copy(),
        test,
        slices_beta[iexcit],
    )
    assert numpy.allclose(ref, test)
    ref = numpy.zeros((nwalkers, ndets), dtype=numpy.complex128)
    test = numpy.zeros((nwalkers, ndets), dtype=numpy.complex128)
    iexcit = 2
    wk.get_ss_doubles(
        trial_ref.cre_ex_b[iexcit],
        trial_ref.anh_ex_b[iexcit],
        trial_ref.occ_map_b,
        Lbb,
        ref,
        slices_beta[iexcit],
    )
    wk.get_ss_doubles(
        trial_test.cre_ex_b[iexcit],
        trial_test.anh_ex_b[iexcit],
        trial_test.occ_map_b,
        Lbb[:, act_orb, occ_orb, :].copy(),
        test,
        slices_beta[iexcit],
    )
    assert numpy.allclose(ref, test)
    iexcit = 4
    ndets_level = len(trial_ref.cre_ex_b[iexcit])
    det_mat_ref = numpy.zeros(
        (nwalkers, ndets_level, iexcit, iexcit), dtype=numpy.complex128
    )
    det_mat_test = numpy.zeros(
        (nwalkers, ndets_level, iexcit, iexcit), dtype=numpy.complex128
    )
    cof_mat = numpy.zeros(
        (nwalkers, ndets_level, iexcit - 1, iexcit - 1), dtype=numpy.complex128
    )
    wk.build_det_matrix(
        trial_ref.cre_ex_b[iexcit],
        trial_ref.anh_ex_b[iexcit],
        trial_ref.occ_map_b,
        trial_ref.nfrozen,
        walker_batch_ref.G0b,
        det_mat_ref,
    )
    wk.build_det_matrix(
        trial_test.cre_ex_b[iexcit],
        trial_test.anh_ex_b[iexcit],
        trial_test.occ_map_b,
        trial_test.nfrozen,
        walker_batch_test.G0b,
        det_mat_test,
    )
    assert numpy.allclose(det_mat_test, det_mat_ref)
    ref = numpy.zeros((nwalkers, ndets, nchol), dtype=numpy.complex128)
    test = numpy.zeros((nwalkers, ndets, nchol), dtype=numpy.complex128)
    from ipie.estimators.local_energy_wicks import fill_opp_spin_factors_batched_chol

    cof_mat = numpy.zeros(
        (nwalkers, ndets_level, iexcit - 1, iexcit - 1), dtype=numpy.complex128
    )
    wk.fill_os_nfold(
        trial_ref.cre_ex_b[iexcit],
        trial_ref.anh_ex_b[iexcit],
        trial_ref.occ_map_b,
        det_mat_ref,
        cof_mat,
        Lbb,
        ref,
        slices_beta[iexcit],
    )
    cof_mat = numpy.zeros(
        (nwalkers, ndets_level, iexcit - 1, iexcit - 1), dtype=numpy.complex128
    )
    wk.fill_os_nfold(
        trial_test.cre_ex_b[iexcit],
        trial_test.anh_ex_b[iexcit],
        trial_test.occ_map_b,
        det_mat_test,
        cof_mat,
        Lbb[:, act_orb, occ_orb, :].copy(),
        test,
        slices_beta[iexcit],
    )
    assert numpy.allclose(ref, test)
    ref = numpy.zeros((nwalkers, ndets), dtype=numpy.complex128)
    test = numpy.zeros((nwalkers, ndets), dtype=numpy.complex128)
    from ipie.estimators.local_energy_wicks import (
        fill_same_spin_contribution_batched_contr,
    )

    cof_mat = numpy.zeros(
        (nwalkers, ndets_level, iexcit - 1, iexcit - 1), dtype=numpy.complex128
    )
    wk.get_ss_nfold(
        trial_ref.cre_ex_b[iexcit],
        trial_ref.anh_ex_b[iexcit],
        trial_ref.occ_map_b,
        det_mat_ref,
        cof_mat,
        Lbb,
        ref,
        slices_beta[iexcit],
    )
    cof_mat = numpy.zeros(
        (nwalkers, ndets_level, iexcit - 1, iexcit - 1), dtype=numpy.complex128
    )
    wk.get_ss_nfold(
        trial_test.cre_ex_b[iexcit],
        trial_test.anh_ex_b[iexcit],
        trial_test.occ_map_b,
        det_mat_test,
        cof_mat,
        Lbb[:, act_orb, occ_orb, :].copy(),
        test,
        slices_beta[iexcit],
    )
    assert numpy.allclose(ref, test)


@pytest.mark.unit
def test_phmsd_local_energy_active_space():
    numpy.random.seed(7)
    nmo = 30
    nelec = (9, 9)
    nwalkers = 1
    nsteps = 100
    nact = 12
    ncore = 2
    core = [0, 1]
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    system = Generic(nelec=nelec)
    ham = HamGeneric[chol.dtype](
        h1e=numpy.array([h1e, h1e]),
        chol=chol.reshape((-1, nmo * nmo)).T.copy(),
        ecore=0,
        # options={"symmetry": False},
    )
    wfn, init = get_random_phmsd(7, 7, nact, ndet=5000, init=True)
    ci, occa, occb = wfn
    wfn_2 = (
        ci[::50],
        occa[::50],
        occb[::50],
    )  # Get high excitation determinants too
    ci, occa, occb = wfn_2
    with_core_a = numpy.array(
        [numpy.array(core + [orb + 2 for orb in oa], dtype=numpy.int32) for oa in occa]
    )
    with_core_b = numpy.array(
        [numpy.array(core + [orb + 2 for orb in ob], dtype=numpy.int32) for ob in occb]
    )

    trial_ref = ParticleHoleWicksSlow(
        (ci, with_core_a, with_core_b),
        nelec,
        nmo,
    )
    trial_ref.build()
    trial_ref.half_rotate(system, ham)

    trial_test = ParticleHoleWicksNonChunked(
        wfn_2,
        nelec,
        nmo,
    )
    trial_test.build()
    trial_test.half_rotate(system, ham)
    I = numpy.eye(nmo)
    init = numpy.hstack([I[:, : nelec[0]], I[:, : nelec[1]]])

    numpy.random.seed(7)
    qmc = dotdict({"dt": 0.005, "nstblz": 5, "batched": True, "nwalkers": nwalkers})
    prop = PhaselessGeneric(qmc["dt"])
    prop.build(ham,trial_ref)

    walker_batch_ref = UHFWalkersTrial[type(trial_ref)](init,system.nup,system.ndown,ham.nbasis,nwalkers)
    walker_batch_ref.build(trial_ref)

    walker_batch_test = UHFWalkersTrial[type(trial_test)](init,system.nup,system.ndown,ham.nbasis,nwalkers)
    walker_batch_test.build(trial_test)

    numpy.random.seed(7)
    for i in range(nsteps):
        prop.propagate_walkers(walker_batch_ref, ham, trial_ref, 0)
        walker_batch_ref.reortho()

    import copy

    walker_batch_test.phia = walker_batch_ref.phia.copy()
    walker_batch_test.phib = walker_batch_ref.phib.copy()
    walker_batch_test.ovlp = walker_batch_ref.ovlp
    greens_function_multi_det_wicks(walker_batch_ref, trial_ref)
    greens_function_multi_det_wicks_opt(walker_batch_test, trial_test)
    assert numpy.allclose(walker_batch_ref.Ga, walker_batch_test.Ga)
    assert numpy.allclose(walker_batch_test.Ghalfa, walker_batch_ref.Ghalfa)
    assert numpy.allclose(walker_batch_test.Ghalfb, walker_batch_ref.Ghalfb)
    assert numpy.allclose(walker_batch_test.Ga, walker_batch_ref.Ga)
    assert numpy.allclose(walker_batch_test.Gb, walker_batch_ref.Gb)
    assert numpy.allclose(walker_batch_test.det_ovlpas, walker_batch_ref.det_ovlpas)
    assert numpy.allclose(walker_batch_test.det_ovlpbs, walker_batch_ref.det_ovlpbs)
    # assert numpy.allclose(walker_batch_test.CIa, walker_batch_ref.CIa)
    # assert numpy.allclose(walker_batch_test.CIb, walker_batch_ref.CIb)
    assert trial_ref.nfrozen != trial_test.nfrozen
    e_wicks_opt = local_energy_multi_det_trial_wicks_batch(
        system, ham, walker_batch_ref, trial_ref
    )
    e_wicks_opt_act = local_energy_multi_det_trial_wicks_batch_opt(
        system, ham, walker_batch_test, trial_test
    )

    assert numpy.allclose(e_wicks_opt, e_wicks_opt_act)


@pytest.mark.unit
def test_phmsd_local_energy_active_space_polarised():
    numpy.random.seed(7)
    nelec = (9, 7)
    nwalkers = 1
    nsteps = 10
    nact = 12
    nmo = 20
    ncore = 2
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    system = Generic(nelec=nelec)
    ham = HamGeneric[chol.dtype](
        h1e=numpy.array([h1e, h1e]),
        chol=chol.reshape((-1, nmo * nmo)).T.copy(),
        ecore=0,
        # options={"symmetry": False},
    )
    from ipie.utils.testing import get_random_phmsd_opt, shaped_normal

    wfn, init = get_random_phmsd_opt(7, 5, nact, ndet=100, init=True)
    init = shaped_normal((nmo, system.ne))
    ci, occa, occb = wfn
    core = [0, 1]
    with_core_a = numpy.array(
        [numpy.array(core + [orb + 2 for orb in oa], dtype=numpy.int32) for oa in occa]
    )
    with_core_b = numpy.array(
        [numpy.array(core + [orb + 2 for orb in ob], dtype=numpy.int32) for ob in occb]
    )
    trial = ParticleHoleNaive(
        (ci, with_core_a, with_core_b),
        # wfn,
        nelec,
        nmo,
    )
    trial.build()
    trial.half_rotate(system, ham)
    trial_test = ParticleHoleWicksNonChunked(
        wfn,
        nelec,
        nmo,
    )
    trial_test.build()
    trial_test.half_rotate(system, ham)
    trial_test_chunked = ParticleHoleWicks(
        wfn,
        nelec,
        nmo,
        num_det_chunks=4,
    )
    trial_test_chunked.build()
    trial_test_chunked.half_rotate(system, ham)
    qmc = dotdict({"dt": 0.005, "nstblz": 5, "batched": True, "nwalkers": nwalkers})
    options = {"hybrid": True}

    walker_batch = UHFWalkersTrial[type(trial)](init,system.nup,system.ndown,ham.nbasis,nwalkers)
    walker_batch.build(trial)

    walker_batch_test = UHFWalkersTrial[type(trial_test)](init,system.nup,system.ndown,ham.nbasis,nwalkers)
    walker_batch_test.build(trial_test)
    
    walker_batch_test_chunked = UHFWalkersTrial[type(trial_test_chunked)](init,system.nup,system.ndown,ham.nbasis,nwalkers)
    walker_batch_test_chunked.build(trial_test_chunked)

    numpy.random.seed(7)
    prop = PhaselessGeneric(qmc["dt"])
    prop.build(ham,trial)

    for i in range(nsteps):
        prop.propagate_walkers(walker_batch, ham, trial, 0)
        walker_batch.reortho()
    numpy.random.seed(7)
    prop = PhaselessGeneric(qmc["dt"])
    prop.build(ham,trial_test)
    for i in range(nsteps):
        prop.propagate_walkers(walker_batch_test, ham, trial_test, 0)
        walker_batch_test.reortho()
    numpy.random.seed(7)
    prop = PhaselessGeneric(qmc["dt"])
    prop.build(ham,trial_test_chunked)
    for i in range(nsteps):
        prop.propagate_walkers(
         walker_batch_test_chunked, ham, trial_test_chunked, 0
        )
        walker_batch_test_chunked.reortho()

    greens_function_multi_det(walker_batch, trial, build_full=True)
    greens_function_multi_det_wicks_opt(walker_batch_test, trial_test, build_full=True)
    greens_function_multi_det_wicks_opt(
        walker_batch_test_chunked, trial_test_chunked, build_full=True
    )
    assert numpy.allclose(walker_batch.Ga, walker_batch_test.Ga)
    assert numpy.allclose(walker_batch.Gb, walker_batch_test.Gb)
    assert numpy.allclose(walker_batch.Ga, walker_batch_test_chunked.Ga)
    assert numpy.allclose(walker_batch.Gb, walker_batch_test_chunked.Gb)
    e_ref = local_energy_multi_det_trial_batch(system, ham, walker_batch, trial)
    e_wicks = local_energy_multi_det_trial_wicks_batch_opt(
        system, ham, walker_batch_test, trial_test
    )
    assert numpy.allclose(e_ref, e_wicks)
    e_wicks_chunked = local_energy_multi_det_trial_wicks_batch_opt_chunked(
        system, ham, walker_batch_test_chunked, trial_test_chunked
    )


@pytest.mark.unit
def test_phmsd_local_energy_active_space_non_aufbau():
    numpy.random.seed(7)
    nelec = (9, 9)
    nwalkers = 4
    nsteps = 10
    nact = 12
    nmo = 20
    ncore = 2
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    system = Generic(nelec=nelec)
    ham = HamGeneric[chol.dtype](
        h1e=numpy.array([h1e, h1e]),
        chol=chol.reshape((-1, nmo * nmo)).T.copy(),
        ecore=0,
        # options={"symmetry": False},
    )
    from ipie.utils.testing import get_random_phmsd_opt, shaped_normal

    wfn, init = get_random_phmsd_opt(
        7, 7, nact, ndet=100, init=True, cmplx_coeffs=False
    )
    init = shaped_normal((nmo, system.ne))
    ci, occa, occb = wfn
    tmp = occa[0]
    occa[0] = occa[2]
    occa[2] = tmp
    tmp = occb[0]
    occb[0] = occb[2]
    occb[2] = tmp
    core = [0, 1]
    ci, occa, occb = wfn
    with_core_a = numpy.array(
        [numpy.array(core + [orb + 2 for orb in oa], dtype=numpy.int32) for oa in occa]
    )
    with_core_b = numpy.array(
        [numpy.array(core + [orb + 2 for orb in ob], dtype=numpy.int32) for ob in occb]
    )
    nskip = 5
    wfn_2_no_act = (
        ci[::nskip],
        with_core_a[::nskip],
        with_core_b[::nskip],
    )  # Get high excitation determinants too
    wfn_2 = (
        ci[::nskip],
        occa[::nskip],
        occb[::nskip],
    )  # Get high excitation determinants too
    ci, occa, occb = wfn_2
    # Naive MSD with looping
    trial = ParticleHoleNaive(
        wfn_2_no_act,
        nelec,
        nmo,
    )
    trial.build()
    trial.half_rotate(system, ham)
    trial_tmp = ParticleHoleWicksNonChunked(
        wfn_2_no_act, nelec, nmo, use_active_space=False
    )
    trial_tmp.build()
    # Original implementation
    trial_ref = ParticleHoleWicksSlow(
        wfn_2_no_act,
        nelec,
        nmo,
    )
    trial_ref.build()
    trial_ref.half_rotate(system, ham)
    # Hack to ensure cre_ex_a structures are present for testing.
    trial_ref.__dict__.update(trial_tmp.__dict__)
    trial_ref.optimized = False
    # Chunked wicks algorithm
    trial_tmp = ParticleHoleWicksNonChunked(wfn_2, nelec, nmo, verbose=True)
    trial_tmp.build()
    trial_test = ParticleHoleWicks(wfn_2, nelec, nmo, num_det_chunks=10)
    trial_test.build()
    trial_test.half_rotate(system, ham)

    qmc = dotdict({"dt": 0.005, "nstblz": 5, "batched": True, "nwalkers": nwalkers})
    options = {"hybrid": True}

    walker_batch = UHFWalkersTrial[type(trial)](init,system.nup,system.ndown,ham.nbasis,nwalkers)
    walker_batch.build(trial)

    walker_batch_ref = UHFWalkersTrial[type(trial_ref)](init,system.nup,system.ndown,ham.nbasis,nwalkers)
    walker_batch_ref.build(trial_ref)

    walker_batch_test = UHFWalkersTrial[type(trial_test)](init,system.nup,system.ndown,ham.nbasis,nwalkers)
    walker_batch_test.build(trial_test)
    
    # Naive
    numpy.random.seed(7)
    prop = PhaselessGeneric(qmc["dt"])
    prop.build(ham,trial_ref)
    for i in range(nsteps):
        prop.propagate_walkers(walker_batch_ref, ham, trial_ref, 0)
        walker_batch_ref.reortho()

    # No optimization
    numpy.random.seed(7)
    prop = PhaselessGeneric(qmc["dt"])
    prop.build(ham,trial)
    for i in range(nsteps):
        prop.propagate_walkers(walker_batch, ham, trial, 0)
        walker_batch.reortho()

    # chunked
    numpy.random.seed(7)
    prop = PhaselessGeneric(qmc["dt"])
    prop.build(ham,trial_test)
    for i in range(nsteps):
        prop.propagate_walkers(walker_batch_test, ham, trial_test, 0)
        walker_batch_test.reortho()

    import copy

    from ipie.propagation.overlap import compute_determinants_batched

    G0a, G0b = walker_batch_ref.G0a, walker_batch_ref.G0b
    dets_a_ref, dets_b_ref = compute_determinants_batched(G0a, G0b, trial_ref)
    G0a, G0b = walker_batch_test.G0a, walker_batch_test.G0b
    dets_a_test, dets_b_test = compute_determinants_batched(G0a, G0b, trial_test)
    assert numpy.allclose(dets_a_ref, dets_a_test)
    assert numpy.allclose(dets_b_ref, dets_b_test)
    greens_function_multi_det(walker_batch, trial)
    greens_function_multi_det_wicks(walker_batch_ref, trial_ref)
    greens_function_multi_det_wicks_opt(walker_batch_test, trial_test)
    assert numpy.allclose(walker_batch_test.Ghalfa, walker_batch_ref.Ghalfa)
    assert numpy.allclose(walker_batch_test.Ghalfb, walker_batch_ref.Ghalfb)
    assert numpy.allclose(walker_batch.Ga, walker_batch_ref.Ga)
    assert numpy.allclose(walker_batch.Gb, walker_batch_ref.Gb)
    assert numpy.allclose(walker_batch_test.Ga, walker_batch_ref.Ga)
    assert numpy.allclose(walker_batch_test.Gb, walker_batch_ref.Gb)
    assert numpy.allclose(walker_batch_test.det_ovlpas, walker_batch_ref.det_ovlpas)
    assert numpy.allclose(walker_batch_test.det_ovlpbs, walker_batch_ref.det_ovlpbs)
    assert trial_ref.nfrozen != trial_test.nfrozen
    e_wicks = local_energy_multi_det_trial_batch(system, ham, walker_batch, trial)
    e_wicks_opt = local_energy_multi_det_trial_wicks_batch(
        system, ham, walker_batch_ref, trial_ref
    )
    e_wicks_opt_act = local_energy_multi_det_trial_wicks_batch_opt(
        system, ham, walker_batch_test, trial_test
    )
    e_wicks_opt_chunk = local_energy_multi_det_trial_wicks_batch_opt_chunked(
        system, ham, walker_batch_test, trial_test
    )

    assert numpy.allclose(e_wicks, e_wicks_opt)
    assert numpy.allclose(e_wicks_opt, e_wicks_opt_act)
    assert numpy.allclose(e_wicks_opt_chunk, e_wicks_opt_act)


if __name__ == "__main__":
    test_phmsd_local_energy()
