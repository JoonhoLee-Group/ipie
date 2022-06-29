import numpy
import os
import pytest
from ipie.systems.generic import Generic
from ipie.hamiltonians.generic import Generic as HamGeneric
from ipie.trial_wavefunction.multi_slater import MultiSlater
from ipie.propagation.continuous import Continuous
from ipie.utils.misc import dotdict
from ipie.utils.testing import (
        generate_hamiltonian,
        get_random_nomsd,
        get_random_phmsd
        )
from ipie.legacy.walkers.multi_det import MultiDetWalker
from ipie.walkers.multi_det_batch import MultiDetTrialWalkerBatch
from ipie.legacy.estimators.greens_function import gab_spin, gab

from ipie.estimators.greens_function_batch import (
        greens_function_multi_det_wicks,
        greens_function_multi_det,
        greens_function_multi_det_wicks_opt
        )
from ipie.estimators.local_energy_batch import (
        local_energy_multi_det_trial_batch)
from ipie.estimators.local_energy_wicks import (
        local_energy_multi_det_trial_wicks_batch,
        local_energy_multi_det_trial_wicks_batch_opt,
        fill_opp_spin_factors_batched,
        fill_same_spin_contribution_batched
        )

from ipie.legacy.estimators.local_energy import local_energy_multi_det
from ipie.propagation.overlap import (
        get_overlap_one_det_wicks,
        get_det_matrix_batched,
        compute_determinants_batched)
from ipie.utils.linalg import minor_mask4, minor_mask

@pytest.mark.unit
def test_greens_function_wicks_opt():
    numpy.random.seed(7)
    nmo = 12
    nelec = (7,7)
    nwalkers = 10
    nsteps = 100
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    system = Generic(nelec=nelec)
    ham = HamGeneric(h1e=numpy.array([h1e,h1e]),
                     chol=chol.reshape((-1,nmo*nmo)).T.copy(),
                     ecore=0, options = {"symmetry":False})
    # Test PH type wavefunction.
    wfn, init = get_random_phmsd(system.nup, system.ndown, ham.nbasis, ndet=3000, init=True)
    ci, oa, ob = wfn
    wfn_2 = (ci[::50], oa[::50], ob[::50])
    # wfn_2 = (ci[:100], oa[:100], ob[:100])
    trial = MultiSlater(system, ham, wfn_2, init=init, options = {'wicks':True,
        'use_wicks_helper': False})
    trial.calculate_energy(system, ham)
    trial_slow = MultiSlater(system, ham, wfn_2, init=init, options = {'wicks': False,
        'use_wicks_helper': False})
    trial_slow.calculate_energy(system, ham)
    trial_opt = MultiSlater(
            system,
            ham,
            wfn_2,
            init=init,
            options={
                'wicks': True,
                'use_wicks_helper': False,
                'optimized': True
                }
            )
    numpy.random.seed(7)
    walker_batch_wick = MultiDetTrialWalkerBatch(system, ham, trial, nwalkers)
    # walker_batch_slow = multidettrialwalkerbatch(system, ham, trial_slow, nwalkers)
    walker_batch_slow = MultiDetTrialWalkerBatch(system, ham, trial_slow, nwalkers)
    walker_batch_opt  = MultiDetTrialWalkerBatch(system, ham, trial_opt, nwalkers)
    options = {'hybrid': True}
    qmc = dotdict({'dt': 0.005, 'nstblz': 5, 'batched': True, 'nwalkers':
        nwalkers})
    numpy.random.seed(7)
    prop = Continuous(system, ham, trial, qmc, options=options)
    for i in range(nsteps):
        prop.propagate_walker_batch(walker_batch_wick, system, ham, trial, 0)
        walker_batch_wick.reortho()
    numpy.random.seed(7)
    prop = Continuous(system, ham, trial_opt, qmc, options=options)
    for i in range(nsteps):
        prop.propagate_walker_batch(walker_batch_opt, system, ham, trial_opt, 0)
        walker_batch_opt.reortho()
    numpy.random.seed(7)
    walker_batch_slow.phia = walker_batch_wick.phia
    walker_batch_slow.phib = walker_batch_wick.phib
    nbasis = walker_batch_wick.Ga.shape[-1]
    from ipie.propagation.overlap import calc_overlap_multi_det_wicks_opt
    ovlps_ref_wick = greens_function_multi_det_wicks(walker_batch_wick, trial)
    ovlps_ref_slow = greens_function_multi_det(walker_batch_slow, trial_slow)
    ovlps_ref_opt = greens_function_multi_det_wicks_opt(walker_batch_opt, trial_opt)
    assert numpy.allclose(ovlps_ref_wick, ovlps_ref_slow)
    assert numpy.allclose(ovlps_ref_opt, ovlps_ref_slow)
    assert numpy.allclose(walker_batch_wick.Ga, walker_batch_slow.Ga)
    assert numpy.allclose(walker_batch_wick.Gb, walker_batch_slow.Gb)
    assert numpy.allclose(walker_batch_wick.Ghalfa, walker_batch_slow.Gihalfa[:,0])
    assert numpy.allclose(walker_batch_wick.Ghalfb, walker_batch_slow.Gihalfb[:,0])
    assert numpy.allclose(walker_batch_opt.Ga, walker_batch_slow.Ga)
    assert numpy.allclose(walker_batch_opt.Gb, walker_batch_slow.Gb)
    assert numpy.allclose(walker_batch_opt.det_ovlpas, walker_batch_wick.det_ovlpas)
    assert numpy.allclose(walker_batch_opt.det_ovlpbs, walker_batch_wick.det_ovlpbs)
    CIa_full = numpy.zeros_like(walker_batch_wick.CIa)
    CIb_full = numpy.zeros_like(walker_batch_wick.CIb)
    CIa_full[:,trial_opt.act_orb_alpha, trial_opt.occ_orb_alpha] = walker_batch_opt.CIa
    CIb_full[:,trial_opt.act_orb_beta, trial_opt.occ_orb_beta] = walker_batch_opt.CIb
    assert numpy.allclose(CIa_full, walker_batch_wick.CIa)
    assert numpy.allclose(CIb_full, walker_batch_wick.CIb)

# Move to propagator tests
@pytest.mark.unit
def test_cofactor_matrix():
    numpy.random.seed(7)
    nexcit = 7
    nwalkers = 10
    ndets_b  = 11
    det_mat = numpy.random.random((nwalkers, ndets_b, nexcit, nexcit)).astype(numpy.complex128)
    cofactor_matrix = numpy.zeros((nwalkers, ndets_b, nexcit-1, nexcit-1), dtype=numpy.complex128)
    iex = 6
    jex = 6
    from ipie.propagation.wicks_kernels import get_cofactor_matrix_batched
    from ipie.utils.linalg import minor_mask
    get_cofactor_matrix_batched(
            nwalkers,
            ndets_b,
            nexcit,
            iex,
            jex,
            det_mat,
            cofactor_matrix)
    for iw in range(nwalkers):
        for jdet in range(ndets_b):
            assert numpy.allclose(cofactor_matrix[iw, jdet], minor_mask(det_mat[iw, jdet], iex, jex))

    from ipie.estimators.kernels.cpu import wicks as wk
    cofactor_matrix = numpy.zeros((nwalkers, ndets_b, nexcit-1, nexcit-1), dtype=numpy.complex128)
    wk.build_cofactor_matrix(
            iex,
            jex,
            det_mat,
            cofactor_matrix)
    for iw in range(nwalkers):
        for jdet in range(ndets_b):
            assert numpy.allclose(cofactor_matrix[iw, jdet], minor_mask(det_mat[iw, jdet], iex, jex))

# Move to propagator tests
@pytest.mark.unit
def test_cofactor_matrix_4():
    numpy.random.seed(7)
    nexcit = 3
    nwalkers = 1
    ndets_b  = 1
    det_mat = numpy.random.random((nwalkers, ndets_b, nexcit, nexcit)).astype(numpy.complex128)
    cofactor_matrix = numpy.zeros((nwalkers, ndets_b, nexcit-2, nexcit-2), dtype=numpy.complex128)
    from ipie.propagation.wicks_kernels import get_cofactor_matrix_4_batched
    from ipie.utils.linalg import minor_mask4
    import itertools
    for iex, jex, kex, lex in itertools.product(range(nexcit), repeat=4):
        if kex > iex and lex > jex:
            get_cofactor_matrix_4_batched(
                    nwalkers,
                    ndets_b,
                    nexcit,
                    iex,
                    jex,
                    kex,
                    lex,
                    det_mat,
                    cofactor_matrix)
            for iw in range(nwalkers):
                for jdet in range(ndets_b):
                    ref = minor_mask4(det_mat[iw, jdet], iex, jex, kex, lex)
                    assert numpy.allclose(cofactor_matrix[iw, jdet], ref)
    nexcit = 6
    nwalkers = 4
    ndets_b  = 10
    det_mat = numpy.random.random((nwalkers, ndets_b, nexcit, nexcit)).astype(numpy.complex128)
    cofactor_matrix = numpy.zeros((nwalkers, ndets_b, nexcit-2, nexcit-2), dtype=numpy.complex128)
    from ipie.estimators.kernels.cpu import wicks as wk
    for iex, jex, kex, lex in itertools.product(range(nexcit), repeat=4):
        if kex > iex and lex > jex:
            get_cofactor_matrix_4_batched(
                    nwalkers,
                    ndets_b,
                    nexcit,
                    iex,
                    jex,
                    kex,
                    lex,
                    det_mat,
                    cofactor_matrix)
            for iw in range(nwalkers):
                for jdet in range(ndets_b):
                    ref = minor_mask4(det_mat[iw, jdet], iex, jex, kex, lex)
                    assert numpy.allclose(cofactor_matrix[iw, jdet], ref)

    # test numba
    det_mat = numpy.random.random((nwalkers, ndets_b, nexcit, nexcit)).astype(numpy.complex128)
    cofactor_matrix = numpy.zeros((nwalkers, ndets_b, nexcit-2, nexcit-2), dtype=numpy.complex128)
    for iex, jex, kex, lex in itertools.product(range(nexcit), repeat=4):
        if kex > iex and lex > jex:
            wk.build_cofactor_matrix_4(
                    iex,
                    jex,
                    kex,
                    lex,
                    det_mat,
                    cofactor_matrix)
            for iw in range(nwalkers):
                for jdet in range(ndets_b):
                    ref = minor_mask4(det_mat[iw, jdet], iex, jex, kex, lex)
                    assert numpy.allclose(cofactor_matrix[iw, jdet], ref)

@pytest.mark.unit
def test_det_matrix():
    numpy.random.seed(7)
    nmo = 12
    nelec = (7,7)
    nwalkers = 10
    nsteps = 100
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    system = Generic(nelec=nelec)
    ham = HamGeneric(h1e=numpy.array([h1e,h1e]),
                     chol=chol.reshape((-1,nmo*nmo)).T.copy(),
                     ecore=0, options = {"symmetry":False})
    # Test PH type wavefunction.
    wfn, init = get_random_phmsd(system.nup, system.ndown, ham.nbasis, ndet=3000, init=True)
    ci, oa, ob = wfn
    wfn_2 = (ci[::50], oa[::50], ob[::50]) # Get high excitation determinants too
    trial = MultiSlater(system, ham, wfn_2, init=init, options = {'wicks':True,
        'use_wicks_helper': False})
    trial.calculate_energy(system, ham)
    trial.half_rotate(system, ham)

    numpy.random.seed(7)
    qmc = dotdict({'dt': 0.005, 'nstblz': 5, 'batched': True, 'nwalkers':
        nwalkers})
    options = {'hybrid': True}
    prop = Continuous(system, ham, trial, qmc, options=options)
    walker_batch = MultiDetTrialWalkerBatch(system, ham, trial, nwalkers)
    numpy.random.seed(7)
    for i in range (nsteps):
        prop.propagate_walker_batch(walker_batch, system, ham, trial, 0)
        walker_batch.reortho()

    nexcit = 4
    from ipie.propagation.wicks_kernels import get_det_matrix_batched
    from ipie.utils.testing import shaped_normal
    ndets_b = len(trial.cre_ex_b[nexcit])
    det_mat = numpy.zeros((nwalkers, ndets_b, nexcit, nexcit),
            dtype=numpy.complex128)
    get_det_matrix_batched(
            nexcit,
            trial.cre_ex_b[nexcit],
            trial.anh_ex_b[nexcit],
            walker_batch.G0b,
            det_mat
            )
    from ipie.estimators.kernels.cpu import wicks as wk
    det_mat_2 = numpy.zeros((nwalkers, ndets_b, nexcit, nexcit),
            dtype=numpy.complex128)
    wk.build_det_matrix(
            trial.cre_ex_b[nexcit],
            trial.anh_ex_b[nexcit],
            trial.occ_map_b,
            trial.nfrozen,
            walker_batch.G0b,
            det_mat_2
            )

    assert numpy.allclose(det_mat, det_mat_2)



@pytest.mark.unit
def test_phmsd_local_energy():
    numpy.random.seed(7)
    nmo = 12
    nelec = (7,7)
    nwalkers = 10
    nsteps = 100
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    system = Generic(nelec=nelec)
    ham = HamGeneric(h1e=numpy.array([h1e,h1e]),
                     chol=chol.reshape((-1,nmo*nmo)).T.copy(),
                     ecore=0, options = {"symmetry":False})
    # Test PH type wavefunction.
    # wfn, init = get_random_phmsd(system.nup, system.ndown, ham.nbasis, ndet=5, init=True)
    wfn, init = get_random_phmsd(system.nup, system.ndown, ham.nbasis, ndet=3000, init=True)
    ci, oa, ob = wfn
    wfn_2 = (ci[::50], oa[::50], ob[::50]) # Get high excitation determinants too
    trial_slow = MultiSlater(system, ham, wfn_2, init=init, options={'wicks': False,
        'use_wicks_helper': False, 'optimized': False})
    trial_slow.calculate_energy(system, ham)
    trial_slow.half_rotate(system, ham)
    trial = MultiSlater(system, ham, wfn_2, init=init, options={'wicks':  True,
        'use_wicks_helper': False, 'optimized': False})
    trial.calculate_energy(system, ham)
    trial.half_rotate(system, ham)
    trial_test = MultiSlater(system, ham, wfn_2, init=init, options={'wicks':  True,
        'use_wicks_helper': False, 'optimized': True})
    trial_test.half_rotate(system, ham)

    numpy.random.seed(7)
    qmc = dotdict({'dt': 0.005, 'nstblz': 5, 'batched': True, 'nwalkers': nwalkers})
    options = {'hybrid': True}
    walker_batch = MultiDetTrialWalkerBatch(system, ham, trial_slow, nwalkers)
    walker_batch_test = MultiDetTrialWalkerBatch(system, ham, trial_test, nwalkers)
    walker_batch_test2 = MultiDetTrialWalkerBatch(system, ham, trial, nwalkers)
    numpy.random.seed(7)
    prop = Continuous(system, ham, trial_slow, qmc, options=options)
    for i in range (nsteps):
        prop.propagate_walker_batch(walker_batch, system, ham, trial_slow, 0)
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
    assert numpy.allclose(walker_batch_test.Ghalfa, walker_batch.Gihalfa[:,0])
    assert numpy.allclose(walker_batch_test.Ghalfb, walker_batch.Gihalfb[:,0])
    assert numpy.allclose(walker_batch_test.Ga, walker_batch.Ga)
    assert numpy.allclose(walker_batch_test.Gb, walker_batch.Gb)
    assert numpy.allclose(walker_batch_test2.Ga, walker_batch.Ga)
    assert numpy.allclose(walker_batch_test2.Gb, walker_batch.Gb)
    assert numpy.allclose(walker_batch_test.det_ovlpas, walker_batch_test2.det_ovlpas)
    assert numpy.allclose(walker_batch_test.det_ovlpbs, walker_batch_test2.det_ovlpbs)
    e_simple = local_energy_multi_det_trial_batch(system, ham, walker_batch, trial)
    e_wicks = local_energy_multi_det_trial_wicks_batch(system, ham, walker_batch_test2, trial)
    e_wicks_opt = local_energy_multi_det_trial_wicks_batch_opt(system, ham, walker_batch_test, trial_test)

    assert numpy.allclose(e_simple, e_wicks)
    assert numpy.allclose(e_simple, e_wicks_opt)
    assert numpy.allclose(e_wicks, e_wicks_opt)

@pytest.mark.unit
def test_kernels_energy():
    numpy.random.seed(7)
    nmo = 12
    nelec = (7,7)
    nwalkers = 10
    nsteps = 100
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    system = Generic(nelec=nelec)
    ham = HamGeneric(h1e=numpy.array([h1e,h1e]),
                     chol=chol.reshape((-1,nmo*nmo)).T.copy(),
                     ecore=0, options = {"symmetry":False})
    wfn, init = get_random_phmsd(system.nup, system.ndown, ham.nbasis, ndet=5000, init=True)
    ci, oa, ob = wfn
    wfn_2 = (ci[::50], oa[::50], ob[::50]) # Get high excitation determinants too
    trial = MultiSlater(
            system,
            ham,
            wfn_2,
            init=init,
            options={
                'wicks': True,
                'optimized': True,
                'use_wicks_helper': False}
            )

    numpy.random.seed(7)
    qmc = dotdict({'dt': 0.005, 'nstblz': 5, 'batched': True, 'nwalkers':
        nwalkers})
    options = {'hybrid': True}
    prop = Continuous(system, ham, trial, qmc, options=options)
    walker_batch = MultiDetTrialWalkerBatch(system, ham, trial, nwalkers)
    numpy.random.seed(7)
    for i in range (nsteps):
        prop.propagate_walker_batch(walker_batch, system, ham, trial, 0)
        walker_batch.reortho()

    import copy
    greens_function_multi_det_wicks_opt(walker_batch, trial)
    from ipie.estimators.kernels.cpu import wicks as wk
    from ipie.estimators.local_energy_wicks import (
            fill_opp_spin_factors_batched_singles,
            fill_opp_spin_factors_batched_doubles_chol,
            fill_opp_spin_factors_batched_triples_chol,
            get_same_spin_double_contribution_batched_contr,
            fill_same_spin_contribution_batched_contr,
            build_slices
            )

    ndets = trial.ndets
    nchol = ham.nchol
    ref = numpy.zeros((nwalkers, ndets, nchol), dtype=numpy.complex128)
    test = numpy.zeros((nwalkers, ndets, nchol), dtype=numpy.complex128)
    slices_alpha, slices_beta = build_slices(trial)
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
            slices_beta[1]
            )
    wk.fill_os_singles(
            trial.cre_ex_b[1],
            trial.anh_ex_b[1],
            trial.occ_map_b,
            trial.nfrozen,
            Lbb,
            test,
            slices_beta[1]
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
            slices_beta[2]
            )
    wk.fill_os_doubles(
            trial.cre_ex_b[2],
            trial.anh_ex_b[2],
            trial.occ_map_b,
            trial.nfrozen,
            G0,
            Lbb,
            test,
            slices_beta[2]
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
            slices_beta[iexcit]
            )
    wk.fill_os_triples(
            trial.cre_ex_b[iexcit],
            trial.anh_ex_b[iexcit],
            trial.occ_map_b,
            trial.nfrozen,
            G0,
            Lbb,
            test,
            slices_beta[iexcit]
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
            slices_beta[iexcit]
            )
    wk.get_ss_doubles(
            trial.cre_ex_b[iexcit],
            trial.anh_ex_b[iexcit],
            trial.occ_map_b,
            Lbb,
            test,
            slices_beta[iexcit]
            )
    assert numpy.allclose(ref, test)
    iexcit = 4
    ndets_level = len(trial.cre_ex_b[iexcit])
    det_mat = numpy.zeros((nwalkers, ndets_level, iexcit, iexcit), dtype=numpy.complex128)
    cof_mat = numpy.zeros((nwalkers, ndets_level, iexcit-1, iexcit-1), dtype=numpy.complex128)
    wk.build_det_matrix(
            trial.cre_ex_b[iexcit],
            trial.anh_ex_b[iexcit],
            trial.occ_map_b,
            trial.nfrozen,
            walker_batch.G0b,
            det_mat)
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
            slices_beta[iexcit]
            )
    wk.fill_os_nfold(
            trial.cre_ex_b[iexcit],
            trial.anh_ex_b[iexcit],
            trial.occ_map_b,
            det_mat,
            cof_mat,
            Lbb,
            test,
            slices_beta[iexcit]
            )
    assert numpy.allclose(ref, test)
    det_mat = numpy.zeros((nwalkers, ndets_level, iexcit, iexcit), dtype=numpy.complex128)
    cof_mat = numpy.zeros((nwalkers, ndets_level, iexcit-2, iexcit-2), dtype=numpy.complex128)
    get_det_matrix_batched(
            iexcit,
            trial.cre_ex_b[iexcit],
            trial.anh_ex_b[iexcit],
            walker_batch.G0b,
            det_mat)
    ref = numpy.zeros((nwalkers, ndets), dtype=numpy.complex128)
    test = numpy.zeros((nwalkers, ndets), dtype=numpy.complex128)
    from ipie.estimators.local_energy_wicks import fill_same_spin_contribution_batched_contr
    fill_same_spin_contribution_batched_contr(
            iexcit,
            trial.cre_ex_b[iexcit],
            trial.anh_ex_b[iexcit],
            det_mat,
            cof_mat,
            Lbb,
            ref,
            trial.excit_map_b[iexcit],
            slices_beta[iexcit]
            )
    wk.get_ss_nfold(
            trial.cre_ex_b[iexcit],
            trial.anh_ex_b[iexcit],
            trial.occ_map_b,
            det_mat,
            cof_mat,
            Lbb,
            test,
            slices_beta[iexcit]
            )
    assert numpy.allclose(ref, test)

@pytest.mark.unit
def test_kernels_gf():
    numpy.random.seed(7)
    nmo = 12
    nelec = (7,7)
    nwalkers = 10
    nsteps = 100
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    system = Generic(nelec=nelec)
    ham = HamGeneric(h1e=numpy.array([h1e,h1e]),
                     chol=chol.reshape((-1,nmo*nmo)).T.copy(),
                     ecore=0, options = {"symmetry":False})
    wfn, init = get_random_phmsd(system.nup, system.ndown, ham.nbasis, ndet=5000, init=True)
    ci, oa, ob = wfn
    wfn_2 = (ci[::50], oa[::50], ob[::50]) # Get high excitation determinants too
    trial = MultiSlater(system, ham, wfn_2, init=init, options = {'wicks':True,
        'use_wicks_helper': False})

    numpy.random.seed(7)
    qmc = dotdict({'dt': 0.005, 'nstblz': 5, 'batched': True, 'nwalkers':
        nwalkers})
    options = {'hybrid': True}
    prop = Continuous(system, ham, trial, qmc, options=options)
    walker_batch = MultiDetTrialWalkerBatch(system, ham, trial, nwalkers)
    numpy.random.seed(7)
    for i in range (nsteps):
        prop.propagate_walker_batch(walker_batch, system, ham, trial, 0)
        walker_batch.reortho()

    greens_function_multi_det_wicks(walker_batch, trial)

    from ipie.estimators.kernels.cpu import wicks as wk
    from ipie.estimators.local_energy_wicks import (
            fill_opp_spin_factors_batched_singles,
            fill_opp_spin_factors_batched_doubles_chol,
            fill_opp_spin_factors_batched_triples_chol,
            get_same_spin_double_contribution_batched_contr,
            fill_same_spin_contribution_batched_contr,
            build_slices
            )

    ndets = trial.ndets
    nchol = ham.nchol
    ref = numpy.zeros((nwalkers, ndets, nchol), dtype=numpy.complex128)
    test = numpy.zeros((nwalkers, ndets, nchol), dtype=numpy.complex128)
    ovlpa = walker_batch.det_ovlpas
    ovlpb = walker_batch.det_ovlpbs

    c_phasea_ovlpb = numpy.einsum(
                        'wJ,J->wJ',
                        ovlpb,
                        trial.phase_a*trial.coeffs.conj(),
                        optimize=True)
    c_phaseb_ovlpa = numpy.einsum(
                        'wJ,J->wJ',
                        ovlpa,
                        trial.phase_b*trial.coeffs.conj(),
                        optimize=True)
    walker_batch.CIa.fill(0.0+0.0j)
    walker_batch.CIb.fill(0.0+0.0j)
    from ipie.estimators.greens_function_batch import (
            build_CI_single_excitation,
            build_CI_single_excitation_opt,
            )
    build_CI_single_excitation(
            walker_batch,
            trial,
            c_phasea_ovlpb,
            c_phaseb_ovlpa
            )
    refa = walker_batch.CIa.copy()
    refb = walker_batch.CIb.copy()
    walker_batch.CIa.fill(0.0+0.0j)
    walker_batch.CIb.fill(0.0+0.0j)
    build_CI_single_excitation_opt(
            walker_batch,
            trial,
            c_phasea_ovlpb,
            c_phaseb_ovlpa
            )
    assert numpy.allclose(refa, walker_batch.CIa)
    assert numpy.allclose(refb, walker_batch.CIb)
    from ipie.estimators.greens_function_batch import (
            build_CI_double_excitation,
            build_CI_double_excitation_opt,
            )
    walker_batch.CIa.fill(0.0+0.0j)
    walker_batch.CIb.fill(0.0+0.0j)
    build_CI_double_excitation(
            walker_batch,
            trial,
            c_phasea_ovlpb,
            c_phaseb_ovlpa
            )
    refa = walker_batch.CIa.copy()
    refb = walker_batch.CIb.copy()
    walker_batch.CIa.fill(0.0+0.0j)
    walker_batch.CIb.fill(0.0+0.0j)
    build_CI_double_excitation_opt(
            walker_batch,
            trial,
            c_phasea_ovlpb,
            c_phaseb_ovlpa
            )
    assert numpy.allclose(refa, walker_batch.CIa)
    assert numpy.allclose(refb, walker_batch.CIb)
    from ipie.estimators.greens_function_batch import (
            build_CI_triple_excitation,
            build_CI_triple_excitation_opt,
            )
    walker_batch.CIa.fill(0.0+0.0j)
    walker_batch.CIb.fill(0.0+0.0j)
    build_CI_triple_excitation(
            walker_batch,
            trial,
            c_phasea_ovlpb,
            c_phaseb_ovlpa
            )
    refa = walker_batch.CIa.copy()
    refb = walker_batch.CIb.copy()
    walker_batch.CIa.fill(0.0+0.0j)
    walker_batch.CIb.fill(0.0+0.0j)
    build_CI_triple_excitation_opt(
            walker_batch,
            trial,
            c_phasea_ovlpb,
            c_phaseb_ovlpa
            )
    assert numpy.allclose(refa, walker_batch.CIa)
    assert numpy.allclose(refb, walker_batch.CIb)
    from ipie.estimators.greens_function_batch import (
            build_CI_nfold_excitation,
            build_CI_nfold_excitation_opt,
            )
    walker_batch.CIa.fill(0.0+0.0j)
    walker_batch.CIb.fill(0.0+0.0j)
    build_CI_nfold_excitation(
            4,
            walker_batch,
            trial,
            c_phasea_ovlpb,
            c_phaseb_ovlpa
            )
    refa = walker_batch.CIa.copy()
    refb = walker_batch.CIb.copy()
    walker_batch.CIa.fill(0.0+0.0j)
    walker_batch.CIb.fill(0.0+0.0j)
    build_CI_nfold_excitation_opt(
            4,
            walker_batch,
            trial,
            c_phasea_ovlpb,
            c_phaseb_ovlpa
            )
    assert numpy.allclose(refa, walker_batch.CIa)
    assert numpy.allclose(refb, walker_batch.CIb)

@pytest.mark.unit
def test_kernels_gf_active_space():
    numpy.random.seed(7)
    nmo = 30
    nelec = (9,9)
    nwalkers = 1
    nsteps = 100
    nact = 12
    ncore = 2
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    system = Generic(nelec=nelec)
    ham = HamGeneric(h1e=numpy.array([h1e,h1e]),
                     chol=chol.reshape((-1,nmo*nmo)).T.copy(),
                     ecore=0, options = {"symmetry":False})
    wfn, init = get_random_phmsd(7, 7, nact, ndet=5000, init=True)
    ci, occa, occb = wfn
    core = [0, 1]
    with_core_a = [numpy.array(core + [orb + 2 for orb in oa], dtype=numpy.int32) for oa in occa]
    with_core_b = [numpy.array(core + [orb + 2 for orb in ob], dtype=numpy.int32) for ob in occb]
    wfn_2 = (ci[::50], with_core_a[::50], with_core_b[::50]) # Get high excitation determinants too

    trial_ref = MultiSlater(
            system,
            ham,
            wfn_2,
            options={
                'wicks': True,
                'optimized': False,
                'use_wicks_helper': False,
                }
            )
    trial_test = MultiSlater(
            system,
            ham,
            wfn_2,
            options={
                'wicks': True,
                'optimized': True,
                'use_wicks_helper': False,
                'nact': 12,
                'ncas': 14,
                }
            )

    numpy.random.seed(7)
    qmc = dotdict(
            {'dt': 0.005, 'nstblz': 5, 'batched': True, 'nwalkers': nwalkers}
            )
    options = {'hybrid': True}
    prop = Continuous(system, ham, trial_ref, qmc, options=options)
    walker_batch_ref = MultiDetTrialWalkerBatch(system, ham, trial_ref, nwalkers)
    walker_batch_test = MultiDetTrialWalkerBatch(system, ham, trial_test, nwalkers)
    numpy.random.seed(7)
    for i in range (nsteps):
        prop.propagate_walker_batch(walker_batch_ref, system, ham, trial_ref, 0)
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

    ndets = trial_ref.ndets
    nchol = ham.nchol
    ref = numpy.zeros((nwalkers, ndets, nchol), dtype=numpy.complex128)
    test = numpy.zeros((nwalkers, ndets, nchol), dtype=numpy.complex128)
    ovlpa = walker_batch_ref.det_ovlpas
    ovlpb = walker_batch_ref.det_ovlpbs

    c_phasea_ovlpb = numpy.einsum(
                        'wJ,J->wJ',
                        ovlpb,
                        trial_ref.phase_a*trial_ref.coeffs.conj(),
                        optimize=True)
    c_phaseb_ovlpa = numpy.einsum(
                        'wJ,J->wJ',
                        ovlpa,
                        trial_ref.phase_b*trial_ref.coeffs.conj(),
                        optimize=True)
    walker_batch_ref.CIa.fill(0.0+0.0j)
    walker_batch_ref.CIb.fill(0.0+0.0j)
    from ipie.estimators.greens_function_batch import (
            build_CI_single_excitation_opt,
            build_CI_double_excitation_opt,
            build_CI_triple_excitation_opt,
            build_CI_nfold_excitation_opt
            )
    methods = [
            build_CI_single_excitation_opt,
            build_CI_double_excitation_opt,
            build_CI_triple_excitation_opt,
            ]
    for method in methods:
        walker_batch_ref.CIa.fill(0.0+0.0j)
        walker_batch_ref.CIb.fill(0.0+0.0j)
        method(
                walker_batch_ref,
                trial_ref,
                c_phasea_ovlpb,
                c_phaseb_ovlpa
                )
        refa = walker_batch_ref.CIa.copy()
        refb = walker_batch_ref.CIb.copy()
        walker_batch_test.CIa.fill(0.0+0.0j)
        walker_batch_test.CIb.fill(0.0+0.0j)
        method(
                walker_batch_test,
                trial_test,
                c_phasea_ovlpb,
                c_phaseb_ovlpa
                )
        CIa_full = numpy.zeros_like(refa)
        CIb_full = numpy.zeros_like(refb)
        CIa_full[:,trial_test.act_orb_alpha, trial_test.occ_orb_alpha] = walker_batch_test.CIa
        CIb_full[:,trial_test.act_orb_beta, trial_test.occ_orb_beta] = walker_batch_test.CIb
        assert numpy.allclose(refa, CIa_full)
        assert numpy.allclose(refb, CIb_full)
    walker_batch_test.Q0a = numpy.eye(nmo)[None, :] - walker_batch_test.G0a.copy()
    walker_batch_test.Q0b = numpy.eye(nmo)[None, :] - walker_batch_test.G0b.copy()
    from ipie.estimators.greens_function_batch import (
            contract_CI
            )
    act_orb = trial_ref.act_orb_alpha
    occ_orb = trial_ref.occ_orb_alpha
    contract_CI(
        walker_batch_ref.Q0a[:,:,act_orb].copy(),
        walker_batch_ref.CIa,
        walker_batch_ref.G0a,
        walker_batch_ref.Ga
        )
    act_orb = trial_test.act_orb_alpha
    occ_orb = trial_test.occ_orb_alpha
    contract_CI(
        walker_batch_test.Q0a[:,:,act_orb].copy(),
        walker_batch_test.CIa,
        walker_batch_test.Ghalfa[:,act_orb].copy(),
        walker_batch_test.Ga
        )
    assert numpy.allclose(walker_batch_ref.Ga, walker_batch_test.Ga)
    assert numpy.allclose(walker_batch_ref.Ga, walker_batch_test.Ga)
    for nexcit in range(4, trial_ref.max_excite+1):
        walker_batch_ref.CIa.fill(0.0+0.0j)
        walker_batch_ref.CIb.fill(0.0+0.0j)
        build_CI_nfold_excitation_opt(
                nexcit,
                walker_batch_ref,
                trial_ref,
                c_phasea_ovlpb,
                c_phaseb_ovlpa
                )
        refa = walker_batch_ref.CIa.copy()
        refb = walker_batch_ref.CIb.copy()
        walker_batch_test.CIa.fill(0.0+0.0j)
        walker_batch_test.CIb.fill(0.0+0.0j)
        build_CI_nfold_excitation_opt(
                nexcit,
                walker_batch_test,
                trial_test,
                c_phasea_ovlpb,
                c_phaseb_ovlpa
                )
        CIa_full = numpy.zeros_like(refa)
        CIb_full = numpy.zeros_like(refb)
        CIa_full[:,trial_test.act_orb_alpha, trial_test.occ_orb_alpha] = walker_batch_test.CIa
        CIb_full[:,trial_test.act_orb_beta, trial_test.occ_orb_beta] = walker_batch_test.CIb
        assert numpy.allclose(refa, CIa_full)
        assert numpy.allclose(refb, CIb_full)
    contract_CI(
        walker_batch_test.Q0b[:,:,act_orb].copy(),
        walker_batch_test.CIb,
        walker_batch_test.Ghalfb[:,act_orb].copy(),
        walker_batch_test.Gb
        )
    contract_CI(
        walker_batch_ref.Q0b,
        walker_batch_ref.CIb,
        walker_batch_ref.G0b,
        walker_batch_ref.Gb
        )
    assert numpy.allclose(walker_batch_ref.Ga, walker_batch_test.Ga)

@pytest.mark.unit
def test_kernels_energy_active_space():
    numpy.random.seed(7)
    nmo = 30
    nelec = (9,9)
    nwalkers = 1
    nsteps = 100
    nact = 12
    ncore = 2
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    system = Generic(nelec=nelec)
    ham = HamGeneric(h1e=numpy.array([h1e,h1e]),
                     chol=chol.reshape((-1,nmo*nmo)).T.copy(),
                     ecore=0, options = {"symmetry":False})
    wfn, init = get_random_phmsd(7, 7, nact, ndet=5000, init=True)
    ci, occa, occb = wfn
    core = [0, 1]
    with_core_a = [numpy.array(core + [orb + 2 for orb in oa], dtype=numpy.int32) for oa in occa]
    with_core_b = [numpy.array(core + [orb + 2 for orb in ob], dtype=numpy.int32) for ob in occb]
    wfn_2 = (ci[::50], with_core_a[::50], with_core_b[::50]) # Get high excitation determinants too

    trial_ref = MultiSlater(
            system,
            ham,
            wfn_2,
            options={
                'wicks': True,
                'optimized': False,
                'use_wicks_helper': False,
                }
            )
    trial_test = MultiSlater(
            system,
            ham,
            wfn_2,
            options={
                'wicks': True,
                'optimized': True,
                'use_wicks_helper': False,
                'nact': 12,
                'ncas': 14,
                }
            )

    numpy.random.seed(7)
    qmc = dotdict(
            {'dt': 0.005, 'nstblz': 5, 'batched': True, 'nwalkers': nwalkers}
            )
    options = {'hybrid': True}
    prop = Continuous(system, ham, trial_ref, qmc, options=options)
    walker_batch_ref = MultiDetTrialWalkerBatch(system, ham, trial_ref, nwalkers)
    walker_batch_test = MultiDetTrialWalkerBatch(system, ham, trial_test, nwalkers)
    numpy.random.seed(7)
    for i in range (nsteps):
        prop.propagate_walker_batch(walker_batch_ref, system, ham, trial_ref, 0)
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

    ndets = trial_ref.ndets
    nchol = ham.nchol
    ref = numpy.zeros((nwalkers, ndets, nchol), dtype=numpy.complex128)
    test = numpy.zeros((nwalkers, ndets, nchol), dtype=numpy.complex128)
    ovlpa = walker_batch_ref.det_ovlpas
    ovlpb = walker_batch_ref.det_ovlpbs

    c_phasea_ovlpb = numpy.einsum(
                        'wJ,J->wJ',
                        ovlpb,
                        trial_ref.phase_a*trial_ref.coeffs.conj(),
                        optimize=True)
    c_phaseb_ovlpa = numpy.einsum(
                        'wJ,J->wJ',
                        ovlpa,
                        trial_ref.phase_b*trial_ref.coeffs.conj(),
                        optimize=True)
    walker_batch_ref.CIa.fill(0.0+0.0j)
    walker_batch_ref.CIb.fill(0.0+0.0j)
    from ipie.utils.testing import shaped_normal
    from ipie.estimators.local_energy_wicks import (
            build_slices
            )
    Lbb = shaped_normal((nwalkers, nmo, system.ndown, nchol))
    slices_alpha, slices_beta = build_slices(trial_ref)
    assert trial_ref.nfrozen != trial_test.nfrozen
    # 1.
    wk.fill_os_singles(
            trial_ref.cre_ex_b[1],
            trial_ref.anh_ex_b[1],
            trial_ref.occ_map_b,
            trial_ref.nfrozen,
            Lbb,
            ref,
            slices_beta[1]
            )
    act_orb = trial_test.act_orb_beta
    occ_orb = trial_test.occ_orb_beta
    wk.fill_os_singles(
            trial_test.cre_ex_b[1],
            trial_test.anh_ex_b[1],
            trial_test.occ_map_b,
            trial_test.nfrozen,
            Lbb[:,act_orb,occ_orb,:].copy(),
            test,
            slices_beta[1]
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
            slices_beta[2]
            )
    wk.fill_os_doubles(
            trial_test.cre_ex_b[2],
            trial_test.anh_ex_b[2],
            trial_test.occ_map_b,
            trial_test.nfrozen,
            G0,
            Lbb[:,act_orb,occ_orb,:].copy(),
            test,
            slices_beta[2]
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
            slices_beta[iexcit]
            )
    wk.fill_os_triples(
            trial_test.cre_ex_b[iexcit],
            trial_test.anh_ex_b[iexcit],
            trial_test.occ_map_b,
            trial_test.nfrozen,
            G0,
            Lbb[:,act_orb,occ_orb,:].copy(),
            test,
            slices_beta[iexcit]
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
            slices_beta[iexcit]
            )
    wk.get_ss_doubles(
            trial_test.cre_ex_b[iexcit],
            trial_test.anh_ex_b[iexcit],
            trial_test.occ_map_b,
            Lbb[:,act_orb,occ_orb,:].copy(),
            test,
            slices_beta[iexcit]
            )
    assert numpy.allclose(ref, test)
    iexcit = 4
    ndets_level = len(trial_ref.cre_ex_b[iexcit])
    det_mat_ref = numpy.zeros((nwalkers, ndets_level, iexcit, iexcit), dtype=numpy.complex128)
    det_mat_test = numpy.zeros((nwalkers, ndets_level, iexcit, iexcit), dtype=numpy.complex128)
    cof_mat = numpy.zeros((nwalkers, ndets_level, iexcit-1, iexcit-1), dtype=numpy.complex128)
    wk.build_det_matrix(
            trial_ref.cre_ex_b[iexcit],
            trial_ref.anh_ex_b[iexcit],
            trial_ref.occ_map_b,
            trial_ref.nfrozen,
            walker_batch_ref.G0b,
            det_mat_ref)
    wk.build_det_matrix(
            trial_test.cre_ex_b[iexcit],
            trial_test.anh_ex_b[iexcit],
            trial_test.occ_map_b,
            trial_test.nfrozen,
            walker_batch_test.G0b,
            det_mat_test)
    assert numpy.allclose(det_mat_test, det_mat_ref)
    ref = numpy.zeros((nwalkers, ndets, nchol), dtype=numpy.complex128)
    test = numpy.zeros((nwalkers, ndets, nchol), dtype=numpy.complex128)
    from ipie.estimators.local_energy_wicks import fill_opp_spin_factors_batched_chol
    cof_mat = numpy.zeros((nwalkers, ndets_level, iexcit-1, iexcit-1), dtype=numpy.complex128)
    wk.fill_os_nfold(
            trial_ref.cre_ex_b[iexcit],
            trial_ref.anh_ex_b[iexcit],
            trial_ref.occ_map_b,
            det_mat_ref,
            cof_mat,
            Lbb,
            ref,
            slices_beta[iexcit]
            )
    cof_mat = numpy.zeros((nwalkers, ndets_level, iexcit-1, iexcit-1), dtype=numpy.complex128)
    wk.fill_os_nfold(
            trial_test.cre_ex_b[iexcit],
            trial_test.anh_ex_b[iexcit],
            trial_test.occ_map_b,
            det_mat_test,
            cof_mat,
            Lbb[:,act_orb,occ_orb,:].copy(),
            test,
            slices_beta[iexcit]
            )
    assert numpy.allclose(ref, test)
    ref = numpy.zeros((nwalkers, ndets), dtype=numpy.complex128)
    test = numpy.zeros((nwalkers, ndets), dtype=numpy.complex128)
    from ipie.estimators.local_energy_wicks import fill_same_spin_contribution_batched_contr
    cof_mat = numpy.zeros((nwalkers, ndets_level, iexcit-1, iexcit-1), dtype=numpy.complex128)
    wk.get_ss_nfold(
            trial_ref.cre_ex_b[iexcit],
            trial_ref.anh_ex_b[iexcit],
            trial_ref.occ_map_b,
            det_mat_ref,
            cof_mat,
            Lbb,
            ref,
            slices_beta[iexcit]
            )
    cof_mat = numpy.zeros((nwalkers, ndets_level, iexcit-1, iexcit-1), dtype=numpy.complex128)
    wk.get_ss_nfold(
            trial_test.cre_ex_b[iexcit],
            trial_test.anh_ex_b[iexcit],
            trial_test.occ_map_b,
            det_mat_test,
            cof_mat,
            Lbb[:,act_orb,occ_orb,:].copy(),
            test,
            slices_beta[iexcit]
            )
    assert numpy.allclose(ref, test)

@pytest.mark.unit
def test_phmsd_local_energy_active_space():
    numpy.random.seed(7)
    nmo = 30
    nelec = (9,9)
    nwalkers = 1
    nsteps = 100
    nact = 12
    ncore = 2
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    system = Generic(nelec=nelec)
    ham = HamGeneric(h1e=numpy.array([h1e,h1e]),
                     chol=chol.reshape((-1,nmo*nmo)).T.copy(),
                     ecore=0, options = {"symmetry":False})
    wfn, init = get_random_phmsd(7, 7, nact, ndet=5000, init=True)
    ci, occa, occb = wfn
    core = [0, 1]
    with_core_a = [numpy.array(core + [orb + 2 for orb in oa], dtype=numpy.int32) for oa in occa]
    with_core_b = [numpy.array(core + [orb + 2 for orb in ob], dtype=numpy.int32) for ob in occb]
    wfn_2 = (ci[::50], with_core_a[::50], with_core_b[::50]) # Get high excitation determinants too

    trial_ref = MultiSlater(
            system,
            ham,
            wfn_2,
            options={
                'wicks': True,
                'optimized': True,
                'use_wicks_helper': False,
                }
            )
    trial_ref.half_rotate(system, ham)
    trial_test = MultiSlater(
            system,
            ham,
            wfn_2,
            options={
                'wicks': True,
                'optimized': True,
                'use_wicks_helper': False,
                'nact': 12,
                'ncas': 14,
                }
            )
    trial_test.half_rotate(system, ham)

    numpy.random.seed(7)
    qmc = dotdict(
            {'dt': 0.005, 'nstblz': 5, 'batched': True, 'nwalkers': nwalkers}
            )
    options = {'hybrid': True}
    prop = Continuous(system, ham, trial_ref, qmc, options=options)
    walker_batch_ref = MultiDetTrialWalkerBatch(system, ham, trial_ref, nwalkers)
    walker_batch_test = MultiDetTrialWalkerBatch(system, ham, trial_test, nwalkers)
    numpy.random.seed(7)
    for i in range (nsteps):
        prop.propagate_walker_batch(walker_batch_ref, system, ham, trial_ref, 0)
        walker_batch_ref.reortho()

    import copy
    walker_batch_test.phia = walker_batch_ref.phia.copy()
    walker_batch_test.phib = walker_batch_ref.phib.copy()
    walker_batch_test.ovlp = walker_batch_ref.ovlp
    greens_function_multi_det_wicks_opt(walker_batch_ref, trial_ref)
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
    e_wicks_opt = local_energy_multi_det_trial_wicks_batch_opt(
                        system,
                        ham,
                        walker_batch_ref,
                        trial_ref)
    e_wicks_opt_act = local_energy_multi_det_trial_wicks_batch_opt(
                        system,
                        ham,
                        walker_batch_test,
                        trial_test)

    assert numpy.allclose(e_wicks_opt, e_wicks_opt_act)

@pytest.mark.unit
def test_phmsd_local_energy_active_space_non_aufbau():
    numpy.random.seed(7)
    nelec = (9,9)
    nwalkers = 1
    nsteps = 10
    nact = 12
    nmo = 20
    ncore = 2
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    system = Generic(nelec=nelec)
    ham = HamGeneric(h1e=numpy.array([h1e,h1e]),
                     chol=chol.reshape((-1,nmo*nmo)).T.copy(),
                     ecore=0, options = {"symmetry":False})
    from ipie.utils.testing import (
            shaped_normal,
            get_random_phmsd_opt
            )
    wfn, init = get_random_phmsd_opt(7, 7, nact, ndet=100, init=True)
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
    with_core_a = numpy.array([numpy.array(core + [orb + 2 for orb in oa], dtype=numpy.int32) for oa in occa])
    with_core_b = numpy.array([numpy.array(core + [orb + 2 for orb in ob], dtype=numpy.int32) for ob in occb])
    nskip = 10
    wfn_2 = (ci[::nskip], with_core_a[::nskip], with_core_b[::nskip]) # Get high excitation determinants too
    ci, occa, occb = wfn_2
    trial = MultiSlater(
            system,
            ham,
            wfn_2,
            init=init,
            options={
                'wicks': False,
                'optimized': False,
                'use_wicks_helper': False,
                }
            )
    trial.half_rotate(system, ham)
    trial_ref = MultiSlater(
            system,
            ham,
            wfn_2,
            init=init,
            options={
                'wicks': True,
                'optimized': False,
                'use_wicks_helper': False,
                }
            )
    trial_ref.half_rotate(system, ham)
    trial_test = MultiSlater(
            system,
            ham,
            wfn_2,
            init=init,
            options={
                'wicks': True,
                'optimized': True,
                'use_wicks_helper': False,
                'nact': nact,
                'ncas': 14,
                }
            )
    trial_test.half_rotate(system, ham)

    qmc = dotdict(
            {'dt': 0.005, 'nstblz': 5, 'batched': True, 'nwalkers': nwalkers}
            )
    options = {'hybrid': True}
    walker_batch = MultiDetTrialWalkerBatch(system, ham, trial, nwalkers)
    walker_batch_ref = MultiDetTrialWalkerBatch(system, ham, trial_ref, nwalkers)
    walker_batch_test = MultiDetTrialWalkerBatch(system, ham, trial_test, nwalkers)
    numpy.random.seed(7)
    prop = Continuous(system, ham, trial_ref, qmc, options=options)
    for i in range(nsteps):
        prop.propagate_walker_batch(walker_batch_ref, system, ham, trial_ref, 0)
        walker_batch_ref.reortho()
    numpy.random.seed(7)
    prop = Continuous(system, ham, trial_test, qmc, options=options)
    for i in range(nsteps):
        prop.propagate_walker_batch(walker_batch_test, system, ham, trial_test, 0)
        walker_batch_test.reortho()
    numpy.random.seed(7)
    prop = Continuous(system, ham, trial, qmc, options=options)
    for i in range(nsteps):
        prop.propagate_walker_batch(walker_batch, system, ham, trial, 0)
        walker_batch.reortho()

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
    e_wicks = local_energy_multi_det_trial_batch(
                        system,
                        ham,
                        walker_batch,
                        trial)
    e_wicks_opt = local_energy_multi_det_trial_wicks_batch(
                        system,
                        ham,
                        walker_batch_ref,
                        trial_ref)
    e_wicks_opt_act = local_energy_multi_det_trial_wicks_batch_opt(
                        system,
                        ham,
                        walker_batch_test,
                        trial_test)

    assert numpy.allclose(e_wicks, e_wicks_opt)
    assert numpy.allclose(e_wicks_opt, e_wicks_opt_act)

if __name__ == '__main__':
    test_phmsd_local_energy()
