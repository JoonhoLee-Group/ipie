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
    numpy.random.seed(7)
    walker_batch_wick = MultiDetTrialWalkerBatch(system, ham, trial, nwalkers)
    walker_batch_slow = MultiDetTrialWalkerBatch(system, ham, trial, nwalkers)
    walker_batch_opt  = MultiDetTrialWalkerBatch(system, ham, trial, nwalkers)
    options = {'hybrid': True}
    numpy.random.seed(7)
    qmc = dotdict({'dt': 0.005, 'nstblz': 5, 'batched': True, 'nwalkers':
        nwalkers})
    prop = Continuous(system, ham, trial, qmc, options=options)
    for i in range(nsteps):
        prop.propagate_walker_batch(walker_batch_wick, system, ham, trial, 0)
        walker_batch_wick.reortho()
    walker_batch_opt.phia = walker_batch_wick.phia
    walker_batch_opt.phib = walker_batch_wick.phib
    walker_batch_slow.phia = walker_batch_wick.phia
    walker_batch_slow.phib = walker_batch_wick.phib
    nbasis = walker_batch_wick.Ga.shape[-1]
    from ipie.propagation.overlap import calc_overlap_multi_det_wicks_opt
    ovlps_ref_wick = greens_function_multi_det_wicks(walker_batch_wick, trial)
    ovlps_ref_slow = greens_function_multi_det(walker_batch_slow, trial)
    ovlps_ref_opt = greens_function_multi_det_wicks_opt(walker_batch_opt, trial)
    assert numpy.allclose(ovlps_ref_wick, ovlps_ref_slow)
    assert numpy.allclose(ovlps_ref_opt, ovlps_ref_slow)
    assert numpy.allclose(walker_batch_wick.Ga, walker_batch_slow.Ga)
    assert numpy.allclose(walker_batch_wick.Ga, walker_batch_slow.Ga)
    assert numpy.allclose(walker_batch_wick.Ghalfa, walker_batch_slow.Gihalfa[:,0])
    assert numpy.allclose(walker_batch_wick.Ghalfb, walker_batch_slow.Gihalfb[:,0])
    assert numpy.allclose(walker_batch_opt.Ga, walker_batch_slow.Ga)
    assert numpy.allclose(walker_batch_opt.Gb, walker_batch_slow.Gb)
    assert numpy.allclose(walker_batch_opt.det_ovlpas, walker_batch_wick.det_ovlpas)
    assert numpy.allclose(walker_batch_opt.det_ovlpbs, walker_batch_wick.det_ovlpbs)
    assert numpy.allclose(walker_batch_opt.CIa, walker_batch_wick.CIa)
    assert numpy.allclose(walker_batch_opt.CIb, walker_batch_wick.CIb)

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

    import copy
    walker_batch_test = copy.deepcopy(walker_batch)
    walker_batch_test2 = copy.deepcopy(walker_batch)
    greens_function_multi_det(walker_batch, trial)
    greens_function_multi_det_wicks_opt(walker_batch_test, trial)
    greens_function_multi_det_wicks(walker_batch_test2, trial)
    e_wicks = local_energy_multi_det_trial_wicks_batch(system, ham, walker_batch_test, trial)
    # from ipie.estimators.local_energy_wicks_old import local_energy_multi_det_trial_wicks_batch_opt as wicks_old
    # e_wicks_opt_old = wicks_old(system, ham, walker_batch_test, trial)
    e_wicks_opt = local_energy_multi_det_trial_wicks_batch_opt(system, ham,
            walker_batch_test, trial)
    e_simple = local_energy_multi_det_trial_batch(system, ham, walker_batch, trial)

    assert numpy.allclose(e_simple, e_wicks)
    # assert numpy.allclose(e_simple, e_wicks_opt_old)
    assert numpy.allclose(e_simple, e_wicks_opt)

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
    print(trial.ndet_a, trial.ndet_b)
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
            Lbb,
            test,
            slices_beta[iexcit]
            )
    assert numpy.allclose(ref, test)
    iexcit = 4
    ndets_level = len(trial.cre_ex_b[iexcit])
    det_mat = numpy.zeros((nwalkers, ndets_level, iexcit, iexcit), dtype=numpy.complex128)
    cof_mat = numpy.zeros((nwalkers, ndets_level, iexcit-1, iexcit-1), dtype=numpy.complex128)
    get_det_matrix_batched(
            iexcit,
            trial.cre_ex_b[iexcit],
            trial.anh_ex_b[iexcit],
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

if __name__ == '__main__':
    test_phmsd_local_energy()
