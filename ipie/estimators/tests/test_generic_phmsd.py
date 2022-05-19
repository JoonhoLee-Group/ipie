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
    assert numpy.allclose(walker_batch_wick.Ghalf0a, walker_batch_slow.Gihalfa[:,0])
    assert numpy.allclose(walker_batch_wick.Ghalf0b, walker_batch_slow.Gihalfb[:,0])
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
    wk.cofactor_matrix(
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
            wk.cofactor_matrix_4(
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
    wk.det_matrix(
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
def test_kernels():
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

def compute_alpha_ss(jdet, trial, Laa, cphasea, ovlpb, det_a):
    cont3 = 0.0
    nex_a = len(trial.cre_a[jdet])
    if nex_a == 2: # 4-leg same spin block aaaa
        p = trial.cre_a[jdet][0]
        q = trial.anh_a[jdet][0]
        r = trial.cre_a[jdet][1]
        s = trial.anh_a[jdet][1]
        const = cphasea * ovlpb
        cont3 += (numpy.dot(Laa[q,p,:],Laa[s,r,:])-numpy.dot(Laa[q,r,:],Laa[s,p,:])) * const
    elif nex_a > 2:
        cofactor = numpy.zeros((nex_a-2, nex_a-2), dtype=numpy.complex128)
        for iex in range(nex_a):
            for jex in range(nex_a):
                p = trial.cre_a[jdet][iex]
                q = trial.anh_a[jdet][jex]
                for kex in range(iex+1, nex_a):
                    for lex in range(jex+1, nex_a):
                        r = trial.cre_a[jdet][kex]
                        s = trial.anh_a[jdet][lex]
                        cofactor[:,:] = minor_mask4(det_a, iex, jex, kex, lex)
                        det_cofactor = (-1)**(kex+lex+iex+jex) * numpy.linalg.det(cofactor)
                        const = cphasea * det_cofactor * ovlpb
                        cont3 +=  (numpy.dot(Laa[q,p,:],Laa[s,r,:])-numpy.dot(Laa[q,r,:],Laa[s,p,:])) * const

    return cont3

def compute_beta_ss(jdet, trial, Lbb, cphaseb, ovlpa, det_b):
    cont3 = 0.0 + 0.0j
    nex_b = len(trial.cre_b[jdet])
    if (nex_b == 2): # 4-leg same spin block bbbb
        p = trial.cre_b[jdet][0]
        q = trial.anh_b[jdet][0]
        r = trial.cre_b[jdet][1]
        s = trial.anh_b[jdet][1]
        const = cphaseb * ovlpa
        cont3 += (numpy.dot(Lbb[q,p,:],Lbb[s,r,:])-numpy.dot(Lbb[q,r,:],Lbb[s,p,:])) * const
    elif (nex_b > 2):
        cofactor = numpy.zeros((nex_b-2, nex_b-2), dtype=numpy.complex128)
        for iex in range(nex_b):
            for jex in range(nex_b):
                p = trial.cre_b[jdet][iex]
                q = trial.anh_b[jdet][jex]
                for kex in range(iex+1,nex_b):
                    for lex in range(jex+1,nex_b):
                        r = trial.cre_b[jdet][kex]
                        s = trial.anh_b[jdet][lex]
                        cofactor[:,:] = minor_mask4(det_b, iex, jex, kex, lex)
                        det_cofactor = (-1)**(kex+lex+iex+jex)* numpy.linalg.det(cofactor)
                        # if jdet == 5:
                            # print(iex, jex, kex, lex,
                                    # (numpy.dot(Lbb[q,p,0],Lbb[s,r,0])-numpy.dot(Lbb[q,r,0],Lbb[s,p,0])),
                                    # cont3)
                            # print(det_b)
                            # print(cofactor)
                            # print("ovlp: ", iex, jex, kex, lex, p, q, r, s,
                                    # (-1)**(kex+lex+iex+jex), det_cofactor)
                        const = cphaseb * det_cofactor * ovlpa
                        # const = 1.0
                        # cont3 += (numpy.dot(Lbb[q,p,0],Lbb[s,r,0])-numpy.dot(Lbb[q,r,0],Lbb[s,p,0])) * const
                        cont3 += (numpy.dot(Lbb[q,p,:],Lbb[s,r,:])-numpy.dot(Lbb[q,r,:],Lbb[s,p,:])) * const
    return cont3

def compute_os(jdet, trial, Laa, Lbb, G0a, G0b, det_a, det_b, cphaseab):
    nex_a = len(trial.cre_a[jdet])
    nex_b = len(trial.cre_b[jdet])
    cont3 = 0.0 + 0.0j
    if (nex_a > 0 and nex_b > 0): # 2-leg opposite spin block
        if (nex_a == 1 and nex_b == 1):
            p = trial.cre_a[jdet][0]
            q = trial.anh_a[jdet][0]
            r = trial.cre_b[jdet][0]
            s = trial.anh_b[jdet][0]
            # if jdet == 80:
                # # print(p,q,r,s, cphaseab, cphaseab *
                        # numpy.dot(Laa[q,p,0],Lbb[s,r,0]), Laa[q,p,0], Lbb[s,r,0])
            cont3 += cphaseab * numpy.dot(Laa[q,p],Lbb[s,r])
        elif (nex_a == 2 and nex_b == 1):
            p = trial.cre_a[jdet][0]
            q = trial.anh_a[jdet][0]
            r = trial.cre_a[jdet][1]
            s = trial.anh_a[jdet][1]
            t = trial.cre_b[jdet][0]
            u = trial.anh_b[jdet][0]
            cofactor = [cphaseab * G0a[r,s],
            cphaseab * G0a[r,q],
            cphaseab * G0a[p,s],
            cphaseab * G0a[p,q]]
            cont3 += numpy.dot(Laa[q,p,:],Lbb[u,t,:]) * cofactor[0]
            cont3 -= numpy.dot(Laa[s,p,:],Lbb[u,t,:]) * cofactor[1]
            cont3 -= numpy.dot(Laa[q,r,:],Lbb[u,t,:]) * cofactor[2]
            cont3 += numpy.dot(Laa[s,r,:],Lbb[u,t,:]) * cofactor[3]
        elif (nex_a == 1 and nex_b == 2):
            p = trial.cre_a[jdet][0]
            q = trial.anh_a[jdet][0]
            r = trial.cre_b[jdet][0]
            s = trial.anh_b[jdet][0]
            t = trial.cre_b[jdet][1]
            u = trial.anh_b[jdet][1]
            cofactor = [cphaseab * G0b[t,u],
                        cphaseab * G0b[t,s],
                        cphaseab * G0b[r,u],
                        cphaseab * G0b[r,s]]
            cont3 +=  numpy.dot(Laa[q,p,:],Lbb[s,r,:]) * cofactor[0]
            cont3 -=  numpy.dot(Laa[q,p,:],Lbb[u,r,:]) * cofactor[1]
            cont3 -=  numpy.dot(Laa[q,p,:],Lbb[s,t,:]) * cofactor[2]
            cont3 +=  numpy.dot(Laa[q,p,:],Lbb[u,t,:]) * cofactor[3]
        elif (nex_a == 2 and nex_b == 2):
            p = trial.cre_a[jdet][0]
            q = trial.anh_a[jdet][0]
            r = trial.cre_a[jdet][1]
            s = trial.anh_a[jdet][1]

            t = trial.cre_b[jdet][0]
            u = trial.anh_b[jdet][0]
            v = trial.cre_b[jdet][1]
            w = trial.anh_b[jdet][1]
            cofactor = [cphaseab * G0a[r,s] * G0b[v,w],
                        cphaseab * G0a[r,q] * G0b[v,w],
                        cphaseab * G0a[p,s] * G0b[v,w],
                        cphaseab * G0a[p,q] * G0b[v,w],
                        cphaseab * G0a[r,s] * G0b[t,u],
                        cphaseab * G0a[r,q] * G0b[t,u],
                        cphaseab * G0a[p,s] * G0b[t,u],
                        cphaseab * G0a[p,q] * G0b[t,u],
                        cphaseab * G0a[r,s] * G0b[v,u],
                        cphaseab * G0a[r,q] * G0b[v,u],
                        cphaseab * G0a[p,s] * G0b[v,u],
                        cphaseab * G0a[p,q] * G0b[v,u],
                        cphaseab * G0a[r,s] * G0b[t,w],
                        cphaseab * G0a[r,q] * G0b[t,w],
                        cphaseab * G0a[p,s] * G0b[t,w],
                        cphaseab * G0a[p,q] * G0b[t,w]]
            cont3 += numpy.dot(Laa[q,p,:],Lbb[u,t,:]) * cofactor[0]
            cont3 -= numpy.dot(Laa[s,p,:],Lbb[u,t,:]) * cofactor[1]
            cont3 -= numpy.dot(Laa[q,r,:],Lbb[u,t,:]) * cofactor[2]
            cont3 += numpy.dot(Laa[s,r,:],Lbb[u,t,:]) * cofactor[3]

            cont3 += numpy.dot(Laa[q,p,:],Lbb[w,v,:]) * cofactor[4]
            cont3 -= numpy.dot(Laa[s,p,:],Lbb[w,v,:]) * cofactor[5]
            cont3 -= numpy.dot(Laa[q,r,:],Lbb[w,v,:]) * cofactor[6]
            cont3 += numpy.dot(Laa[s,r,:],Lbb[w,v,:]) * cofactor[7]

            cont3 -= numpy.dot(Laa[q,p,:],Lbb[w,t,:]) * cofactor[8]
            cont3 += numpy.dot(Laa[s,p,:],Lbb[w,t,:]) * cofactor[9]
            cont3 += numpy.dot(Laa[q,r,:],Lbb[w,t,:]) * cofactor[10]
            cont3 -= numpy.dot(Laa[s,r,:],Lbb[w,t,:]) * cofactor[11]

            cont3 -= numpy.dot(Laa[q,p,:],Lbb[u,v,:]) * cofactor[12]
            cont3 += numpy.dot(Laa[s,p,:],Lbb[u,v,:]) * cofactor[13]
            cont3 += numpy.dot(Laa[q,r,:],Lbb[u,v,:]) * cofactor[14]
            cont3 -= numpy.dot(Laa[s,r,:],Lbb[u,v,:]) * cofactor[15]

        elif (nex_a == 3 and nex_b == 1):
            p = trial.cre_a[jdet][0]
            q = trial.anh_a[jdet][0]
            r = trial.cre_a[jdet][1]
            s = trial.anh_a[jdet][1]
            t = trial.cre_a[jdet][2]
            u = trial.anh_a[jdet][2]
            v = trial.cre_b[jdet][0]
            w = trial.anh_b[jdet][0]
            cofactor = [G0a[r,s]*G0a[t,u] - G0a[r,u]*G0a[t,s],
                        G0a[r,q]*G0a[t,u] - G0a[r,u]*G0a[t,q],
                        G0a[r,q]*G0a[t,s] - G0a[r,s]*G0a[t,q],
                        G0a[p,s]*G0a[t,u] - G0a[t,s]*G0a[p,u],
                        G0a[p,q]*G0a[t,u] - G0a[t,q]*G0a[p,u],
                        G0a[p,q]*G0a[t,s] - G0a[t,q]*G0a[p,s],
                        G0a[p,s]*G0a[r,u] - G0a[r,s]*G0a[p,u],
                        G0a[p,q]*G0a[r,u] - G0a[r,q]*G0a[p,u],
                        G0a[p,q]*G0a[r,s] - G0a[r,q]*G0a[p,s]]
            cont3 += cphaseab * numpy.dot(Laa[q,p], Lbb[w,v]) * cofactor[0]
            cont3 -= cphaseab * numpy.dot(Laa[s,p], Lbb[w,v]) * cofactor[1]
            cont3 += cphaseab * numpy.dot(Laa[u,p], Lbb[w,v]) * cofactor[2]
            cont3 -= cphaseab * numpy.dot(Laa[q,r], Lbb[w,v]) * cofactor[3]
            cont3 += cphaseab * numpy.dot(Laa[s,r], Lbb[w,v]) * cofactor[4]
            cont3 -= cphaseab * numpy.dot(Laa[u,r], Lbb[w,v]) * cofactor[5]
            cont3 += cphaseab * numpy.dot(Laa[q,t], Lbb[w,v]) * cofactor[6]
            cont3 -= cphaseab * numpy.dot(Laa[s,t], Lbb[w,v]) * cofactor[7]
            cont3 += cphaseab * numpy.dot(Laa[u,t], Lbb[w,v]) * cofactor[8]
        elif (nex_a == 1 and nex_b == 3):
            p = trial.cre_b[jdet][0]
            q = trial.anh_b[jdet][0]
            r = trial.cre_b[jdet][1]
            s = trial.anh_b[jdet][1]
            t = trial.cre_b[jdet][2]
            u = trial.anh_b[jdet][2]
            v = trial.cre_a[jdet][0]
            w = trial.anh_a[jdet][0]
            cofactor = [G0b[r,s]*G0b[t,u] - G0b[r,u]*G0b[t,s],
                       G0b[r,q]*G0b[t,u] - G0b[r,u]*G0b[t,q],
                       G0b[r,q]*G0b[t,s] - G0b[r,s]*G0b[t,q],
                       G0b[p,s]*G0b[t,u] - G0b[t,s]*G0b[p,u],
                       G0b[p,q]*G0b[t,u] - G0b[t,q]*G0b[p,u],
                       G0b[p,q]*G0b[t,s] - G0b[t,q]*G0b[p,s],
                       G0b[p,s]*G0b[r,u] - G0b[r,s]*G0b[p,u],
                       G0b[p,q]*G0b[r,u] - G0b[r,q]*G0b[p,u],
                       G0b[p,q]*G0b[r,s] - G0b[r,q]*G0b[p,s]]

            cont3 += cphaseab * numpy.dot(Lbb[q,p],Laa[w,v]) * cofactor[0]
            cont3 -= cphaseab * numpy.dot(Lbb[s,p],Laa[w,v]) * cofactor[1]
            cont3 += cphaseab * numpy.dot(Lbb[u,p],Laa[w,v]) * cofactor[2]
            cont3 -= cphaseab * numpy.dot(Lbb[q,r],Laa[w,v]) * cofactor[3]
            cont3 += cphaseab * numpy.dot(Lbb[s,r],Laa[w,v]) * cofactor[4]
            cont3 -= cphaseab * numpy.dot(Lbb[u,r],Laa[w,v]) * cofactor[5]
            cont3 += cphaseab * numpy.dot(Lbb[q,t],Laa[w,v]) * cofactor[6]
            cont3 -= cphaseab * numpy.dot(Lbb[s,t],Laa[w,v]) * cofactor[7]
            cont3 += cphaseab * numpy.dot(Lbb[u,t],Laa[w,v]) * cofactor[8]

        else:
            cofactor_a = numpy.zeros((nex_a-1, nex_a-1), dtype=numpy.complex128)
            cofactor_b = numpy.zeros((nex_b-1, nex_b-1), dtype=numpy.complex128)
            for iex in range(nex_a):
                for jex in range(nex_a):
                    p = trial.cre_a[jdet][iex]
                    q = trial.anh_a[jdet][jex]
                    cofactor_a[:,:] = minor_mask(det_a, iex, jex)
                    det_cofactor_a = (-1)**(iex+jex)* numpy.linalg.det(cofactor_a)
                    for kex in range(nex_b):
                        for lex in range(nex_b):
                            r = trial.cre_b[jdet][kex]
                            s = trial.anh_b[jdet][lex]
                            cofactor_b[:,:] = minor_mask(det_b, kex, lex)
                            det_cofactor_b = (-1)**(kex+lex)* numpy.linalg.det(cofactor_b)
                            const = cphaseab * det_cofactor_a * det_cofactor_b
                            cont3 += numpy.dot(Laa[q,p,:], Lbb[s,r,:]) * const

    return cont3

# @pytest.mark.unit
# def test_same_spin_batched():
    # numpy.random.seed(7)
    # nmo = 12
    # nelec = (7,7)
    # nwalkers = 1
    # nsteps = 100
    # h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    # system = Generic(nelec=nelec)
    # ham = HamGeneric(h1e=numpy.array([h1e,h1e]),
                     # chol=chol.reshape((-1,nmo*nmo)).T.copy(),
                     # ecore=0,options = {"symmetry":False})
    # wfn, init = get_random_phmsd(system.nup, system.ndown, ham.nbasis, ndet=3000, init=True)
    # ci, oa, ob = wfn
    # # wfn_2 = ([ci[0],ci[9]], [oa[0], oa[9]], [ob[0], ob[9]])
    # wfn_2 = (ci[::100], oa[::100], ob[::100])
    # trial = MultiSlater(system, ham, wfn_2, init=init, options = {'wicks':True})
    # # trial.calculate_energy(system, ham)
    # options = {'hybrid': True}
    # # qmc = dotdict({'dt': 0.005, 'nstblz': 5})
    # # prop = Continuous(system, ham, trial, qmc, options=options)

    # numpy.random.seed(7)
    # qmc = dotdict({'dt': 0.005, 'nstblz': 5, 'batched': True, 'nwalkers':
        # nwalkers})
    # prop = Continuous(system, ham, trial, qmc, options=options)
    # walker_batch = MultiDetTrialWalkerBatch(system, ham, trial, nwalkers)
    # for i in range (nsteps):
        # prop.propagate_walker_batch(walker_batch, system, ham, trial, 0.0)
        # walker_batch.reortho()

    # greens_function_multi_det_wicks(walker_batch, trial) # compute green's function using Wick's theorem
    # same_spin = []
    # nbasis = nmo
    # nchol = ham.chol_vecs.shape[-1]
    # for iwalker in range(nwalkers):
        # ovlpa0 = walker_batch.det_ovlpas[iwalker,0]
        # ovlpb0 = walker_batch.det_ovlpbs[iwalker,0]
        # ovlp0 = ovlpa0 * ovlpb0
        # ovlp = walker_batch.ovlp[iwalker]

        # # useful variables
        # G0a = walker_batch.G0a[iwalker]
        # G0b = walker_batch.G0b[iwalker]
        # Q0a = walker_batch.Q0a[iwalker]
        # Q0b = walker_batch.Q0b[iwalker]
        # CIa = walker_batch.CIa[iwalker]
        # CIb = walker_batch.CIb[iwalker]
        # G0 = [G0a, G0b]

        # # contribution 1 (disconnected)
        # # cont1 = local_energy_generic_cholesky(system, ham, G0)[2]

        # # contribution 2 (half-connected, two-leg, one-body-like)
        # # First, Coulomb-like term
        # P0 = G0[0] + G0[1]
        # Xa = ham.chol_vecs.T.dot(G0[0].ravel()) #numpy.einsum("m,xm->x", G0[0].ravel(), ham.chol_vecs)
        # Xb = ham.chol_vecs.T.dot(G0[1].ravel()) #numpy.einsum("m,xm->x", G0[1].ravel(), ham.chol_vecs)

        # LXa = numpy.einsum("x,mx->m", Xa, ham.chol_vecs, optimize=True)
        # LXb = numpy.einsum("x,mx->m", Xb, ham.chol_vecs, optimize=True)
        # LXa = LXa.reshape((nbasis,nbasis))
        # LXb = LXb.reshape((nbasis,nbasis))

        # # useful intermediate
        # QCIGa = Q0a.dot(CIa).dot(G0a)
        # QCIGb = Q0b.dot(CIb).dot(G0b)

        # cont2_Jaa = numpy.sum(QCIGa * LXa)
        # cont2_Jbb = numpy.sum(QCIGb * LXb)
        # cont2_Jab = numpy.sum(QCIGb * LXa) + numpy.sum(QCIGa * LXb)
        # cont2_J = cont2_Jaa + cont2_Jbb + cont2_Jab
        # cont2_J *= (ovlp0/ovlp)

        # # Second, Exchange-like term
        # cont2_Kaa = 0.0 + 0.0j
        # cont2_Kbb = 0.0 + 0.0j
        # for x in range(nchol):
            # Lmn = ham.chol_vecs[:,x].reshape((nbasis, nbasis))
            # LGL = Lmn.dot(G0a.T).dot(Lmn)
            # cont2_Kaa -= numpy.sum(LGL*QCIGa)

            # LGL = Lmn.dot(G0b.T).dot(Lmn)
            # cont2_Kbb -= numpy.sum(LGL*QCIGb)

        # cont2_Kaa *= (ovlp0/ovlp)
        # cont2_Kbb *= (ovlp0/ovlp)

        # cont2_K = cont2_Kaa + cont2_Kbb

        # Laa = numpy.einsum("iq,pj,ijx->qpx",Q0a, G0a, ham.chol_vecs.reshape((nbasis, nbasis, nchol)), optimize=True)
        # Lbb = numpy.einsum("iq,pj,ijx->qpx",Q0b, G0b, ham.chol_vecs.reshape((nbasis, nbasis, nchol)), optimize=True)

        # cont3 = 0.0 + 0.0j

        # ss_alpha = numpy.zeros((nwalkers, trial.ndets), dtype=numpy.complex128)
        # ss_beta = numpy.zeros((nwalkers, trial.ndets), dtype=numpy.complex128)
        # opp_spin = numpy.zeros((nwalkers, trial.ndets), dtype=numpy.complex128)
        # for iwalker in range(nwalkers):
            # ovlpa0 = walker_batch.det_ovlpas[iwalker,0]
            # ovlpb0 = walker_batch.det_ovlpbs[iwalker,0]
            # ovlp0 = ovlpa0 * ovlpb0
            # ovlp = walker_batch.ovlp[iwalker]

            # # useful variables
            # G0a = walker_batch.G0a[iwalker]
            # G0b = walker_batch.G0b[iwalker]
            # Q0a = walker_batch.Q0a[iwalker]
            # Q0b = walker_batch.Q0b[iwalker]
            # CIa = walker_batch.CIa[iwalker]
            # CIb = walker_batch.CIb[iwalker]
            # G0 = [G0a, G0b]

            # # contribution 1 (disconnected)
            # # cont1 = local_energy_generic_cholesky(system, ham, G0)[2]

            # # contribution 2 (half-connected, two-leg, one-body-like)
            # # First, Coulomb-like term
            # P0 = G0[0] + G0[1]
            # Xa = ham.chol_vecs.T.dot(G0[0].ravel()) #numpy.einsum("m,xm->x", G0[0].ravel(), ham.chol_vecs)
            # Xb = ham.chol_vecs.T.dot(G0[1].ravel()) #numpy.einsum("m,xm->x", G0[1].ravel(), ham.chol_vecs)

            # LXa = numpy.einsum("x,mx->m", Xa, ham.chol_vecs, optimize=True)
            # LXb = numpy.einsum("x,mx->m", Xb, ham.chol_vecs, optimize=True)
            # LXa = LXa.reshape((nbasis,nbasis))
            # LXb = LXb.reshape((nbasis,nbasis))

            # # useful intermediate
            # QCIGa = Q0a.dot(CIa).dot(G0a)
            # QCIGb = Q0b.dot(CIb).dot(G0b)

            # cont2_Jaa = numpy.sum(QCIGa * LXa)
            # cont2_Jbb = numpy.sum(QCIGb * LXb)
            # cont2_Jab = numpy.sum(QCIGb * LXa) + numpy.sum(QCIGa * LXb)
            # cont2_J = cont2_Jaa + cont2_Jbb + cont2_Jab
            # cont2_J *= (ovlp0/ovlp)

            # # Second, Exchange-like term
            # cont2_Kaa = 0.0 + 0.0j
            # cont2_Kbb = 0.0 + 0.0j
            # for x in range(nchol):
                # Lmn = ham.chol_vecs[:,x].reshape((nbasis, nbasis))
                # LGL = Lmn.dot(G0a.T).dot(Lmn)
                # cont2_Kaa -= numpy.sum(LGL*QCIGa)

                # LGL = Lmn.dot(G0b.T).dot(Lmn)
                # cont2_Kbb -= numpy.sum(LGL*QCIGb)

            # cont2_Kaa *= (ovlp0/ovlp)
            # cont2_Kbb *= (ovlp0/ovlp)

            # cont2_K = cont2_Kaa + cont2_Kbb

            # Laa = numpy.einsum("iq,pj,ijx->qpx",Q0a, G0a, ham.chol_vecs.reshape((nbasis, nbasis, nchol)), optimize=True)
            # Lbb = numpy.einsum("iq,pj,ijx->qpx",Q0b, G0b, ham.chol_vecs.reshape((nbasis, nbasis, nchol)), optimize=True)

            # cont3 = 0.0 + 0.0j

            # for jdet in range(1, trial.ndets):
                # nex_a = len(trial.cre_a[jdet])
                # nex_b = len(trial.cre_b[jdet])

                # ovlpa, ovlpb = get_overlap_one_det_wicks(nex_a, trial.cre_a[jdet], trial.anh_a[jdet], G0a,\
                    # nex_b, trial.cre_b[jdet], trial.anh_b[jdet], G0b)
                # ovlpa *= trial.phase_a[jdet]
                # ovlpb *= trial.phase_b[jdet]

                # det_a = numpy.zeros((nex_a,nex_a), dtype=numpy.complex128)
                # det_b = numpy.zeros((nex_b,nex_b), dtype=numpy.complex128)

                # for iex in range(nex_a):
                    # det_a[iex,iex] = G0a[trial.cre_a[jdet][iex],trial.anh_a[jdet][iex]]
                    # for jex in range(iex+1, nex_a):
                        # det_a[iex, jex] = G0a[trial.cre_a[jdet][iex],trial.anh_a[jdet][jex]]
                        # det_a[jex, iex] = G0a[trial.cre_a[jdet][jex],trial.anh_a[jdet][iex]]
                # for iex in range(nex_b):
                    # det_b[iex,iex] = G0b[trial.cre_b[jdet][iex],trial.anh_b[jdet][iex]]
                    # for jex in range(iex+1, nex_b):
                        # det_b[iex, jex] = G0b[trial.cre_b[jdet][iex],trial.anh_b[jdet][jex]]
                        # det_b[jex, iex] = G0b[trial.cre_b[jdet][jex],trial.anh_b[jdet][iex]]

                # cphasea = trial.coeffs[jdet].conj() * trial.phase_a[jdet]
                # cphaseb = trial.coeffs[jdet].conj() * trial.phase_b[jdet]
                # cphaseab = trial.coeffs[jdet].conj() * trial.phase_a[jdet] * trial.phase_b[jdet]

                # cont3_alpha = compute_alpha_ss(jdet, trial, Laa,
                        # cphasea, ovlpb, det_a)
                # ss_alpha[iwalker, jdet] = cont3_alpha
                # cont3_beta = compute_beta_ss(jdet, trial, Lbb,
                        # cphaseb, ovlpa, det_b)
                # ss_beta[iwalker, jdet] = cont3_beta
                # opp_spin[iwalker, jdet] = compute_os(jdet, trial, Laa, Lbb, G0a,
                        # G0b, det_a, det_b, cphaseab)
                # cont3 += cont3_alpha + cont3_beta + opp_spin[iwalker, jdet]
                # # if jdet == 80:
                    # # print(opp_spin[iwalker,jdet], Laa[7,6,0], Lbb[9, 5, 0])

    # # print("here: ", cont3, sum(opp_spin[0]), sum(ss_alpha[0]), sum(ss_beta[0]), sum(ss_alpha[0]+ss_beta[0]+opp_spin[0]))
    # # print(ss_alpha[1])
    # # print(ss_beta[1])
    # G0a = walker_batch.G0a
    # G0b = walker_batch.G0b
    # Q0a = walker_batch.Q0a
    # Q0b = walker_batch.Q0b
    # CIa = walker_batch.CIa
    # CIb = walker_batch.CIb
    # dets_a_full, dets_b_full = compute_determinants_batched(G0a, G0b, trial)

    # ndets = len(trial.coeffs)
    # chol_vecs = ham.chol_vecs.reshape((nchol, nbasis, nbasis))
    # energy_aa = numpy.zeros((nwalkers,ndets), dtype=numpy.complex128)
    # energy_bb = numpy.zeros((nwalkers,ndets), dtype=numpy.complex128)
    # energy_os = numpy.zeros((nwalkers,ndets), dtype=numpy.complex128)
    # cphase_a = trial.coeffs.conj() * trial.phase_a
    # cphase_b = trial.coeffs.conj() * trial.phase_b
    # ovlpa = dets_a_full * trial.phase_a[None,:]
    # ovlpb = dets_b_full * trial.phase_b[None,:]
    # c_phasea_ovlpb = cphase_a[None,:] * ovlpb
    # c_phaseb_ovlpa = cphase_b[None,:] * ovlpa
    # cphase_ab = cphase_a * trial.phase_b
    # out = numpy.zeros_like(ovlpb)
    # chol_vecs = ham.chol_vecs.reshape((nbasis, nbasis, -1))
    # na = walker_batch.nup
    # nb = walker_batch.ndown
    # for ix in range(nchol):
        # # print(Q0a.shape, G0a.shape, Lx.shape)
        # Lx = chol_vecs[:,:,ix]
        # Laa = numpy.einsum('wiq,wpj,ij->wqp', Q0a, G0a, Lx, optimize=True)
        # Lbb = numpy.einsum('wiq,wpj,ij->wqp', Q0b, G0b, Lx, optimize=True)
        # # Same-spin contributions
        # alpha_os_buffer = numpy.zeros((nwalkers, ndets), dtype=numpy.complex128)
        # beta_os_buffer = numpy.zeros((nwalkers, ndets), dtype=numpy.complex128)
        # alpha_ss_buffer = numpy.zeros((nwalkers, ndets), dtype=numpy.complex128)
        # beta_ss_buffer = numpy.zeros((nwalkers, ndets), dtype=numpy.complex128)
        # for iexcit in range(1, max(na, nb)):
            # ndets_a = len(trial.cre_ex_a[iexcit])
            # det_mat_a = numpy.zeros((nwalkers, ndets_a, iexcit, iexcit), dtype=numpy.complex128)
            # get_det_matrix_batched(
                    # iexcit,
                    # trial.cre_ex_a[iexcit],
                    # trial.anh_ex_a[iexcit],
                    # walker_batch.G0a,
                    # det_mat_a)
            # # defined here for reuse
            # cofactor_matrix_a = numpy.zeros((nwalkers, ndets_a, max(iexcit-1,1), max(iexcit-1,1)), dtype=numpy.complex128)
            # if ndets_a > 0:
                # # print(iexcit, ndets_a)
                # fill_opp_spin_factors_batched(
                                # iexcit,
                                # trial.cre_ex_a[iexcit],
                                # trial.anh_ex_a[iexcit],
                                # trial.excit_map_a,
                                # det_mat_a,
                                # cofactor_matrix_a,
                                # Laa,
                                # alpha_os_buffer)
            # if iexcit >= 2 and ndets_a > 0:
                # fill_same_spin_contribution_batched(
                                    # iexcit,
                                    # trial.cre_ex_a[iexcit],
                                    # trial.anh_ex_a[iexcit],
                                    # det_mat_a,
                                    # cofactor_matrix_a[:,:,:max(iexcit-2,1),:max(iexcit-2,1)],
                                    # Laa,
                                    # alpha_ss_buffer,
                                    # trial.excit_map_a
                                    # )
            # ndets_b = len(trial.cre_ex_b[iexcit])
            # det_mat_b = numpy.zeros((nwalkers, ndets_b, iexcit, iexcit), dtype=numpy.complex128)
            # get_det_matrix_batched(
                    # iexcit,
                    # trial.cre_ex_b[iexcit],
                    # trial.anh_ex_b[iexcit],
                    # walker_batch.G0b,
                    # det_mat_b)
            # cofactor_matrix_b = numpy.zeros((nwalkers, ndets_b, max(iexcit-1,1), max(iexcit-1,1)), dtype=numpy.complex128)
            # if ndets_b > 0:
                # fill_opp_spin_factors_batched(
                                # iexcit,
                                # trial.cre_ex_b[iexcit],
                                # trial.anh_ex_b[iexcit],
                                # trial.excit_map_b,
                                # det_mat_b,
                                # cofactor_matrix_b,
                                # Lbb,
                                # beta_os_buffer)
            # if iexcit >= 2 and ndets_b > 0:
                # fill_same_spin_contribution_batched(
                                    # iexcit,
                                    # trial.cre_ex_b[iexcit],
                                    # trial.anh_ex_b[iexcit],
                                    # det_mat_b,
                                    # cofactor_matrix_b[:,:,:max(iexcit-2,1),:max(iexcit-2,1)],
                                    # Lbb,
                                    # beta_ss_buffer,
                                    # trial.excit_map_b
                                    # )
            # # ndets_b = len(trial.cre_ex_b[iexcit])
            # # det_mat_b = numpy.zeros((nwalkers, ndets_b, iexcit, iexcit), dtype=numpy.complex128)
            # # get_det_matrix_batched(
                    # # iexcit,
                    # # trial.cre_ex_b[iexcit],
                    # # trial.anh_ex_b[iexcit],
                    # # walker_batch.G0b,
                    # # det_mat_b)
            # # cofactor_matrix_b = numpy.zeros((nwalkers, ndets_b, iexcit-1, iexcit-1), dtype=numpy.complex128)
            # # if ndets_b > 0:
                # # fill_opp_spin_factors_batched(
                                # # iexcit,
                                # # trial.cre_ex_b[iexcit],
                                # # trial.anh_ex_b[iexcit],
                                # # trial.excit_map_b,
                                # # det_mat_b,
                                # # cofactor_matrix_b,
                                # # Lbb,
                                # # beta_os_buffer)
            # # if iexcit >= 2 and ndets_b > 0:
                # # # if ix == 0 and iexcit == 3:
                # # cofactor_matrix_b_ = cofactor_matrix_b[:,:,:max(iexcit-2,1),:max(iexcit-2,1)]
                # # fill_same_spin_contribution_batched(
                                    # # iexcit,
                                    # # trial.cre_ex_b[iexcit],
                                    # # trial.anh_ex_b[iexcit],
                                    # # det_mat_b,
                                    # # cofactor_matrix_b_,
                                    # # Lbb,
                                    # # beta_ss_buffer,
                                    # # trial.excit_map_b
                                    # # )

        # # accumulating over x (cholesky vector)
        # energy_os += numpy.einsum('wJ,J->wJ', alpha_os_buffer*beta_os_buffer, cphase_ab, optimize=True)
        # energy_aa += numpy.einsum('wJ,wJ->wJ', alpha_ss_buffer, c_phasea_ovlpb, optimize=True)
        # energy_bb += numpy.einsum('wJ,wJ->wJ', beta_ss_buffer, c_phaseb_ovlpa, optimize=True)
        # # energy_ss += beta_ss_buffer

    # # print(
    # # print(sum(energy_os[0]+energy_aa[0]+energy_bb[0]))
    # # print("this: ", sum(energy_os[0]), sum(energy_aa[0]), sum(energy_bb[0]))
    # # print("this: ", sum(opp_spin[0]), sum(ss_alpha[0]), sum(ss_beta[0]))
    # # print(energy_os+energy_aa+energy_bb)
    # assert numpy.allclose(energy_aa, ss_alpha)
    # assert numpy.allclose(energy_bb, ss_beta)
    # assert numpy.allclose(energy_os, opp_spin)

if __name__ == '__main__':
    test_phmsd_local_energy()
