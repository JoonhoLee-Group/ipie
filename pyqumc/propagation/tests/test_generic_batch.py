import pytest
import numpy
from pyqumc.utils.misc import dotdict
from pyqumc.trial_wavefunction.multi_slater import MultiSlater
from pyqumc.systems.generic import Generic
from pyqumc.propagation.continuous import Continuous
from pyqumc.propagation.force_bias import construct_force_bias_batch
from pyqumc.hamiltonians.generic import Generic as HamGeneric
from pyqumc.walkers.single_det_batch import SingleDetWalkerBatch
from pyqumc.walkers.multi_det_batch import MultiDetTrialWalkerBatch
from pyqumc.walkers.single_det import SingleDetWalker
from pyqumc.walkers.multi_det import MultiDetWalker
from pyqumc.utils.testing import (
        generate_hamiltonian,
        get_random_nomsd,
        get_random_phmsd
        )
from pyqumc.estimators.greens_function_batch import greens_function_single_det_batch
from pyqumc.propagation.overlap import calc_overlap_single_det_batch

@pytest.mark.unit
def test_overlap_rhf_batch():
    numpy.random.seed(7)
    nmo = 10
    nelec = (5,5)
    nwalkers = 2
    nsteps = 25
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    system = Generic(nelec=nelec)
    ham = HamGeneric(h1e=numpy.array([h1e,h1e]),
                     chol=chol.reshape((-1,nmo*nmo)).T.copy(),
                     ecore=0)
    # Test PH type wavefunction.
    wfn, init = get_random_phmsd(system.nup, system.ndown, ham.nbasis, ndet=1, init=True)
    init[:,:nelec[0]] = init[:,nelec[0]:].copy()
    trial = MultiSlater(system, ham, wfn, init=init)
    trial.half_rotate(system, ham)
    trial.psi = trial.psi[0]
    trial.psia = trial.psia[0]
    trial.psib = trial.psib[0]
    trial.psib = trial.psia.copy()
    trial.psi[:,nelec[0]:] = trial.psia.copy()
    trial.calculate_energy(system, ham)

    numpy.random.seed(7)
    options = {'hybrid': True}
    qmc = dotdict({'dt': 0.005, 'nstblz': 5})
    prop = Continuous(system, ham, trial, qmc, options=options)

    walkers = [SingleDetWalker(system, ham, trial) for iw in range(nwalkers)]
    for i in range (nsteps):
        for walker in walkers:
            prop.propagate_walker(walker, system, ham, trial, trial.energy)
            detR = walker.reortho(trial) # reorthogonalizing to stablize
            walker.greens_function(trial)

    walker_opts = {'rhf': True}
    walker_batch = SingleDetWalkerBatch(system, ham, trial, nwalkers, walker_opts=walker_opts)
    for iw in range(nwalkers):
        walker_batch.phia[iw] = walkers[iw].phi[:,:nelec[0]].copy()
    
    ovlp = calc_overlap_single_det_batch(walker_batch, trial)
    ovlp_gf = greens_function_single_det_batch(walker_batch, trial)
    ot = [walkers[iw].ot for iw in range(walker_batch.nwalkers)]
    assert numpy.allclose(ovlp, ot)
    assert numpy.allclose(ovlp_gf, ot)

    for iw in range(nwalkers):
        assert numpy.allclose(walker_batch.Ga[iw], walkers[iw].G[0])
        assert numpy.allclose(walker_batch.Ghalfa[iw], walkers[iw].Ghalf[0])

@pytest.mark.unit
def test_overlap_batch():
    numpy.random.seed(7)
    nmo = 10
    nelec = (6,5)
    nwalkers = 2
    nsteps = 25
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    system = Generic(nelec=nelec)
    ham = HamGeneric(h1e=numpy.array([h1e,h1e]),
                     chol=chol.reshape((-1,nmo*nmo)).T.copy(),
                     ecore=0)
    # Test PH type wavefunction.
    wfn, init = get_random_phmsd(system.nup, system.ndown, ham.nbasis, ndet=1, init=True)
    trial = MultiSlater(system, ham, wfn, init=init)
    trial.half_rotate(system, ham)
    trial.psi = trial.psi[0]
    trial.psia = trial.psia[0]
    trial.psib = trial.psib[0]
    trial.calculate_energy(system, ham)

    numpy.random.seed(7)
    options = {'hybrid': True}
    qmc = dotdict({'dt': 0.005, 'nstblz': 5})
    prop = Continuous(system, ham, trial, qmc, options=options)

    walkers = [SingleDetWalker(system, ham, trial) for iw in range(nwalkers)]
    for i in range (nsteps):
        for walker in walkers:
            prop.propagate_walker(walker, system, ham, trial, trial.energy)
            detR = walker.reortho(trial) # reorthogonalizing to stablize
            walker.greens_function(trial)

    walker_batch = SingleDetWalkerBatch(system, ham, trial, nwalkers)
    for iw in range(nwalkers):
        walker_batch.phia[iw] = walkers[iw].phi[:,:nelec[0]].copy()
        walker_batch.phib[iw] = walkers[iw].phi[:,nelec[0]:].copy()
    
    ovlp = calc_overlap_single_det_batch(walker_batch, trial)
    ovlp_gf = greens_function_single_det_batch(walker_batch, trial)
    ot = [walkers[iw].ot for iw in range(walker_batch.nwalkers)]
    assert numpy.allclose(ovlp, ot)
    assert numpy.allclose(ovlp_gf, ot)

    for iw in range(nwalkers):
        assert numpy.allclose(walker_batch.Ga[iw], walkers[iw].G[0])
        assert numpy.allclose(walker_batch.Gb[iw], walkers[iw].G[1])
        assert numpy.allclose(walker_batch.Ghalfa[iw], walkers[iw].Ghalf[0])
        assert numpy.allclose(walker_batch.Ghalfb[iw], walkers[iw].Ghalf[1])

@pytest.mark.unit
def test_two_body_rhf_batch():
    numpy.random.seed(7)
    nmo = 10
    nelec = (5,5)
    nwalkers = 8
    nsteps = 25
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    system = Generic(nelec=nelec)
    ham = HamGeneric(h1e=numpy.array([h1e,h1e]),
                     chol=chol.reshape((-1,nmo*nmo)).T.copy(),
                     ecore=0)
    # Test PH type wavefunction.
    wfn, init = get_random_phmsd(system.nup, system.ndown, ham.nbasis, ndet=1, init=True)
    init[:,:nelec[0]] = init[:,nelec[0]:].copy()
    trial = MultiSlater(system, ham, wfn, init=init)
    trial.half_rotate(system, ham)
    trial.psi = trial.psi[0]
    trial.psia = trial.psia[0]
    trial.psib = trial.psib[0]
    trial.psib = trial.psia.copy()
    trial.psi[:,nelec[0]:] = trial.psia.copy()
    trial.calculate_energy(system, ham)

    numpy.random.seed(7)
    options = {'hybrid': True}
    qmc = dotdict({'dt': 0.005, 'nstblz': 5})
    prop = Continuous(system, ham, trial, qmc, options=options)

    walkers = [SingleDetWalker(system, ham, trial) for iw in range(nwalkers)]
    for i in range (nsteps):
        for walker in walkers:
            prop.two_body_propagator(walker, system, ham, trial)
            detR = walker.reortho(trial) # reorthogonalizing to stablize
            walker.greens_function(trial)

    numpy.random.seed(7)
    options = {'hybrid': True}
    qmc = dotdict({'dt': 0.005, 'nstblz': 5, 'batched': True, 'nwalkers':nwalkers})
    prop = Continuous(system, ham, trial, qmc, options=options)

    walker_opts = {'rhf': True}
    walker_batch = SingleDetWalkerBatch(system, ham, trial, nwalkers, walker_opts=walker_opts)
    for i in range (nsteps):
        prop.two_body_propagator_batch(walker_batch, system, ham, trial)
        detR = walker_batch.reortho() # reorthogonalizing to stablize
        prop.compute_greens_function(walker_batch, trial)

    for iw in range(nwalkers):
        assert numpy.allclose(walker_batch.Ga[iw], walkers[iw].G[0])
        assert numpy.allclose(walker_batch.Ghalfa[iw], walkers[iw].Ghalf[0])

@pytest.mark.unit
def test_two_body_batch():
    numpy.random.seed(7)
    nmo = 10
    nelec = (6,5)
    nwalkers = 2
    nsteps = 25
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    system = Generic(nelec=nelec)
    ham = HamGeneric(h1e=numpy.array([h1e,h1e]),
                     chol=chol.reshape((-1,nmo*nmo)).T.copy(),
                     ecore=0)
    # Test PH type wavefunction.
    wfn, init = get_random_phmsd(system.nup, system.ndown, ham.nbasis, ndet=1, init=True)
    trial = MultiSlater(system, ham, wfn, init=init)
    trial.half_rotate(system, ham)
    trial.psi = trial.psi[0]
    trial.psia = trial.psia[0]
    trial.psib = trial.psib[0]
    trial.calculate_energy(system, ham)

    numpy.random.seed(7)
    options = {'hybrid': True}
    qmc = dotdict({'dt': 0.005, 'nstblz': 5})
    prop = Continuous(system, ham, trial, qmc, options=options)

    walkers = [SingleDetWalker(system, ham, trial) for iw in range(nwalkers)]
    for i in range (nsteps):
        for walker in walkers:
            prop.two_body_propagator(walker, system, ham, trial)
            detR = walker.reortho(trial) # reorthogonalizing to stablize
            walker.greens_function(trial)

    numpy.random.seed(7)
    options = {'hybrid': True}
    qmc = dotdict({'dt': 0.005, 'nstblz': 5, 'batched': True, 'nwalkers':nwalkers})
    prop = Continuous(system, ham, trial, qmc, options=options)

    walker_batch = SingleDetWalkerBatch(system, ham, trial, nwalkers)
    for i in range (nsteps):
        prop.two_body_propagator_batch(walker_batch, system, ham, trial)
        detR = walker_batch.reortho() # reorthogonalizing to stablize
        prop.compute_greens_function(walker_batch, trial)

    for iw in range(nwalkers):
        assert numpy.allclose(walker_batch.Ga[iw], walkers[iw].G[0])
        assert numpy.allclose(walker_batch.Gb[iw], walkers[iw].G[1])
        assert numpy.allclose(walker_batch.Ghalfa[iw], walkers[iw].Ghalf[0])
        assert numpy.allclose(walker_batch.Ghalfb[iw], walkers[iw].Ghalf[1])

@pytest.mark.unit
def test_hybrid_rhf_batch():
    numpy.random.seed(7)
    nmo = 10
    nelec = (5,5)
    nwalkers = 8
    nsteps = 25
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    system = Generic(nelec=nelec)
    ham = HamGeneric(h1e=numpy.array([h1e,h1e]),
                     chol=chol.reshape((-1,nmo*nmo)).T.copy(),
                     ecore=0)
    # Test PH type wavefunction.
    wfn, init = get_random_phmsd(system.nup, system.ndown, ham.nbasis, ndet=1, init=True)
    init[:,:nelec[0]] = init[:,nelec[0]:].copy()
    trial = MultiSlater(system, ham, wfn, init=init)
    trial.half_rotate(system, ham)
    trial.psi = trial.psi[0]
    trial.psia = trial.psia[0]
    trial.psib = trial.psib[0]
    trial.psib = trial.psia.copy()
    trial.psi[:,nelec[0]:] = trial.psia.copy()
    trial.calculate_energy(system, ham)
    options = {'hybrid': True}
    qmc = dotdict({'dt': 0.005, 'nstblz': 5})
    prop = Continuous(system, ham, trial, qmc, options=options)

    walkers = [SingleDetWalker(system, ham, trial) for iw in range(nwalkers)]
    
    numpy.random.seed(7)
    for i in range (nsteps):
        for walker in walkers:
            prop.propagate_walker(walker, system, ham, trial, trial.energy)
            detR = walker.reortho(trial) # reorthogonalizing to stablize

    numpy.random.seed(7)
    qmc = dotdict({'dt': 0.005, 'nstblz': 5, 'batched': True, 'nwalkers': nwalkers})
    prop = Continuous(system, ham, trial, qmc, options=options)
    walker_batch = SingleDetWalkerBatch(system, ham, trial, nwalkers)
    for i in range (nsteps):
        prop.propagate_walker_batch(walker_batch, system, ham, trial, trial.energy)
        walker_batch.reortho()

    for iw in range(nwalkers):
        assert numpy.allclose(walker_batch.Ga[iw], walkers[iw].G[0])
        assert numpy.allclose(walker_batch.phia[iw], walkers[iw].phi[:,:nelec[0]])

@pytest.mark.unit
def test_hybrid_batch():
    numpy.random.seed(7)
    nmo = 10
    nelec = (6,5)
    nwalkers = 8
    nsteps = 25
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    system = Generic(nelec=nelec)
    ham = HamGeneric(h1e=numpy.array([h1e,h1e]),
                     chol=chol.reshape((-1,nmo*nmo)).T.copy(),
                     ecore=0)
    # Test PH type wavefunction.
    wfn, init = get_random_phmsd(system.nup, system.ndown, ham.nbasis, ndet=1, init=True)
    trial = MultiSlater(system, ham, wfn, init=init)
    trial.half_rotate(system, ham)
    trial.psi = trial.psi[0]
    trial.psia = trial.psia[0]
    trial.psib = trial.psib[0]

    numpy.random.seed(7)

    trial.calculate_energy(system, ham)
    options = {'hybrid': True}
    qmc = dotdict({'dt': 0.005, 'nstblz': 5})
    prop = Continuous(system, ham, trial, qmc, options=options)

    walkers = [SingleDetWalker(system, ham, trial) for iw in range(nwalkers)]
    for i in range (nsteps):
        for walker in walkers:
            prop.propagate_walker(walker, system, ham, trial, trial.energy)
            detR = walker.reortho(trial) # reorthogonalizing to stablize

    numpy.random.seed(7)
    qmc = dotdict({'dt': 0.005, 'nstblz': 5, 'batched': True, 'nwalkers': nwalkers})
    prop = Continuous(system, ham, trial, qmc, options=options)
    walker_batch = SingleDetWalkerBatch(system, ham, trial, nwalkers)
    for i in range (nsteps):
        prop.propagate_walker_batch(walker_batch, system, ham, trial, trial.energy)
        walker_batch.reortho()

    for iw in range(nwalkers):
        assert numpy.allclose(walker_batch.Ga[iw], walkers[iw].G[0])
        assert numpy.allclose(walker_batch.Gb[iw], walkers[iw].G[1])
        assert numpy.allclose(walker_batch.phia[iw], walkers[iw].phi[:,:nelec[0]])
        assert numpy.allclose(walker_batch.phib[iw], walkers[iw].phi[:,nelec[0]:])

if __name__ == '__main__':
    test_overlap_rhf_batch()
    test_overlap_batch()
    test_two_body_batch()
    test_two_body_rhf_batch()
    test_hybrid_rhf_batch()
    test_hybrid_batch()
