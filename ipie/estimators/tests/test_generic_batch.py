import pytest
import numpy
from ipie.utils.misc import dotdict
from ipie.trial_wavefunction.multi_slater import MultiSlater
from ipie.systems.generic import Generic
from ipie.propagation.continuous import Continuous
from ipie.propagation.force_bias import construct_force_bias_batch
from ipie.hamiltonians.generic import Generic as HamGeneric
from ipie.walkers.single_det_batch import SingleDetWalkerBatch
from ipie.walkers.multi_det_batch import MultiDetTrialWalkerBatch
from ipie.walkers.single_det import SingleDetWalker
from ipie.walkers.multi_det import MultiDetWalker
from ipie.utils.testing import (
        generate_hamiltonian,
        get_random_nomsd,
        get_random_phmsd
        )
from ipie.estimators.local_energy import local_energy_generic_cholesky_opt
from ipie.estimators.local_energy_batch import local_energy_single_det_batch, local_energy_single_det_rhf_batch
from ipie.estimators.greens_function_batch import greens_function_single_det_batch
from ipie.propagation.overlap import calc_overlap_single_det_batch

@pytest.mark.unit
def test_greens_function_batch():
    numpy.random.seed(7)
    nmo = 10
    nelec = (6,5)
    nwalkers = 12
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
    ovlp = greens_function_single_det_batch(walker_batch, trial)

    ot = [walkers[iw].ot for iw in range(walker_batch.nwalkers)]
    assert numpy.allclose(ovlp, ot)

    for iw in range(nwalkers):
        assert numpy.allclose(walker_batch.Ga[iw], walkers[iw].G[0])
        assert numpy.allclose(walker_batch.Gb[iw], walkers[iw].G[1])
        assert numpy.allclose(walker_batch.Ghalfa[iw], walkers[iw].Ghalf[0])
        assert numpy.allclose(walker_batch.Ghalfb[iw], walkers[iw].Ghalf[1])

@pytest.mark.unit
def test_local_energy_single_det_batch():
    numpy.random.seed(7)
    nmo = 10
    nelec = (5,5)
    nwalkers = 10
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

    etots = []
    e1s = []
    e2s = []
    for iw, walker in enumerate(walkers):
        e = local_energy_generic_cholesky_opt(system, ham, walker.G[0], walker.G[1], walker.Ghalf[0],walker.Ghalf[1], trial._rchola, trial._rcholb)
        etots += [e[0]]
        e1s += [e[1]]
        e2s += [e[2]]

    numpy.random.seed(7)
    qmc = dotdict({'dt': 0.005, 'nstblz': 5, 'batched': True, 'nwalkers': 10})
    prop = Continuous(system, ham, trial, qmc, options=options)
    walker_batch = SingleDetWalkerBatch(system, ham, trial, nwalkers)
    for i in range (nsteps):
        prop.propagate_walker_batch(walker_batch, system, ham, trial, trial.energy)
        walker_batch.reortho()

    energies = local_energy_single_det_batch(system, ham, walker_batch, trial)

    for iw in range(nwalkers):
        energy = local_energy_single_det_batch(system, ham, walker_batch, trial, iw = iw)
        assert numpy.allclose(walker_batch.phia[iw], walkers[iw].phi[:,:nelec[0]])
        assert numpy.allclose(walker_batch.phib[iw], walkers[iw].phi[:,nelec[0]:])
        assert numpy.allclose(energy, energies[iw])
        assert numpy.allclose(etots[iw], energies[iw,0])
        assert numpy.allclose(e1s[iw], energies[iw,1])
        assert numpy.allclose(e2s[iw], energies[iw,2])


@pytest.mark.unit
def test_local_energy_single_det_batch_rhf():
    numpy.random.seed(7)
    nmo = 10
    nelec = (5,5)
    nwalkers = 10
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

    etots = []
    e1s = []
    e2s = []
    for iw, walker in enumerate(walkers):
        e = local_energy_generic_cholesky_opt(system, ham, walker.G[0], walker.G[1], walker.Ghalf[0],walker.Ghalf[1], trial._rchola, trial._rcholb)
        etots += [e[0]]
        e1s += [e[1]]
        e2s += [e[2]]

    numpy.random.seed(7)
    qmc = dotdict({'dt': 0.005, 'nstblz': 5, 'batched': True, 'nwalkers': 10})
    prop = Continuous(system, ham, trial, qmc, options=options)
    walker_opts = dotdict({'rhf': True})
    walker_batch = SingleDetWalkerBatch(system, ham, trial, nwalkers, walker_opts=walker_opts)
    for i in range (nsteps):
        prop.propagate_walker_batch(walker_batch, system, ham, trial, trial.energy)
        walker_batch.reortho()

    energies = local_energy_single_det_rhf_batch(system, ham, walker_batch, trial)

    for iw in range(nwalkers):
        assert numpy.allclose(walker_batch.phia[iw], walkers[iw].phi[:,:nelec[0]])
        # assert numpy.allclose(energy, energies[iw])
        assert numpy.allclose(etots[iw], energies[iw,0])
        assert numpy.allclose(e1s[iw], energies[iw,1])
        assert numpy.allclose(e2s[iw], energies[iw,2])

if __name__ == '__main__':
    test_greens_function_batch()
    test_local_energy_single_det_batch()
    test_local_energy_single_det_batch_rhf()
