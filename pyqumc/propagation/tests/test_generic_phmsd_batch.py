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

@pytest.mark.unit
def test_phmsd_force_bias_batch():
    numpy.random.seed(7)
    nmo = 10
    nelec = (5,4)
    nwalkers = 10
    ndets = 5
    nsteps = 100
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    nchols = chol.shape[0]
    system = Generic(nelec=nelec)
    ham = HamGeneric(h1e=numpy.array([h1e,h1e]),
                     chol=chol.reshape((-1,nmo*nmo)).T.copy(),
                     ecore=0)
    # Test PH type wavefunction.
    wfn, init = get_random_phmsd(system.nup, system.ndown, ham.nbasis, ndet=ndets, init=True)
    trial = MultiSlater(system, ham, wfn, init=init)

    numpy.random.seed(7)

    trial.calculate_energy(system, ham)
    options = {'hybrid': True}
    qmc = dotdict({'dt': 0.005, 'nstblz': 5})
    prop = Continuous(system, ham, trial, qmc, options=options)

    walkers = [MultiDetWalker(system, ham, trial) for iw in range(nwalkers)]
    fb_ref = numpy.zeros((nwalkers, nchols), dtype=numpy.complex128)
    for iw in range(nwalkers):
        fb_ref[iw,:] = prop.propagator.construct_force_bias(ham, walkers[iw], trial)

    numpy.random.seed(7)
    qmc = dotdict({'dt': 0.005, 'nstblz': 5, 'batched':True,'nwalkers':nwalkers})
    prop = Continuous(system, ham, trial, qmc, options=options)

    walker_batch = MultiDetTrialWalkerBatch(system, ham, trial, nwalkers)
    prop.propagator.vbias_batch = construct_force_bias_batch(ham, walker_batch, trial)
    fb = - prop.propagator.sqrt_dt * (1j*prop.propagator.vbias_batch-prop.propagator.mf_shift)

    for iw in range(nwalkers):
        assert numpy.allclose(fb_ref[iw], fb[iw])

@pytest.mark.unit
def test_phmsd_greens_function_batch():
    numpy.random.seed(7)
    nmo = 10
    nelec = (5,4)
    nwalkers = 1
    ndets = 5
    nsteps = 100
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    nchols = chol.shape[0]
    system = Generic(nelec=nelec)
    ham = HamGeneric(h1e=numpy.array([h1e,h1e]),
                     chol=chol.reshape((-1,nmo*nmo)).T.copy(),
                     ecore=0)
    # Test PH type wavefunction.
    wfn, init = get_random_phmsd(system.nup, system.ndown, ham.nbasis, ndet=ndets, init=True)
    trial = MultiSlater(system, ham, wfn, init=init)

    numpy.random.seed(7)

    trial.calculate_energy(system, ham)
    options = {'hybrid': True}
    qmc = dotdict({'dt': 0.005, 'nstblz': 5})
    prop = Continuous(system, ham, trial, qmc, options=options)

    walkers = [MultiDetWalker(system, ham, trial) for iw in range(nwalkers)]
    walker_batch = MultiDetTrialWalkerBatch(system, ham, trial, nwalkers)

    for iw in range(nwalkers):
        assert numpy.allclose(walker_batch.Gia[iw], walkers[iw].Gi[:,0,:,:])
        assert numpy.allclose(walker_batch.Gib[iw], walkers[iw].Gi[:,1,:,:])
        assert numpy.allclose(walker_batch.Ga[iw], walkers[iw].G[0,:,:])
        assert numpy.allclose(walker_batch.Gb[iw], walkers[iw].G[1,:,:])

@pytest.mark.unit
def test_phmsd_overlap_batch():
    numpy.random.seed(70)
    nmo = 10
    nelec = (5,4)
    nwalkers = 1
    ndets = 5
    nsteps = 100
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    nchols = chol.shape[0]
    system = Generic(nelec=nelec)
    ham = HamGeneric(h1e=numpy.array([h1e,h1e]),
                     chol=chol.reshape((-1,nmo*nmo)).T.copy(),
                     ecore=0)
    # Test PH type wavefunction.
    wfn, init = get_random_phmsd(system.nup, system.ndown, ham.nbasis, ndet=ndets, init=True)
    trial = MultiSlater(system, ham, wfn, init=init)

    numpy.random.seed(70)

    trial.calculate_energy(system, ham)
    options = {'hybrid': True}
    qmc = dotdict({'dt': 0.005, 'nstblz': 5})
    prop = Continuous(system, ham, trial, qmc, options=options)

    walkers = [MultiDetWalker(system, ham, trial) for iw in range(nwalkers)]
    walker_batch = MultiDetTrialWalkerBatch(system, ham, trial, nwalkers)

    for iw in range(nwalkers):
        assert numpy.allclose(walkers[iw].ovlp,walker_batch.ovlp[iw])

@pytest.mark.unit
def test_phmsd_propagation_batch():
    numpy.random.seed(7)
    nmo = 10
    nelec = (5,4)
    nwalkers = 10
    ndets = 5
    nsteps = 20
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    nchols = chol.shape[0]
    system = Generic(nelec=nelec)
    ham = HamGeneric(h1e=numpy.array([h1e,h1e]),
                     chol=chol.reshape((-1,nmo*nmo)).T.copy(),
                     ecore=0)
    # Test PH type wavefunction.
    wfn, init = get_random_phmsd(system.nup, system.ndown, ham.nbasis, ndet=ndets, init=True)
    trial = MultiSlater(system, ham, wfn, init=init)

    numpy.random.seed(7)

    trial.calculate_energy(system, ham)
    options = {'hybrid': True}
    qmc = dotdict({'dt': 0.005, 'nstblz': 5})
    prop = Continuous(system, ham, trial, qmc, options=options)

    walkers = [MultiDetWalker(system, ham, trial) for iw in range(nwalkers)]
    fb_ref = numpy.zeros((nwalkers, nchols), dtype=numpy.complex128)
    for iw in range(nwalkers):
        fb_ref[iw,:] = prop.propagator.construct_force_bias(ham, walkers[iw], trial)

    numpy.random.seed(7)
    qmc = dotdict({'dt': 0.005, 'nstblz': 5, 'batched':True,'nwalkers':nwalkers})
    prop_batch = Continuous(system, ham, trial, qmc, options=options)

    walker_batch = MultiDetTrialWalkerBatch(system, ham, trial, nwalkers)
    prop_batch.propagator.vbias_batch = construct_force_bias_batch(ham, walker_batch, trial)
    fb = - prop_batch.propagator.sqrt_dt * (1j*prop_batch.propagator.vbias_batch-prop_batch.propagator.mf_shift)

    for istep in range(nsteps):
        for iw in range(nwalkers):
            prop.propagate_walker(walkers[iw], system, ham, trial, trial.energy)

    numpy.random.seed(7)

    for istep in range(nsteps):
        prop_batch.propagate_walker_batch(walker_batch, system, ham, trial, trial.energy)

if __name__ == '__main__':
    test_phmsd_greens_function_batch()
    test_phmsd_force_bias_batch()
    test_phmsd_overlap_batch()
    test_phmsd_propagation_batch()