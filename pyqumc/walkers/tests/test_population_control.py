import numpy
import pytest
from pyqumc.utils.misc import dotdict
from pyqumc.qmc.afqmc_batch import AFQMCBatch
from pyqumc.qmc.options import QMCOpts
from pyqumc.propagation.continuous import Continuous
from pyqumc.propagation.utils import get_propagator_driver
from pyqumc.hamiltonians.utils import get_hamiltonian
from pyqumc.trial_wavefunction.multi_slater import MultiSlater
from pyqumc.trial_wavefunction.utils import get_trial_wavefunction
from pyqumc.estimators.local_energy import local_energy
from pyqumc.systems.generic import Generic
from pyqumc.hamiltonians.generic import Generic as HamGeneric
from pyqumc.utils.mpi import get_shared_comm
from pyqumc.utils.io import  get_input_value
from pyqumc.walkers.walker_batch_handler import WalkerBatchHandler
from pyqumc.walkers.single_det_batch import SingleDetWalkerBatch
from pyqumc.walkers.single_det import SingleDetWalker
from pyqumc.walkers.handler import Walkers
from pyqumc.utils.testing import (
        generate_hamiltonian,
        get_random_nomsd,
        get_random_phmsd
        )

@pytest.mark.unit
def test_pair_branch_batch():
    import mpi4py
    mpi4py.rc.recv_mprobe = False
    from mpi4py import MPI

    numpy.random.seed(7)
    comm = MPI.COMM_WORLD

    nelec = (5,5)
    nwalkers = 10
    nsteps = 10
    nmo = 10

    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    sys = Generic(nelec=nelec)
    ham = HamGeneric(h1e=numpy.array([h1e,h1e]),
                     chol=chol.reshape((-1,nmo*nmo)).T.copy(),
                     ecore=0)
    # Test PH type wavefunction.
    wfn, init = get_random_phmsd(sys.nup, sys.ndown, ham.nbasis, ndet=1, init=True)
    trial = MultiSlater(sys, ham, wfn, init=init)
    trial.half_rotate(sys, ham)

    trial.psi = trial.psi[0]
    trial.psia = trial.psia[0]
    trial.psib = trial.psib[0]
    trial.calculate_energy(sys, ham)
    
    numpy.random.seed(7)
    options = {'hybrid': True, 'population_control': "pair_branch"}
    qmc = dotdict({'dt': 0.005, 'nstblz': 5, 'nwalkers': nwalkers, 'batched': True})
    qmc.ntot_walkers = qmc.nwalkers * comm.size
    prop = Continuous(sys, ham, trial, qmc, options=options)
    handler_batch = WalkerBatchHandler(sys, ham, trial, qmc, options, verbose=False, comm=comm)

    for i in range (nsteps):
        prop.propagate_walker_batch(handler_batch.walkers_batch, sys, ham, trial, trial.energy)
        handler_batch.walkers_batch.reortho()
        handler_batch.pop_control(comm)
    numpy.random.seed(7)
    qmc = dotdict({'dt': 0.005, 'nstblz': 5, 'nwalkers': nwalkers, 'batched': False})
    qmc.ntot_walkers = qmc.nwalkers * comm.size
    prop = Continuous(sys, ham, trial, qmc, options=options)
    handler = Walkers(sys, ham, trial, qmc, options, verbose=False, comm=comm)

    for i in range (nsteps):
        for walker in handler.walkers:
            prop.propagate_walker(walker, sys, ham, trial, trial.energy)
            detR = walker.reortho(trial) # reorthogonalizing to stablize
        handler.pop_control(comm)

    for iw in range(nwalkers):
        assert numpy.allclose(handler_batch.walkers_batch.phia[iw], handler.walkers[iw].phi[:,:sys.nup])
        assert numpy.allclose(handler_batch.walkers_batch.phib[iw], handler.walkers[iw].phi[:,sys.nup:])
        assert numpy.allclose(handler_batch.walkers_batch.weight[iw], handler.walkers[iw].weight)
    assert pytest.approx (handler_batch.walkers_batch.weight[0]) == 0.2571750688329709
    assert pytest.approx (handler_batch.walkers_batch.weight[1]) == 1.0843219322894988
    assert pytest.approx (handler_batch.walkers_batch.weight[2]) == 0.8338283613093604
    assert pytest.approx(handler_batch.walkers_batch.phia[iw][0,0]) == -0.0005573508035052743+0.12432250308987346j

@pytest.mark.unit
def test_comb_batch():
    import mpi4py
    mpi4py.rc.recv_mprobe = False
    from mpi4py import MPI

    numpy.random.seed(7)
    comm = MPI.COMM_WORLD

    nelec = (5,5)
    nwalkers = 10
    nsteps = 10
    nmo = 10

    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    sys = Generic(nelec=nelec)
    ham = HamGeneric(h1e=numpy.array([h1e,h1e]),
                     chol=chol.reshape((-1,nmo*nmo)).T.copy(),
                     ecore=0)
    # Test PH type wavefunction.
    wfn, init = get_random_phmsd(sys.nup, sys.ndown, ham.nbasis, ndet=1, init=True)
    trial = MultiSlater(sys, ham, wfn, init=init)
    trial.half_rotate(sys, ham)

    trial.psi = trial.psi[0]
    trial.psia = trial.psia[0]
    trial.psib = trial.psib[0]
    trial.calculate_energy(sys, ham)
    
    numpy.random.seed(7)
    options = {'hybrid': True, 'population_control': "comb"}
    qmc = dotdict({'dt': 0.005, 'nstblz': 5, 'nwalkers': nwalkers, 'batched': True})
    qmc.ntot_walkers = qmc.nwalkers * comm.size
    prop = Continuous(sys, ham, trial, qmc, options=options)
    handler_batch = WalkerBatchHandler(sys, ham, trial, qmc, options, verbose=False, comm=comm)
    for i in range (nsteps):
        prop.propagate_walker_batch(handler_batch.walkers_batch, sys, ham, trial, trial.energy)
        handler_batch.walkers_batch.reortho()
        handler_batch.pop_control(comm)

    numpy.random.seed(7)
    qmc = dotdict({'dt': 0.005, 'nstblz': 5, 'nwalkers': nwalkers, 'batched': False})
    qmc.ntot_walkers = qmc.nwalkers * comm.size
    prop = Continuous(sys, ham, trial, qmc, options=options)
    handler = Walkers(sys, ham, trial, qmc, options, verbose=False, comm=comm)

    for i in range (nsteps):
        for walker in handler.walkers:
            prop.propagate_walker(walker, sys, ham, trial, trial.energy)
            detR = walker.reortho(trial) # reorthogonalizing to stablize
        handler.pop_control(comm)

    for iw in range(nwalkers):
        assert numpy.allclose(handler_batch.walkers_batch.phia[iw], handler.walkers[iw].phi[:,:sys.nup])
        assert numpy.allclose(handler_batch.walkers_batch.phib[iw], handler.walkers[iw].phi[:,sys.nup:])
        assert numpy.allclose(handler_batch.walkers_batch.weight[iw], handler.walkers[iw].weight)
    assert pytest.approx(handler_batch.walkers_batch.phia[iw][0,0]) == -0.0597200851442905-0.002353281222663805j

if __name__ == '__main__':
    test_pair_branch_batch()
    test_comb_batch()
