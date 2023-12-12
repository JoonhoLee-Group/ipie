import mpi4py
import numpy

from ipie.hamiltonians.utils import get_hamiltonian
from ipie.legacy.walkers.handler import Walkers
from ipie.propagation.utils import get_propagator_driver
from ipie.qmc.options import QMCOpts
from ipie.systems.generic import Generic
from ipie.trial_wavefunction.utils import get_trial_wavefunction
from ipie.utils.io import get_input_value
from ipie.utils.mpi import get_shared_comm
from ipie.walkers.walker_batch_handler import WalkerBatchHandler

mpi4py.rc.recv_mprobe = False
from mpi4py import MPI

nelec = (5,5)
nwalkers = 20
nsteps = 50

options = {
    "system": {
        "nup": nelec[0],
        "ndown": nelec[1],
    },
    "hamiltonian": {
        "name": "Generic",
        "integrals": "afqmc.h5"
    },
    "qmc": {
        "dt": 0.01,
        "nsteps": nsteps,
        "nwalkers": nwalkers,
        "blocks": 1,
        "batched": True
    },
    "trial": {
        "filename": "afqmc.h5"
    },
    "walker":{
    "population_control":"pair_branch"
    },
    "estimators": {}
}

numpy.random.seed(7)
sys = Generic(nelec=nelec)
comm = MPI.COMM_WORLD
verbose = True
shared_comm = get_shared_comm(comm, verbose=verbose)

qmc_opts = get_input_value(options, 'qmc',
                           default={},
                           verbose=verbose)
ham_opts = get_input_value(options, 'hamiltonian',
                           default={},
                           verbose=verbose)
twf_opts = get_input_value(options, 'trial',
                           default={},
                           verbose=verbose)
prop_opts = get_input_value(options, 'propoagator',
                           default={},
                           verbose=verbose)
wlk_opts = get_input_value(options, 'walkers', default={},
                           alias=['walker', 'walker_opts'],
                           verbose=verbose)
est_opts = get_input_value(options, 'estimators', default={},
                           alias=['estimates','estimator'],
                           verbose=verbose)

qmc = QMCOpts(qmc_opts, sys, verbose=True)
qmc.ntot_walkers = qmc.nwalkers * comm.size

ham = get_hamiltonian (sys, ham_opts, verbose = True, comm=shared_comm)

trial = ( get_trial_wavefunction(sys, ham, options=twf_opts,
                       comm=comm,
                       scomm=shared_comm,
                       verbose=verbose) )
trial.calculate_energy(sys, ham) # this is to get the energy shift

print(trial.psi.shape)
prop = get_propagator_driver(sys, ham, trial, qmc, options=prop_opts,verbose=verbose)
print(trial.psi.shape)

handler_batch = WalkerBatchHandler(sys, ham, trial, qmc, wlk_opts, verbose=False, comm=comm)
for i in range (nsteps):
    prop.propagate_walker_batch(handler_batch.walkers_batch, sys, ham, trial, trial.energy)
    handler_batch.walkers_batch.reortho()
    handler_batch.pop_control(comm)

numpy.random.seed(7)
sys = Generic(nelec=nelec)
comm = MPI.COMM_WORLD
verbose = True
shared_comm = get_shared_comm(comm, verbose=verbose)

options = {
    "system": {
        "nup": nelec[0],
        "ndown": nelec[1],
    },
    "hamiltonian": {
        "name": "Generic",
        "integrals": "afqmc.h5"
    },
    "qmc": {
        "dt": 0.01,
        "nsteps": nsteps,
        "nwalkers": nwalkers,
        "blocks": 1,
        "batched": False
    },
    "trial": {
        "filename": "afqmc.h5"
    },
    "walker":{
    "population_control":"pair_branch"
    },
    "estimators": {}
}

qmc_opts = get_input_value(options, 'qmc',
                           default={},
                           verbose=verbose)
ham_opts = get_input_value(options, 'hamiltonian',
                           default={},
                           verbose=verbose)
twf_opts = get_input_value(options, 'trial',
                           default={},
                           verbose=verbose)
prop_opts = get_input_value(options, 'propoagator',
                           default={},
                           verbose=verbose)
qmc = QMCOpts(qmc_opts, sys,verbose=True)
qmc.ntot_walkers = qmc.nwalkers * comm.size
prop = get_propagator_driver(sys, ham, trial, qmc, options=prop_opts,verbose=verbose)

handler = Walkers(sys, ham, trial, qmc, wlk_opts, verbose=False, comm=comm)

for i in range (nsteps):
    for walker in handler.walkers:
        prop.propagate_walker(walker, sys, ham, trial, trial.energy)
        detR = walker.reortho(trial) # reorthogonalizing to stablize
    handler.pop_control(comm)

for iw in range(nwalkers):
    assert numpy.allclose(handler_batch.walkers_batch.phi[iw], handler.walkers[iw].phi)
    assert numpy.allclose(handler_batch.walkers_batch.weight[iw], handler.walkers[iw].weight)