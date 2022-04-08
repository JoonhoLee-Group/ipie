import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import numpy
import pytest
from ipie.qmc.options import QMCOpts
from ipie.propagation.utils import get_propagator_driver
from ipie.hamiltonians.utils import get_hamiltonian
from ipie.trial_wavefunction.utils import get_trial_wavefunction
from ipie.estimators.local_energy import local_energy, local_energy_G, local_energy_generic_cholesky
from ipie.estimators.local_energy_batch import local_energy_multi_det_trial_batch
from ipie.estimators.greens_function import greens_function_multi_det_wicks
from ipie.systems.generic import Generic
from ipie.utils.mpi import get_shared_comm
from ipie.utils.linalg import minor_mask, minor_mask4
from ipie.utils.io import  get_input_value
from ipie.walkers.multi_det_batch import MultiDetTrialWalkerBatch
from ipie.qmc.afqmc_batch import AFQMCBatch
from ipie.qmc.afqmc import AFQMC
from ipie.analysis.extraction import (
        extract_mixed_estimates,
        )
from ipie.propagation.overlap import get_calc_overlap, calc_overlap_multi_det
from ipie.estimators.greens_function import gab_spin, greens_function_multi_det
import mpi4py
mpi4py.rc.recv_mprobe = False
from mpi4py import MPI
import scipy

import cProfile

def map_orb(orb, nbasis):
    """Map spin orbital to spatial index."""
    if orb // nbasis == 0:
        s = 0
        ix = orb
    else:
        s = 1
        ix = orb - nbasis
    return ix, s

nelec = (8,6)
nwalkers = 1
nsteps = 1
nblocks = 1
seed = 7

options = {
    "system": {
        "nup": nelec[0],
        "ndown": nelec[1],
    },
    "hamiltonian": {
        "name": "Generic",
        "integrals": "afqmc_msd.h5"
    },
    "qmc": {
        "dt": 0.01,
        "nsteps": nsteps,
        "nwalkers": nwalkers,
        "blocks": nblocks,
        "batched": True,
        "rng_seed": seed
    },
    "trial": {
        "filename": "afqmc_msd.h5",
        "wicks": True
    },
    "estimators": {}
}

numpy.random.seed(seed)
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
qmc = QMCOpts(qmc_opts, sys,verbose=True)
ham = get_hamiltonian (sys, ham_opts, verbose = True, comm=shared_comm)

trial = ( get_trial_wavefunction(sys, ham, options=twf_opts,
                       comm=comm,
                       scomm=shared_comm,
                       verbose=verbose) )
trial.calculate_energy(sys, ham) # this is to get the energy shift

afqmc = AFQMCBatch(comm=comm, system=sys, hamiltonian = ham, trial=trial, options=options, verbose=verbose)
afqmc.run(comm=comm, verbose=0)

