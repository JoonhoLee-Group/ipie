import numpy
import pytest
from typing import Union

from ipie.systems.generic import Generic
from ipie.qmc.options import QMCOpts
from ipie.utils.mpi import MPIHandler

from ipie.legacy.systems.ueg import UEG as LegacyUEG
from ipie.legacy.hamiltonians.ueg import UEG as LegacyHamUEG
from ipie.legacy.hamiltonians._generic import Generic as LegacyHamGeneric
from ipie.legacy.trial_density_matrices.onebody import OneBody as LegacyOneBody
from ipie.legacy.trial_density_matrices.mean_field import MeanField as LegacyMeanField
from ipie.legacy.walkers.thermal import ThermalWalker
from ipie.legacy.walkers.handler import Walkers
from ipie.legacy.thermal_propagation.continuous import Continuous
from ipie.legacy.thermal_propagation.planewave import PlaneWave
from ipie.legacy.qmc.thermal_afqmc import ThermalAFQMC as LegacyThermalAFQMC


def legacy_propagate_walkers(legacy_hamiltonian, legacy_trial, legacy_walkers, legacy_propagator, xi=None):
    if xi is None:
        xi = [None] * len(legacy_walkers)

    for iw, walker in enumerate(legacy_walkers.walkers):
        legacy_propagator.propagate_walker(
                legacy_hamiltonian, walker, legacy_trial, xi=xi[iw])

    return legacy_walkers


def build_legacy_generic_test_case_handlers(hamiltonian,
                                            comm,
                                            options: dict,
                                            seed: Union[int, None],
                                            verbose: bool = False):
    # Unpack options
    nelec = options['nelec']
    nbasis = options['nbasis']
    mu = options['mu']
    beta = options['beta']
    timestep = options['timestep']
    nwalkers = options.get('nwalkers', 100)
    stack_size = options.get('stack_size', 10)
    nblocks = options.get('nblocks', 100)
    nsteps_per_block = 1
    stabilize_freq = options.get('stabilize_freq', 5)
    pop_control_freq = options.get('pop_control_freq', 1)
    pop_control_method = options.get('pop_control_method', 'pair_branch')
    lowrank = options.get('lowrank', False)
    lowrank_thresh = options.get('lowrank_thresh', 1e-6)
    alt_convention = options.get('alt_convention', False)
    sparse = options.get('sparse', False)

    mf_trial = options.get('mf_trial', True)
    propagate = options.get('propagate', False)
    numpy.random.seed(seed)
    
    legacy_options = {
        "walkers": {
            "stack_size": stack_size,
            "low_rank": lowrank,
            "low_rank_thresh": lowrank_thresh,
            "pop_control_freq": pop_control_freq,
            "pop_control": pop_control_method
        },

        "propagator": {
            "optimised": False
        },
    }

    # 1. Build system.
    legacy_system = Generic(nelec, verbose=verbose)
    legacy_system.mu = mu
    
    # 2. Build Hamiltonian.
    legacy_hamiltonian = LegacyHamGeneric(
                            h1e=hamiltonian.H1,
                            chol=hamiltonian.chol,
                            ecore=hamiltonian.ecore)
    legacy_hamiltonian.hs_pot = numpy.copy(hamiltonian.chol)
    legacy_hamiltonian.hs_pot = legacy_hamiltonian.hs_pot.T.reshape(
            (hamiltonian.nchol, hamiltonian.nbasis, hamiltonian.nbasis))
    legacy_hamiltonian.mu = mu
    legacy_hamiltonian._alt_convention = alt_convention
    legacy_hamiltonian.sparse = sparse
    
    # 3. Build trial.
    legacy_trial = LegacyOneBody(legacy_system, legacy_hamiltonian, beta, timestep, 
                                 verbose=verbose)
    if mf_trial:
        legacy_trial = LegacyMeanField(legacy_system, legacy_hamiltonian, beta, timestep, 
                                       verbose=verbose)
    # 4. Build walkers.    
    qmc_opts = QMCOpts()
    qmc_opts.nwalkers = nwalkers
    qmc_opts.ntot_walkers = nwalkers
    qmc_opts.beta = beta
    qmc_opts.nsteps = nsteps_per_block
    qmc_opts.dt = timestep
    qmc_opts.seed = seed

    legacy_walkers = Walkers(legacy_system, legacy_hamiltonian, legacy_trial,
                             qmc_opts, walker_opts=legacy_options['walkers'],
                             verbose=verbose, comm=comm)

    # 5. Build propagator.
    legacy_propagator = Continuous(
                            legacy_options["propagator"], qmc_opts, legacy_system, 
                            legacy_hamiltonian, legacy_trial, verbose=verbose, 
                            lowrank=lowrank)

    if propagate:
        for t in range(legacy_walkers[0].stack.ntime_slices):
            for iw, walker in enumerate(legacy_walkers):
                legacy_propagator.propagate_walker(
                        legacy_hamiltonian, walker, legacy_trial)

    legacy_objs = {'system': legacy_system,
                   'trial': legacy_trial,
                   'hamiltonian': legacy_hamiltonian,
                   'walkers': legacy_walkers,
                   'propagator': legacy_propagator}
    return legacy_objs
    

def build_legacy_generic_test_case_handlers_mpi(hamiltonian,
                                                mpi_handler: MPIHandler,
                                                options: dict,
                                                seed: Union[int, None],
                                                verbose: bool = False):
    # Unpack options
    nelec = options['nelec']
    nbasis = options['nbasis']
    mu = options['mu']
    beta = options['beta']
    timestep = options['timestep']
    nwalkers = options.get('nwalkers', 100)
    stack_size = options.get('stack_size', 10)
    nblocks = options.get('nblocks', 100)
    nsteps_per_block = 1
    stabilize_freq = options.get('stabilize_freq', 5)
    pop_control_freq = options.get('pop_control_freq', 1)
    pop_control_method = options.get('pop_control_method', 'pair_branch')
    lowrank = options.get('lowrank', False)
    lowrank_thresh = options.get('lowrank_thresh', 1e-6)
    alt_convention = options.get('alt_convention', False)
    sparse = options.get('sparse', False)

    mf_trial = options.get('mf_trial', True)
    propagate = options.get('propagate', False)
    numpy.random.seed(seed)
    comm = mpi_handler.comm

    legacy_options = {
        "walkers": {
            "stack_size": stack_size,
            "low_rank": lowrank,
            "low_rank_thresh": lowrank_thresh,
            "pop_control_freq": pop_control_freq,
            "pop_control": pop_control_method
        },

        "propagator": {
            "optimised": False
        },
    }
    
    # 1. Build system.
    legacy_system = Generic(nelec, verbose=verbose)
    legacy_system.mu = mu
    
    # 2. Build Hamiltonian.
    legacy_hamiltonian = LegacyHamGeneric(
                            h1e=hamiltonian.H1,
                            chol=hamiltonian.chol,
                            ecore=hamiltonian.ecore)
    legacy_hamiltonian.hs_pot = numpy.copy(hamiltonian.chol)
    legacy_hamiltonian.hs_pot = legacy_hamiltonian.hs_pot.T.reshape(
            (hamiltonian.nchol, hamiltonian.nbasis, hamiltonian.nbasis))
    legacy_hamiltonian.mu = mu
    legacy_hamiltonian._alt_convention = alt_convention
    legacy_hamiltonian.sparse = sparse
    
    # 3. Build trial.
    legacy_trial = LegacyOneBody(legacy_system, legacy_hamiltonian, beta, timestep, 
                                 verbose=verbose)
    if mf_trial:
        legacy_trial = LegacyMeanField(legacy_system, legacy_hamiltonian, beta, timestep, 
                                       verbose=verbose)
    # 4. Build walkers.    
    qmc_opts = QMCOpts()
    qmc_opts.nwalkers = nwalkers
    qmc_opts.ntot_walkers = nwalkers * comm.size
    qmc_opts.beta = beta
    qmc_opts.nsteps = nsteps_per_block
    qmc_opts.dt = timestep
    qmc_opts.seed = seed

    legacy_walkers = Walkers(legacy_system, legacy_hamiltonian, legacy_trial,
                             qmc_opts, walker_opts=legacy_options['walkers'],
                             verbose=verbose, comm=comm)

    # 5. Build propagator.
    legacy_propagator = Continuous(
                            legacy_options["propagator"], qmc_opts, legacy_system, 
                            legacy_hamiltonian, legacy_trial, verbose=verbose, 
                            lowrank=lowrank)

    if propagate:
        for t in range(legacy_walkers[0].stack.ntime_slices):
            for iw, walker in enumerate(legacy_walkers):
                legacy_propagator.propagate_walker(
                        legacy_hamiltonian, walker, legacy_trial)

    legacy_objs = {'system': legacy_system,
                   'trial': legacy_trial,
                   'hamiltonian': legacy_hamiltonian,
                   'walkers': legacy_walkers,
                   'propagator': legacy_propagator}
    return legacy_objs
    

def build_legacy_driver_generic_test_instance(hamiltonian,
                                              comm,
                                              options: Union[dict, None],
                                              seed: Union[int, None],
                                              verbose: bool = False):
    # Unpack options
    nelec = options['nelec']
    nup, ndown = nelec
    mu = options['mu']
    beta = options['beta']
    timestep = options['timestep']
    nwalkers = options['nwalkers']
    stack_size = options.get('stack_size', 10)
    nblocks = options.get('nblocks', 100)
    nsteps_per_block = 1
    stabilize_freq = options.get('stabilize_freq', 5)
    pop_control_freq = options.get('pop_control_freq', 1)
    pop_control_method = options.get('pop_control_method', 'pair_branch')
    lowrank = options.get('lowrank', False)
    lowrank_thresh = options.get('lowrank_thresh', 1e-6)
    alt_convention = options.get('alt_convention', False)
    sparse = options.get('sparse', False)
    numpy.random.seed(seed)
    
    legacy_options = {
        "qmc": {
            "dt": timestep,
            # Input of `nwalkers` refers to the total number of walkers in
            # legacy `ThermalAFQMC`.
            "nwalkers": nwalkers * comm.size,
            "blocks": nblocks,
            "nsteps": nsteps_per_block,
            "beta": beta,
            "rng_seed": seed,
            "pop_control_freq": pop_control_freq,
            "stabilise_freq": stabilize_freq,
            "batched": False
        },

        "propagator": {
            "optimised": False
        },

        "walkers": {
            "stack_size": stack_size,
            "low_rank": lowrank,
            "low_rank_thresh": lowrank_thresh,
            "pop_control_freq": pop_control_freq,
            "pop_control": pop_control_method
        },

        "system": {
            "name": "Generic",
            "nup": nup,
            "ndown": ndown,
            "mu": mu
        },

        "estimators": options["estimators"]
    }

    legacy_system = Generic(nelec)
    legacy_system.mu = mu
    legacy_hamiltonian = LegacyHamGeneric(
                            h1e=hamiltonian.H1,
                            chol=hamiltonian.chol,
                            ecore=hamiltonian.ecore)
    legacy_hamiltonian.hs_pot = numpy.copy(hamiltonian.chol)
    legacy_hamiltonian.hs_pot = legacy_hamiltonian.hs_pot.T.reshape(
            (hamiltonian.nchol, hamiltonian.nbasis, hamiltonian.nbasis))
    legacy_hamiltonian.mu = mu
    legacy_hamiltonian._alt_convention = alt_convention
    legacy_hamiltonian.sparse = sparse
    legacy_trial = LegacyMeanField(legacy_system, legacy_hamiltonian, beta, timestep)
    
    afqmc = LegacyThermalAFQMC(comm, legacy_options, legacy_system, 
                               legacy_hamiltonian, legacy_trial, verbose=verbose)
    return afqmc


def build_legacy_ueg_test_case_handlers(hamiltonian,
                                        comm,
                                        options: dict,
                                        seed: Union[int, None],
                                        verbose: bool = False):
    # Unpack options
    mu = options['mu']
    beta = options['beta']
    timestep = options['timestep']
    nwalkers = options.get('nwalkers', 100)
    stack_size = options.get('stack_size', 10)
    nblocks = options.get('nblocks', 100)
    nsteps_per_block = 1
    stabilize_freq = options.get('stabilize_freq', 5)
    pop_control_freq = options.get('pop_control_freq', 1)
    pop_control_method = options.get('pop_control_method', 'pair_branch')
    lowrank = options.get('lowrank', False)
    lowrank_thresh = options.get('lowrank_thresh', 1e-6)
    alt_convention = options.get('alt_convention', False)

    propagate = options.get('propagate', False)
    numpy.random.seed(seed)
    
    legacy_options = {
        "propagator": {
            "optimised": False
        },

        "walkers": {
            "stack_size": stack_size,
            "low_rank": lowrank,
            "low_rank_thresh": lowrank_thresh,
            "pop_control_freq": pop_control_freq,
            "pop_control": pop_control_method
        },
    }

    # 1. Build out system.
    legacy_system = LegacyUEG(options=options["ueg_opts"])
    legacy_system.mu = mu
    
    # 2. Build Hamiltonian.
    legacy_hamiltonian = LegacyHamUEG(legacy_system, options=options["ueg_opts"])
    legacy_hamiltonian.mu = mu
    legacy_hamiltonian._alt_convention = alt_convention

    # 3. Build trial.
    legacy_trial = LegacyOneBody(legacy_system, legacy_hamiltonian, beta, timestep, verbose=verbose)
    
    # 4. Build walkers.    
    qmc_opts = QMCOpts()
    qmc_opts.nwalkers = nwalkers
    qmc_opts.ntot_walkers = nwalkers * comm.size
    qmc_opts.beta = beta
    qmc_opts.nsteps = nsteps_per_block
    qmc_opts.dt = timestep
    qmc_opts.seed = seed

    legacy_walkers = Walkers(legacy_system, legacy_hamiltonian, legacy_trial,
                             qmc_opts, walker_opts=legacy_options['walkers'],
                             verbose=verbose, comm=comm)

    # 5. Build propagator.
    legacy_propagator = PlaneWave(legacy_system, legacy_hamiltonian, legacy_trial, qmc_opts,
                                  options=legacy_options["propagator"], lowrank=lowrank, verbose=verbose)

    if propagate:
        for t in range(legacy_walkers[0].stack.ntime_slices):
            for iw, walker in enumerate(legacy_walkers):
                legacy_propagator.propagate_walker(
                        legacy_hamiltonian, walker, legacy_trial)

    legacy_objs = {'system': legacy_system,
                   'trial': legacy_trial,
                   'hamiltonian': legacy_hamiltonian,
                   'walkers': legacy_walkers,
                   'propagator': legacy_propagator}
    return legacy_objs
    

def build_legacy_driver_ueg_test_instance(hamiltonian,
                                          comm,
                                          options: Union[dict, None],
                                          seed: Union[int, None],
                                          verbose: bool = False):
    # Unpack options
    mu = options['mu']
    beta = options['beta']
    timestep = options['timestep']
    nwalkers = options.get('nwalkers', 100)
    stack_size = options.get('stack_size', 10)
    nblocks = options.get('nblocks', 100)
    nsteps_per_block = 1
    stabilize_freq = options.get('stabilize_freq', 5)
    pop_control_freq = options.get('pop_control_freq', 1)
    pop_control_method = options.get('pop_control_method', 'pair_branch')
    lowrank = options.get('lowrank', False)
    lowrank_thresh = options.get('lowrank_thresh', 1e-6)
    optimised = options.get('optimised', False)
    alt_convention = options.get('alt_convention', False)
    numpy.random.seed(seed)
    
    legacy_options = {
        "qmc": {
            "dt": timestep,
            # Input of `nwalkers` refers to the total number of walkers in
            # legacy `ThermalAFQMC`.
            "nwalkers": nwalkers * comm.size,
            "blocks": nblocks,
            "nsteps": nsteps_per_block,
            "beta": beta,
            "rng_seed": seed,
            "pop_control_freq": pop_control_freq,
            "stabilise_freq": stabilize_freq,
            "batched": False
        },

        "propagator": {
            "optimised": optimised
        },

        "walkers": {
            "stack_size": stack_size,
            "low_rank": lowrank,
            "low_rank_thresh": lowrank_thresh,
            "pop_control_freq": pop_control_freq,
            "pop_control": pop_control_method
        },
        
        "ueg_opts": options["ueg_opts"],
        "estimators": options["estimators"]
    }

    # 1. Build out system.
    legacy_system = LegacyUEG(options=legacy_options["ueg_opts"])
    legacy_system.mu = mu
    
    # 2. Build Hamiltonian.
    legacy_hamiltonian = LegacyHamUEG(legacy_system, options=legacy_options["ueg_opts"])
    legacy_hamiltonian.mu = mu
    legacy_hamiltonian._alt_convention = alt_convention

    # 3. Build trial.
    legacy_trial = LegacyOneBody(legacy_system, legacy_hamiltonian, beta, timestep, verbose=verbose)
    
    # 4. Build Thermal AFQMC.
    afqmc = LegacyThermalAFQMC(comm, legacy_options, legacy_system, 
                               legacy_hamiltonian, legacy_trial, verbose=verbose)
    return afqmc

