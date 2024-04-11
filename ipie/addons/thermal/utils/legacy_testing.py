# Copyright 2022 The ipie Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Fionn Malone <fmalone@google.com>
#          Joonho Lee
#

import numpy
from typing import Tuple, Union

from ipie.systems.generic import Generic
from ipie.utils.mpi import MPIHandler

from ipie.addons.thermal.qmc.options import ThermalQMCOpts

from ipie.legacy.systems.ueg import UEG as LegacyUEG
from ipie.legacy.hamiltonians.ueg import UEG as LegacyHamUEG
from ipie.legacy.hamiltonians._generic import Generic as LegacyHamGeneric
from ipie.legacy.trial_density_matrices.onebody import OneBody as LegacyOneBody
from ipie.legacy.trial_density_matrices.mean_field import MeanField as LegacyMeanField
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


def build_legacy_generic_test_case_handlers(
        hamiltonian,
        comm,
        nelec: Tuple[int, int],
        mu: float,
        beta: float,
        timestep: float,
        nwalkers: int = 100,
        stack_size: int = 10,
        lowrank: bool = False,
        lowrank_thresh: float = 1e-6,
        stabilize_freq: int = 5,
        pop_control_freq: int = 5,
        pop_control_method: str = 'pair_branch',
        alt_convention: bool = False,
        sparse: bool = False,
        mf_trial: bool = True,
        propagate: bool = False,
        seed: Union[int, None] = None,
        verbose: int = 0):
    numpy.random.seed(seed)
    
    legacy_options = {
        "walkers": {
            "stack_size": stack_size,
            "low_rank": lowrank,
            "low_rank_thresh": lowrank_thresh,
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
    legacy_trial = LegacyOneBody(legacy_system, legacy_hamiltonian, beta, 
                                 timestep, verbose=verbose)
    if mf_trial:
        legacy_trial = LegacyMeanField(legacy_system, legacy_hamiltonian, beta, 
                                       timestep, verbose=verbose)
    # 4. Build walkers.    
    qmc_opts = ThermalQMCOpts()
    qmc_opts.nwalkers = nwalkers
    qmc_opts.ntot_walkers = nwalkers
    qmc_opts.beta = beta
    qmc_opts.nsteps = 1
    qmc_opts.dt = timestep
    qmc_opts.nstblz = stabilize_freq
    qmc_opts.npop_control = pop_control_freq
    qmc_opts.pop_control_method = pop_control_method
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
    

def build_legacy_generic_test_case_handlers_mpi(
        hamiltonian,
        mpi_handler: MPIHandler,
        nelec: Tuple[int, int],
        mu: float,
        beta: float,
        timestep: float,
        nwalkers: int = 100,
        stack_size: int = 10,
        lowrank: bool = False,
        lowrank_thresh: float = 1e-6,
        stabilize_freq: int = 5,
        pop_control_freq: int = 5,
        pop_control_method: str = 'pair_branch',
        alt_convention: bool = False,
        sparse: bool = False,
        mf_trial: bool = True,
        propagate: bool = False,
        seed: Union[int, None] = None,
        verbose: int = 0):
    numpy.random.seed(seed)
    comm = mpi_handler.comm

    legacy_options = {
        "walkers": {
            "stack_size": stack_size,
            "low_rank": lowrank,
            "low_rank_thresh": lowrank_thresh,
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
    legacy_trial = LegacyOneBody(legacy_system, legacy_hamiltonian, beta, 
                                 timestep, verbose=verbose)
    if mf_trial:
        legacy_trial = LegacyMeanField(legacy_system, legacy_hamiltonian, beta, 
                                       timestep, verbose=verbose)
    # 4. Build walkers.    
    qmc_opts = ThermalQMCOpts()
    qmc_opts.nwalkers = nwalkers
    qmc_opts.ntot_walkers = nwalkers * comm.size
    qmc_opts.beta = beta
    qmc_opts.nsteps = 1
    qmc_opts.dt = timestep
    qmc_opts.nstblz = stabilize_freq
    qmc_opts.npop_control = pop_control_freq
    qmc_opts.pop_control_method = pop_control_method
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
    

def build_legacy_driver_generic_test_instance(
        hamiltonian,
        comm,
        nelec: Tuple[int, int],
        mu: float,
        beta: float,
        timestep: float,
        nblocks: int,
        nwalkers: int = 100,
        stack_size: int = 10,
        lowrank: bool = False,
        lowrank_thresh: float = 1e-6,
        stabilize_freq: int = 5,
        pop_control_freq: int = 5,
        pop_control_method: str = 'pair_branch',
        alt_convention: bool = False,
        sparse: bool = False,
        seed: Union[int, None] = None,
        estimator_filename: Union[str, None] = None,
        verbose: int = 0):
    nup, ndown = nelec
    numpy.random.seed(seed)
    
    legacy_options = {
        "qmc": {
            "dt": timestep,
            # Input of `nwalkers` refers to the total number of walkers in
            # legacy `ThermalAFQMC`.
            "nwalkers": nwalkers * comm.size,
            "blocks": nblocks,
            "nsteps": 1,
            "beta": beta,
            "stabilise_freq": stabilize_freq,
            "pop_control_freq": pop_control_freq,
            "pop_control_method": pop_control_method,
            "rng_seed": seed,
            "batched": False
        },

        "propagator": {
            "optimised": False
        },

        "walkers": {
            "stack_size": stack_size,
            "low_rank": lowrank,
            "low_rank_thresh": lowrank_thresh,
            "pop_control": pop_control_method
        },

        "system": {
            "name": "Generic",
            "nup": nup,
            "ndown": ndown,
            "mu": mu
        },

        "estimators": {
            "filename": estimator_filename,
        },
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


def build_legacy_ueg_test_case_handlers(
        comm,
        nelec: Tuple[int, int],
        rs: float,
        ecut: float,
        mu: float,
        beta: float,
        timestep: float,
        nwalkers: int = 100,
        stack_size: int = 10,
        lowrank: bool = False,
        lowrank_thresh: float = 1e-6,
        propagate: bool = False,
        stabilize_freq: int = 5,
        pop_control_freq: int = 5,
        pop_control_method: str = 'pair_branch',
        alt_convention: bool = False,
        sparse: bool = False,
        seed: Union[int, None] = None,
        verbose: int = 0):
    numpy.random.seed(seed)
    nup, ndown = nelec
    legacy_options = {
        "ueg": {
            "nup": nup,
            "ndown": ndown,
            "rs": rs,
            "ecut": ecut,
            "thermal": True,
            "write_integrals": False,
            "low_rank": lowrank
        },

        "propagator": {
            "optimised": False
        },

        "walkers": {
            "stack_size": stack_size,
            "low_rank": lowrank,
            "low_rank_thresh": lowrank_thresh,
            "pop_control": pop_control_method
        },
    }

    # 1. Build out system.
    legacy_system = LegacyUEG(options=legacy_options['ueg'])
    legacy_system.mu = mu
    
    # 2. Build Hamiltonian.
    legacy_hamiltonian = LegacyHamUEG(legacy_system, options=legacy_options['ueg'])
    legacy_hamiltonian.mu = mu
    legacy_hamiltonian._alt_convention = alt_convention

    # 3. Build trial.
    legacy_trial = LegacyOneBody(legacy_system, legacy_hamiltonian, beta, 
                                 timestep, verbose=verbose)
    
    # 4. Build walkers.    
    qmc_opts = ThermalQMCOpts()
    qmc_opts.nwalkers = nwalkers
    qmc_opts.ntot_walkers = nwalkers * comm.size
    qmc_opts.beta = beta
    qmc_opts.nsteps = 1
    qmc_opts.dt = timestep
    qmc_opts.nstblz = stabilize_freq
    qmc_opts.npop_control = pop_control_freq
    qmc_opts.pop_control_method = pop_control_method
    qmc_opts.seed = seed

    legacy_walkers = Walkers(legacy_system, legacy_hamiltonian, legacy_trial,
                             qmc_opts, walker_opts=legacy_options['walkers'],
                             verbose=verbose, comm=comm)

    # 5. Build propagator.
    legacy_propagator = PlaneWave(legacy_system, legacy_hamiltonian, legacy_trial, 
                                  qmc_opts, options=legacy_options["propagator"], 
                                  lowrank=lowrank, verbose=verbose)

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
    

def build_legacy_driver_ueg_test_instance(
        comm,
        nelec: Tuple[int, int],
        rs: float,
        ecut: float,
        mu: float,
        beta: float,
        timestep: float,
        nblocks: int,
        nwalkers: int = 100,
        stack_size: int = 10,
        lowrank: bool = False,
        lowrank_thresh: float = 1e-6,
        stabilize_freq: int = 5,
        pop_control_freq: int = 5,
        pop_control_method: str = 'pair_branch',
        alt_convention: bool = False,
        sparse: bool = False,
        seed: Union[int, None] = None,
        estimator_filename: Union[str, None] = None,
        verbose: int = 0):
    numpy.random.seed(seed)
    nup, ndown = nelec
    legacy_options = {
        "ueg": {
            "nup": nup,
            "ndown": ndown,
            "rs": rs,
            "ecut": ecut,
            "thermal": True,
            "write_integrals": False,
            "low_rank": lowrank
        },

        "qmc": {
            "dt": timestep,
            # Input of `nwalkers` refers to the total number of walkers in
            # legacy `ThermalAFQMC`.
            "nwalkers": nwalkers * comm.size,
            "blocks": nblocks,
            "nsteps": 1,
            "beta": beta,
            "stabilise_freq": stabilize_freq,
            "pop_control_freq": pop_control_freq,
            "pop_control_method": pop_control_method,
            "rng_seed": seed,
            "batched": False
        },

        "propagator": {
            "optimised": False
        },

        "walkers": {
            "stack_size": stack_size,
            "low_rank": lowrank,
            "low_rank_thresh": lowrank_thresh,
            "pop_control": pop_control_method
        },
        
        "estimators": {
            "filename": estimator_filename,
        },
    }

    # 1. Build out system.
    legacy_system = LegacyUEG(options=legacy_options['ueg'])
    legacy_system.mu = mu
    
    # 2. Build Hamiltonian.
    legacy_hamiltonian = LegacyHamUEG(legacy_system, options=legacy_options['ueg'])
    legacy_hamiltonian.mu = mu
    legacy_hamiltonian._alt_convention = alt_convention

    # 3. Build trial.
    legacy_trial = LegacyOneBody(legacy_system, legacy_hamiltonian, beta, 
                                 timestep, verbose=verbose)
    
    # 4. Build Thermal AFQMC.
    afqmc = LegacyThermalAFQMC(comm, legacy_options, legacy_system, 
                               legacy_hamiltonian, legacy_trial, verbose=verbose)
    return afqmc

