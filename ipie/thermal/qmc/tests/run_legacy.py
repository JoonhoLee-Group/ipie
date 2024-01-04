import json
import os
import sys
import tempfile
import h5py
import uuid
import pytest
import numpy
from typing import Tuple, Union

from pyscf import gto, scf, lo
from ipie.config import MPI
from ipie.systems.generic import Generic
from ipie.hamiltonians.utils import get_hamiltonian
from ipie.qmc.options import QMCOpts

from ipie.thermal.trial.mean_field import MeanField
from ipie.thermal.trial.one_body import OneBody
from ipie.thermal.walkers.uhf_walkers import UHFThermalWalkers
from ipie.thermal.propagation.phaseless_generic import PhaselessGeneric
from ipie.thermal.qmc.options import ThermalQMCParams
from ipie.thermal.qmc.thermal_afqmc import ThermalAFQMC
from ipie.thermal.estimators.energy import local_energy
from ipie.analysis.extraction import (
        extract_test_data_hdf5, 
        extract_data,
        extract_observable, 
        extract_mixed_estimates)

from ipie.legacy.hamiltonians._generic import Generic as LegacyHamGeneric
from ipie.legacy.trial_density_matrices.mean_field import MeanField as LegacyMeanField
from ipie.legacy.trial_density_matrices.onebody import OneBody as LegacyOneBody
from ipie.legacy.walkers.thermal import ThermalWalker
from ipie.legacy.thermal_propagation.continuous import Continuous
from ipie.legacy.estimators.generic import local_energy_generic_cholesky as legacy_local_energy_generic_cholesky
from ipie.legacy.estimators.thermal import one_rdm_from_G as legacy_one_rdm_from_G
from ipie.legacy.qmc.thermal_afqmc import ThermalAFQMC as LegacyThermalAFQMC

comm = MPI.COMM_WORLD
serial_test = comm.size == 1

# Unique filename to avoid name collision when running through CI.
if comm.rank == 0:
    test_id = str(uuid.uuid1())

else:
    test_id = None

test_id = comm.bcast(test_id, root=0)


def build_legacy_driver_instance(nelec: Tuple[int, int],
                                 hamiltonian,
                                 options: Union[dict, None] = None,
                                 seed: Union[int, None] = None,
                                 verbose: bool = False):
    if seed is not None:
        numpy.random.seed(seed)

    # Unpack options
    nup, ndown = nelec
    mu = options['mu']
    beta = options['beta']
    timestep = options['timestep']
    nwalkers = options['nwalkers']
    nsteps_per_block = options['nsteps_per_block']
    nblocks = options['nblocks']
    stabilize_freq = options['stabilize_freq']
    pop_control_freq = options['pop_control_freq']
    pop_control_method = options['pop_control_method']
    lowrank = options['lowrank']
    
    legacy_options = {
        "qmc": {
            "dt": timestep,
            "nwalkers": nwalkers,
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
            "low_rank": lowrank,
            "pop_control": pop_control_method
        },

        "system": {
            "name": "Generic",
            "nup": nup,
            "ndown": ndown,
            "mu": mu
        },

        "hamiltonian": options["hamiltonian"],
        "estimators": options["estimators"]
    }

    legacy_system = Generic(nelec)
    legacy_system.mu = mu
    legacy_hamiltonian = LegacyHamGeneric(
                            h1e=hamiltonian.H1,
                            chol=hamiltonian.chol,
                            ecore=hamiltonian.ecore,
                            options=legacy_options["hamiltonian"])
    legacy_hamiltonian.hs_pot = numpy.copy(hamiltonian.chol)
    legacy_hamiltonian.hs_pot = legacy_hamiltonian.hs_pot.T.reshape(
            (hamiltonian.nchol, hamiltonian.nbasis, hamiltonian.nbasis))
    legacy_hamiltonian._alt_convention = legacy_options["hamiltonian"]["_alt_convention"]
    legacy_hamiltonian.mu = legacy_options["hamiltonian"]["mu"]
    legacy_trial = LegacyMeanField(legacy_system, legacy_hamiltonian, beta, timestep)
        
    legacy_walkers = [
            ThermalWalker(
                legacy_system, legacy_hamiltonian, legacy_trial, 
                walker_opts=options, verbose=i == 0) for i in range(nwalkers)]

    #legacy_hamiltonian.chol_vecs = legacy_hamiltonian.chol_vecs.T.reshape(
    #                (hamiltonian.nchol, hamiltonian.nbasis, hamiltonian.nbasis))
    afqmc = LegacyThermalAFQMC(comm, legacy_options, legacy_system, 
                               legacy_hamiltonian, legacy_trial, verbose=verbose)
    return afqmc
    

def build_driver_test_instance(nelec: Tuple[int, int],
                               options: Union[dict, None] = None,
                               seed: Union[int, None] = None,
                               debug: bool = False,
                               verbose: bool = False):
    if seed is not None:
        numpy.random.seed(seed)

    # Unpack options
    mu = options['mu']
    beta = options['beta']
    timestep = options['timestep']
    nwalkers = options['nwalkers']
    nsteps_per_block = options['nsteps_per_block']
    nblocks = options['nblocks']
    stabilize_freq = options['stabilize_freq']
    pop_control_freq = options['pop_control_freq']
    pop_control_method = options['pop_control_method']
    lowrank = options['lowrank']
    
    params = ThermalQMCParams(
                num_walkers=nwalkers,
                total_num_walkers=nwalkers * comm.size,
                num_blocks=nblocks,
                num_steps_per_block=nsteps_per_block,
                timestep=timestep,
                beta=beta,
                num_stblz=stabilize_freq,
                pop_control_freq=pop_control_freq,
                pop_control_method=pop_control_method,
                rng_seed=seed)

    system = Generic(nelec)
    hamiltonian = get_hamiltonian(system, options["hamiltonian"])
    trial = MeanField(hamiltonian, nelec, beta, timestep)

    nbasis = trial.dmat.shape[-1]
    walkers = UHFThermalWalkers(trial, nbasis, nwalkers, lowrank=lowrank)
    propagator = PhaselessGeneric(timestep, mu, lowrank=lowrank)
    propagator.build(hamiltonian, trial=trial, walkers=walkers)

    eloc = local_energy(hamiltonian, walkers)
    print(f'# Initial energy = {eloc[0]}')

    afqmc = ThermalAFQMC(system, hamiltonian, trial, walkers, propagator, 
                         params, debug=debug, verbose=verbose)
    return afqmc, hamiltonian, walkers


def run_legacy():
    ref_path = "reference_data/generic/"

    # Thermal AFQMC params.
    nocca = 5
    noccb = 5
    nelec = (nocca, noccb)

    mu = -10.0
    beta = 0.1
    timestep = 0.01
    nwalkers = 9
    seed = 7
    nsteps_per_block = 10
    nblocks = 10
    stabilize_freq = 10
    pop_control_freq = 1
    pop_control_method = 'comb'
    lowrank = False
    verbose = True
    debug = True

    with tempfile.NamedTemporaryFile() as tmpf:
        options = {
                    'mu': mu,
                    'beta': beta,
                    'timestep': timestep,
                    'nwalkers': nwalkers,
                    'seed': 7,
                    'nsteps_per_block': nsteps_per_block,
                    'nblocks': nblocks,
                    'stabilize_freq': stabilize_freq,
                    'pop_control_freq': pop_control_freq,
                    'pop_control_method': pop_control_method,
                    'lowrank': lowrank,

                    "hamiltonian": {
                        "name": "Generic",
                        "integrals": ref_path + "generic_integrals.h5",
                        "_alt_convention": False,
                        "symmetry": False,
                        "sparse": False,
                        "mu": mu
                    },

                    "estimators": {
                        "filename": tmpf.name,
                    }
                }
    
        # ---------------------------------------------------------------------
        # Test.
        # ---------------------------------------------------------------------
        afqmc, hamiltonian, walkers = build_driver_test_instance(nelec, options, seed, debug, verbose)
        
        # ---------------------------------------------------------------------
        # Legacy.
        # ---------------------------------------------------------------------
        print('\n------------------------------')
        print('Running Legacy ThermalAFQMC...')
        print('------------------------------')
        legacy_afqmc = build_legacy_driver_instance(nelec, hamiltonian, options, seed, verbose)
        legacy_afqmc.run(comm=comm)
        legacy_afqmc.finalise(verbose=False)
        legacy_afqmc.estimators.estimators["mixed"].update(
            legacy_afqmc.qmc,
            legacy_afqmc.system,
            legacy_afqmc.hamiltonian,
            legacy_afqmc.trial,
            legacy_afqmc.walk,
            0,
            legacy_afqmc.propagators.free_projection)
        legacy_mixed_data = extract_mixed_estimates(legacy_afqmc.estimators.filename)

        enum = legacy_afqmc.estimators.estimators["mixed"].names
        legacy_energy_numer = legacy_afqmc.estimators.estimators["mixed"].estimates[enum.enumer]
        legacy_energy_denom = legacy_afqmc.estimators.estimators["mixed"].estimates[enum.edenom]
        
        if verbose:
            print(f'\nThermal AFQMC options: \n{json.dumps(options, indent=4)}\n')
            print(f'legacy_mixed_data: \n{legacy_mixed_data}\n')



if __name__ == '__main__':
    run_legacy()

