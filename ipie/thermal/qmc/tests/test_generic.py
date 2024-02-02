import json
import tempfile
import h5py
import uuid
import pytest
import numpy
from typing import Union

from ipie.config import MPI
from ipie.utils.testing import generate_hamiltonian
from ipie.systems.generic import Generic
from ipie.hamiltonians.generic import Generic as HamGeneric
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


def build_driver_test_instance(options: Union[dict, None],
                               seed: Union[int, None],
                               debug: bool = False,
                               verbose: bool = False):
    # Unpack options
    nelec = options['nelec']
    nbasis = options['nbasis']
    mu = options['mu']
    beta = options['beta']
    timestep = options['timestep']
    nwalkers = options['nwalkers']
    nsteps_per_block = options['nsteps_per_block']
    nblocks = options['nblocks']
    stabilize_freq = options['stabilize_freq']
    pop_control_freq = options['pop_control_freq']
    lowrank = options['lowrank']
    numpy.random.seed(seed)

    complex_integrals = False
    sym = 8
    if complex_integrals: sym = 4
    
    params = ThermalQMCParams(
                num_walkers=nwalkers,
                total_num_walkers=nwalkers * comm.size,
                num_blocks=nblocks,
                timestep=timestep,
                beta=beta,
                num_stblz=stabilize_freq,
                pop_control_freq=pop_control_freq,
                rng_seed=seed)
    
    system = Generic(nelec=nelec)
    h1e, chol, _, _ = generate_hamiltonian(nbasis, nelec, cplx=complex_integrals, 
                                           sym=sym, tol=1e-10)
    hamiltonian = HamGeneric(h1e=numpy.array([h1e, h1e]),
                             chol=chol.reshape((-1, nbasis**2)).T.copy(),
                             ecore=0)
    hamiltonian.name = options["hamiltonian"]["name"]
    hamiltonian._alt_convention = options["hamiltonian"]["_alt_convention"]
    hamiltonian.sparse = options["hamiltonian"]["sparse"]
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


def build_legacy_driver_instance(hamiltonian,
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
    nsteps_per_block = options['nsteps_per_block']
    nblocks = options['nblocks']
    stabilize_freq = options['stabilize_freq']
    pop_control_freq = options['pop_control_freq']
    lowrank = options['lowrank']
    numpy.random.seed(seed)
    
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
            "low_rank": lowrank
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


@pytest.mark.unit
def test_thermal_afqmc():
    # Thermal AFQMC params.
    nup = 5
    ndown = 5
    nelec = (nup, ndown)
    nbasis = 20

    mu = -10.0
    beta = 0.1
    timestep = 0.01
    nwalkers = 10
    seed = 7
    # Must be fixed at 1 for Thermal AFQMC--legacy code overides whatever input!
    nsteps_per_block = 1
    nblocks = 12
    stabilize_freq = 10
    pop_control_freq = 1
    lowrank = False
    verbose = True
    debug = True

    with tempfile.NamedTemporaryFile() as tmpf1, tempfile.NamedTemporaryFile() as tmpf2:
        options = {
                    'nelec': nelec,
                    'nbasis': nbasis,
                    'mu': mu,
                    'beta': beta,
                    'timestep': timestep,
                    'nwalkers': nwalkers,
                    'seed': seed,
                    'nsteps_per_block': nsteps_per_block,
                    'nblocks': nblocks,
                    'stabilize_freq': stabilize_freq,
                    'pop_control_freq': pop_control_freq,
                    'lowrank': lowrank,

                    "hamiltonian": {
                        "name": "Generic",
                        "_alt_convention": False,
                        "sparse": False,
                        "mu": mu
                    },

                    "estimators": {
                        "filename": tmpf2.name, # For legacy.
                    },
                }
    
        # ---------------------------------------------------------------------
        # Test.
        # ---------------------------------------------------------------------
        print('\n-----------------------')
        print('Running ThermalAFQMC...')
        print('-----------------------')
        afqmc, hamiltonian, walkers = build_driver_test_instance(options, seed, debug, verbose)
        afqmc.run(walkers, verbose, estimator_filename=tmpf1.name)
        afqmc.finalise()
        afqmc.estimators.compute_estimators(afqmc.hamiltonian, afqmc.trial, afqmc.walkers)

        test_energy_data = extract_observable(afqmc.estimators.filename, "energy")
        test_energy_numer = afqmc.estimators["energy"]["ENumer"]
        test_energy_denom = afqmc.estimators["energy"]["EDenom"]
        test_number_data = extract_observable(afqmc.estimators.filename, "nav")
        
        # ---------------------------------------------------------------------
        # Legacy.
        # ---------------------------------------------------------------------
        print('\n------------------------------')
        print('Running Legacy ThermalAFQMC...')
        print('------------------------------')
        legacy_afqmc = build_legacy_driver_instance(hamiltonian, options, seed, verbose)
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
            print(f'test filename: {afqmc.estimators.filename}')
            print(f'legacy filename: {legacy_afqmc.estimators.filename}')
            print(f'\ntest_energy_data: \n{test_energy_data}\n')
            print(f'test_number_data: \n{test_number_data}\n')
            print(f'legacy_mixed_data: \n{legacy_mixed_data}\n')
        
        # Check.
        assert test_energy_numer.real == pytest.approx(legacy_energy_numer.real)
        assert test_energy_denom.real == pytest.approx(legacy_energy_denom.real)
        assert test_energy_numer.imag == pytest.approx(legacy_energy_numer.imag)
        assert test_energy_denom.imag == pytest.approx(legacy_energy_denom.imag)
    
        assert numpy.mean(test_energy_data.WeightFactor.values[1:-1].real) == pytest.approx(
            numpy.mean(legacy_mixed_data.WeightFactor.values[1:-1].real))
        assert numpy.mean(test_energy_data.Weight.values[1:-1].real) == pytest.approx(
            numpy.mean(legacy_mixed_data.Weight.values[1:-1].real))
        assert numpy.mean(test_energy_data.ENumer.values[:-1].real) == pytest.approx(
            numpy.mean(legacy_mixed_data.ENumer.values[:-1].real))
        assert numpy.mean(test_energy_data.EDenom.values[:-1].real) == pytest.approx(
            numpy.mean(legacy_mixed_data.EDenom.values[:-1].real))
        assert numpy.mean(test_energy_data.ETotal.values[:-1].real) == pytest.approx(
            numpy.mean(legacy_mixed_data.ETotal.values[:-1].real))
        assert numpy.mean(test_energy_data.E1Body.values[:-1].real) == pytest.approx(
            numpy.mean(legacy_mixed_data.E1Body.values[:-1].real))
        assert numpy.mean(test_energy_data.E2Body.values[:-1].real) == pytest.approx(
            numpy.mean(legacy_mixed_data.E2Body.values[:-1].real))
        assert numpy.mean(test_energy_data.HybridEnergy.values[:-1].real) == pytest.approx(
            numpy.mean(legacy_mixed_data.EHybrid.values[:-1].real))
        assert numpy.mean(test_number_data.Nav.values[:-1].real) == pytest.approx(
            numpy.mean(legacy_mixed_data.Nav.values[:-1].real))
    

if __name__ == '__main__':
    test_thermal_afqmc()

