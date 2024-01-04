import os
import sys
import json
import pprint
import tempfile
import h5py
import uuid
import pytest
import numpy
from typing import Union

from ueg import UEG
from ipie.config import MPI
from ipie.systems.generic import Generic
from ipie.hamiltonians.generic import Generic as HamGeneric
from ipie.qmc.options import QMCOpts

from ipie.thermal.trial.one_body import OneBody
from ipie.thermal.walkers.uhf_walkers import UHFThermalWalkers
from ipie.thermal.propagation.phaseless_generic import PhaselessGeneric
from ipie.thermal.qmc.options import ThermalQMCParams
from ipie.thermal.qmc.thermal_afqmc import ThermalAFQMC
from ipie.thermal.estimators.energy import local_energy
from ipie.analysis.extraction import (
        get_metadata,
        extract_test_data_hdf5, 
        extract_data,
        extract_observable, 
        extract_mixed_estimates)

from ipie.legacy.systems.ueg import UEG as LegacyUEG
from ipie.legacy.hamiltonians.ueg import UEG as LegacyHamUEG
from ipie.legacy.trial_density_matrices.onebody import OneBody as LegacyOneBody
from ipie.legacy.walkers.thermal import ThermalWalker
from ipie.legacy.thermal_propagation.planewave import PlaneWave
from ipie.legacy.estimators.ueg import local_energy_ueg as legacy_local_energy_ueg
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


def compare_test_data(ref_data, test_data):
    comparison = {}

    for k, v in ref_data.items():
        alias = [k]

        if k == "sys_info":
            continue

        elif k == "EHybrid":
            alias.append("HybridEnergy")
        
        err = 0
        ref = ref_data[k]

        for a in alias:
            try:
                test = test_data[a]
                comparison[k] = (
                    numpy.array(ref),
                    numpy.array(test),
                    numpy.max(numpy.abs(numpy.array(ref) - numpy.array(test))) < 1e-10)

            except KeyError:
                err += 1

        if err == len(alias):
            print(f"# Issue with test data key {k}")

    return comparison


def build_driver_test_instance(ueg: UEG,
                               options: Union[dict, None],
                               seed: Union[int, None],
                               debug: bool = False,
                               verbose: bool = False):
    # Unpack options
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

    nbasis = ueg.nbasis
    nchol = ueg.nchol
    nelec = (ueg.nup, ueg.ndown)
    nup, ndown = nelec

    h1 = ueg.H1[0]
    chol = 2. * ueg.chol_vecs.toarray().copy()
    #ecore = ueg.ecore
    ecore = 0.

    params = ThermalQMCParams(
                num_walkers=nwalkers,
                total_num_walkers=nwalkers * comm.size,
                num_blocks=nblocks,
                timestep=timestep,
                beta=beta,
                num_stblz=stabilize_freq,
                pop_control_freq=pop_control_freq,
                rng_seed=seed)
    
    print('\n----------------------------')
    print('Constructing test objects...')
    print('----------------------------')
    
    # 1. Build Hamiltonian.
    hamiltonian = HamGeneric(
            numpy.array([h1, h1], dtype=numpy.complex128), 
            numpy.array(chol, dtype=numpy.complex128), 
            ecore,
            verbose=verbose)
    hamiltonian.name = options["hamiltonian"]["name"]
    hamiltonian._alt_convention = options["hamiltonian"]["_alt_convention"]
    hamiltonian.sparse = options["hamiltonian"]["sparse"]

    # 2. Build trial.
    trial = OneBody(hamiltonian, nelec, beta, timestep, verbose=verbose)

    # 3. Build walkers.
    walkers = UHFThermalWalkers(trial, nbasis, nwalkers, lowrank=lowrank, 
                                verbose=verbose)

    # 4. Build propagator.
    propagator = PhaselessGeneric(timestep, mu, lowrank=lowrank, verbose=verbose)
    propagator.build(hamiltonian, trial=trial, walkers=walkers, verbose=verbose)

    eloc = local_energy(hamiltonian, walkers)
    print(f'# Initial energy = {eloc[0]}')
    
    # 5. Build Thermal AFQMC driver.
    # Dummy system.
    system = None
    afqmc = ThermalAFQMC(system, hamiltonian, trial, walkers, propagator, 
                         params, debug=debug, verbose=verbose)
    return afqmc, hamiltonian, walkers


def build_legacy_driver_instance(hamiltonian,
                                 options: Union[dict, None],
                                 seed: Union[int, None],
                                 verbose: bool = False):
    # Unpack options
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
        
        "ueg_opts": options["ueg_opts"],
        "hamiltonian": options["hamiltonian"],
        "estimators": options["estimators"]
    }

    # 1. Build out system.
    legacy_system = LegacyUEG(options=legacy_options["ueg_opts"])
    legacy_system.mu = mu
    
    # 2. Build Hamiltonian.
    legacy_hamiltonian = LegacyHamUEG(legacy_system, options=legacy_options["ueg_opts"])
    legacy_hamiltonian._alt_convention = legacy_options["hamiltonian"]["_alt_convention"]
    legacy_hamiltonian.mu = legacy_options["hamiltonian"]["mu"]

    # 3. Build trial.
    legacy_trial = LegacyOneBody(legacy_system, legacy_hamiltonian, beta, timestep, verbose=verbose)
    
    # 4. Build walkers.
    legacy_walkers = [
            ThermalWalker(
                legacy_system, legacy_hamiltonian, legacy_trial,
                walker_opts=legacy_options, verbose=i == 0) for i in range(nwalkers)]

    # 5. Build propagator.
    qmc_opts = QMCOpts()
    qmc_opts.nwalkers = nwalkers
    qmc_opts.ntot_walkers = nwalkers
    qmc_opts.beta = beta
    qmc_opts.nsteps = nsteps_per_block
    qmc_opts.dt = timestep
    qmc_opts.seed = seed

    legacy_propagator = PlaneWave(legacy_system, legacy_hamiltonian, legacy_trial, qmc_opts,
                                  options=legacy_options["propagator"], lowrank=lowrank, verbose=verbose)
    afqmc = LegacyThermalAFQMC(comm, legacy_options, legacy_system, 
                               legacy_hamiltonian, legacy_trial, verbose=verbose)
    return afqmc


@pytest.mark.unit
def test_thermal_afqmc(against_ref=False):
    # Thermal AFQMC params.
    seed = 7
    mu = -1.
    beta = 0.1
    timestep = 0.01
    nwalkers = 1
    # Must be fixed at 1 for Thermal AFQMC--legacy code overides whatever input!
    nsteps_per_block = 1
    nblocks = 11
    stabilize_freq = 10
    pop_control_freq = 1
    lowrank = False
    verbose = True
    debug = True
    numpy.random.seed(seed)
    
    with tempfile.NamedTemporaryFile() as tmpf1, tempfile.NamedTemporaryFile() as tmpf2:
        ueg_opts = {
                "nup": 7,
                "ndown": 7,
                "rs": 1.,
                "ecut": 1.,
                "thermal": True,
                "write_integrals": False,
                "low_rank": lowrank
                }

        # Generate UEG integrals.
        ueg = UEG(ueg_opts, verbose=verbose)
        ueg.build(verbose=verbose)
        nbasis = ueg.nbasis
        nchol = ueg.nchol
        nelec = (ueg.nup, ueg.ndown)
        nup, ndown = nelec

        if verbose:
            print(f"# nbasis = {nbasis}")
            print(f"# nchol = {nchol}")
            print(f"# nup = {nup}")
            print(f"# ndown = {ndown}")
        
        # ---------------------------------------------------------------------
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
                        "name": "UEG",
                        "_alt_convention": False,
                        "sparse": False,
                        "mu": mu
                    },

                    "estimators": {
                        "filename": tmpf2.name, # For legacy.
                    },

                    "ueg_opts": ueg_opts
                }
    
        # ---------------------------------------------------------------------
        # Test.
        # ---------------------------------------------------------------------
        print('\n-----------------------')
        print('Running ThermalAFQMC...')
        print('-----------------------')
        afqmc, hamiltonian, walkers = build_driver_test_instance(ueg, options, seed, debug, verbose)
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
        
        # ---------------------------------------------------------------------
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
        
        # ---------------------------------------------------------------------
        # Test against reference data.
        if against_ref:
            _data_dir = os.path.abspath(os.path.dirname(__file__)).split("ueg")[0] + "/reference_data/"
            _legacy_test_dir = "ueg"
            _legacy_test = _data_dir + _legacy_test_dir + "/reference_1walker.json"
            
            test_name = _legacy_test_dir
            with open(_legacy_test, "r") as f:
                ref_data = json.load(f)

            skip_val = ref_data.get("extract_skip_value", 10)
            _test_energy_data = test_energy_data[::skip_val].to_dict(orient="list")
            _test_number_data = test_number_data[::skip_val].to_dict(orient="list")
            energy_comparison = compare_test_data(ref_data, _test_energy_data)
            number_comparison = compare_test_data(ref_data, _test_number_data)
            
            print('\nenergy comparison:')
            pprint.pprint(energy_comparison)
            print('\nnumber comparison:')
            pprint.pprint(number_comparison)

            local_err_count = 0

            for k, v in energy_comparison.items():
                if not v[-1]:
                    local_err_count += 1
                    print(f"\n *** FAILED *** : mismatch between benchmark and test run: {test_name}")
                    print(f" name = {k}\n ref = {v[0]}\n test = {v[1]}\n delta = {v[0]-v[1]}\n")

            for k, v in number_comparison.items():
                if not v[-1]:
                    local_err_count += 1
                    print(f"\n *** FAILED *** : mismatch between benchmark and test run: {test_name}")
                    print(f" name = {k}\n ref = {v[0]}\n test = {v[1]}\n delta = {v[0]-v[1]}\n")

            if local_err_count == 0:
                print(f"\n*** PASSED : {test_name} ***")


if __name__ == '__main__':
    test_thermal_afqmc(against_ref=True)

