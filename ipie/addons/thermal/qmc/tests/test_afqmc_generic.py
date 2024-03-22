import json
import tempfile
import h5py
import uuid
import pytest
import numpy
from typing import Union

try:
    from ipie.addons.thermal.utils.legacy_testing import build_legacy_driver_generic_test_instance
    _no_cython = False

except ModuleNotFoundError:
    _no_cython = True

from ipie.config import MPI
from ipie.analysis.extraction import (
        extract_test_data_hdf5, 
        extract_data,
        extract_observable, 
        extract_mixed_estimates)
from ipie.addons.thermal.utils.testing import build_driver_generic_test_instance

comm = MPI.COMM_WORLD
serial_test = comm.size == 1

# Unique filename to avoid name collision when running through CI.
if comm.rank == 0:
    test_id = str(uuid.uuid1())

else:
    test_id = None

test_id = comm.bcast(test_id, root=0)


@pytest.mark.skipif(_no_cython, reason="Need to build cython modules.")
@pytest.mark.unit
def test_thermal_afqmc():
    # System params.
    nup = 5
    ndown = 5
    nelec = (nup, ndown)
    nbasis = 10

    # Thermal AFQMC params.
    mu = -10.0
    beta = 0.1
    timestep = 0.01
    nwalkers = 32 // comm.size
    # Must be fixed at 1 for Thermal AFQMC--legacy code overides whatever input!
    nsteps_per_block = 1
    nblocks = 12
    stabilize_freq = 10
    pop_control_freq = 1
    pop_control_method = 'pair_branch'
    #pop_control_method = 'comb'
    lowrank = False

    verbose = 0 if (comm.rank != 0) else 1
    complex_integrals = False
    debug = True
    seed = 7
    numpy.random.seed(seed)

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
                    'pop_control_method': pop_control_method,
                    'lowrank': lowrank,
                    'complex_integrals': complex_integrals,

                    "estimators": {
                        "filename": tmpf2.name, # For legacy.
                    },
                }
    
        # ---------------------------------------------------------------------
        # Test.
        # ---------------------------------------------------------------------
        if verbose:
            print('\n-----------------------')
            print('Running ThermalAFQMC...')
            print('-----------------------')
    
        afqmc = build_driver_generic_test_instance(options, seed, debug, verbose)
        afqmc.run(verbose=verbose, estimator_filename=tmpf1.name)
        afqmc.finalise()
        afqmc.estimators.compute_estimators(afqmc.hamiltonian, afqmc.trial, afqmc.walkers)
        
        test_energy_data = None
        test_energy_numer = None
        test_energy_denom = None
        test_number_data = None

        if comm.rank == 0:
            test_energy_data = extract_observable(afqmc.estimators.filename, "energy")
            test_energy_numer = afqmc.estimators["energy"]["ENumer"]
            test_energy_denom = afqmc.estimators["energy"]["EDenom"]
            test_number_data = extract_observable(afqmc.estimators.filename, "nav")
        
        # ---------------------------------------------------------------------
        # Legacy.
        # ---------------------------------------------------------------------
        if verbose:
            print('\n------------------------------')
            print('Running Legacy ThermalAFQMC...')
            print('------------------------------')

        legacy_afqmc = build_legacy_driver_generic_test_instance(
                        afqmc.hamiltonian, comm, options, seed, verbose)
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

        legacy_mixed_data = None
        enum = None
        legacy_energy_numer = None
        legacy_energy_denom = None

        if comm.rank == 0:
            legacy_mixed_data = extract_mixed_estimates(legacy_afqmc.estimators.filename)
            enum = legacy_afqmc.estimators.estimators["mixed"].names
            legacy_energy_numer = legacy_afqmc.estimators.estimators["mixed"].estimates[enum.enumer]
            legacy_energy_denom = legacy_afqmc.estimators.estimators["mixed"].estimates[enum.edenom]
        
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

