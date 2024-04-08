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

try:
    from ipie.addons.thermal.utils.legacy_testing import build_legacy_driver_ueg_test_instance
    _no_cython = False

except ModuleNotFoundError:
    _no_cython = True

from ipie.config import MPI
from ipie.analysis.extraction import (
        get_metadata,
        extract_test_data_hdf5, 
        extract_data,
        extract_observable, 
        extract_mixed_estimates)
from ipie.addons.thermal.utils.testing import build_driver_ueg_test_instance

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


@pytest.mark.skipif(_no_cython, reason="Need to build cython modules.")
@pytest.mark.unit
def test_thermal_afqmc_1walker(against_ref=False):
    # UEG params.
    nup = 7
    ndown = 7
    nelec = (nup, ndown)
    rs = 1.
    ecut = 1.

    # Thermal AFQMC params.
    mu = -1.
    beta = 0.1
    timestep = 0.01
    nwalkers = 1
    nblocks = 11
    
    stabilize_freq = 10
    pop_control_freq = 1
    # `pop_control_method` doesn't matter for 1 walker.
    pop_control_method = "pair_branch"
    #pop_control_method = "comb"
    lowrank = False
    
    verbose = False if (comm.rank != 0) else True
    debug = True
    seed = 7
    numpy.random.seed(seed)
    
    with tempfile.NamedTemporaryFile() as tmpf1, tempfile.NamedTemporaryFile() as tmpf2:
        # ---------------------------------------------------------------------
        # Test.
        # ---------------------------------------------------------------------
        if verbose:
            print('\n-----------------------')
            print('Running ThermalAFQMC...')
            print('-----------------------')
        
        afqmc = build_driver_ueg_test_instance(
                nelec, rs, ecut, mu, beta, timestep, nblocks, nwalkers=nwalkers, 
                lowrank=lowrank, pop_control_method=pop_control_method, 
                stabilize_freq=stabilize_freq, pop_control_freq=pop_control_freq,
                debug=debug, seed=seed, verbose=verbose)
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

        legacy_afqmc = build_legacy_driver_ueg_test_instance(
                        comm, nelec, rs, ecut, mu, beta, timestep, nblocks, 
                        nwalkers=nwalkers, lowrank=lowrank, 
                        stabilize_freq=stabilize_freq,
                        pop_control_freq=pop_control_freq,
                        pop_control_method=pop_control_method, seed=seed, 
                        estimator_filename=tmpf2.name, verbose=verbose)
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
        
            print(f'\ntest filename: {afqmc.estimators.filename}')
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
                _data_dir = os.path.abspath(os.path.dirname(__file__)).split("qmc")[0] + "/reference_data/"
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
                    print(f"\n*** PASSED : {test_name} ***\n")


@pytest.mark.skipif(_no_cython, reason="Need to build cython modules.")
@pytest.mark.unit
def test_thermal_afqmc(against_ref=False):
    # UEG params.
    nup = 7
    ndown = 7
    nelec = (nup, ndown)
    rs = 1.
    ecut = 1.

    # Thermal AFQMC params.
    mu = -1.
    beta = 0.1
    timestep = 0.01
    nwalkers = 32
    # Must be fixed at 1 for Thermal AFQMC--legacy code overides whatever input!
    nsteps_per_block = 1
    nblocks = 10
    stabilize_freq = 10
    pop_control_freq = 1
    pop_control_method = "pair_branch"
    #pop_control_method = "comb"
    lowrank = False
    
    verbose = False if (comm.rank != 0) else True
    debug = True
    seed = 7
    numpy.random.seed(seed)
    
    with tempfile.NamedTemporaryFile() as tmpf1, tempfile.NamedTemporaryFile() as tmpf2:
        # ---------------------------------------------------------------------
        # Test.
        # ---------------------------------------------------------------------
        if verbose:
            print('\n-----------------------')
            print('Running ThermalAFQMC...')
            print('-----------------------')
        
        afqmc = build_driver_ueg_test_instance(
                nelec, rs, ecut, mu, beta, timestep, nblocks, nwalkers=nwalkers, 
                lowrank=lowrank, pop_control_method=pop_control_method, 
                stabilize_freq=stabilize_freq, pop_control_freq=pop_control_freq,
                debug=debug, seed=seed, verbose=verbose)
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

        legacy_afqmc = build_legacy_driver_ueg_test_instance(
                        comm, nelec, rs, ecut, mu, beta, timestep, nblocks, 
                        nwalkers=nwalkers, lowrank=lowrank,
                        stabilize_freq=stabilize_freq,
                        pop_control_freq=pop_control_freq,
                        pop_control_method=pop_control_method, seed=seed, 
                        estimator_filename=tmpf2.name, verbose=verbose)
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
        
            print(f'\ntest filename: {afqmc.estimators.filename}')
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
                _data_dir = os.path.abspath(os.path.dirname(__file__)).split("qmc")[0] + "/reference_data/"
                _legacy_test_dir = "ueg"
                _legacy_test = _data_dir + _legacy_test_dir + "/reference_nompi.json"
                
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
                    print(f"\n*** PASSED : {test_name} ***\n")


@pytest.mark.skipif(_no_cython, reason="Need to build cython modules.")
@pytest.mark.mpi
def test_thermal_afqmc_mpi(against_ref=False):
    # UEG params.
    nup = 7
    ndown = 7
    nelec = (nup, ndown)
    rs = 1.
    ecut = 1.

    # Thermal AFQMC params.
    mu = -1.
    beta = 0.1
    timestep = 0.01
    nwalkers = 32 // comm.size
    # Must be fixed at 1 for Thermal AFQMC--legacy code overides whatever input!
    nsteps_per_block = 1
    nblocks = 10
    stabilize_freq = 10
    pop_control_freq = 1
    pop_control_method = "pair_branch"
    #pop_control_method = "comb"
    lowrank = False
    
    verbose = False if (comm.rank != 0) else True
    debug = True
    seed = 7
    numpy.random.seed(seed)
    
    with tempfile.NamedTemporaryFile() as tmpf1, tempfile.NamedTemporaryFile() as tmpf2:
        # ---------------------------------------------------------------------
        # Test.
        # ---------------------------------------------------------------------
        if verbose:
            print('\n-----------------------')
            print('Running ThermalAFQMC...')
            print('-----------------------')
        
        afqmc = build_driver_ueg_test_instance(
                nelec, rs, ecut, mu, beta, timestep, nblocks, nwalkers=nwalkers, 
                lowrank=lowrank, pop_control_method=pop_control_method, 
                stabilize_freq=stabilize_freq, pop_control_freq=pop_control_freq,
                debug=debug, seed=seed, verbose=verbose)
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

        legacy_afqmc = build_legacy_driver_ueg_test_instance(
                        comm, nelec, rs, ecut, mu, beta, timestep, nblocks, 
                        nwalkers=nwalkers, lowrank=lowrank,
                        stabilize_freq=stabilize_freq,
                        pop_control_freq=pop_control_freq,
                        pop_control_method=pop_control_method, seed=seed, 
                        estimator_filename=tmpf2.name, verbose=verbose)
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
        
            print(f'\ntest filename: {afqmc.estimators.filename}')
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
                _data_dir = os.path.abspath(os.path.dirname(__file__)).split("qmc")[0] + "/reference_data/"
                _legacy_test_dir = "ueg"
                _legacy_test = _data_dir + _legacy_test_dir + "/reference.json"
                
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
                    print(f"\n*** PASSED : {test_name} ***\n")


if __name__ == '__main__':
    test_thermal_afqmc_1walker(against_ref=True)
    test_thermal_afqmc(against_ref=True)
    #test_thermal_afqmc_mpi(against_ref=True)

