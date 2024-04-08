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

import pytest
import tempfile
import numpy
from typing import Tuple, Union

from ipie.config import MPI
from ipie.hamiltonians.generic import Generic as HamGeneric
from ipie.hamiltonians.generic import GenericRealChol, GenericComplexChol

from ipie.addons.thermal.estimators.energy import ThermalEnergyEstimator
from ipie.addons.thermal.estimators.particle_number import ThermalNumberEstimator
from ipie.addons.thermal.estimators.handler import ThermalEstimatorHandler
from ipie.addons.thermal.utils.testing import build_generic_test_case_handlers

# System params.
nup = 5
ndown = 5
nelec = (nup, ndown)
ne = nup + ndown
nbasis = 10

# Thermal AFQMC params.
mu = -10.
beta = 0.1
timestep = 0.01
nwalkers = 10
lowrank = False

mf_trial = True
complex_integrals = False
debug = True
verbose = True
seed = 7
numpy.random.seed(seed)

@pytest.mark.unit
def test_energy_estimator():
    # Test.
    print('\n----------------------------')
    print('Constructing test objects...')
    print('----------------------------')
    objs = build_generic_test_case_handlers(
            nelec, nbasis, mu, beta, timestep, nwalkers=nwalkers, lowrank=lowrank, 
            mf_trial=mf_trial, complex_integrals=complex_integrals, debug=debug, 
            seed=seed, verbose=verbose)
    trial = objs['trial']
    hamiltonian = objs['hamiltonian']
    walkers = objs['walkers']

    assert isinstance(hamiltonian, GenericRealChol)
    chol = hamiltonian.chol

    # GenericRealChol.
    re_estim = ThermalEnergyEstimator(hamiltonian=hamiltonian, trial=trial)
    re_estim.compute_estimator(walkers, hamiltonian, trial)
    assert len(re_estim.names) == 5
    assert re_estim["ENumer"].real == pytest.approx(24.66552451455761)
    assert re_estim["ETotal"] == pytest.approx(0.0)
    tmp = re_estim.data.copy()
    re_estim.post_reduce_hook(tmp)
    assert tmp[re_estim.get_index("ETotal")] == pytest.approx(2.4665524514557613)
    assert re_estim.print_to_stdout
    assert re_estim.ascii_filename == None
    assert re_estim.shape == (5,)
    header = re_estim.header_to_text
    data_to_text = re_estim.data_to_text(tmp)
    assert len(data_to_text.split()) == 5
    
    # GenericComplexChol.
    cx_chol = numpy.array(chol, dtype=numpy.complex128)
    cx_hamiltonian = HamGeneric(
        numpy.array(hamiltonian.H1, dtype=numpy.complex128), cx_chol, 
                    hamiltonian.ecore, verbose=False)

    assert isinstance(cx_hamiltonian, GenericComplexChol)

    cx_estim = ThermalEnergyEstimator(hamiltonian=cx_hamiltonian, trial=trial)
    cx_estim.compute_estimator(walkers, cx_hamiltonian, trial)
    assert len(cx_estim.names) == 5
    assert cx_estim["ENumer"].real == pytest.approx(24.66552451455761)
    assert cx_estim["ETotal"] == pytest.approx(0.0)
    tmp = cx_estim.data.copy()
    cx_estim.post_reduce_hook(tmp)
    assert tmp[cx_estim.get_index("ETotal")] == pytest.approx(2.4665524514557613)
    assert cx_estim.print_to_stdout
    assert cx_estim.ascii_filename == None
    assert cx_estim.shape == (5,)
    header = cx_estim.header_to_text
    data_to_text = cx_estim.data_to_text(tmp)
    assert len(data_to_text.split()) == 5
    
    numpy.testing.assert_allclose(re_estim.data, cx_estim.data)


@pytest.mark.unit
def test_number_estimator():
    # Test.
    print('\n----------------------------')
    print('Constructing test objects...')
    print('----------------------------')
    objs =  build_generic_test_case_handlers(
            nelec, nbasis, mu, beta, timestep, nwalkers=nwalkers, lowrank=lowrank, 
            mf_trial=mf_trial, complex_integrals=True, debug=debug, 
            seed=seed, verbose=verbose)
    trial = objs['trial']
    hamiltonian = objs['hamiltonian']
    walkers = objs['walkers']

    estim = ThermalNumberEstimator(hamiltonian=hamiltonian, trial=trial)
    estim.compute_estimator(walkers, hamiltonian, trial)
    assert len(estim.names) == 3
    assert estim["NavNumer"].real == pytest.approx(ne * nwalkers)
    assert estim["Nav"] == pytest.approx(0.0)
    tmp = estim.data.copy()
    estim.post_reduce_hook(tmp)
    assert tmp[estim.get_index("Nav")] == pytest.approx(ne)
    assert estim.print_to_stdout
    assert estim.ascii_filename == None
    assert estim.shape == (3,)
    header = estim.header_to_text
    data_to_text = estim.data_to_text(tmp)
    assert len(data_to_text.split()) == 3
    

@pytest.mark.unit
def test_estimator_handler():
    with tempfile.NamedTemporaryFile() as tmp1, tempfile.NamedTemporaryFile() as tmp2:
        # Test.
        print('\n----------------------------')
        print('Constructing test objects...')
        print('----------------------------')
        objs =  build_generic_test_case_handlers(
                nelec, nbasis, mu, beta, timestep, nwalkers=nwalkers, lowrank=lowrank, 
                mf_trial=mf_trial, complex_integrals=True, debug=debug, 
                seed=seed, verbose=verbose)
        trial = objs['trial']
        hamiltonian = objs['hamiltonian']
        walkers = objs['walkers']

        estim = ThermalEnergyEstimator(hamiltonian=hamiltonian, trial=trial, 
                                       filename=tmp1.name)
        estim.print_to_stdout = False

        comm = MPI.COMM_WORLD
        handler = ThermalEstimatorHandler(
                    comm,
                    hamiltonian,
                    trial,
                    block_size=10,
                    observables=("energy",),
                    filename=tmp2.name)
        handler["energy1"] = estim
        handler.json_string = ""
        handler.initialize(comm)
        handler.compute_estimators(hamiltonian, trial, walkers)


if __name__ == "__main__":
    test_energy_estimator()
    test_number_estimator()
    test_estimator_handler()



