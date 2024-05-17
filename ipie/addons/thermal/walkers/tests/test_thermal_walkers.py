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
import pytest
from typing import Union

try:
    from ipie.addons.thermal.utils.legacy_testing import build_legacy_generic_test_case_handlers
    from ipie.addons.thermal.utils.legacy_testing import legacy_propagate_walkers
    _no_cython = False

except ModuleNotFoundError:
    _no_cython = True

from ipie.config import MPI
from ipie.addons.thermal.estimators.generic import local_energy_generic_cholesky
from ipie.addons.thermal.estimators.thermal import one_rdm_from_G
from ipie.addons.thermal.utils.testing import build_generic_test_case_handlers

from ipie.legacy.estimators.generic import local_energy_generic_cholesky as legacy_local_energy_generic_cholesky
from ipie.legacy.estimators.thermal import one_rdm_from_G as legacy_one_rdm_from_G

comm = MPI.COMM_WORLD

@pytest.mark.skipif(_no_cython, reason="Need to build cython modules.")
@pytest.mark.unit
def test_thermal_walkers_fullrank():
    # System params.
    nup = 5
    ndown = 5
    nelec = (nup, ndown)
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
    
    # Test.
    objs =  build_generic_test_case_handlers(
            nelec, nbasis, mu, beta, timestep, nwalkers=nwalkers, lowrank=lowrank, 
            mf_trial=mf_trial, complex_integrals=complex_integrals, debug=debug, 
            seed=seed, verbose=verbose)
    trial = objs['trial']
    hamiltonian = objs['hamiltonian']
    walkers = objs['walkers']
    
    # Legacy.
    legacy_objs = build_legacy_generic_test_case_handlers(
                    hamiltonian, comm, nelec, mu, beta, timestep, nwalkers=nwalkers, 
                    lowrank=lowrank, mf_trial=mf_trial, seed=seed, verbose=verbose)
    legacy_system = legacy_objs['system']
    legacy_trial = legacy_objs['trial']
    legacy_hamiltonian = legacy_objs['hamiltonian']
    legacy_walkers = legacy_objs['walkers']
    
    for iw in range(walkers.nwalkers):
        P = one_rdm_from_G(numpy.array([walkers.Ga[iw], walkers.Gb[iw]])) 
        eloc = local_energy_generic_cholesky(hamiltonian, P)

        legacy_P = legacy_one_rdm_from_G(numpy.array(legacy_walkers.walkers[iw].G))
        legacy_eloc = legacy_local_energy_generic_cholesky(
                        legacy_system, legacy_hamiltonian, legacy_P)

        numpy.testing.assert_almost_equal(legacy_eloc, eloc, decimal=10)
        numpy.testing.assert_almost_equal(legacy_walkers.walkers[iw].G[0], walkers.Ga[iw], decimal=10)
        numpy.testing.assert_almost_equal(legacy_walkers.walkers[iw].G[1], walkers.Gb[iw], decimal=10)
        numpy.testing.assert_almost_equal(legacy_walkers.walkers[iw].stack.ovlp[0], walkers.stack[iw].ovlp[0], decimal=10)
        numpy.testing.assert_almost_equal(legacy_walkers.walkers[iw].stack.ovlp[1], walkers.stack[iw].ovlp[1], decimal=10)


# TODO: Lowrank code is WIP.
#@pytest.mark.skipif(_no_cython, reason="Need to build cython modules.")
#@pytest.mark.unit
def test_thermal_walkers_lowrank():
    # System params.
    nup = 5
    ndown = 5
    nelec = (nup, ndown)
    nbasis = 10

    # Thermal AFQMC params.
    mu = -10.
    beta = 0.1
    timestep = 0.01
    nwalkers = 10
    lowrank = True

    mf_trial = False
    diagonal = True
    complex_integrals = False
    debug = True
    verbose = True
    seed = 7
    numpy.random.seed(seed)
    
    # Test.
    objs =  build_generic_test_case_handlers(
            nelec, nbasis, mu, beta, timestep, nwalkers=nwalkers, lowrank=lowrank, 
            mf_trial=mf_trial, complex_integrals=complex_integrals, debug=debug, 
            seed=seed, verbose=verbose)
    trial = objs['trial']
    hamiltonian = objs['hamiltonian']
    walkers = objs['walkers']
    
    # Legacy.
    legacy_objs = build_legacy_generic_test_case_handlers(
                    hamiltonian, comm, nelec, mu, beta, timestep, nwalkers=nwalkers, 
                    lowrank=lowrank, mf_trial=mf_trial, seed=seed, verbose=verbose)
    legacy_system = legacy_objs['system']
    legacy_trial = legacy_objs['trial']
    legacy_hamiltonian = legacy_objs['hamiltonian']
    legacy_walkers = legacy_objs['walkers']
    
    for iw in range(walkers.nwalkers):
        P = one_rdm_from_G(numpy.array([walkers.Ga[iw], walkers.Gb[iw]])) 
        eloc = local_energy_generic_cholesky(hamiltonian, P)

        legacy_P = legacy_one_rdm_from_G(numpy.array(legacy_walkers.walkers[iw].G))
        legacy_eloc = legacy_local_energy_generic_cholesky(
                        legacy_system, legacy_hamiltonian, legacy_P)

        numpy.testing.assert_almost_equal(legacy_eloc, eloc, decimal=10)
        numpy.testing.assert_almost_equal(legacy_walkers.walkers[iw].G[0], walkers.Ga[iw], decimal=10)
        numpy.testing.assert_almost_equal(legacy_walkers.walkers[iw].G[1], walkers.Gb[iw], decimal=10)
        numpy.testing.assert_almost_equal(legacy_walkers.walkers[iw].stack.ovlp[0], walkers.stack[iw].ovlp[0], decimal=10)
        numpy.testing.assert_almost_equal(legacy_walkers.walkers[iw].stack.ovlp[1], walkers.stack[iw].ovlp[1], decimal=10)

if __name__ == "__main__":
    test_thermal_walkers_fullrank()
    #test_thermal_walkers_lowrank()
