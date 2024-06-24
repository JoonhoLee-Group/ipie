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

import json
import tempfile
import uuid
from typing import Union

import h5py
import numpy
import pytest

try:
    from ipie.addons.thermal.utils.legacy_testing import build_legacy_driver_generic_test_instance

    _no_cython = False

except ModuleNotFoundError:
    _no_cython = True

from ipie.addons.thermal.utils.testing import build_driver_generic_test_instance
from ipie.analysis.extraction import (
    extract_data,
    extract_mixed_estimates,
    extract_observable,
    extract_test_data_hdf5,
)
from ipie.config import MPI

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
    nblocks = 12
    stabilize_freq = 10
    pop_control_freq = 1
    pop_control_method = "pair_branch"
    # pop_control_method = 'comb'
    lowrank = False

    verbose = 0 if (comm.rank != 0) else 1
    # Local energy evaluation in legacy code seems wrong.
    complex_integrals = False
    debug = True
    seed = 7
    numpy.random.seed(seed)

    with tempfile.NamedTemporaryFile() as tmpf1, tempfile.NamedTemporaryFile() as tmpf2:
        # ---------------------------------------------------------------------
        # Test.
        # ---------------------------------------------------------------------
        afqmc = build_driver_generic_test_instance(
            nelec,
            nbasis,
            mu,
            beta,
            timestep,
            nblocks,
            nwalkers=nwalkers,
            lowrank=lowrank,
            pop_control_method=pop_control_method,
            stabilize_freq=stabilize_freq,
            pop_control_freq=pop_control_freq,
            complex_integrals=complex_integrals,
            debug=debug,
            seed=seed,
            verbose=verbose,
        )
        afqmc.run(verbose=verbose, estimator_filename=tmpf1.name)
        afqmc.finalise()
        afqmc.estimators.compute_estimators(
            hamiltonian=afqmc.hamiltonian, trial=afqmc.trial, walker_batch=afqmc.walkers
        )

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
        legacy_afqmc = build_legacy_driver_generic_test_instance(
            afqmc.hamiltonian,
            comm,
            nelec,
            mu,
            beta,
            timestep,
            nblocks,
            nwalkers=nwalkers,
            lowrank=lowrank,
            stabilize_freq=stabilize_freq,
            pop_control_freq=pop_control_freq,
            pop_control_method=pop_control_method,
            seed=seed,
            estimator_filename=tmpf2.name,
            verbose=verbose,
        )
        legacy_afqmc.run(comm=comm)
        legacy_afqmc.finalise(verbose=False)
        legacy_afqmc.estimators.estimators["mixed"].update(
            legacy_afqmc.qmc,
            legacy_afqmc.system,
            legacy_afqmc.hamiltonian,
            legacy_afqmc.trial,
            legacy_afqmc.walk,
            0,
            legacy_afqmc.propagators.free_projection,
        )

        legacy_mixed_data = None
        enum = None
        legacy_energy_numer = None
        legacy_energy_denom = None

        if comm.rank == 0:
            legacy_mixed_data = extract_mixed_estimates(legacy_afqmc.estimators.filename)
            enum = legacy_afqmc.estimators.estimators["mixed"].names
            legacy_energy_numer = legacy_afqmc.estimators.estimators["mixed"].estimates[enum.enumer]
            legacy_energy_denom = legacy_afqmc.estimators.estimators["mixed"].estimates[enum.edenom]

            # Check.
            assert test_energy_numer.real == pytest.approx(legacy_energy_numer.real)
            assert test_energy_denom.real == pytest.approx(legacy_energy_denom.real)
            assert test_energy_numer.imag == pytest.approx(legacy_energy_numer.imag)
            assert test_energy_denom.imag == pytest.approx(legacy_energy_denom.imag)

            assert numpy.mean(test_energy_data.WeightFactor.values[1:-1].real) == pytest.approx(
                numpy.mean(legacy_mixed_data.WeightFactor.values[1:-1].real)
            )
            assert numpy.mean(test_energy_data.Weight.values[1:-1].real) == pytest.approx(
                numpy.mean(legacy_mixed_data.Weight.values[1:-1].real)
            )
            assert numpy.mean(test_energy_data.ENumer.values[:-1].real) == pytest.approx(
                numpy.mean(legacy_mixed_data.ENumer.values[:-1].real)
            )
            assert numpy.mean(test_energy_data.EDenom.values[:-1].real) == pytest.approx(
                numpy.mean(legacy_mixed_data.EDenom.values[:-1].real)
            )
            assert numpy.mean(test_energy_data.ETotal.values[:-1].real) == pytest.approx(
                numpy.mean(legacy_mixed_data.ETotal.values[:-1].real)
            )
            assert numpy.mean(test_energy_data.E1Body.values[:-1].real) == pytest.approx(
                numpy.mean(legacy_mixed_data.E1Body.values[:-1].real)
            )
            assert numpy.mean(test_energy_data.E2Body.values[:-1].real) == pytest.approx(
                numpy.mean(legacy_mixed_data.E2Body.values[:-1].real)
            )
            assert numpy.mean(test_energy_data.HybridEnergy.values[:-1].real) == pytest.approx(
                numpy.mean(legacy_mixed_data.EHybrid.values[:-1].real)
            )
            assert numpy.mean(test_number_data.Nav.values[:-1].real) == pytest.approx(
                numpy.mean(legacy_mixed_data.Nav.values[:-1].real)
            )


if __name__ == "__main__":
    test_thermal_afqmc()
