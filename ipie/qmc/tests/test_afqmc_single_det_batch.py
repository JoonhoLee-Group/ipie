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
import tempfile

import numpy
import pytest

from ipie.analysis.extraction import extract_mixed_estimates, extract_observable
from ipie.config import MPI
from ipie.qmc.calc import AFQMC
from ipie.utils.io import write_hamiltonian, write_wavefunction
from ipie.utils.legacy_testing import build_legacy_driver_instance
from ipie.utils.testing import build_driver_test_instance

steps = 25
blocks = 10
seed = 7
nwalkers = 10
nmo = 14
nelec = (4, 3)
# steps = 1
# blocks = 10
# seed = 7
# nwalkers = 1
# nmo = 4
# nelec = (2, 1)

pop_control_freq = 5
stabilise_freq = 5
comm = MPI.COMM_WORLD

options = {
    "dt": 0.005,
    "nstblz": 5,
    "nwalkers": nwalkers,
    "nwalkers_per_task": nwalkers,
    "batched": True,
    "hybrid": True,
    "steps": steps,
    "blocks": blocks,
    "pop_control_freq": pop_control_freq,
    "stabilise_freq": stabilise_freq,
    "rng_seed": seed,
}


@pytest.mark.driver
def test_generic_single_det_batch():
    with tempfile.NamedTemporaryFile() as tmpf:
        driver_options = {
            "verbosity": 0,
            "get_sha1": False,
            "qmc": options,
            "estimates": {"filename": tmpf.name, "observables": {"energy": {}}},
            "walkers": {"population_control": "pair_branch"},
        }

        afqmc = build_driver_test_instance(
            nelec, nmo, trial_type="single_det", options=driver_options, seed=7
        )
        afqmc.run(verbose=False, estimator_filename=tmpf.name)
        afqmc.finalise(verbose=0)
        afqmc.estimators.compute_estimators(
            comm, afqmc.system, afqmc.hamiltonian, afqmc.trial, afqmc.walkers
        )
        numer_batch = afqmc.estimators["energy"]["ENumer"]
        denom_batch = afqmc.estimators["energy"]["EDenom"]
        data_batch = extract_observable(tmpf.name, "energy")
        driver_options["estimates"] = {
            "filename": tmpf.name,
            "mixed": {"energy_eval_freq": options["steps"]},
        }
        options["batched"] = False
        legacy_afqmc = build_legacy_driver_instance(
            nelec, nmo, trial_type="single_det", options=driver_options, seed=7
        )
        legacy_afqmc.run(comm=comm, verbose=1)
        legacy_afqmc.finalise(verbose=0)
        legacy_afqmc.estimators.estimators["mixed"].update(
            legacy_afqmc.qmc,
            legacy_afqmc.system,
            legacy_afqmc.hamiltonian,
            legacy_afqmc.trial,
            legacy_afqmc.psi,
            0,
        )

        enum = legacy_afqmc.estimators.estimators["mixed"].names
        numer = legacy_afqmc.estimators.estimators["mixed"].estimates[enum.enumer]
        denom = legacy_afqmc.estimators.estimators["mixed"].estimates[enum.edenom]
        weight = legacy_afqmc.estimators.estimators["mixed"].estimates[enum.weight]

        assert numer.real == pytest.approx(numer_batch.real)
        assert denom.real == pytest.approx(denom_batch.real)
        # assert weight.real == pytest.approx(weight_batch.real)
        assert numer.imag == pytest.approx(numer_batch.imag)
        assert denom.imag == pytest.approx(denom_batch.imag)
        # assert weight.imag == pytest.approx(weight_batch.imag)
        data = extract_mixed_estimates(tmpf.name)

        assert numpy.mean(data_batch.WeightFactor.values[1:-1].real) == pytest.approx(
            numpy.mean(data.WeightFactor.values[1:-1].real)
        )
        assert numpy.mean(data_batch.Weight.values[1:-1].real) == pytest.approx(
            numpy.mean(data.Weight.values[1:-1].real)
        )
        assert numpy.mean(data_batch.ENumer.values[:-1].real) == pytest.approx(
            numpy.mean(data.ENumer.values[:-1].real)
        )
        assert numpy.mean(data_batch.EDenom.values[:-1].real) == pytest.approx(
            numpy.mean(data.EDenom.values[:-1].real)
        )
        assert numpy.mean(data_batch.ETotal.values[:-1].real) == pytest.approx(
            numpy.mean(data.ETotal.values[:-1].real)
        )
        assert numpy.mean(data_batch.E1Body.values[:-1].real) == pytest.approx(
            numpy.mean(data.E1Body.values[:-1].real)
        )
        assert numpy.mean(data_batch.E2Body.values[:-1].real) == pytest.approx(
            numpy.mean(data.E2Body.values[:-1].real)
        )
        assert numpy.mean(data_batch.HybridEnergy.values[:-1].real) == pytest.approx(
            numpy.mean(data.EHybrid.values[:-1].real)
        )
        # no longer computed
        # assert numpy.mean(data_batch.Overlap.values[:-2].real) == pytest.approx(
        # numpy.mean(data.Overlap.values[:-1].real)
        # )


@pytest.mark.driver
def test_generic_single_det_batch_density_diff():
    with tempfile.NamedTemporaryFile() as tmpf:
        driver_options = {
            "verbosity": 0,
            "get_sha1": False,
            "qmc": options,
            "estimates": {"filename": tmpf.name, "observables": {"energy": {}}},
            "walkers": {"population_control": "pair_branch"},
        }
        driver_options["estimates"] = {"filename": tmpf.name, "observables": {"energy": {}}}
        comm = MPI.COMM_WORLD

        driver_options["qmc"]["batched"] = True
        afqmc = build_driver_test_instance(
            nelec, nmo, trial_type="single_det", options=driver_options, seed=7, density_diff=True
        )
        afqmc.run(verbose=False, estimator_filename=tmpf.name)
        afqmc.finalise(verbose=0)
        afqmc.estimators.compute_estimators(
            comm, afqmc.system, afqmc.hamiltonian, afqmc.trial, afqmc.walkers
        )

        numer_batch = afqmc.estimators["energy"]["ENumer"]
        denom_batch = afqmc.estimators["energy"]["EDenom"]
        # weight_batch = afqmc.estimators['energy']['Weight']

        data_batch = extract_observable(tmpf.name, "energy")

        numpy.random.seed(seed)
        driver_options["estimates"] = {"filename": tmpf.name, "mixed": {"energy_eval_freq": steps}}
        driver_options["qmc"]["batched"] = False
        legacy_afqmc = build_legacy_driver_instance(
            nelec, nmo, trial_type="single_det", options=driver_options, seed=7, density_diff=True
        )
        legacy_afqmc.run(comm=comm, verbose=1)
        legacy_afqmc.finalise(verbose=0)
        legacy_afqmc.estimators.estimators["mixed"].update(
            legacy_afqmc.qmc,
            legacy_afqmc.system,
            legacy_afqmc.hamiltonian,
            legacy_afqmc.trial,
            legacy_afqmc.psi,
            0,
        )
        enum = legacy_afqmc.estimators.estimators["mixed"].names
        numer = legacy_afqmc.estimators.estimators["mixed"].estimates[enum.enumer]
        denom = legacy_afqmc.estimators.estimators["mixed"].estimates[enum.edenom]
        weight = legacy_afqmc.estimators.estimators["mixed"].estimates[enum.weight]

        assert numer.real == pytest.approx(numer_batch.real)
        assert denom.real == pytest.approx(denom_batch.real)
        # assert weight.real == pytest.approx(weight_batch.real)
        assert numer.imag == pytest.approx(numer_batch.imag)
        assert denom.imag == pytest.approx(denom_batch.imag)
        # assert weight.imag == pytest.approx(weight_batch.imag)
        data = extract_mixed_estimates(tmpf.name)

        # print(data_batch.ENumer)
        # print(data.ENumer)
        assert numpy.mean(data_batch.WeightFactor.values[1:-1].real) == pytest.approx(
            numpy.mean(data.WeightFactor.values[1:-1].real)
        )
        assert numpy.mean(data_batch.Weight.values[1:-1].real) == pytest.approx(
            numpy.mean(data.Weight.values[1:-1].real)
        )
        assert numpy.mean(data_batch.ENumer.values[:-1].real) == pytest.approx(
            numpy.mean(data.ENumer.values[:-1].real)
        )
        assert numpy.mean(data_batch.EDenom.values[:-1].real) == pytest.approx(
            numpy.mean(data.EDenom.values[:-1].real)
        )
        assert numpy.mean(data_batch.ETotal.values[:-1].real) == pytest.approx(
            numpy.mean(data.ETotal.values[:-1].real)
        )
        assert numpy.mean(data_batch.E1Body.values[:-1].real) == pytest.approx(
            numpy.mean(data.E1Body.values[:-1].real)
        )
        assert numpy.mean(data_batch.E2Body.values[:-1].real) == pytest.approx(
            numpy.mean(data.E2Body.values[:-1].real)
        )
        assert numpy.mean(data_batch.HybridEnergy.values[:-1].real) == pytest.approx(
            numpy.mean(data.EHybrid.values[:-1].real)
        )
        # assert numpy.mean(data_batch.Overlap.values[:-1].real) == pytest.approx(
        # numpy.mean(data.Overlap.values[:-1].real)
        # )


@pytest.mark.driver
def test_factory_method():
    with tempfile.NamedTemporaryFile() as hamilf, tempfile.NamedTemporaryFile() as wfnf:
        numpy.random.seed(7)
        nmo = 17
        nelec = (7, 3)
        nmo = 10
        naux = 100
        hcore = numpy.random.random((nmo, nmo))
        LXmn = numpy.random.random((naux, nmo, nmo))
        write_hamiltonian(hcore, LXmn, 0.0, filename=hamilf.name)
        nalpha = nelec[0]
        nbeta = nelec[1]
        wfna = numpy.random.random((nmo, nalpha))
        wfnb = numpy.random.random((nmo, nbeta))
        wfn = [wfna, wfnb]
        write_wavefunction(wfn, filename=wfnf.name)
        AFQMC.build_from_hdf5(nelec, hamilf.name, wfnf.name)


if __name__ == "__main__":
    test_generic_single_det_batch()
    test_generic_single_det_batch_density_diff()
