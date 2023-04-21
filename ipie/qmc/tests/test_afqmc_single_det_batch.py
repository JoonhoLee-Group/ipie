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

import numpy
import pytest
from mpi4py import MPI

from ipie.analysis.extraction import extract_observable, extract_mixed_estimates
from ipie.utils.testing import build_driver_test_instance
from ipie.utils.legacy_testing import build_legacy_driver_instance

steps = 25
blocks = 7
seed = 7
nwalkers = 2
nmo = 14
nelec = (4, 3)
# steps = 1
# blocks = 10
# seed = 7
# nwalkers = 1
# nmo = 4
# nelec = (2, 1)

pop_control_freq = 1
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
driver_options = {
    "verbosity": 0,
    "get_sha1": False,
    "qmc": options,
    "estimates": {
        "filename": "estimates.test_generic_single_det_batch.h5",
        "observables": {
            "energy": {},
        },
    },
    "walkers": {"population_control": "pair_branch"},
}


@pytest.mark.driver
def test_generic_single_det_batch():
    afqmc = build_driver_test_instance(
        nelec, nmo, trial_type="single_det", options=driver_options, seed=7
    )
    afqmc.run(comm=comm, verbose=0)
    afqmc.finalise(verbose=0)
    afqmc.estimators.compute_estimators(
        comm,
        afqmc.system,
        afqmc.hamiltonian,
        afqmc.trial,
        afqmc.psi,
    )
    numer_batch = afqmc.estimators["energy"]["ENumer"]
    denom_batch = afqmc.estimators["energy"]["EDenom"]
    data_batch = extract_observable(
        "estimates.test_generic_single_det_batch.h5", "energy"
    )
#        Iteration            WeightFactor                  Weight                  ENumer                  EDenom                  ETotal                  E1Body                  E2Body                 EHybrid                 Overlap                    Time
#  0.0000000000000000e+00  1.5000000000000000e+01  1.5000000000000000e+01  1.8283924369667727e+02  1.5000000000000000e+01  1.2189282913111818e+01  1.3205721945538620e+01 -1.0164390324268044e+00  0.0000000000000000e+00  2.4716604120159354e+02  9.8626852035522461e-02
#  2.5000000000000000e+01  1.1527867112417985e+01  1.3282034152988215e+01  1.6221804652440252e+02  1.5000000000000000e+01  1.0814536434960168e+01  1.2100564418095576e+01 -1.2860279831354082e+00  1.3050901010864038e+01  4.0565543129426864e+01  1.1511631011962890e-02
#  5.0000000000000000e+01  1.4896020623853685e+01  1.5253633766656975e+01  1.1805199679385174e+02  1.4999999999999998e+01  7.8701331195901174e+00  9.7601396793074588e+00 -1.8900065597173394e+00  1.0921185408182316e+01  2.9091658045254776e+00  1.1408643722534180e-02
#  7.5000000000000000e+01  1.5575250170916464e+01  1.5203506191303671e+01  8.0076659245395177e+01  1.5000000000000000e+01  5.3384439496930121e+00  7.5537811744291377e+00 -2.2153372247361278e+00  8.3464297835721464e+00  6.2653083007085741e-01  1.1470594406127931e-02
#  1.0000000000000000e+02  1.4685990166716158e+01  1.4759057793986456e+01  4.4771332002854798e+01  1.4999999999999998e+01  2.9847554668569871e+00  5.9106157049838135e+00 -2.9258602381268273e+00  5.7622926310103519e+00  1.7808709732637956e-01  1.1413364410400391e-02
#  1.2500000000000000e+02  1.2847676298901408e+01  1.3400423727443831e+01  8.1647862427794749e+01  1.4999999999999996e+01  5.4431908285196506e+00  7.4339575133770595e+00 -1.9907666848574090e+00  5.2795790387174746e+00  6.0272619212751982e-02  1.1377649307250976e-02
#  1.5000000000000000e+02  1.2026138448763263e+01  1.2644579343079368e+01  4.1667759028082173e+01  1.4999999999999998e+01  2.7778506018721454e+00  5.2538151777159694e+00 -2.4759645758438249e+00  4.8521079574979513e+00  3.4146673265942284e-02  1.1453714370727539e-02
#  1.7500000000000000e+02  1.1812495416467232e+01  1.2736677607240553e+01 -8.7797166670036773e+01  1.5000000000000000e+01 -5.8531444446691179e+00  1.9360411077357835e+00 -7.7891855524049003e+00  3.6150809364445471e+00  2.7527356767741970e-02  1.1449117660522461e-02

    driver_options["estimates"] = {
        "filename": "estimates.test_generic_single_det_batch.h5",
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
    data = extract_mixed_estimates("estimates.test_generic_single_det_batch.h5")

    print("data_batch.WeightFactor = {}".format(data_batch.WeightFactor))
    print("data_batch.WeightFactor.shape = {}".format(data_batch.WeightFactor.shape))
    print("data.WeightFactor = {}".format(data.WeightFactor))
    print("data.WeightFactor.shape = {}".format(data.WeightFactor.shape))

    print("data_batch.WeightFactor.values[1:-1].real = {}".format(data_batch.WeightFactor.values[1:-1].real))
    print("data.WeightFactor.values[1:-1].real = {}".format(data.WeightFactor.values[1:-1].real))

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
    driver_options["estimates"] = {
        "filename": "estimates.test_generic_single_det_batch_density_diff.h5",
        "observables": {
            "energy": {},
        },
    }
    comm = MPI.COMM_WORLD

    driver_options["qmc"]["batched"] = True
    afqmc = build_driver_test_instance(
        nelec,
        nmo,
        trial_type="single_det",
        options=driver_options,
        seed=7,
        density_diff=True,
    )
    afqmc.run(comm=comm, verbose=0)
    afqmc.finalise(verbose=0)
    afqmc.estimators.compute_estimators(
        comm,
        afqmc.system,
        afqmc.hamiltonian,
        afqmc.trial,
        afqmc.psi,
    )

    numer_batch = afqmc.estimators["energy"]["ENumer"]
    denom_batch = afqmc.estimators["energy"]["EDenom"]
    # weight_batch = afqmc.estimators['energy']['Weight']

    data_batch = extract_observable(
        "estimates.test_generic_single_det_batch_density_diff.h5", "energy"
    )

    numpy.random.seed(seed)
    driver_options["estimates"] = {
        "filename": "estimates.test_generic_single_det_batch_density_diff.h5",
        "mixed": {"energy_eval_freq": steps},
    }
    driver_options["qmc"]["batched"] = False
    legacy_afqmc = build_legacy_driver_instance(
        nelec,
        nmo,
        trial_type="single_det",
        options=driver_options,
        seed=7,
        density_diff=True,
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
    data = extract_mixed_estimates(
        "estimates.test_generic_single_det_batch_density_diff.h5"
    )

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


def teardown_module():
    cwd = os.getcwd()
    files = [
        "estimates.test_generic_single_det_batch_density_diff.h5",
        "estimates.test_generic_single_det_batch.h5",
    ]
    for f in files:
        try:
            os.remove(cwd + "/" + f)
        except OSError:
            pass


if __name__ == "__main__":
    test_generic_single_det_batch()
    test_generic_single_det_batch_density_diff()
