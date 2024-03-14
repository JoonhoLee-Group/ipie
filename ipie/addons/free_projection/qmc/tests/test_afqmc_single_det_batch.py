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

import tempfile

import numpy
import pytest

from ipie.addons.free_projection.analysis.extraction import extract_observable
from ipie.addons.free_projection.utils.testing import build_driver_test_instance_fp
from ipie.config import MPI

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
def test_generic_single_det_batch_fp():
    with tempfile.NamedTemporaryFile() as tmpf:
        driver_options = {
            "verbosity": 0,
            "get_sha1": False,
            "qmc": options,
            "estimates": {"filename": tmpf.name, "observables": {"energy": {}}},
            "walkers": {"population_control": "pair_branch"},
        }

        afqmc = build_driver_test_instance_fp(
            nelec,
            nmo,
            trial_type="single_det",
            options=driver_options,
            seed=7,
        )
        afqmc.setup_estimators(tmpf.name)
        afqmc.run(verbose=False, estimator_filename=tmpf.name)
        afqmc.finalise(verbose=0)
        for i in range(len(afqmc.estimators)):
            data_batch = extract_observable(f"{tmpf.name}.{i}", "energy")
            numer_batch = data_batch["ENumer"]
            denom_batch = data_batch["EDenom"]
            etot_batch = data_batch["ETotal"]
            assert etot_batch.dtype == numpy.complex128

        data_batch = extract_observable(f"{tmpf.name}.0", "energy")
        numer_batch = data_batch["ENumer"]
        denom_batch = data_batch["EDenom"]
        etot_batch = data_batch["ETotal"]
        assert numpy.allclose(numpy.sum(numer_batch), 89026.91053310843 + 37.16899096646583j)
        assert numpy.allclose(numpy.sum(denom_batch), 7431.790242711337 + 12.22172751384279j)
        assert numpy.allclose(numpy.sum(etot_batch), 35.93783190822862 - 0.04412020753601597j)


if __name__ == "__main__":
    test_generic_single_det_batch_fp()
