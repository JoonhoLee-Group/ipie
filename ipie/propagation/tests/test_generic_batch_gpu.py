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

from ipie.utils.misc import dotdict
from ipie.utils.testing import build_test_case_handlers
from ipie.utils.legacy_testing import build_legacy_test_case_handlers


@pytest.mark.gpu
def test_hybrid_batch():
    numpy.random.seed(7)
    nmo = 10
    nelec = (5, 5)
    nwalkers = 10
    nsteps = 25
    qmc = dotdict(
        {
            "dt": 0.005,
            "nstblz": 5,
            "nwalkers": nwalkers,
            "batched": False,
            "hybrid": True,
            "rhf": True,
            "num_steps": nsteps,
        }
    )
    legacy_data = build_legacy_test_case_handlers(
        nelec,
        nmo,
        num_dets=1,
        options=qmc,
        seed=7,
        trial_type="nomsd",
        rhf_trial=True,
    )
    qmc.batched = True
    batched_data = build_test_case_handlers(
        nelec,
        nmo,
        num_dets=1,
        options=qmc,
        seed=7,
        trial_type="single_det",
        rhf_trial=True,
    )
    walkers = legacy_data.walker_handler.walkers
    walker_batch = batched_data.walker_handler.walkers_batch
    assert False, "FDM FIX THIS, rng state between gpu and cpu"


if __name__ == "__main__":
    test_hybrid_batch()
