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
# Authors: Joonho Lee
#          Fionn Malone <fionn.malone@gmail.com>
#

import numpy
import pytest

from ipie.estimators.greens_function_batch import greens_function_single_det
from ipie.utils.testing import build_test_case_handlers
from ipie.utils.legacy_testing import build_legacy_test_case_handlers
from ipie.utils.misc import dotdict

@pytest.mark.unit
def test_greens_function_batch():
    nelec = (5, 5)
    nwalkers = 10
    nsteps = 10
    nmo = 10
    qmc = dotdict(
        {
            "dt": 0.005,
            "nstblz": 5,
            "nwalkers": nwalkers,
            "batched": False,
            "hybrid": True,
            "num_steps": nsteps,
        }
    )
    legacy_data = build_legacy_test_case_handlers(
        nelec,
        nmo,
        num_dets=1,
        complex_trial=True,
        options=qmc,
        seed=7,
    )
    qmc.batched = True
    batched_data = build_test_case_handlers(
        nelec,
        nmo,
        num_dets=1,
        complex_trial=True,
        options=qmc,
        seed=7,
    )
    legacy_walkers = legacy_data.walker_handler.walkers
    walkers = batched_data.walkers
    ovlp = greens_function_single_det(walkers, batched_data.trial, build_full=True)
    for iw in range(nwalkers):
        numpy.testing.assert_allclose(
            legacy_walkers[iw].Ghalf[0], walkers.Ghalfa[iw], atol=1e-12
        )
        numpy.testing.assert_allclose(
            legacy_walkers[iw].Ghalf[1], walkers.Ghalfb[iw], atol=1e-12
        )
        numpy.testing.assert_allclose(
            legacy_walkers[iw].G[0], walkers.Ga[iw], atol=1e-12
        )
        numpy.testing.assert_allclose(
            legacy_walkers[iw].G[1], walkers.Gb[iw], atol=1e-12
        )


@pytest.mark.unit
def test_overlap_batch():
    nelec = (5, 5)
    nwalkers = 10
    nsteps = 10
    nmo = 10
    qmc = dotdict(
        {
            "dt": 0.005,
            "nstblz": 5,
            "nwalkers": nwalkers,
            "batched": False,
            "hybrid": True,
            "num_steps": nsteps,
        }
    )
    legacy_data = build_legacy_test_case_handlers(
        nelec, nmo, num_dets=1, complex_trial=True, options=qmc, seed=7
    )
    qmc.batched = True
    batched_data = build_test_case_handlers(
        nelec, nmo, num_dets=1, complex_trial=True, options=qmc, seed=7
    )
    legacy_walkers = legacy_data.walker_handler.walkers
    walkers = batched_data.walkers
    ovlp = greens_function_single_det(walkers, batched_data.trial, build_full=True)
    ovlp_legacy = [w.calc_overlap(legacy_data.trial) for w in legacy_walkers]
    assert numpy.allclose(ovlp_legacy, ovlp)


@pytest.mark.unit
def test_reortho_batch():
    nelec = (5, 5)
    nwalkers = 10
    nsteps = 10
    nmo = 10
    qmc = dotdict(
        {
            "dt": 0.005,
            "nstblz": 5,
            "nwalkers": nwalkers,
            "batched": False,
            "hybrid": True,
            "num_steps": nsteps,
        }
    )
    legacy_data = build_legacy_test_case_handlers(
        nelec, nmo, num_dets=1, complex_trial=True, options=qmc, seed=7
    )
    qmc.batched = True
    batched_data = build_test_case_handlers(
        nelec, nmo, num_dets=1, complex_trial=True, options=qmc, seed=7
    )
    legacy_walkers = legacy_data.walker_handler.walkers
    detR = batched_data.walkers.orthogonalise(False)
    detR_legacy = [w.reortho(legacy_data.trial) for w in legacy_walkers]
    assert numpy.allclose(detR_legacy, detR)


if __name__ == "__main__":
    test_overlap_batch()
    test_greens_function_batch()
    test_reortho_batch()
