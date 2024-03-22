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
#          Ankit Mahajan <ankitmahajan76@gmail.com>
#

import numpy
import pytest

from ipie.config import MPI
from ipie.utils.legacy_testing import build_legacy_test_case_handlers_mpi
from ipie.utils.misc import dotdict
from ipie.utils.mpi import MPIHandler
from ipie.utils.testing import build_test_case_handlers_mpi


@pytest.mark.unit
def test_pair_branch_batch():
    mpi_handler = MPIHandler()

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
            "population_control": "pair_branch",
        }
    )
    legacy_data = build_legacy_test_case_handlers_mpi(
        nelec, nmo, mpi_handler, num_dets=1, complex_trial=True, options=qmc, seed=7
    )
    qmc.batched = True
    batched_data = build_test_case_handlers_mpi(
        nelec, nmo, mpi_handler, num_dets=1, complex_trial=True, options=qmc, seed=7
    )
    nup = nelec[0]
    for iw in range(nwalkers):
        assert numpy.allclose(
            batched_data.walkers.phia[iw],
            legacy_data.walker_handler.walkers[iw].phi[:, :nup],
        )
        assert numpy.allclose(
            batched_data.walkers.phib[iw],
            legacy_data.walker_handler.walkers[iw].phi[:, nup:],
        )
        assert numpy.allclose(
            batched_data.walkers.weight[iw],
            legacy_data.walker_handler.walkers[iw].weight,
        )
        assert numpy.allclose(
            batched_data.walkers.unscaled_weight[iw],
            legacy_data.walker_handler.walkers[iw].unscaled_weight,
        )

    assert pytest.approx(batched_data.walkers.weight[0]) == 0.2571750688329709
    assert pytest.approx(batched_data.walkers.weight[1]) == 1.0843219322894988
    assert pytest.approx(batched_data.walkers.weight[2]) == 0.8338283613093604
    assert (
        pytest.approx(batched_data.walkers.phia[9][0, 0])
        == -0.0005573508035052743 + 0.12432250308987346j
    )


@pytest.mark.unit
def test_comb_batch():
    mpi_handler = MPIHandler()

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
            "population_control": "comb",
        }
    )
    legacy_data = build_legacy_test_case_handlers_mpi(
        nelec,
        nmo,
        mpi_handler,
        num_dets=1,
        complex_trial=True,
        options=qmc,
        seed=7,
    )
    qmc.batched = True
    batched_data = build_test_case_handlers_mpi(
        nelec,
        nmo,
        mpi_handler,
        num_dets=1,
        complex_trial=True,
        options=qmc,
        seed=7,
    )
    nup = nelec[0]

    for iw in range(nwalkers):
        assert numpy.allclose(
            batched_data.walkers.phia[iw],
            legacy_data.walker_handler.walkers[iw].phi[:, :nup],
        )
        assert numpy.allclose(
            batched_data.walkers.phib[iw],
            legacy_data.walker_handler.walkers[iw].phi[:, nup:],
        )
        assert numpy.allclose(
            batched_data.walkers.weight[iw],
            legacy_data.walker_handler.walkers[iw].weight,
        )
    assert (
        pytest.approx(batched_data.walkers.phia[9][0, 0])
        == -0.0597200851442905 - 0.002353281222663805j
    )


@pytest.mark.unit
def test_stochastic_reconfiguration_batch():
    numpy.random.seed(7)

    mpi_handler = MPIHandler()

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
            "population_control": "stochastic_reconfiguration",
            "reconfiguration_freq": 2,
        }
    )
    qmc.batched = True
    batched_data = build_test_case_handlers_mpi(
        nelec, nmo, mpi_handler, num_dets=1, complex_trial=True, options=qmc, seed=7
    )

    assert pytest.approx(batched_data.walkers.weight[0]) == 1.0
    assert pytest.approx(batched_data.walkers.phia[0][0, 0]) == 0.0305067 + 0.01438442j


if __name__ == "__main__":
    test_pair_branch_batch()
    test_comb_batch()
    test_stochastic_reconfiguration_batch()
