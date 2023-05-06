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
# Author: Fionn Malone <fmalone@google.com>
#

import os

import numpy as np
import pytest

from ipie.estimators.energy import EnergyEstimator
from ipie.estimators.handler import EstimatorHandler
from ipie.utils.testing import gen_random_test_instances


@pytest.mark.unit
def test_energy_estimator():
    nmo = 10
    nocc = 8
    naux = 30
    nwalker = 10
    system, ham, walker_batch, trial = gen_random_test_instances(nmo, nocc, naux, nwalker)
    estim = EnergyEstimator(system=system, ham=ham, trial=trial)
    estim.compute_estimator(system, walker_batch, ham, trial)
    assert len(estim.names) == 5
    assert estim["ENumer"].real == pytest.approx(-754.0373585215561)
    assert estim["ETotal"] == pytest.approx(0.0)
    tmp = estim.data.copy()
    estim.post_reduce_hook(tmp)
    assert tmp[estim.get_index("ETotal")] == pytest.approx(-75.40373585215562)
    assert estim.print_to_stdout
    assert estim.ascii_filename == None
    assert estim.shape == (5,)
    header = estim.header_to_text
    data_to_text = estim.data_to_text(tmp)
    assert len(data_to_text.split()) == 5


@pytest.mark.unit
def test_estimator_handler():
    nmo = 10
    nocc = 8
    naux = 30
    nwalker = 10
    system, ham, walker_batch, trial = gen_random_test_instances(nmo, nocc, naux, nwalker)
    estim = EnergyEstimator(system=system, ham=ham, trial=trial, options={"filename": "test.txt"})
    estim.print_to_stdout = False
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    options = {"block_size": 10, "observables": {"energy": {"filename": "test2.txt"}}}
    handler = EstimatorHandler(comm, system, ham, trial, options=options)
    handler["energy1"] = estim
    handler.json_string = ""
    handler.initialize(comm)
    handler.compute_estimators(comm, system, ham, trial, walker_batch)
    handler.compute_estimators(comm, system, ham, trial, walker_batch)


def teardown_module():
    cwd = os.getcwd()
    files = ["estimates.0.h5", "test.txt", "test2.txt"]
    for f in files:
        try:
            os.remove(cwd + "/" + f)
        except OSError:
            pass
