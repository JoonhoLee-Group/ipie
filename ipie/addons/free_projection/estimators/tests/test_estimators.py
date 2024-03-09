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

import tempfile

import pytest

from ipie.addons.free_projection.estimators.energy import EnergyEstimatorFP
from ipie.addons.free_projection.estimators.handler import EstimatorHandlerFP
from ipie.utils.testing import gen_random_test_instances


@pytest.mark.unit
def test_energy_fp_estimator():
    nmo = 10
    nocc = 8
    naux = 30
    nwalker = 10
    system, ham, walker_batch, trial = gen_random_test_instances(nmo, nocc, naux, nwalker)
    estim = EnergyEstimatorFP(system=system, ham=ham, trial=trial)
    estim.compute_estimator(system, walker_batch, ham, trial)
    assert len(estim.names) == 5
    tmp = estim.data.copy()
    estim.post_reduce_hook(tmp)
    assert estim.print_to_stdout
    assert estim.ascii_filename == None
    assert estim.shape == (5,)
    data_to_text = estim.data_to_text(tmp)
    assert len(data_to_text.split()) == 5


@pytest.mark.unit
def test_estimator_handler_fp():
    with tempfile.NamedTemporaryFile() as tmp1, tempfile.NamedTemporaryFile() as tmp2:
        nmo = 10
        nocc = 8
        naux = 30
        nwalker = 10
        system, ham, walker_batch, trial = gen_random_test_instances(nmo, nocc, naux, nwalker)
        estim = EnergyEstimatorFP(system=system, ham=ham, trial=trial, filename=tmp1.name)
        estim.print_to_stdout = False
        from ipie.config import MPI

        comm = MPI.COMM_WORLD
        handler = EstimatorHandlerFP(
            comm,
            system,
            ham,
            trial,
            block_size=10,
            observables=("energy",),
            filename=tmp2.name,
        )
        handler["energy1"] = estim
        handler.json_string = ""
        handler.initialize(comm)
        handler.compute_estimators(comm, system, ham, trial, walker_batch)
        handler.compute_estimators(comm, system, ham, trial, walker_batch)


if __name__ == "__main__":
    test_energy_fp_estimator()
    test_estimator_handler_fp()
