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

import pytest

from ipie.addons.free_projection.utils.testing import build_test_case_handlers_fp
from ipie.utils.misc import dotdict


@pytest.mark.unit
def test_reortho_batch_fp():
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
    qmc.batched = True
    batched_data = build_test_case_handlers_fp(
        nelec, nmo, num_dets=1, complex_trial=True, options=qmc, seed=7
    )
    batched_data.walkers.orthogonalise()
    assert batched_data.walkers.phia.shape == (nwalkers, nmo, nelec[0])
    assert batched_data.walkers.phib.shape == (nwalkers, nmo, nelec[1])


if __name__ == "__main__":
    test_reortho_batch_fp()
