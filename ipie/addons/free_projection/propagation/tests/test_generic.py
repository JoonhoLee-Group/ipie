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

from ipie.addons.free_projection.propagation.free_propagation import FreePropagation
from ipie.utils.misc import dotdict
from ipie.utils.testing import build_test_case_handlers


@pytest.mark.unit
def test_free_projection():
    numpy.random.seed(7)
    nmo = 10
    nelec = (6, 5)
    nwalkers = 8
    nsteps = 25
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
    batched_data = build_test_case_handlers(nelec, nmo, num_dets=1, options=qmc, seed=7)
    prop_fp = FreePropagation(time_step=0.005, verbose=False, ene_0=-1.0)
    prop_fp.build(batched_data.hamiltonian, batched_data.trial)

    prop_fp.propagate_walkers(
        batched_data.walkers, batched_data.hamiltonian, batched_data.trial, -1.0
    )
    assert batched_data.walkers.phia.shape == (nwalkers, nmo, nelec[0])
    assert batched_data.walkers.phib.shape == (nwalkers, nmo, nelec[1])
    assert numpy.allclose(
        numpy.sum(batched_data.walkers.phase), 7.926221838159645 + 0.3971467053264697j
    )
    assert numpy.allclose(numpy.sum(batched_data.walkers.weight), 1.7901505653712695)
    assert numpy.allclose(
        numpy.sum(batched_data.walkers.ovlp), -6.40187371404052e-05 - 2.34160780650416e-05j
    )
    assert numpy.allclose(
        numpy.sum(batched_data.walkers.phia), 33.95629475599705 - 0.30274130601759786j
    )
    assert numpy.allclose(
        numpy.sum(batched_data.walkers.phib), 41.45587700725909 - 2.8023497141639413j
    )


if __name__ == "__main__":
    test_free_projection()
