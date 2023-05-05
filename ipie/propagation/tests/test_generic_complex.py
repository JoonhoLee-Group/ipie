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

from ipie.utils.misc import dotdict
from ipie.utils.testing import (
    build_test_case_handlers,
)

@pytest.mark.unit
def test_vhs():
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
            "hybrid": True,
            "num_steps": nsteps,
        }
    )
    
    batched_data = build_test_case_handlers(nelec, nmo, num_dets=1, options=qmc, seed=7, complex_integrals=True, complex_trial = True, trial_type="single_det")
    xshifted = numpy.random.normal(
        0.0, 1.0, nwalkers * batched_data.hamiltonian.nfields
    ).reshape(batched_data.hamiltonian.nfields,nwalkers)

    vhs_batch = batched_data.propagator.construct_VHS(
        batched_data.hamiltonian, xshifted
    )

if __name__ == "__main__":
    test_vhs()