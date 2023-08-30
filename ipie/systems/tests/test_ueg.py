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
# Authors: Fionn Malone <fionn.malone@gmail.com>
#          Joonho Lee
#

# TODO: write UEG test

import os
import tempfile

import math
import numpy
import pytest

from ipie.config import MPI
from ipie.systems.ueg import UEG


@pytest.mark.unit
def test_ueg():
    numpy.random.seed(7)
    nup = 7
    ndown = 5
    ne = nup + ndown
    rs = 1.
    mu = -1.
    ecut = 1.
    sys_opts = {
                "nup": nup,
                "ndown": ndown,
                "rs": rs,
                "mu": mu,
                "ecut": ecut
                }
    sys = UEG(sys_opts, verbose=True)
    assert sys.nup == nup
    assert sys.ndown == ndown
    assert sys.rs == rs
    assert sys.mu == mu
    assert sys.ecut == ecut
    assert sys.ne == nup + ndown
    assert sys.zeta == (nup - ndown) / ne
    assert sys.rho == ((4.0 * math.pi) / 3.0 * rs**3.0)**(-1.0)
    assert sys.L == rs * (4.0 * ne * math.pi / 3.0) ** (1.0 / 3.0)
    assert sys.vol == sys.L**3.0
    assert sys.kfac == 2 * math.pi / sys.L
    assert sys.kf == (3 * (sys.zeta + 1) * math.pi**2 * ne / sys.L**3) ** (1.0 / 3.0)
    assert sys.ef == 0.5 * sys.kf**2


if __name__ == "__main__":
    test_ueg()
