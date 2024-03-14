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

from ipie.addons.free_projection.analysis.jackknife import jackknife_ratios


@pytest.mark.unit
def test_jackknife_ratios():
    numpy.random.seed(0)
    num = numpy.random.randn(100) + 0.0j
    denom = numpy.ones(100)
    mean, sigma = jackknife_ratios(num, denom)
    assert numpy.isclose(mean, num.sum() / denom.sum())
    assert numpy.isclose(sigma, 0.1, atol=0.01)


if __name__ == "__main__":
    test_jackknife_ratios()
