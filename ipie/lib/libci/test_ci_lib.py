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
# Author: Fionn Malone <fionn.malone@gmail.com>
#

import numpy as np
import pytest

from ipie.lib.libci import one_rdm
from ipie.utils.testing import get_random_phmsd_opt


@pytest.mark.unit
def test_one_rdm():
    num_spat = 8
    num_alpha = 2
    num_beta = 2
    (coeff, occa, occb), _ = get_random_phmsd_opt(num_alpha, num_beta, num_spat)
    one_rdm(coeff, occa, occb, num_spat)
