
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
from numba import jit


@jit(nopython=True, fastmath=True)
def unpack_VHS_batch(idx_i, idx_j, VHS_packed, VHS):
    nwalkers = VHS.shape[0]
    nbsf = VHS.shape[1]
    nut = round(nbsf * (nbsf + 1) / 2)

    for iw in range(nwalkers):
        for i in range(nut):
            VHS[iw, idx_i[i], idx_j[i]] = VHS_packed[iw, i]
            VHS[iw, idx_j[i], idx_i[i]] = VHS_packed[iw, i]

    return
