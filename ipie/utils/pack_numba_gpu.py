
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

from numba import cuda


@cuda.jit("void(int32[:],int32[:],complex128[:,:],complex128[:,:,:])")
def unpack_VHS_batch_gpu(idx_i, idx_j, VHS_packed, VHS):
    nwalkers = VHS.shape[0]
    nbsf = VHS.shape[1]
    nut = round(nbsf * (nbsf + 1) / 2)
    pos = cuda.grid(1)
    pos1 = pos // nut
    pos2 = pos - pos1 * nut
    if pos1 < nwalkers and pos2 < nut:
        VHS[pos1, idx_i[pos2], idx_j[pos2]] = VHS_packed[pos1, pos2]
        VHS[pos1, idx_j[pos2], idx_i[pos2]] = VHS_packed[pos1, pos2]
