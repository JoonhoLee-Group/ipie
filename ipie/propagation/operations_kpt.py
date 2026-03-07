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
# Authors: Jinghong Zhang <jinghongzhang@fas.harvard.edu>
#

from numba import jit

from ipie.utils.backend import arraylib as xp
from ipie.utils.misc import is_cupy


def propagate_one_body_kpt(phi, bt2):
    r"""
    Propagate by the kinetic term by direct matrix multiplication for k-point calculations.
    Only one spin component. Assuming phi is a batch.
    For use with the continuus algorithm and free propagation.
    Parameters
    ----------
    phi : xp.ndarray
        Walker object to be updated. on output we have acted on
        :math:`|\phi_i\rangle` by :math:`B_{T/2}` and updated the weight
        appropriately.  updates inplace.
    bt2 : xp.ndarray
        The kinetic propagator for k-point calculations.
    """
    if is_cupy(bt2):
        phi = xp.einsum("kpr,wkrs->wkps", bt2, phi, optimize=True)
        return phi
    else:
        return propagate_one_body_kpt_cpu(phi, bt2)


@jit(nopython=True, fastmath=True)
def propagate_one_body_kpt_cpu(phi, bt2):
    for iw in range(phi.shape[0]):
        for ik1 in range(bt2.shape[0]):
            phi[iw][ik1] = xp.dot(bt2[ik1], phi[iw][ik1])
    return phi
