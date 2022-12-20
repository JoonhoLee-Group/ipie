
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

import numpy

from ipie.utils.backend import arraylib as xp
from ipie.utils.misc import is_cupy


# TODO: Rename this
def kinetic_real(phi, system, bt2, H1diag=False):
    r"""Propagate by the kinetic term by direct matrix multiplication.

    For use with the continuus algorithm and free propagation.

    todo : this is the same a propagating by an arbitrary matrix, remove.

    Parameters
    ----------
    walker : :class:`pie.walker.Walker`
        Walker object to be updated. on output we have acted on
        :math:`|\phi_i\rangle` by :math:`B_{T/2}` and updated the weight
        appropriately.  updates inplace.
    state : :class:`pie.state.State`
        Simulation state.
    """
    nup = system.nup
    # Assuming that our walker is in UHF form.
    if H1diag:
        phi[:, :nup] = xp.einsum("ii,ij->ij", bt2[0], phi[:, :nup])
        phi[:, nup:] = xp.einsum("ii,ij->ij", bt2[1], phi[:, nup:])
    else:
        phi[:, :nup] = bt2[0].dot(phi[:, :nup])
        phi[:, nup:] = bt2[1].dot(phi[:, nup:])


def kinetic_spin_real_batch(phi, bt2, H1diag=False):
    r"""Propagate by the kinetic term by direct matrix multiplication. Only one spin component. Assuming phi is a batch.

    For use with the continuus algorithm and free propagation.

    todo : this is the same a propagating by an arbitrary matrix, remove.

    Parameters
    ----------
    walker : :class:`pie.walker.Walker`
        Walker object to be updated. on output we have acted on
        :math:`|\phi_i\rangle` by :math:`B_{T/2}` and updated the weight
        appropriately.  updates inplace.
    state : :class:`pie.state.State`
        Simulation state.
    """
    # Assuming that our walker is in UHF form.
    if H1diag:
        phi[:, :] = xp.einsum("ii,wij->ij", bt2, phi)
    else:
        if is_cupy(bt2):
            phi = xp.einsum("ik,wkj->wij", bt2, phi, optimize=True)
        else:
            # Loop is O(10x) times faster on CPU for FeP benchmark
            for iw in range(phi.shape[0]):
                phi[iw] = xp.dot(bt2, phi[iw])

    return phi
