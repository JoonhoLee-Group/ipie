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
#          Jinghong Zhang <jinghongzhang@fas.harvard.edu>
#
from ipie.config import config
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import synchronize
from ipie.utils.misc import is_cupy

# TODO: Rename this


def propagate_one_body(phi, bt2, H1diag=False):
    r"""Propagate by the kinetic term by direct matrix multiplication.

    Only one spin component. Assuming phi is a batch.

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


def apply_exponential(phi, VHS, exp_nmax):
    """Apply exponential propagator of the HS transformation
    Parameters
    ----------
    system :
        system class
    phi : numpy array
        a state
    VHS : numpy array
        HS transformation potential
    Returns
    -------
    phi : numpy array
        Exp(VHS) * phi
    """
    # Temporary array for matrix exponentiation.
    Temp = xp.zeros(phi.shape, dtype=phi.dtype)

    xp.copyto(Temp, phi)
    for n in range(1, exp_nmax + 1):
        Temp = VHS.dot(Temp) / n
        phi += Temp

    synchronize()
    return phi


def apply_exponential_batch(phi, VHS, exp_nmax):
    """Apply exponential propagator of the HS transformation
    Parameters
    ----------
    system :
        system class
    phi : numpy array
        a state
    VHS : numpy array
        HS transformation potential
    Returns
    -------
    phi : numpy array
        Exp(VHS) * phi
    """
    # Temporary array for matrix exponentiation.
    Temp = xp.zeros(phi.shape, dtype=phi.dtype)

    xp.copyto(Temp, phi)
    if config.get_option("use_gpu"):
        for n in range(1, exp_nmax + 1):
            Temp = xp.matmul(VHS, Temp) / n  # matmul use much less GPU memory than einsum
            phi += Temp
        del Temp
    else:
        for iw in range(phi.shape[0]):
            for n in range(1, exp_nmax + 1):
                Temp[iw] = VHS[iw].dot(Temp[iw]) / n
                phi[iw] += Temp[iw]

    synchronize()

    return phi

def conjmat(M): #conjugates a matrix
    return xp.conjugate(M.T)

def apply_isometry(phi, Isometry,reverse=False):
    r"""Propagate by the unitary transfrom by direct matrix multiplication.

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
    n_walkers= phi.shape[0]
    n_electrons= phi.shape[2]
    
 
    if reverse:
        Ueff = conjmat(Isometry)

    else:
        Ueff = Isometry
    n_old=xp.shape(Ueff)[1]
    n_new=xp.shape(Ueff)[0]


    if is_cupy(Ueff):
        # GPU-efficient batched matrix multiply
        phi_trans = xp.einsum("ik,wkj->wij", Ueff, phi, optimize=True)
    else:
        # CPU: explicit loop is much faster
        phi_trans = xp.zeros((n_walkers ,n_new, n_electrons), dtype=phi.dtype)
        for iw in range(n_walkers):
            phi_trans[iw] = Ueff@ phi[iw] #replace with tensor operation?

    return phi_trans