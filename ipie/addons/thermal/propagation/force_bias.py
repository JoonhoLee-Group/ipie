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

import plum
import numpy

from ipie.hamiltonians.generic import GenericRealChol, GenericComplexChol
from ipie.addons.thermal.estimators.thermal import one_rdm_from_G
from ipie.utils.backend import arraylib as xp


@plum.dispatch
def construct_force_bias(hamiltonian: GenericRealChol, walkers):
    r"""Compute optimal force bias.

    Parameters
    ----------
    G: :class:`numpy.ndarray`
        Walker's 1RDM: <c_i^{\dagger}c_j>.

    Returns
    -------
    xbar : :class:`numpy.ndarray`
        Force bias.
    """
    vbias = xp.empty((walkers.nwalkers, hamiltonian.nchol), dtype=walkers.Ga.dtype)

    for iw in range(walkers.nwalkers):
        P = one_rdm_from_G(numpy.array([walkers.Ga[iw], walkers.Gb[iw]]))
        vbias[iw] = hamiltonian.chol.T.dot(P[0].ravel()) + hamiltonian.chol.T.dot(P[1].ravel())

    return vbias


@plum.dispatch
def construct_force_bias(hamiltonian: GenericComplexChol, walkers):
    r"""Compute optimal force bias.

    Parameters
    ----------
    G: :class:`numpy.ndarray`
        Walker's 1RDM: <c_i^{\dagger}c_j>.

    Returns
    -------
    xbar : :class:`numpy.ndarray`
        Force bias.
    """
    nchol = hamiltonian.nchol
    vbias = xp.empty((walkers.nwalkers, hamiltonian.nfields), dtype=walkers.Ga.dtype)

    for iw in range(walkers.nwalkers):
        P = one_rdm_from_G(numpy.array([walkers.Ga[iw], walkers.Gb[iw]]))
        vbias[iw, :nchol] = hamiltonian.A.T.dot(P[0].ravel()) + hamiltonian.A.T.dot(P[1].ravel())
        vbias[iw, nchol:] = hamiltonian.B.T.dot(P[0].ravel()) + hamiltonian.B.T.dot(P[1].ravel())

    return vbias
