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

from ipie.hamiltonians.generic import GenericRealChol, GenericComplexChol
from ipie.addons.thermal.propagation.phaseless_base import PhaselessBase
from ipie.utils.backend import arraylib as xp


class PhaselessGeneric(PhaselessBase):
    """A class for performing phaseless propagation with real, generic, hamiltonian."""

    def __init__(self, time_step, mu, exp_nmax=6, lowrank=False, verbose=False):
        super().__init__(time_step, mu, lowrank=lowrank, exp_nmax=exp_nmax, verbose=verbose)
    
    @plum.dispatch
    def construct_VHS(self, hamiltonian: GenericRealChol, xshifted: xp.ndarray):
        """Includes `nwalkers`.
        """
        nwalkers = xshifted.shape[-1] # Shape (nfields, nwalkers).
        VHS = hamiltonian.chol.dot(xshifted) # Shape (nbasis^2, nwalkers).
        VHS = self.isqrt_dt * VHS.T.reshape(nwalkers, hamiltonian.nbasis, hamiltonian.nbasis)
        return VHS # Shape (nwalkers, nbasis, nbasis).
    
    @plum.dispatch
    def construct_VHS(self, hamiltonian: GenericComplexChol, xshifted: xp.ndarray):
        """Includes `nwalkers`.
        """
        nwalkers = xshifted.shape[-1]
        nchol = hamiltonian.nchol
        VHS = self.isqrt_dt * (
            hamiltonian.A.dot(xshifted[:nchol]) + hamiltonian.B.dot(xshifted[nchol:])
        )
        VHS = VHS.T.copy()
        VHS = VHS.reshape(nwalkers, hamiltonian.nbasis, hamiltonian.nbasis)
        return VHS
