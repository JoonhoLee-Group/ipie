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
# Author: Fionn Malone <fmalone@google.com>
#

import plum
import numpy

from ipie.utils.backend import arraylib as xp
from ipie.hamiltonians.generic import GenericComplexChol, GenericRealChol
from ipie.estimators.estimator_base import EstimatorBase

from ipie.thermal.walkers.uhf_walkers import UHFThermalWalkers
from ipie.thermal.estimators.thermal import one_rdm_from_G
from ipie.thermal.estimators.generic import local_energy_generic_cholesky


def local_energy(
        hamiltonian: GenericRealChol, 
        walkers: UHFThermalWalkers):
    energies = numpy.zeros((walkers.nwalkers, 3), dtype=numpy.complex128)
    
    for iw in range(walkers.nwalkers):
        walkers.calc_greens_function(iw) # In-place update of GF.
        P = one_rdm_from_G(numpy.array([walkers.Ga[iw], walkers.Gb[iw]]))
        energy = local_energy_generic_cholesky(hamiltonian, P)

        for i in range(3):
            energies[iw, i] = energy[i]

    return energies

class ThermalEnergyEstimator(EstimatorBase):
    def __init__(self, hamiltonian=None, trial=None, filename=None):
        super().__init__()
        self._eshift = 0.0
        self.scalar_estimator = True
        self._data = {
            "ENumer": 0.0j,
            "EDenom": 0.0j,
            "ETotal": 0.0j,
            "E1Body": 0.0j,
            "E2Body": 0.0j,
        }

        self._shape = (len(self.names),)
        self._data_index = {k: i for i, k in enumerate(list(self._data.keys()))}
        self.print_to_stdout = True
        self.ascii_filename = filename

    def compute_estimator(self, walkers, hamiltonian, trial, istep=1):
        # Need to be able to dispatch here.
        # Re-calculated Green's function in `local_energy`.
        energy = local_energy(hamiltonian, walkers)
        self._data["ENumer"] = xp.sum(walkers.weight * energy[:, 0].real)
        self._data["EDenom"] = xp.sum(walkers.weight)
        self._data["E1Body"] = xp.sum(walkers.weight * energy[:, 1].real)
        self._data["E2Body"] = xp.sum(walkers.weight * energy[:, 2].real)
        return self.data

    def get_index(self, name):
        index = self._data_index.get(name, None)

        if index is None:
            raise RuntimeError(f"Unknown estimator {name}")

        return index

    def post_reduce_hook(self, data):
        ix_proj = self._data_index["ETotal"]
        ix_nume = self._data_index["ENumer"]
        ix_deno = self._data_index["EDenom"]
        data[ix_proj] = data[ix_nume] / data[ix_deno]
        ix_nume = self._data_index["E1Body"]
        data[ix_nume] = data[ix_nume] / data[ix_deno]
        ix_nume = self._data_index["E2Body"]
        data[ix_nume] = data[ix_nume] / data[ix_deno]
