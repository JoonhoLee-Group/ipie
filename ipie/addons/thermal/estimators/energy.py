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

from typing import Union

from ipie.addons.thermal.estimators.generic import local_energy_generic_cholesky
from ipie.addons.thermal.estimators.thermal import one_rdm_from_G
from ipie.addons.thermal.walkers.uhf_walkers import UHFThermalWalkers
from ipie.estimators.energy import EnergyEstimator
from ipie.hamiltonians.generic import GenericComplexChol, GenericRealChol
from ipie.utils.backend import arraylib as xp


def local_energy(
    hamiltonian: Union[GenericRealChol, GenericComplexChol], walkers: UHFThermalWalkers
):
    energies = xp.zeros((walkers.nwalkers, 3), dtype=xp.complex128)

    for iw in range(walkers.nwalkers):
        # Want the full Green's function when calculating observables.
        walkers.calc_greens_function(iw, slice_ix=walkers.stack[iw].nslice)
        P = one_rdm_from_G(xp.array([walkers.Ga[iw], walkers.Gb[iw]]))
        energy = local_energy_generic_cholesky(hamiltonian, P)
        energies[iw] = energy

    return energies


class ThermalEnergyEstimator(EnergyEstimator):
    def __init__(self, system=None, hamiltonian=None, trial=None, filename=None):
        super().__init__(system=system, ham=hamiltonian, trial=trial, filename=filename)

    def compute_estimator(self, system=None, walkers=None, hamiltonian=None, trial=None):
        # Need to be able to dispatch here.
        # Re-calculated Green's function in `local_energy`.
        if hamiltonian is None:
            raise ValueError("Hamiltonian must not be none.")
        if walkers is None:
            raise ValueError("walkers must not be none.")
        energy = local_energy(hamiltonian, walkers)
        self._data["ENumer"] = xp.sum(walkers.weight * energy[:, 0].real)
        self._data["EDenom"] = xp.sum(walkers.weight)
        self._data["E1Body"] = xp.sum(walkers.weight * energy[:, 1].real)
        self._data["E2Body"] = xp.sum(walkers.weight * energy[:, 2].real)
        return self.data
