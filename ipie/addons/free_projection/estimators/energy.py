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


from ipie.estimators.energy import EnergyEstimator, local_energy
from ipie.utils.backend import arraylib as xp


class EnergyEstimatorFP(EnergyEstimator):
    def __init__(
        self,
        system=None,
        ham=None,
        trial=None,
        filename=None,
    ):
        super().__init__(system, ham, trial, filename)

    def compute_estimator(self, system, walkers, hamiltonian, trial, istep=1):
        trial.calc_greens_function(walkers)
        # Need to be able to dispatch here
        energy = local_energy(system, hamiltonian, walkers, trial)
        self._data["ENumer"] = xp.sum(walkers.weight * walkers.phase * walkers.ovlp * energy[:, 0])
        self._data["EDenom"] = xp.sum(walkers.weight * walkers.phase * walkers.ovlp)
        self._data["E1Body"] = xp.sum(walkers.weight * walkers.phase * walkers.ovlp * energy[:, 1])
        self._data["E2Body"] = xp.sum(walkers.weight * walkers.phase * walkers.ovlp * energy[:, 2])

        return self.data
