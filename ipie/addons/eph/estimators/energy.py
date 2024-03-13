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

import plum

from ipie.estimators.estimator_base import EstimatorBase
from ipie.addons.eph.estimators.local_energy_holstein import local_energy_holstein
from ipie.addons.eph.hamiltonians.holstein import HolsteinModel
from ipie.systems.generic import Generic
from ipie.addons.eph.trial_wavefunction.eph_trial_base import EPhTrialWavefunctionBase
from ipie.utils.backend import arraylib as xp
from ipie.addons.eph.walkers.eph_walkers import EPhWalkers


@plum.dispatch
def local_energy(
    system: Generic,
    hamiltonian: HolsteinModel,
    walkers: EPhWalkers,
    trial: EPhTrialWavefunctionBase,
) -> xp.ndarray:
    return local_energy_holstein(system, hamiltonian, walkers, trial)


class EnergyEstimator(EstimatorBase):
    def __init__(
        self,
        system=None,
        ham=None,
        trial=None,
        filename=None,
    ):
        assert system is not None
        assert ham is not None
        assert trial is not None
        super().__init__()
        self._eshift = 0.0
        self.scalar_estimator = True
        self._data = {
            "ENumer": 0.0j,
            "EDenom": 0.0j,
            "ETotal": 0.0j,
            "EEl": 0.0j,
            "EElPh": 0.0j,
            "EPh": 0.0j,
        }
        self._shape = (len(self.names),)
        self._data_index = {k: i for i, k in enumerate(list(self._data.keys()))}
        self.print_to_stdout = True
        self.ascii_filename = filename

    def compute_estimator(self, system, walkers, hamiltonian, trial, istep=1):
        # Need to be able to dispatch here
        energy = local_energy(system, hamiltonian, walkers, trial)
        self._data["ENumer"] = xp.sum(walkers.weight * energy[:, 0].real)
        self._data["EDenom"] = xp.sum(walkers.weight)
        self._data["EEl"] = xp.sum(walkers.weight * energy[:, 1].real)
        self._data["EElPh"] = xp.sum(walkers.weight * energy[:, 2].real)
        self._data["EPh"] = xp.sum(walkers.weight * energy[:, 3].real)

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
        ix_nume = self._data_index["EEl"]
        data[ix_nume] = data[ix_nume] / data[ix_deno]
        ix_nume = self._data_index["EElPh"]
        data[ix_nume] = data[ix_nume] / data[ix_deno]
        ix_nume = self._data_index["EPh"]
        data[ix_nume] = data[ix_nume] / data[ix_deno]
