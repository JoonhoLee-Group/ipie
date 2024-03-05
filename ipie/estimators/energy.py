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

from ipie.estimators.estimator_base import EstimatorBase
from ipie.estimators.local_energy_batch import (
    local_energy_batch,
    local_energy_multi_det_trial_batch,
)
from ipie.estimators.local_energy_noci import local_energy_noci
from ipie.estimators.local_energy_sd import local_energy_single_det_uhf
from ipie.estimators.local_energy_wicks import (
    local_energy_multi_det_trial_wicks_batch,
    local_energy_multi_det_trial_wicks_batch_opt,
    local_energy_multi_det_trial_wicks_batch_opt_chunked,
)
from ipie.hamiltonians.generic import GenericComplexChol, GenericRealChol
from ipie.systems.generic import Generic
from ipie.trial_wavefunction.noci import NOCI
from ipie.trial_wavefunction.particle_hole import (
    ParticleHole,
    ParticleHoleNaive,
    ParticleHoleNonChunked,
    ParticleHoleSlow,
)
from ipie.trial_wavefunction.single_det import SingleDet
from ipie.utils.backend import arraylib as xp
from ipie.walkers.uhf_walkers import UHFWalkers


@plum.dispatch
def local_energy(
    system: Generic, hamiltonian: GenericRealChol, walkers: UHFWalkers, trial: SingleDet
):
    return local_energy_batch(system, hamiltonian, walkers, trial)


@plum.dispatch
def local_energy(
    system: Generic,
    hamiltonian: GenericComplexChol,
    walkers: UHFWalkers,
    trial: SingleDet,
):
    return local_energy_single_det_uhf(system, hamiltonian, walkers, trial)


@plum.dispatch
def local_energy(
    system: Generic,
    hamiltonian: GenericRealChol,
    walkers: UHFWalkers,
    trial: ParticleHoleNaive,
):
    return local_energy_multi_det_trial_batch(system, hamiltonian, walkers, trial)


@plum.dispatch
def local_energy(
    system: Generic,
    hamiltonian: GenericRealChol,
    walkers: UHFWalkers,
    trial: ParticleHole,
):
    return local_energy_multi_det_trial_wicks_batch_opt_chunked(system, hamiltonian, walkers, trial)


@plum.dispatch
def local_energy(
    system: Generic,
    hamiltonian: GenericRealChol,
    walkers: UHFWalkers,
    trial: ParticleHoleNonChunked,
):
    return local_energy_multi_det_trial_wicks_batch_opt(system, hamiltonian, walkers, trial)


@plum.dispatch
def local_energy(
    system: Generic,
    hamiltonian: GenericRealChol,
    walkers: UHFWalkers,
    trial: ParticleHoleSlow,
):
    return local_energy_multi_det_trial_wicks_batch(system, hamiltonian, walkers, trial)


@plum.dispatch
def local_energy(system: Generic, hamiltonian: GenericRealChol, walkers: UHFWalkers, trial: NOCI):
    return local_energy_noci(system, hamiltonian, walkers, trial)


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
            "E1Body": 0.0j,
            "E2Body": 0.0j,
        }
        self._shape = (len(self.names),)
        self._data_index = {k: i for i, k in enumerate(list(self._data.keys()))}
        self.print_to_stdout = True
        self.ascii_filename = filename

    def compute_estimator(self, system, walkers, hamiltonian, trial, istep=1):
        trial.calc_greens_function(walkers)
        # Need to be able to dispatch here
        energy = local_energy(system, hamiltonian, walkers, trial)
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
