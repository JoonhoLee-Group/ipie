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

import numpy as np

from ipie.estimators.estimator_base import EstimatorBase
from ipie.estimators.local_energy_batch import (
    local_energy_batch,
    local_energy_multi_det_trial_batch,
)
from ipie.utils.io import get_input_value
from ipie.utils.backend import arraylib as xp

from ipie.trial_wavefunction.particle_hole import (
    ParticleHoleNaive,
    ParticleHoleWicks,
    ParticleHoleWicksNonChunked,
    ParticleHoleWicksSlow,
)
from ipie.trial_wavefunction.single_det import SingleDet

from ipie.estimators.local_energy_wicks import (
    local_energy_multi_det_trial_wicks_batch,
    local_energy_multi_det_trial_wicks_batch_opt,
    local_energy_multi_det_trial_wicks_batch_opt_chunked,
)
from ipie.estimators.local_energy_sd import (
    local_energy_single_det_batch_gpu,
    local_energy_single_det_rhf_batch,
    local_energy_single_det_uhf_batch,
)
from ipie.estimators.local_energy_sd_chunked import (
    local_energy_single_det_uhf_batch_chunked,
    local_energy_single_det_uhf_batch_chunked_gpu,
)


# Single dispatch
_dispatcher = {
    ParticleHoleNaive: local_energy_multi_det_trial_batch,
    ParticleHoleWicks: local_energy_multi_det_trial_wicks_batch_opt_chunked,
    ParticleHoleWicksNonChunked: local_energy_multi_det_trial_wicks_batch_opt,
    ParticleHoleWicksSlow: local_energy_multi_det_trial_wicks_batch,
    SingleDet: local_energy_batch,
}


class EnergyEstimator(EstimatorBase):
    def __init__(
        self,
        comm=None,
        qmc=None,
        system=None,
        ham=None,
        trial=None,
        verbose=False,
        options={},
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
        self.ascii_filename = get_input_value(options, "filename", default=None)

    def compute_estimator(
        self, system, walker_batch, hamiltonian, trial_wavefunction, istep=1
    ):
        trial_wavefunction.calc_greens_function(walker_batch)
        # Need to be able to dispatch here
        energy = _dispatcher[type(trial_wavefunction)](
            system, hamiltonian, walker_batch, trial_wavefunction
        )
        self._data["ENumer"] = xp.sum(walker_batch.weight * energy[:, 0].real)
        self._data["EDenom"] = xp.sum(walker_batch.weight)
        self._data["E1Body"] = xp.sum(walker_batch.weight * energy[:, 1].real)
        self._data["E2Body"] = xp.sum(walker_batch.weight * energy[:, 2].real)

        return self.data

    def get_index(self, name):
        index = self._data_index.get(name, None)
        if index is None:
            raise RuntimeError(f"Unknown estimator {name}")
        return index

    def post_reduce_hook(self, reduced_data):
        ix_proj = self._data_index["ETotal"]
        ix_nume = self._data_index["ENumer"]
        ix_deno = self._data_index["EDenom"]
        reduced_data[ix_proj] = reduced_data[ix_nume] / reduced_data[ix_deno]
        ix_nume = self._data_index["E1Body"]
        reduced_data[ix_nume] = reduced_data[ix_nume] / reduced_data[ix_deno]
        ix_nume = self._data_index["E2Body"]
        reduced_data[ix_nume] = reduced_data[ix_nume] / reduced_data[ix_deno]
