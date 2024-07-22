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

"""Routines and classes for estimation of observables."""

import os
from typing import Tuple, Union

from ipie.addons.thermal.estimators.energy import ThermalEnergyEstimator
from ipie.addons.thermal.estimators.particle_number import ThermalNumberEstimator
from ipie.config import config, MPI
from ipie.estimators.handler import EstimatorHandler

# Some supported (non-custom) estimators
_predefined_estimators = {
    "energy": ThermalEnergyEstimator,
    "nav": ThermalNumberEstimator,
}


class ThermalEstimatorHandler(EstimatorHandler):
    """Container for qmc options of observables.

    Parameters
    ----------
    comm : MPI.COMM_WORLD
        MPI Communicator.
    hamiltonian : :class:`ipie.hamiltonian.X' object
        Hamiltonian describing the system.
    trial : :class:`ipie.trial_wavefunction.X' object
        Trial wavefunction class.
    walker_state : :class:`WalkerAccumulator` object
        WalkerAccumulator class.
    verbose : bool
        If true we print out additional setup information.
    filename : str
        .h5 file name for saving data.
    basename : str
        .h5 base name for saving data.
    overwrite : bool
        Whether to overwrite .h5 files.
    observables : tuple
        Tuple listing observables to be calculated.

    Attributes
    ----------
    estimators : dict
        Dictionary of estimator objects.
    """

    def __init__(
        self,
        comm,
        hamiltonian,
        trial,
        walker_state=None,
        verbose: bool = False,
        filename: Union[str, None] = None,
        basename: str = "estimates",
        overwrite=True,
        observables: Tuple[str] = ("energy", "nav"),  # TODO: Use factory method!
    ):
        if verbose:
            print("# Setting up estimator object.")
        if comm.rank == 0:
            self.basename = basename
            self.filename = filename
            self.index = 0
            if self.filename is None:
                self.filename = f"{self.basename}.{self.index}.h5"
                while os.path.isfile(self.filename) and not overwrite:
                    self.index = int(self.filename.split(".")[1])
                    self.index = self.index + 1
                    self.filename = f"{self.basename}.{self.index}.h5"
            if verbose:
                print(f"# Writing estimator data to {self.filename}")
        else:
            self.filename = None
        self.buffer_size = config.get_option("estimator_buffer_size")
        if walker_state is not None:
            self.num_walker_props = walker_state.size
            self.walker_header = walker_state.names
        else:
            self.num_walker_props = 0
            self.walker_header = ""
        self._estimators = {}
        self._shapes = []
        self._offsets = {}
        self.json_string = "{}"
        # TODO: Replace this, should be built outside
        for obs in observables:
            try:
                est = _predefined_estimators[obs](hamiltonian=hamiltonian, trial=trial)
                self[obs] = est
            except KeyError:
                raise RuntimeError(f"unknown observable: {obs}")
        if verbose:
            print("# Finished settting up estimator object.")

    def compute_estimators(self, system=None, hamiltonian=None, trial=None, walker_batch=None):
        """Update estimators with bached walkers.

        Parameters
        ----------
        hamiltonian : :class:`ipie.hamiltonian.X' object
            Hamiltonian describing the system.
        trial : :class:`ipie.trial_wavefunction.X' object
            Trial wavefunction class.
        walker_batch : :class:`UHFThermalWalkers' object
            Walkers class.
        """
        # Compute all estimators
        # For the moment only consider estimators compute per block.
        # TODO: generalize for different block groups (loop over groups)
        offset = self.num_walker_props
        for k, e in self.items():
            e.compute_estimator(walkers=walker_batch, hamiltonian=hamiltonian, trial=trial)
            start = offset + self.get_offset(k)
            end = start + int(self[k].size)
            self.local_estimates[start:end] += e.data

    def print_time_slice(self, comm, time_slice, walker_state):
        """Print estimators at a time slice of the imgainary time propagation.

        Parameters
        ----------
        comm : MPI.COMM_WORLD
            MPI Communicator.
        time_slice : int
            Time slice.
        walker_state : :class:`WalkerAccumulator` object
            WalkerAccumulator class.
        """
        comm.Reduce(self.local_estimates, self.global_estimates, op=MPI.SUM)
        # Get walker data.
        offset = walker_state.size

        if comm.rank == 0:
            k = "energy"
            e = self[k]
            start = offset + self.get_offset(k)
            end = start + int(self[k].size)
            estim_data = self.global_estimates[start:end]
            e.post_reduce_hook(estim_data)
            etotal = estim_data[e.get_index("ETotal")]

            k = "nav"
            e = self[k]
            start = offset + self.get_offset(k)
            end = start + int(self[k].size)
            estim_data = self.global_estimates[start:end]
            e.post_reduce_hook(estim_data)
            nav = estim_data[e.get_index("Nav")]

            print(f"cut : {time_slice} {nav.real} {etotal.real}")

        self.zero()
