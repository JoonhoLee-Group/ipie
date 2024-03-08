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

from __future__ import print_function

import os
from typing import Tuple, Union

import h5py
import numpy

from ipie.config import config, MPI
from ipie.estimators.energy import EnergyEstimator
from ipie.estimators.estimator_base import EstimatorBase
from ipie.estimators.utils import H5EstimatorHelper
from ipie.utils.io import format_fixed_width_strings

# Some supported (non-custom) estimators
_predefined_estimators = {
    "energy": EnergyEstimator,
}


class EstimatorHandler(object):
    """Container for qmc options of observables.

    Parameters
    ----------
    comm : MPI.COMM_WORLD
        MPI Communicator
    system : :class:`ipie.hubbard.Hubbard` / system object in general.
        Container for model input options.
    trial : :class:`ipie.trial_wavefunction.X' object
        Trial wavefunction class.
    verbose : bool
        If true we print out additional setup information.
    options: dict
        input options detailing which estimators to calculate. By default only
        mixed options will be calculated.

    Attributes
    ----------
    estimators : dict
        Dictionary of estimator objects.
    """

    def __init__(
        self,
        comm,
        system,
        hamiltonian,
        trial,
        walker_state=None,
        verbose: bool = False,
        filename: Union[str, None] = None,
        block_size: int = 1,
        basename: str = "estimates",
        overwrite=True,
        observables: Tuple[str] = ("energy",),  # TODO: Use factory method!
        index: int = 0,
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
                est = _predefined_estimators[obs](
                    system=system,
                    ham=hamiltonian,
                    trial=trial,
                )
                self[obs] = est
            except KeyError:
                raise RuntimeError(f"unknown observable: {obs}")
        if verbose:
            print("# Finished settting up estimator object.")

    def __setitem__(self, name: str, estimator: EstimatorBase) -> None:
        over_writing = self._estimators.get(name) is not None
        self._estimators[name] = estimator
        if not over_writing:
            self._shapes.append(estimator.shape)
            if len(self._offsets.keys()) == 0:
                self._offsets[name] = 0
                prev_obs = name
            else:
                prev_obs = list(self._offsets.keys())[-1]
                offset = self._estimators[prev_obs].size + self._offsets[prev_obs]
                self._offsets[name] = offset

    def get_offset(self, name: str) -> int:
        offset = self._offsets.get(name)
        assert offset is not None, f"Unknown estimator name {name}"
        return offset

    def __getitem__(self, key):
        return self._estimators[key]

    @property
    def items(self):
        return self._estimators.items

    @property
    def size(self):
        return sum(o.size for k, o in self._estimators.items())

    def initialize(self, comm):
        self.local_estimates = numpy.zeros(
            (self.size + self.num_walker_props), dtype=numpy.complex128
        )
        self.global_estimates = numpy.zeros(
            (self.size + self.num_walker_props), dtype=numpy.complex128
        )
        header = f"{'Block':>17s}  "
        header += format_fixed_width_strings(self.walker_header)
        header += " "
        for k, e in self.items():
            if e.print_to_stdout:
                header += e.header_to_text
        if comm.rank == 0:
            with h5py.File(self.filename, "w") as fh5:
                pass
            self.dump_metadata()
        self.output = H5EstimatorHelper(
            self.filename,
            base="block_size_1",
            chunk_size=self.buffer_size,
            shape=(self.size + self.num_walker_props,),
        )
        if comm.rank == 0:
            with h5py.File(self.filename, "r+") as fh5:
                fh5["block_size_1/num_walker_props"] = self.num_walker_props
                fh5["block_size_1/walker_prop_header"] = self.walker_header
                for k, o in self.items():
                    fh5[f"block_size_1/shape/{k}"] = o.shape
                    fh5[f"block_size_1/size/{k}"] = o.size
                    fh5[f"block_size_1/scalar/{k}"] = int(o.scalar_estimator)
                    fh5[f"block_size_1/names/{k}"] = " ".join(name for name in o.names)
                    fh5[f"block_size_1/offset/{k}"] = self.num_walker_props + self.get_offset(k)
        if comm.rank == 0:
            print(header)

    def dump_metadata(self):
        with h5py.File(self.filename, "a") as fh5:
            fh5["metadata"] = self.json_string

    def increment_file_number(self):
        self.index = self.index + 1
        self.filename = self.basename + f".{self.index}.h5"

    def compute_estimators(self, comm, system, hamiltonian, trial, walker_batch):
        """Update estimators with bached psi

        Parameters
        ----------
        """
        # Compute all estimators
        # For the moment only consider estimators compute per block.
        # TODO: generalize for different block groups (loop over groups)
        offset = self.num_walker_props
        for k, e in self.items():
            e.compute_estimator(system, walker_batch, hamiltonian, trial)
            start = offset + self.get_offset(k)
            end = start + int(self[k].size)
            self.local_estimates[start:end] += e.data

    def print_block(self, comm, block, walker_factors, div_factor=None):
        self.local_estimates[: walker_factors.size] = walker_factors.buffer
        comm.Reduce(self.local_estimates, self.global_estimates, op=MPI.SUM)
        output_string = " "
        # Get walker data.
        offset = walker_factors.size
        if comm.rank == 0:
            walker_factors.post_reduce_hook(self.global_estimates[:offset], block)
        output_string += walker_factors.to_text(self.global_estimates[:offset])
        output_string += " "
        for k, e in self.items():
            if comm.rank == 0:
                start = offset + self.get_offset(k)
                end = start + int(self[k].size)
                est_data = self.global_estimates[start:end]
                e.post_reduce_hook(est_data)
                est_string = e.data_to_text(est_data)
                e.to_ascii_file(est_string)
                if e.print_to_stdout:
                    output_string += est_string
        if comm.rank == 0:
            shift = self.global_estimates[walker_factors.get_index("HybridEnergy")]

        else:
            shift = None
        walker_factors.eshift = comm.bcast(shift)
        if comm.rank == 0:
#            self.output.push_to_chunk(self.global_estimates, f"data")
            self.output.increment()
        if comm.rank == 0:
            print(f"{block:>17d} " + output_string)
        self.zero()

    def zero(self):
        self.local_estimates[:] = 0.0
        self.global_estimates[:] = 0.0
        for _, e in self.items():
            e.zero()
