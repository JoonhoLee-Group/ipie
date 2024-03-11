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

from typing import Tuple, Union

import h5py
import numpy

from ipie.addons.free_projection.estimators.energy import EnergyEstimatorFP
from ipie.config import MPI
from ipie.estimators.handler import EstimatorHandler
from ipie.estimators.utils import H5EstimatorHelper


class EstimatorHandlerFP(EstimatorHandler):
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
        super().__init__(
            comm,
            system,
            hamiltonian,
            trial,
            walker_state,
            verbose,
            filename,
            block_size,
            basename,
            overwrite,
            observables,
            index,
        )
        self["energy"] = EnergyEstimatorFP(
            system=system,
            ham=hamiltonian,
            trial=trial,
        )

    def initialize(self, comm, print_header=True):
        self.local_estimates = numpy.zeros(
            (self.size + self.num_walker_props), dtype=numpy.complex128
        )
        self.global_estimates = numpy.zeros(
            (self.size + self.num_walker_props), dtype=numpy.complex128
        )
        header = f"{'Iter':>17s} {'TimeStep':>10s} "
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
        if comm.rank == 0 and print_header:
            print(header)

    def print_block(self, comm, block, walker_factors, div_factor=None, time_step=0):
        self.local_estimates[: walker_factors.size] = walker_factors.buffer
        comm.Reduce(self.local_estimates, self.global_estimates, op=MPI.SUM)
        output_string = " "
        # Get walker data.
        offset = walker_factors.size
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
            self.output.push_to_chunk(self.global_estimates, f"data")
            self.output.increment()
        if comm.rank == 0:
            if time_step == 0:
                print(f"{block:>17d} {time_step:>10d}" + output_string)
            else:
                blank = ""
                print(f"{blank:>17s} {time_step:>10d}" + output_string)
        self.zero()
