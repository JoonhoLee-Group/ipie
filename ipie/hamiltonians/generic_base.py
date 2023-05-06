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
# Authors: Joonho Lee
#
#

from abc import ABCMeta
import numpy as np

from ipie.utils.backend import cast_to_device


class GenericBase(metaclass=ABCMeta):
    """Base class for ab-initio Hamiltonian.
    Can be created by passing the one and two electron integrals directly.
    This is basically a container for integrals.
    Base class only stores 1-electron integrals and the constant term.
    """

    def __init__(self, h1e, ecore=0.0, verbose=False):
        self.verbose = verbose
        self.ecore = ecore
        self.H1 = np.array(h1e, dtype=h1e.dtype)
        self.nbasis = h1e.shape[-1]
        self.nchol = None

    # This function casts relevant member variables into cupy arrays
    # Keeping this specific for the moment as too much logic.
    # A sign it should be split up..
    def cast_to_cupy(self, verbose=False):
        cast_to_device(self, verbose=verbose)

    def chunk(self, handler, verbose=False):
        self.chunked = True  # Boolean to indicate that chunked cholesky is available

        chol_idxs = [i for i in range(self.nchol)]
        self.chol_idxs_chunk = handler.scatter_group(chol_idxs)

        # if handler.srank == 0:  # creating copies for every rank = 0!!!!
        self.chol_packed = self.chol_packed.T.copy()  # [chol, M^2]
        handler.comm.barrier()

        self.chol_packed_chunk = handler.scatter_group(self.chol_packed)  # distribute over chol

        # if handler.srank == 0:
        self.chol_packed = self.chol_packed.T.copy()  # [M^2, chol]
        handler.comm.barrier()

        self.chol_packed_chunk = self.chol_packed_chunk.T.copy()  # [M^2, chol_chunk]

        tot_size = handler.allreduce_group(self.chol_packed_chunk.size)

        assert self.chol_packed.size == tot_size

        # distributing chol
        # if handler.comm.rank == 0:
        self.chol = self.chol.T.copy()  # [chol, M^2]
        handler.comm.barrier()

        self.chol_chunk = handler.scatter_group(self.chol)  # distribute over chol

        # if handler.comm.rank == 0:
        self.chol = self.chol.T.copy()  # [M^2, chol]
        handler.comm.barrier()

        self.chol_chunk = self.chol_chunk.T.copy()  # [M^2, chol_chunk]

        tot_size = handler.allreduce_group(self.chol_chunk.size)
        assert self.chol.size == tot_size
