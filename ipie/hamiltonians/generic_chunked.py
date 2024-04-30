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

import numpy
from ipie.hamiltonians.generic_base import GenericBase
from ipie.utils.pack_numba import pack_cholesky
from ipie.utils.backend import arraylib as xp
from ipie.utils.mpi import make_splits_displacements

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def construct_h1e_mod(chol, h1e, h1e_mod, handler):
    # Subtract one-body bit following reordering of 2-body operators.
    # Eqn (17) of [Motta17]_
    nbasis = h1e.shape[-1]
    nchol = chol.shape[-1]
    chol_view = chol.reshape((nbasis, nbasis * nchol))
    # assert chol_view.__array_interface__['data'][0] == chol.__array_interface__['data'][0]
    v0 = 0.5 * numpy.dot(
        chol_view,
        chol_view.T.conj(),  # conjugate added to account for complex integrals
    )  # einsum('ikn,jkn->ij', chol_3, chol_3, optimize=True)
    v0 = handler.scomm.allreduce(v0, op=MPI.SUM)
    h1e_mod[0, :, :] = h1e[0] - v0
    h1e_mod[1, :, :] = h1e[1] - v0


class GenericRealCholChunked(GenericBase):
    """Class for ab-initio Hamiltonian with 8-fold real symmetric integrals.
    Can be created by passing the one and two electron integrals directly.
    """

    def __init__(
        self,
        h1e,
        chol=None,
        chol_chunk=None,
        chol_packed_chunk=None,
        ecore=0.0,
        handler=None,
        verbose=False,
    ):
        if not (
            (chol is not None and chol_chunk is None and chol_packed_chunk is None)
            or (chol is None and chol_chunk is not None and chol_packed_chunk is not None)
        ):
            raise ValueError(
                "Invalid argument combination. Provide either 'chol' alone or both 'chol_chunk' and 'chol_packed_chunk' together."
            )
        super().__init__(h1e, ecore, verbose)
        self.handler = handler
        assert (
            h1e.shape[0] == 2
        )  # assuming each spin component is given. this should be fixed for GHF...?

        self.sym_idx = numpy.triu_indices(self.nbasis)
        self.sym_idx_i = self.sym_idx[0].copy()
        self.sym_idx_j = self.sym_idx[1].copy()

        if chol is not None:
            self.chol = chol  # [M^2, nchol]
            self.nchol = self.chol.shape[-1]
            self.chol = self.chol.reshape((self.nbasis, self.nbasis, self.nchol))
            cp_shape = (self.nbasis * (self.nbasis + 1) // 2, self.chol.shape[-1])
            self.chol_packed = numpy.zeros(cp_shape, dtype=self.chol.dtype)
            pack_cholesky(self.sym_idx[0], self.sym_idx[1], self.chol_packed, self.chol)
            self.chol = self.chol.reshape((self.nbasis * self.nbasis, self.nchol))
            self.chunk(handler)
        else:
            self.chol_chunk = chol_chunk  # [M^2, nchol]
            self.chol_packed_chunk = chol_packed_chunk

        chunked_chols = self.chol_chunk.shape[-1]
        num_chol = handler.scomm.allreduce(chunked_chols, op=MPI.SUM)
        self.nchol = num_chol

        chol_idxs = [i for i in range(self.nchol)]
        self.chol_idxs_chunk = handler.scatter_group(chol_idxs)

        assert self.chol_chunk.dtype == numpy.dtype("float64")
        assert self.chol_packed_chunk.dtype == numpy.dtype("float64")

        self.nchol_chunk = self.chol_chunk.shape[-1]
        self.nfields = self.nchol
        assert self.nbasis**2 == self.chol_chunk.shape[0]

        self.chunked = True

        # this is the one-body part that comes out of re-ordering the 2-body operators
        h1e_mod = numpy.zeros(self.H1.shape, dtype=self.H1.dtype)
        construct_h1e_mod(self.chol_chunk, self.H1, h1e_mod, handler)
        self.h1e_mod = xp.array(h1e_mod)

        split_size = make_splits_displacements(num_chol, handler.nmembers)[0]
        self.chunk_displacements = [0] + numpy.cumsum(split_size).tolist()

        if verbose:
            mem = self.chol_chunk.nbytes / (1024.0**3)
            mem_packed = self.chol_packed_chunk.nbytes / (1024.0**3)
            print("# Number of orbitals: %d" % self.nbasis)
            print(f"# Approximate memory required by Cholesky vectors {mem:f} GB")
            print(f"# Approximate memory required by packed Cholesky vectors {mem_packed:f} GB")
            print(f"# Approximate memory required total {mem_packed + mem:f} GB")
            print("# Number of Cholesky vectors: %d" % (self.nchol))
            print("# Number of fields: %d" % (self.nchol))
            print("# Finished setting up GenericRealChol object.")

    def hijkl(self, i, j, k, l):  # (ik|jl) somehow physicist notation - terrible!!
        ik = i * self.nbasis + k
        jl = j * self.nbasis + l
        return numpy.dot(self.chol[ik], self.chol[jl])
