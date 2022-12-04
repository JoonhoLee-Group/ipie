
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

import h5py
import numpy

array = numpy.array
zeros = numpy.zeros
einsum = numpy.einsum
isrealobj = numpy.isrealobj

import sys
import time

from ipie.utils.io import (from_qmcpack_dense, from_qmcpack_sparse,
                           write_hamiltonian,
                           read_hamiltonian)
from ipie.utils.backend import cast_to_device


class Generic(object):
    """Ab-initio Hamiltonian.

    Can be created by either passing the one and two electron integrals directly
    or initialised from integrals stored in QMCPACK hdf5 format. If initialising
    from file the `inputs' optional dictionary should be populated.

    Parameters
    ----------
    h1e : :class:`numpy.ndarray'
        One-electron integrals. Optional. Default: None.
    chol : :class:`numpy.ndarray'
        Factorized 2-electron integrals (L_{ik,n}) of shape (nbasis^2, nchol).
        Optional. Default: None.
    ecore : float
        Core energy.
    options : dict
        Input options defined below.
    integrals : string
        Path to file containing one- and two-electron integrals in QMCPACK
        format.
    verbose : bool
        Print extra information.

    Attributes
    ----------
    H1 : :class:`numpy.ndarray`
        One-body part of the Hamiltonian. Spin-dependent by default.
    ecore : float
        Core contribution to the total energy.
    h1e_mod : :class:`numpy.ndarray`
        Modified one-body Hamiltonian.
    chol_vecs : :class:`numpy.ndarray`
        Cholesky vectors. [M^2, nchol]
    nchol : int
        Number of cholesky vectors.
    nfields : int
        Number of auxiliary fields required.
    sparse_cutoff : float
        Screen out integrals below this threshold. Optional. Default 0.
    cplx_chol : bool
        Force setting of interpretation of cholesky decomposition. Optional.
        Default False, i.e. real/complex factorization determined from cholesky
        integrals.
    """

    def __init__(
        self,
        h1e,
        chol,
        ecore,
        h1e_mod=None,
        chol_packed=None,
        options={},
        verbose=False,
        write_ints=False,
    ):
        if verbose:
            print("# Parsing input options for hamiltonians.Generic.")
        self.name = "Generic"
        self.verbose = verbose
        self.exact_eri = options.get("exact_eri", False)
        self.mixed_precision = options.get("mixed_precision", False)
        self.density_diff = options.get("density_diff", False)
        self.symmetry = options.get("symmetry", True)
        self.chunked = False  # chunking disabled by default

        self.ecore = ecore
        self.chol_vecs = chol  # [M^2, nchol]

        if self.symmetry:
            self.chol_packed = chol_packed
            nbsf = h1e.shape[1]
            self.sym_idx = numpy.triu_indices(nbsf)
            self.sym_idx_i = self.sym_idx[0].copy()
            self.sym_idx_j = self.sym_idx[1].copy()
        else:
            self.chol_packed = None  # [M*(M+1)/2, nchol] if used
            self.sym_idx = None

        if self.mixed_precision:
            if not self.density_diff:
                self.density_diff = True
                if self.verbose:
                    print(
                        "# density_diff is switched on for more stable mixed precision"
                    )
            if self.symmetry:
                self.chol_packed = self.chol_packed.astype(numpy.float32)
            else:
                self.chol_vecs = self.chol_vecs.astype(numpy.float32)  # [M^2, nchol]

        if self.exact_eri:
            if self.verbose:
                print("# exact_eri is used for the local energy evaluation")

        if self.density_diff:
            if self.verbose:
                print(
                    "# density_diff is used for the force bias and the local energy evaluation"
                )

        if self.mixed_precision:
            if self.verbose:
                print("# mixed_precision is used for the propagation")

        if isrealobj(self.chol_vecs.dtype):
            if verbose:
                print("# Found real Choleksy integrals.")
            self.cplx_chol = False
        else:
            if verbose:
                print("# Found complex Cholesky integrals.")
            self.cplx_chol = True

        self.H1 = array(h1e)
        self.nbasis = h1e.shape[-1]

        mem = self.chol_vecs.nbytes / (1024.0**3)
        self.sparse = False
        if verbose:
            print("# Number of orbitals: %d" % self.nbasis)
            print("# Approximate memory required by Cholesky vectors %f GB" % mem)
        self.nchol = self.chol_vecs.shape[-1]
        if h1e_mod is not None:
            self.h1e_mod = h1e_mod
        else:
            h1e_mod = zeros(self.H1.shape, dtype=self.H1.dtype)
            construct_h1e_mod(self.chol_vecs, self.H1, h1e_mod)
            self.h1e_mod = h1e_mod

        # For consistency
        self.vol = 1.0
        self.nfields = self.nchol

        if verbose:
            print("# Number of Cholesky vectors: %d" % (self.nchol))
            print("# Number of fields: %d" % (self.nfields))
        if write_ints:
            self.write_integrals()
        if verbose:
            print("# Finished setting up hamiltonians.Generic object.")

    def hijkl(self, i, j, k, l):
        ik = i * self.nbasis + k
        jl = j * self.nbasis + l
        return numpy.dot(self.chol_vecs[ik], self.chol_vecs[jl])

    def write_integrals(self, nelec, filename="hamil.h5"):
        write_hamiltonian(
            self.H1[0],
            self.chol_vecs.T.reshape((self.nchol, self.nbasis, self.nbasis)),
            self.ecore,
            filename=filename,
        )

    def chunk(self, handler, verbose=False):
        self.chunked = True  # Boolean to indicate that chunked cholesky is available

        chol_idxs = [i for i in range(self.nchol)]
        self.chol_idxs_chunk = handler.scatter_group(chol_idxs)

        if self.symmetry:
            # if handler.srank == 0:  # creating copies for every rank = 0!!!!
            self.chol_packed = self.chol_packed.T.copy()  # [chol, M^2]
            handler.comm.barrier()

            self.chol_packed_chunk = handler.scatter_group(
                self.chol_packed
            )  # distribute over chol

            # if handler.srank == 0:
            self.chol_packed = self.chol_packed.T.copy()  # [M^2, chol]
            handler.comm.barrier()

            self.chol_packed_chunk = (
                self.chol_packed_chunk.T.copy()
            )  # [M^2, chol_chunk]

            tot_size = handler.allreduce_group(self.chol_packed_chunk.size)

            assert self.chol_packed.size == tot_size
        else:
            # if handler.comm.rank == 0:
            self.chol_vecs = self.chol_vecs.T.copy()  # [chol, M^2]
            handler.comm.barrier()

            self.chol_vecs_chunk = handler.scatter_group(
                self.chol_vecs
            )  # distribute over chol

            # if handler.comm.rank == 0:
            self.chol_vecs = self.chol_vecs.T.copy()  # [M^2, chol]
            handler.comm.barrier()

            self.chol_vecs_chunk = self.chol_vecs_chunk.T.copy()  # [M^2, chol_chunk]

            tot_size = handler.allreduce_group(self.chol_vecs_chunk.size)
            assert self.chol_vecs.size == tot_size

    # This function casts relevant member variables into cupy arrays
    # Keeping this specific for the moment as too much logic.
    # A sign it should be split up..
    def cast_to_cupy(self, verbose=False):
        import cupy

        size = self.H1.size + self.h1e_mod.size
        if self.chunked:
            if self.symmetry:
                size += self.chol_packed_chunk.size
            else:
                size += self.chol_vecs_chunk.size
        else:
            if self.symmetry:
                size += self.chol_packed.size
            else:
                size += self.chol_vecs.size

        if self.symmetry:
            size += self.sym_idx_i.size
            size += self.sym_idx_j.size

        if verbose:
            expected_bytes = size * 8.0  # float64
            print(
                "# hamiltonians.generic: expected to allocate {:4.3f} GB".format(
                    expected_bytes / 1024**3
                )
            )

        self.H1 = cupy.asarray(self.H1)
        self.h1e_mod = cupy.asarray(self.h1e_mod)
        if self.symmetry:
            self.sym_idx_i = cupy.asarray(self.sym_idx_i)
            self.sym_idx_j = cupy.asarray(self.sym_idx_j)

        if self.chunked:
            if self.symmetry:
                self.chol_packed_chunk = cupy.asarray(self.chol_packed_chunk)
            else:
                self.chol_vecs_chunk = cupy.asarray(self.chol_vecs_chunk)
        else:
            if self.symmetry:
                self.chol_packed = cupy.asarray(self.chol_packed)
            else:
                self.chol_vecs = cupy.asarray(self.chol_vecs)

        free_bytes, total_bytes = cupy.cuda.Device().mem_info
        used_bytes = total_bytes - free_bytes
        if verbose:
            print(
                "# hamiltonians.Generic: using {:4.3f} GB out of {:4.3f} GB memory on GPU".format(
                    used_bytes / 1024**3, total_bytes / 1024**3
                )
            )


def read_integrals(integral_file):
    try:
        (h1e, schol_vecs, ecore, nbasis, nup, ndown) = from_qmcpack_sparse(
            integral_file
        )
        chol_vecs = schol_vecs.toarray()
        return h1e, chol_vecs, ecore
    except KeyError:
        pass
    try:
        (h1e, chol_vecs, ecore, nbasis, nup, ndown) = from_qmcpack_dense(integral_file)
        return h1e, chol_vecs, ecore
    except KeyError:
        pass
    try:
        (h1e, chol_vecs, ecore) = read_hamiltonian(integral_file)
        naux = chol_vecs.shape[0]
        nbsf = chol_vecs.shape[-1]
        return h1e, chol_vecs.T.reshape((nbsf, nbsf, naux)), ecore
    except KeyError:
        return None


def construct_h1e_mod(chol, h1e, h1e_mod):
    # Subtract one-body bit following reordering of 2-body operators.
    # Eqn (17) of [Motta17]_
    nbasis = h1e.shape[-1]
    nchol = chol.shape[-1]
    chol_view = chol.reshape((nbasis, nbasis * nchol))
    # assert chol_view.__array_interface__['data'][0] == chol.__array_interface__['data'][0]
    v0 = 0.5 * numpy.dot(
        chol_view, chol_view.T
    )  # einsum('ikn,jkn->ij', chol_3, chol_3, optimize=True)
    h1e_mod[0, :, :] = h1e[0] - v0
    h1e_mod[1, :, :] = h1e[1] - v0
