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
# Authors: Jinghong Zhang <jinghongzhang@fas.harvard.edu>
#
#

import numpy
from numba import jit

from ipie.hamiltonians.generic_base import GenericBase
from ipie.utils.backend import arraylib as xp
from ipie.utils.kpt_conv import (
    find_gamma_pt,
    find_inverted_index_batched,
    find_translated_index_batched,
    find_self_inverse_set,
    find_Qplus,
)
from ipie.utils.io import read_kpt_hamiltonian


@jit(nopython=True, fastmath=True)
def construct_kpq(kpts_frac):
    nk = kpts_frac.shape[0]
    idx_kpq_mat = numpy.zeros((nk, nk), dtype=numpy.int64)
    for iq in range(nk):
        qvec = kpts_frac[iq]
        idx_kpq = find_translated_index_batched(kpts_frac, qvec)
        idx_kpq_mat[iq] = idx_kpq
    return idx_kpq_mat


@jit(nopython=True, fastmath=True)
def construct_kmq(kpts_frac):
    nk = kpts_frac.shape[0]
    idx_kmq_mat = numpy.zeros((nk, nk), dtype=numpy.int64)
    for iq in range(nk):
        qvec = kpts_frac[iq]
        idx_kmq = find_translated_index_batched(kpts_frac, -qvec)
        idx_kmq_mat[iq] = idx_kmq
    return idx_kmq_mat


def construct_mq(kpts_frac):
    return find_inverted_index_batched(kpts_frac)


def construct_h1e_mod(chol, h1e, ikpq_mat, imq_vec, h1e_mod):
    nk, nbasis = h1e.shape[1], h1e.shape[2]
    v0 = numpy.zeros((nk, nbasis, nbasis), dtype=numpy.complex128)
    for ik in range(nk):
        for iq in range(nk):
            v0[ik] += 0.5 * numpy.einsum(
                "gpr, gqr -> pq", chol[:, ik, :, iq, :], chol[:, ik, :, iq, :].conj()
            )
    h1e_mod[0, :, :, :] = h1e[0, :, :, :] - v0
    h1e_mod[1, :, :, :] = h1e[1, :, :, :] - v0


def construct_h1e_mod_symm(chol, h1e, ikmq_mat, Sset, Qplus, h1e_mod):
    nk, nbasis = h1e.shape[1], h1e.shape[2]
    v0 = numpy.zeros((nk, nbasis, nbasis), dtype=numpy.complex128)
    for iq in range(len(Sset)):
        for ik in range(nk):
            v0[ik] += 0.5 * numpy.einsum(
                "gpr, gqr -> pq", chol[:, ik, :, iq, :], chol[:, ik, :, iq, :].conj()
            )

    for iq in range(len(Sset), len(Sset) + len(Qplus)):
        for ik in range(nk):
            iq_real = Qplus[iq - len(Sset)]
            ikmq = ikmq_mat[iq_real, ik]
            v0[ik] += 0.5 * numpy.einsum(
                "gpr, gqr -> pq", chol[:, ik, :, iq, :], chol[:, ik, :, iq, :].conj()
            ) + 0.5 * numpy.einsum(
                "grp, grq -> pq", chol[:, ikmq, :, iq, :].conj(), chol[:, ikmq, :, iq, :]
            )

    h1e_mod[0, :, :, :] = h1e[0, :, :, :] - v0
    h1e_mod[1, :, :, :] = h1e[1, :, :, :] - v0


class KptComplexCholSymm(GenericBase):
    """Class for ab-initio k-point Hamiltonian with 4-fold complex symmetric integrals. Making use of the cholesky symmetry to only store half of the cholesky vectors.
    Can be created by passing the one and two electron integrals directly.
    """

    def __init__(self, h1e, chol, kpts, ecore=0.0, verbose=False):
        assert h1e.shape[0] == 2
        assert len(h1e.shape) == 4  # shape = nspin, nk, nbasis, nbasis
        super().__init__(h1e, ecore, verbose)

        self.chol = numpy.array(
            chol, dtype=numpy.complex128
        )  # [nchol, Nk, M, unique_nk, M] (gamma, k, p, q, r)
        self.kpts = kpts
        self.Sset = find_self_inverse_set(self.kpts)
        self.Qplus = find_Qplus(self.kpts)
        self.unique_k = numpy.concatenate((self.Sset, self.Qplus))
        self.ikpq_mat = construct_kpq(self.kpts)
        self.ikmq_mat = construct_kmq(self.kpts)
        self.imq_vec = construct_mq(self.kpts)
        self.igamma = find_gamma_pt(self.kpts[self.unique_k])
        self.unique_nk = self.chol.shape[3]
        assert self.unique_nk == len(self.Sset) + len(self.Qplus)
        self.nk = self.kpts.shape[0]
        self.nchol = self.chol.shape[0]

        self.chunked = False

        # this is the one-body part that comes out of re-ordering the 2-body operators
        h1e_mod = numpy.zeros(self.H1.shape, dtype=self.H1.dtype)
        construct_h1e_mod_symm(self.chol, self.H1, self.ikmq_mat, self.Sset, self.Qplus, h1e_mod)
        self.h1e_mod = xp.array(h1e_mod)

        if verbose:
            mem = self.chol.nbytes / (1024.0**3)
            print("# Number of orbitals: %d" % self.nbasis)
            print(f"# Approximate memory required by Cholesky vectors {mem:f} GB")
            print("# Number of Cholesky vectors: %d" % (self.nchol))
            print("# Finished setting up KptComplexCholSymm object.")


class KptComplexChol(GenericBase):
    """Class for ab-initio k-point Hamiltonian with 4-fold complex symmetric integrals.
    Can be created by passing the one and two electron integrals directly.
    """

    def __init__(self, h1e, chol, kpts, ecore=0.0, verbose=False):
        assert h1e.shape[0] == 2
        assert len(h1e.shape) == 4  # shape = nspin, nk, nbasis, nbasis
        super().__init__(h1e, ecore, verbose)

        self.chol = numpy.array(
            chol, dtype=numpy.complex128
        )  # [nchol, Nk, M, Nk, M] (gamma, k, p, q, r)
        self.kpts = kpts
        self.ikpq_mat = construct_kpq(self.kpts)
        self.imq_vec = construct_mq(self.kpts)
        self.igamma = find_gamma_pt(self.kpts)
        self.nk = self.kpts.shape[0]
        self.unique_nk = self.nk
        self.nchol = self.chol.shape[0]

        self.chunked = False

        # this is the one-body part that comes out of re-ordering the 2-body operators
        h1e_mod = numpy.zeros(self.H1.shape, dtype=self.H1.dtype)
        construct_h1e_mod(self.chol, self.H1, self.ikpq_mat, self.imq_vec, h1e_mod)
        self.h1e_mod = xp.array(h1e_mod)

        if verbose:
            mem = self.chol.nbytes / (1024.0**3) * 3
            print("# Number of orbitals: %d" % self.nbasis)
            print(f"# Approximate memory required by Cholesky + A&B vectors {mem:f} GB")
            print("# Number of Cholesky vectors: %d" % (self.nchol))
            print("# Finished setting up KptComplexChol object.")


def read_kpt_integrals(integral_file):
    try:
        h1e, chol_vecs, kpts, ecore = read_kpt_hamiltonian(integral_file)
        return h1e, chol_vecs, kpts, ecore
    except KeyError:
        return None
