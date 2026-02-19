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
from ipie.hamiltonians.generic_base import GenericBase
from ipie.utils.backend import arraylib as xp
from ipie.utils.mpi import make_splits_displacements
from ipie.utils.kpt_conv import find_gamma_pt, find_self_inverse_set, find_Qplus
from ipie.hamiltonians.kpt_hamiltonian import construct_kpq, construct_kmq, construct_mq

from numba import jit


try:
    from mpi4py import MPI
except ImportError:
    MPI = None

def construct_h1e_mod_symm(chol, h1e, ikmq_mat, Sset, Qplus, h1e_mod, handler):
    v0 = calc_v0(chol, ikmq_mat, Sset, Qplus)
    v0 = handler.scomm.allreduce(v0, op=MPI.SUM)
    h1e_mod[0, :, :, :] = h1e[0, :, :, :] - v0
    h1e_mod[1, :, :, :] = h1e[1, :, :, :] - v0

@jit(nopython=True, fastmath=True)
def calc_v0(chol, ikmq_mat, Sset, Qplus):
    nk, nbasis = chol.shape[1], chol.shape[2]
    nchol = chol.shape[0]
    unique_nk = len(Sset) + len(Qplus)
    # chol = chol.reshape((nchol, nk, nbasis, unique_nk, nbasis))
    v0 = numpy.zeros((nk, nbasis, nbasis), dtype=numpy.complex128)
    for iq in range(len(Sset)):
        for ik in range(nk):
            chol_ik = chol[:, ik, :, iq, :]
            chol_pgr = chol_ik.transpose(1, 0, 2).copy() # pgr
            chol_p_gr = chol_pgr.reshape(nbasis, nchol * nbasis)
            chol_gr_p = chol_p_gr.T.copy()
            v0[ik] += .5 * numpy.dot(chol_p_gr, chol_gr_p.conj())
    
    for iq in range(len(Sset), len(Sset) + len(Qplus)):
        for ik in range(nk):
            iq_real = Qplus[iq - len(Sset)]
            ikmq = ikmq_mat[iq_real, ik]
            chol_ik = chol[:, ik, :, iq, :]
            chol_pgr = chol_ik.transpose(1, 0, 2).copy() # pgr
            chol_p_gr = chol_pgr.reshape(nbasis, nchol * nbasis)
            chol_gr_p = chol_p_gr.T.copy()
            chol_ikmq = chol[:, ikmq, :, iq, :].copy() # grp
            chol_gr_p_mq = chol_ikmq.reshape(nchol * nbasis, nbasis)
            chol_p_gr_mq = chol_gr_p_mq.T.copy()
            v0[ik] += .5 * numpy.dot(chol_p_gr, chol_gr_p.conj()) + .5 * numpy.dot(chol_p_gr_mq.conj(), chol_gr_p_mq)
    return v0


class KptComplexCholChunked(GenericBase):
    """Class for ab-initio Hamiltonian with 4-fold complex symmetric integrals & k point symmetry.
    Can be created by passing the one and two electron integrals directly.
    """

    def __init__(
        self,
        h1e,
        kpts,
        chol=None,
        chol_chunk=None,
        ecore=0.0,
        handler=None,
        verbose=False,
    ):
        super().__init__(h1e, ecore, verbose)
        self.handler = handler
        assert (
            h1e.shape[0] == 2
        )  # assuming each spin component is given. this should be fixed for GHF...?

        self.kpts = kpts
        self.Sset = find_self_inverse_set(self.kpts)
        self.Qplus = find_Qplus(self.kpts)
        self.unique_k = numpy.concatenate((self.Sset, self.Qplus))
        self.ikpq_mat = construct_kpq(self.kpts)
        self.ikmq_mat = construct_kmq(self.kpts)
        self.imq_vec = construct_mq(self.kpts)
        self.igamma = find_gamma_pt(self.kpts[self.unique_k])

        self.unique_nk = len(self.Sset) + len(self.Qplus)
        self.nk = self.kpts.shape[0]

        if chol is not None:
            self.chol = chol  # [nchol, nk * M * unique_nk * M]
            self.nchol = self.chol.shape[0]
            self.unique_nk = self.chol.shape[3]
            self.chol = self.chol.reshape(self.nchol, -1)
            self.chunk_kpt(handler)
            self.chol_chunk = self.chol_chunk.reshape(-1, self.nk, self.nbasis, self.unique_nk, self.nbasis)
        else:
            self.chol_chunk = chol_chunk  # [nchol, nk * M * unique_nk * M]
        
        
        chunked_chols = self.chol_chunk.shape[0]
        num_chol = handler.scomm.allreduce(chunked_chols, op=MPI.SUM)
        self.nchol = num_chol

        chol_idxs = [i for i in range(self.nchol)]
        self.chol_idxs_chunk = handler.scatter_group(chol_idxs)

        assert self.chol_chunk.dtype == numpy.dtype("complex128")

        self.nchol_chunk = self.chol_chunk.shape[0]
        assert self.chol_chunk.shape == (self.nchol_chunk, self.nk, self.nbasis, self.unique_nk, self.nbasis)
        
        self.chunked = True

        # this is the one-body part that comes out of re-ordering the 2-body operators
        h1e_mod = numpy.zeros(self.H1.shape, dtype=self.H1.dtype)
        construct_h1e_mod_symm(self.chol_chunk, self.H1, self.ikmq_mat, self.Sset, self.Qplus, h1e_mod, handler)
        self.h1e_mod = xp.array(h1e_mod)

        split_size = make_splits_displacements(num_chol, handler.nmembers)[0]
        self.chunk_displacements = [0] + numpy.cumsum(split_size).tolist()

        if verbose:
            mem = self.chol_chunk.nbytes / (1024.0**3)
            print("# Number of orbitals: %d" % self.nbasis)
            print(f"# Approximate memory required by Cholesky vectors {mem:f} GB")
            print("# Number of Cholesky vectors: %d" % (self.nchol))
            print("# Number of fields: %d" % (self.nchol))
            print("# Finished setting up KptComplexChol object.")

