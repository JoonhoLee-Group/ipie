import numpy
from numba import jit

from ipie.hamiltonians.generic_base import GenericBase
from ipie.utils.backend import arraylib as xp
from ipie.utils.kpt_conv import find_gamma_pt, find_inverted_index_batched, find_translated_index_batched, get_possible_Gs, get_k_from_G_MPmesh, find_self_inverse_set, find_Qplus

import h5py

from ipie.utils.io import (
    from_qmcpack_dense,
    from_qmcpack_sparse,
    read_hamiltonian,
)

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
            ikpq = ikpq_mat[ik, iq]
            imq = imq_vec[iq]
            v0[ik] += .5 * numpy.einsum('gpr, grq -> pq', chol[:, ik, :, iq, :], chol[:, ikpq, :, imq, :])
    h1e_mod[0, :, :, :] = h1e[0, :, :, :] - v0
    h1e_mod[1, :, :, :] = h1e[1, :, :, :] - v0

def construct_h1e_mod_symm(chol, h1e, ikmq_mat, Sset, Qplus, h1e_mod):
    nk, nbasis = h1e.shape[1], h1e.shape[2]
    v0 = numpy.zeros((nk, nbasis, nbasis), dtype=numpy.complex128)
    for iq in range(len(Sset)):
        for ik in range(nk):
            v0[ik] += .5 * numpy.einsum('gpr, gqr -> pq', chol[:, ik, :, iq, :], chol[:, ik, :, iq, :].conj())
    
    for iq in range(len(Sset), len(Sset) + len(Qplus)):
        for ik in range(nk):
            iq_real = Qplus[iq - len(Sset)]
            ikmq = ikmq_mat[iq_real, ik]
            v0[ik] += .5 * numpy.einsum('gpr, gqr -> pq', chol[:, ik, :, iq, :], chol[:, ik, :, iq, :].conj()) + .5 * numpy.einsum('grp, grq -> pq', chol[:, ikmq, :, iq, :].conj(), chol[:, ikmq, :, iq, :])

    h1e_mod[0, :, :, :] = h1e[0, :, :, :] - v0
    h1e_mod[1, :, :, :] = h1e[1, :, :, :] - v0

@jit(nopython=True, fastmath=True)
def construct_h1e_mod_isdf(MPQ, cgto, h1e, ikpq_mat, ikmq_mat, Sset, Qplus, h1e_mod):
    nk, nbasis = h1e.shape[1], h1e.shape[2]
    v0 = numpy.zeros((nk, nbasis, nbasis), dtype=numpy.complex128)
    for iq in range(len(Sset)):
        iq_real = Sset[iq]
        MPQ_iq = MPQ[iq]
        for ik in range(nk):
            ikpq = ikpq_mat[iq_real, ik]
            cgto_ikpq = cgto[ikpq]
            # v0 += .5 * oe.contract('kPp, kPr, PQ, kQr, kQq -> kpq', cgto.conj(), cgto_ikpq, MPQ_iq, cgto_ikpq.conj(), cgto)
            cgto_PQ = cgto_ikpq @ cgto_ikpq.T.conj()
            cgtoM = MPQ_iq * cgto_PQ
            v0[ik] += .5 * cgto[ik].conj().T @ cgtoM @ cgto[ik]
    
    for iq in range(len(Sset), len(Sset) + len(Qplus)):    
        iq_real = Qplus[iq - len(Sset)]
        MPQ_iq = MPQ[iq]
        for ik in range(nk):
            ikpq = ikpq_mat[iq_real, ik]
            cgto_ikpq = cgto[ikpq]
            ikmq = ikmq_mat[iq_real, ik]
            cgto_ikmq = cgto[ikmq]
            
            # v0 += .5 * (oe.contract('kPp, kPr, PQ, kQr, kQq -> kpq', cgto.conj(), cgto_ikpq, MPQ_iq, cgto_ikpq.conj(), cgto) + oe.contract('kPp, kPr, PQ, kQr, kQq -> kpq', cgto.conj(), cgto_ikmq, MPQ_iq.conj(), cgto_ikmq.conj(), cgto))
            cgto_PQ = cgto_ikpq @ cgto_ikpq.T.conj()
            cgtoM = MPQ_iq * cgto_PQ
            v0[ik] += .5 *cgto[ik].conj().T @ cgtoM @ cgto[ik]
            cgto_PQ = cgto_ikmq @ cgto_ikmq.T.conj()
            cgtoM = MPQ_iq.conj() * cgto_PQ
            v0[ik] += .5 * cgto[ik].conj().T @ cgtoM @ cgto[ik]

    h1e_mod[0, :, :, :] = h1e[0, :, :, :] - v0
    h1e_mod[1, :, :, :] = h1e[1, :, :, :] - v0



class KptComplexCholSymm(GenericBase):
    """Class for ab-initio k-point Hamiltonian with 4-fold complex symmetric integrals. Making use of the cholesky symmetry to only store half of the cholesky vectors.
    Can be created by passing the one and two electron integrals directly.
    """

    def __init__(self, h1e, chol, kpts, ecore=0.0, verbose=False):
        assert h1e.shape[0] == 2
        assert len(h1e.shape) == 4 # shape = nspin, nk, nbasis, nbasis
        super().__init__(h1e, ecore, verbose)

        self.chol = numpy.array(chol, dtype=numpy.complex128)  # [nchol, Nk, M, unique_nk, M] (gamma, k, p, q, r)
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
        assert len(h1e.shape) == 4 # shape = nspin, nk, nbasis, nbasis
        super().__init__(h1e, ecore, verbose)

        self.chol = numpy.array(chol, dtype=numpy.complex128)  # [nchol, Nk, M, Nk, M] (gamma, k, p, q, r)
        self.kpts = kpts
        self.ikpq_mat = construct_kpq(self.kpts)
        self.imq_vec = construct_mq(self.kpts)
        self.igamma = find_gamma_pt(self.kpts)
        self.nk = self.kpts.shape[0]
        self.unique_nk = self.nk # for compatibility with KptComplexCholSymm in propagation
        self.nchol = self.chol.shape[0]

        self.chunked = False

        # this is the one-body part that comes out of re-ordering the 2-body operators
        h1e_mod = numpy.zeros(self.H1.shape, dtype=self.H1.dtype)
        construct_h1e_mod(self.chol, self.H1, self.ikpq_mat, self.imq_vec, h1e_mod)
        self.h1e_mod = xp.array(h1e_mod)

        if verbose:
            mem = self.A.nbytes / (1024.0**3) * 3
            print("# Number of orbitals: %d" % self.nbasis)
            print(f"# Approximate memory required by Cholesky + A&B vectors {mem:f} GB")
            print("# Number of Cholesky vectors: %d" % (self.nchol))
            print("# Finished setting up KptComplexChol object.")

class KptISDF(GenericBase):
    """Class for ab-initio k-point Hamiltonian with 4-fold complex symmetric integrals.
    The electron repulsion integrals are approximated by Interpolative Separable Density Fitting (ISDF).
    """
    def __init__(self, h1e, MPQ, cholM, cgto, kpts, ecore=0.0, verbose=False, halfrot_cgto=None, halfrot_M=None):
        assert h1e.shape[0] == 2
        assert len(h1e.shape) == 4 # shape = nspin, nk, nbasis, nbasis
        super().__init__(h1e, ecore, verbose)

        self.MPQ = numpy.array(MPQ, dtype=numpy.complex128)
        self.cholM = numpy.array(cholM, dtype=numpy.complex128)  # [q, P, gamma], M = LL^\dagger
        self.nchol = self.cholM.shape[-1]
        # here we don't have spin indices for cgto because we use OAO basis for UHF cases to avoid extra storage
        self.cgto = numpy.array(cgto, dtype=numpy.complex128) # [k, P, p]
        if halfrot_cgto is not None:
            self.halfrot_cgtoa, self.halfrot_cgtob, self.halfrot_cgto = halfrot_cgto # [k, \tilde{P}, i(a)], [k, \tilde{P}, i(b)], [k, \tilde{P}, p]
        else: 
            self.halfrot_cgtoa = None
            self.halfrot_cgtob = None
            self.halfrot_cgto = self.cgto
        if halfrot_M is not None:
            self.halfrot_M = halfrot_M
        else:
            self.halfrot_M = None
        

        self.kpts = kpts
        self.Sset = find_self_inverse_set(self.kpts)
        self.Qplus = find_Qplus(self.kpts)
        self.unique_k = numpy.concatenate((self.Sset, self.Qplus))
        self.unique_nk = self.unique_k.shape[0]
        self.ikpq_mat = construct_kpq(self.kpts)
        self.ikmq_mat = construct_kmq(self.kpts)
        self.imq_vec = construct_mq(self.kpts)
        self.igamma = find_gamma_pt(self.kpts[self.unique_k])
        self.nk = self.kpts.shape[0]
        self.nisdf = self.cgto.shape[1]
        self.nisdf_halfrot = self.halfrot_cgto.shape[1]

        self.chunked = False

        # this is the one-body part that comes out of re-ordering the 2-body operators
        h1e_mod = numpy.zeros(self.H1.shape, dtype=self.H1.dtype)
        construct_h1e_mod_isdf(self.MPQ, self.cgto, self.H1, self.ikpq_mat, self.ikmq_mat, self.Sset, self.Qplus, h1e_mod)
        self.h1e_mod = xp.array(h1e_mod)

        if verbose:
            mem = 2 * self.cholM.nbytes / (1024.0**3) + 2 * self.cgto.nbytes / (1024.0**3)
            print("# Number of orbitals: %d" % self.nbasis)
            print(f"# Approximate memory required by ISDF vectors {mem:f} GB")
            print("# Number of Cholesky vectors: %d" % (self.nchol))
            print("# Finished setting up KptISDF object.")