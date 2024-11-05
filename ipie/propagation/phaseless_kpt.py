import math
import time

import numpy

import plum

from ipie.config import config
from ipie.hamiltonians.kpt_hamiltonian import KptComplexChol, KptComplexCholSymm, KptISDF
from ipie.hamiltonians.kpt_chunked import KptComplexCholChunked
from ipie.hamiltonians.generic_base import GenericBase
from ipie.propagation.operations import apply_exponential, apply_exponential_batch
from ipie.propagation.phaseless_kpt_base import PhaselessKptBase
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import synchronize
from ipie.walkers.uhf_walkers import UHFWalkers
from numba import jit
from ipie.utils.backend import get_device_memory
from ipie.propagation.kernels import call_kernel_VHS_construction1, call_kernel_VHS_construction2

@jit(nopython=True, fastmath=True)
def construct_VHS_kernel_symm(chol, sqrt_dt, xshifted, nk, nbasis, nwalkers, ikpq_mat, Sset, Qplus):

    VHS = numpy.zeros((nk, nk, nwalkers, nbasis * nbasis), dtype=numpy.complex128)
    for iq in range(len(Sset)):
        iq_real = Sset[iq]
        for ik in range(nk):
            ikpq = ikpq_mat[iq_real, ik]
            x_iq = .5 * (1j * xshifted[0, :, :, iq] + xshifted[1, :, :, iq])
            xconj_iq = .5 * (1j * xshifted[0, :, :, iq] - xshifted[1, :, :, iq])
            cholkq = chol[:, ik, :, iq, :].copy()
            cholkq = cholkq.reshape(-1, nbasis*nbasis)
            VHS[ik, ikpq] += sqrt_dt * x_iq @ cholkq
            XL = sqrt_dt * xconj_iq @ cholkq.conj()
            XL = XL.reshape(nwalkers, nbasis, nbasis).transpose(0, 2, 1).copy()
            VHS[ikpq, ik] += XL.reshape(nwalkers, nbasis * nbasis)

    for iq in range(len(Sset), len(Sset) + len(Qplus)):
        iq_real = Qplus[iq - len(Sset)]
        for ik in range(nk):
            ikpq = ikpq_mat[iq_real, ik]
            x_iq = .5 * (1j * xshifted[0, :, :, iq] + xshifted[1, :, :, iq])
            xconj_iq = .5 * (1j * xshifted[0, :, :, iq] - xshifted[1, :, :, iq])
            cholkq = chol[:, ik, :, iq, :].copy()
            cholkq = cholkq.reshape(-1, nbasis*nbasis)
            VHS[ik, ikpq] += math.sqrt(2) * sqrt_dt * x_iq @ cholkq
            XL = sqrt_dt * xconj_iq @ cholkq.conj()
            XL = XL.reshape(nwalkers, nbasis, nbasis).transpose(0, 2, 1).copy()
            VHS[ikpq, ik] += math.sqrt(2) * XL.reshape(nwalkers, nbasis * nbasis)
    VHS = VHS.reshape(nk, nk, nwalkers, nbasis, nbasis).transpose(2, 0, 3, 1, 4).copy()
    VHS = VHS.reshape(nwalkers, nk * nbasis, nk * nbasis)
    return VHS

def construct_VHS_symm_gpu(chol, sqrt_dt, xshifted, nk, nbasis, nwalkers, ikpq_mat, Sset, Qplus):
    VHS = xp.zeros((nwalkers, nk, nbasis, nk, nbasis), dtype=xp.complex128)
    x= .5 * (1j * xshifted[0] + xshifted[1])
    xconj = .5 * (1j * xshifted[0] - xshifted[1])
    unique_qs = xp.concatenate((Sset, Qplus))
    # print("ikpq_S", ikpq_S)
    idx_lenS = xp.arange(len(Sset))
    idx_lenQ = xp.arange(len(Qplus)) + len(Sset)

    xS = sqrt_dt * x[:, :, idx_lenS]
    xQ = xp.sqrt(2) * sqrt_dt * x[:, :, idx_lenQ]
    xconjS = sqrt_dt * xconj[:, :, idx_lenS]
    xconjQ = xp.sqrt(2) * sqrt_dt * xconj[:, :, idx_lenQ]

    xtot = xp.concatenate((xS, xQ), axis=-1)
    xconjtot = xp.concatenate((xconjS, xconjQ), axis=-1)

    kpq_mat = ikpq_mat[unique_qs]

    naux = chol.shape[0]

    call_kernel_VHS_construction1(chol, xtot, naux, nk, nbasis, nwalkers, kpq_mat, VHS)
    call_kernel_VHS_construction2(chol, xconjtot, naux, nk, nbasis, nwalkers, kpq_mat, VHS)

    VHS = VHS.reshape(nwalkers, nk * nbasis, nk * nbasis)
    return VHS

class PhaselessKptChol(PhaselessKptBase):
    """A class for performing phaseless propagation with k-point Hamiltonian."""

    def __init__(self, time_step, exp_nmax=6, verbose=False):
        super().__init__(time_step, verbose=verbose)
        self.exp_nmax = exp_nmax

    @plum.dispatch
    def apply_VHS(self, walkers: UHFWalkers, hamiltonian: GenericBase, xshifted: xp.ndarray):
        start_time = time.time()
        VHS = self.construct_VHS(hamiltonian, xshifted)
        synchronize()
        self.timer.tvhs += time.time() - start_time
        assert len(VHS.shape) == 3  # shape = nwalkers, nk * nbasis, nk * nbasis
        start_time = time.time()
        if config.get_option("use_gpu"):
            walkers.phia = apply_exponential_batch(walkers.phia, VHS, self.exp_nmax)
            if walkers.ndown > 0 and not walkers.rhf:
                walkers.phib = apply_exponential_batch(walkers.phib, VHS, self.exp_nmax)

        else:
            for iw in range(walkers.nwalkers):
                # 2.b Apply two-body
                walkers.phia[iw] = apply_exponential(walkers.phia[iw], VHS[iw], self.exp_nmax)
                if walkers.ndown > 0 and not walkers.rhf:
                    walkers.phib[iw] = apply_exponential(walkers.phib[iw], VHS[iw], self.exp_nmax)
        synchronize()
        self.timer.tgemm += time.time() - start_time

    @plum.dispatch.abstract
    def construct_VHS(self, hamiltonian: GenericBase, xshifted: xp.ndarray) -> xp.ndarray:
        print("JOONHO here abstract function for construct VHS")
        "abstract function for construct VHS"

    # Any class inherited from PhaselessGeneric should override this method.
    @plum.dispatch
    def construct_VHS(self, hamiltonian: KptComplexChol, xshifted: xp.ndarray) -> xp.ndarray:
        """
        Construct the VHS matrix for phaseless propagation.
        
        xshifted: [2, nwalkers, naux, nk]
        """
        nwalkers = xshifted.shape[1]
        VHS = numpy.zeros((nwalkers, hamiltonian.nk, hamiltonian.nbasis, hamiltonian.nk, hamiltonian.nbasis), dtype=numpy.complex128)

        for iq in range(hamiltonian.nk):
            for ik in range(hamiltonian.nk):
                ikpq = hamiltonian.ikpq_mat[ik, iq]
                imq = hamiltonian.imq_vec[iq]
                xtildepiq = xshifted[0, :, :, iq] + xshifted[0, :, :, imq]
                xtildemiq = xshifted[1, :, :, iq] - xshifted[1, :, :, imq]
                xvhsiq = (1j * xtildepiq + xtildemiq) / 2
                VHS[:, ik, :, ikpq, :] = self.sqrt_dt * numpy.einsum('wx, xpr -> wpr', xvhsiq, hamiltonian.chol[:, ik, :, iq, :])
        VHS = VHS.reshape(nwalkers, hamiltonian.nk * hamiltonian.nbasis, hamiltonian.nk * hamiltonian.nbasis)
        if config.get_option("use_gpu"):
            raise NotImplementedError
        return VHS
    
    @plum.dispatch
    def construct_VHS(self, hamiltonian: KptComplexCholSymm, xshifted: xp.ndarray) -> xp.ndarray:
        """
        Construct the VHS matrix for phaseless propagation.
        
        xshifted: [2, nwalkers, naux, unique_nk]
        """
        nwalkers = xshifted.shape[1]
        if config.get_option("use_gpu"):
            raise NotImplementedError
        else:
            VHS = construct_VHS_kernel_symm(hamiltonian.chol, self.sqrt_dt, xshifted, hamiltonian.nk, hamiltonian.nbasis, nwalkers, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus)
        
        return VHS

class PhaselessKptCholChunked(PhaselessKptChol):
    """A class for performing phaseless propagation with complex hamiltonian with k point symmetry."""

    def __init__(self, time_step, exp_nmax=6, verbose=False):
        super().__init__(time_step, exp_nmax=exp_nmax, verbose=verbose)

    def build(self, hamiltonian, trial=None, walkers=None, mpi_handler=None, verbose=False):
        super().build(hamiltonian, trial, walkers, mpi_handler, verbose)
        self.mpi_handler = mpi_handler

    @plum.dispatch
    def construct_VHS(
        self, hamiltonian: KptComplexCholChunked, xshifted: xp.ndarray
    ) -> xp.ndarray:
        assert hamiltonian.chunked
        nwalkers = xshifted.shape[1]
        
        xshifted_send = xshifted.copy()
        xshifted_recv = xp.zeros_like(xshifted)

        idxs = hamiltonian.chol_idxs_chunk
        chol_chunk = hamiltonian.chol_chunk.reshape(-1, hamiltonian.nk, hamiltonian.nbasis, hamiltonian.unique_nk, hamiltonian.nbasis)
        if config.get_option("use_gpu"):
            VHS_send = construct_VHS_symm_gpu(chol_chunk, self.sqrt_dt, xshifted[:, :, idxs, :], hamiltonian.nk, hamiltonian.nbasis, nwalkers, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus)
        else:
            VHS_send = construct_VHS_kernel_symm(chol_chunk, self.sqrt_dt, xshifted[:, :, idxs, :], hamiltonian.nk, hamiltonian.nbasis, nwalkers, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus)
        VHS_recv = xp.zeros_like(VHS_send)

        srank = self.mpi_handler.scomm.rank
        sender = numpy.where(self.mpi_handler.receivers == srank)[0]

        for _ in range(self.mpi_handler.ssize - 1):
            synchronize()
            self.mpi_handler.scomm.Isend(
                xshifted_send, dest=self.mpi_handler.receivers[srank], tag=1
            )
            self.mpi_handler.scomm.Isend(VHS_send, dest=self.mpi_handler.receivers[srank], tag=2)

            req1 = self.mpi_handler.scomm.Irecv(xshifted_recv, source=sender, tag=1)
            req2 = self.mpi_handler.scomm.Irecv(VHS_recv, source=sender, tag=2)
            req1.wait()
            req2.wait()

            self.mpi_handler.scomm.barrier()
            if config.get_option("use_gpu"):
                VHS_send = construct_VHS_symm_gpu(chol_chunk, self.sqrt_dt, xshifted_recv[:, :, idxs, :], hamiltonian.nk, hamiltonian.nbasis, nwalkers, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus)
            else:
                VHS_send = construct_VHS_kernel_symm(chol_chunk, self.sqrt_dt, xshifted_recv[:, :, idxs, :], hamiltonian.nk, hamiltonian.nbasis, nwalkers, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus)
            VHS_send += VHS_recv

            xshifted_send = xshifted_recv.copy()

        synchronize()
        self.mpi_handler.scomm.Isend(VHS_send, dest=self.mpi_handler.receivers[srank], tag=1)
        req = self.mpi_handler.scomm.Irecv(VHS_recv, source=sender, tag=1)
        req.wait()
        self.mpi_handler.scomm.barrier()

        synchronize()
        return VHS_recv

class PhaselessKptISDF(PhaselessKptBase):
    """A class for performing phaseless propagation with k-point Hamiltonian with ERI approximated by ISDF."""

    def __init__(self, time_step, exp_nmax=6, verbose=False):
        super().__init__(time_step, verbose=verbose)
        self.exp_nmax = exp_nmax

    @plum.dispatch
    def apply_VHS(self, walkers: UHFWalkers, hamiltonian: GenericBase, xshifted: xp.ndarray):
        start_time = time.time()
        VHS = self.construct_VHS(hamiltonian, xshifted)
        synchronize()
        self.timer.tvhs += time.time() - start_time
        assert len(VHS.shape) == 3  # shape = nwalkers, nk * nbasis, nk * nbasis

        start_time = time.time()
        if config.get_option("use_gpu"):
            walkers.phia = apply_exponential_batch(walkers.phia, VHS, self.exp_nmax)
            if walkers.ndown > 0 and not walkers.rhf:
                walkers.phib = apply_exponential_batch(walkers.phib, VHS, self.exp_nmax)
        else:
            for iw in range(walkers.nwalkers):
                # 2.b Apply two-body
                walkers.phia[iw] = apply_exponential(walkers.phia[iw], VHS[iw], self.exp_nmax)
                if walkers.ndown > 0 and not walkers.rhf:
                    walkers.phib[iw] = apply_exponential(walkers.phib[iw], VHS[iw], self.exp_nmax)
        synchronize()
        self.timer.tgemm += time.time() - start_time

    @plum.dispatch.abstract
    def construct_VHS(self, hamiltonian: GenericBase, xshifted: xp.ndarray) -> xp.ndarray:
        print("JOONHO here abstract function for construct VHS")
        "abstract function for construct VHS"

    # Any class inherited from PhaselessGeneric should override this method.
    @plum.dispatch
    def construct_VHS(self, hamiltonian: KptISDF, xshifted: xp.ndarray) -> xp.ndarray:
        """
        Construct the VHS matrix for phaseless propagation.
        
        xshifted: [2, nwalkers, naux, nk]
        """
        nwalkers = xshifted.shape[1]
        Lmat = numpy.zeros((nwalkers, hamiltonian.nk, hamiltonian.nbasis, hamiltonian.nk, hamiltonian.nbasis), dtype=numpy.complex128)
        Lmatdagger

        for iq in range(hamiltonian.nk):
            Glis = hamiltonian.q2G[iq]
            for iG in range(len(Glis)):
                try:
                    ik_lis = hamiltonian.qG2k[(iq, iG)]
                    for ik in ik_lis:
                        ikpq = hamiltonian.ikpq_mat[ik, iq]
                        Lmat[:, ik, :, ikpq, :] += self.sqrt_dt * numpy.einsum("wX, XP, Pp, Pr -> wpr", xshifted[0, :, :, iq], hamiltonian.cholM[:, iq, iG, :], hamiltonian.weights[:, :, ik].conj(), hamiltonian.weights[:, :, ikpq])
                        Lmatdagger[:, ikpq, :, ik, :] += self.sqrt_dt * numpy.einsum("wX, XP, Pp, Pr -> wpr", xshifted[1, :, :, iq], hamiltonian.cholM[:, iq, iG, :].conj(), hamiltonian.weights[:, :, ikpq].conj(), hamiltonian.weights[:, :, ik])
                except KeyError:
                    continue
        Lmat = Lmat.reshape(nwalkers, hamiltonian.nk * hamiltonian.nbasis, hamiltonian.nk * hamiltonian.nbasis)
        Lmatdagger = Lmatdagger.reshape(nwalkers, hamiltonian.nk * hamiltonian.nbasis, hamiltonian.nk * hamiltonian.nbasis)
        VHS = Lmat + Lmatdagger
        if config.get_option("use_gpu"):
            raise NotImplementedError
        return VHS

Phaseless = {"cholesky": PhaselessKptChol, "isdf": PhaselessKptISDF, "cholchunked": PhaselessKptCholChunked}
