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
from cuquantum.bindings import cutensornet
from cuquantum.tensornet import NetworkOptions, contract

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

def construct_VHS_cuquantum(chol, sqrt_dt, xshifted, nk, nbasis, nwalkers, ikpq_mat, Sset, Qplus):
    VHS = xp.zeros((nwalkers, nk, nbasis, nk, nbasis), dtype=xp.complex128)
    x= .5 * (1j * xshifted[0] + xshifted[1])
    xconj = .5 * (1j * xshifted[0] - xshifted[1])
    unique_qs = xp.concatenate((Sset, Qplus))
    unique_nq = len(unique_qs)
    idx_lenS = xp.arange(len(Sset))
    idx_lenQ = xp.arange(len(Qplus)) + len(Sset)

    xS = sqrt_dt * x[:, :, idx_lenS]
    xQ = xp.sqrt(2) * sqrt_dt * x[:, :, idx_lenQ]
    xconjS = sqrt_dt * xconj[:, :, idx_lenS]
    xconjQ = xp.sqrt(2) * sqrt_dt * xconj[:, :, idx_lenQ]

    xtot = xp.concatenate((xS, xQ), axis=-1)
    xconjtot = xp.concatenate((xconjS, xconjQ), axis=-1)

    naux = chol.shape[0]

    handle = cutensornet.create()
    network_opts = NetworkOptions(handle=handle)
    vhs1 = contract('Xkpqr, wXq -> wkpqr', chol, xtot, options=network_opts)
    # feed in the result
    w_idx = xp.arange(nwalkers)[:, None, None, None, None]
    k_idx = xp.arange(nk)[None, :, None, None, None] 
    p_idx = xp.arange(nbasis)[None, None, :, None, None]
    q_idx = unique_qs[None, None, None, :, None]
    r_idx = xp.arange(nbasis)[None, None, None, None, :]
    ikpq_idx = ikpq_mat[k_idx, q_idx]
    VHS[w_idx, k_idx, p_idx, ikpq_idx, r_idx] += vhs1
    del vhs1
    vhs2 = contract('Xkpqr, wXq -> wkrqp', chol.conj(), xconjtot, options=network_opts)
    VHS[w_idx, ikpq_idx, p_idx, k_idx, r_idx] += vhs2
    del vhs2
    cutensornet.destroy(handle)
    xp.cuda.stream.get_current_stream().synchronize()
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
            if hamiltonian.nk > 256:
                # nk^2 > 65536, use cuquantum instead
                VHS_send = construct_VHS_cuquantum(chol_chunk, self.sqrt_dt, xshifted[:, :, idxs, :], hamiltonian.nk, hamiltonian.nbasis, nwalkers, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus)
            else:
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
                if hamiltonian.nk > 256:
                    VHS_send = construct_VHS_cuquantum(chol_chunk, self.sqrt_dt, xshifted_recv[:, :, idxs, :], hamiltonian.nk, hamiltonian.nbasis, nwalkers, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus)
                else:
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
        # if config.get_option("use_gpu"):
        #     xp._default_memory_pool.free_all_blocks()
        return VHS_recv

class PhaselessKptISDF(PhaselessKptBase):
    """A class for performing phaseless propagation with k-point Hamiltonian with ERI approximated by ISDF. Here we do not save VHS to save memory."""

    def __init__(self, time_step, exp_nmax=6, verbose=False):
        super().__init__(time_step, verbose=verbose)
        self.exp_nmax = exp_nmax

    @plum.dispatch
    def apply_VHS(self, walkers: UHFWalkers, hamiltonian: GenericBase, xshifted: xp.ndarray):
        if config.get_option("use_gpu"):
            #determine whether to use bigmem algo or not
            nbsf = hamiltonian.nbasis
            nk = hamiltonian.nk
            nwalkers = walkers.nwalkers
            mem_vhs = 2* nwalkers * nbsf**2 * nk**2 * 16
            if mem_vhs > 0.35 * xp.cuda.Device().mem_info[0]:
                start_time = time.time()
                Lx, Lconjx = self.contract_cholM_xshifted(hamiltonian, xshifted)
                self.timer.tvhs += time.time() - start_time
                assert len(Lx.shape) == 3 # nwalkers, nq, nisdf

                start_time = time.time()
                Temp = xp.zeros(walkers.phia.shape, dtype=walkers.phia.dtype)
                xp.copyto(Temp, walkers.phia)
                handle = cutensornet.create()
                network_opts = NetworkOptions(handle=handle)
                for n in range(1, self.exp_nmax + 1):
                    Temp = apply_VHS_to_phi_batch(hamiltonian.cgto, Lx, Lconjx, Temp, hamiltonian.ikpq_mat, hamiltonian.ikmq_mat, hamiltonian.unique_k, network_opts) / n  # matmul use much less GPU memory than einsum
                    walkers.phia += Temp
                del Temp
                if walkers.ndown > 0 and not walkers.rhf:
                    Temp = xp.zeros(walkers.phib.shape, dtype=walkers.phib.dtype)
                    xp.copyto(Temp, walkers.phib)
                    for n in range(1, self.exp_nmax + 1):
                        Temp = apply_VHS_to_phi_batch(hamiltonian.cgto, Lx, Lconjx, Temp, hamiltonian.ikpq_mat, hamiltonian.ikmq_mat, hamiltonian.unique_k, network_opts) / n  # matmul use much less GPU memory than einsum
                        walkers.phib += Temp
                    del Temp
                cutensornet.destroy(handle)
            else:
                start_time = time.time()
                Lx, Lconjx = self.contract_cholM_xshifted(hamiltonian, xshifted)
                handle = cutensornet.create()
                network_opts = NetworkOptions(handle=handle)
                VHS = construct_VHS_batch(hamiltonian.cgto, Lx, Lconjx, hamiltonian.ikpq_mat, hamiltonian.ikmq_mat, hamiltonian.unique_k, network_opts)
                cutensornet.destroy(handle)
                synchronize()
                self.timer.tvhs += time.time() - start_time
                assert len(Lx.shape) == 3 # nwalkers, nq, nisdf

                start_time = time.time()
                walkers.phia = apply_exponential_batch(walkers.phia, VHS, self.exp_nmax)
                if walkers.ndown > 0 and not walkers.rhf:
                    walkers.phib = apply_exponential_batch(walkers.phib, VHS, self.exp_nmax)
            synchronize()
            self.timer.tgemm += time.time() - start_time                
        else:
            raise NotImplementedError
        

    def contract_cholM_xshifted(self, hamiltonian, xshifted):
        cholM = hamiltonian.cholM # q, P, gamma
        x = .5 * (1j * xshifted[0] + xshifted[1]) # w, gamma, q
        xconj = .5 * (1j * xshifted[0] - xshifted[1]) # w, gamma, q
        unique_qs = xp.concatenate((hamiltonian.Sset, hamiltonian.Qplus))
        # print("ikpq_S", ikpq_S)
        idx_lenS = xp.arange(len(hamiltonian.Sset))
        idx_lenQ = xp.arange(len(hamiltonian.Qplus)) + len(hamiltonian.Sset)

        xS = self.sqrt_dt * x[:, :, idx_lenS]
        xQ = xp.sqrt(2) * self.sqrt_dt * x[:, :, idx_lenQ]
        xconjS = self.sqrt_dt * xconj[:, :, idx_lenS]
        xconjQ = xp.sqrt(2) * self.sqrt_dt * xconj[:, :, idx_lenQ]

        xtot = xp.concatenate((xS, xQ), axis=-1)
        xconjtot = xp.concatenate((xconjS, xconjQ), axis=-1)
        handle = cutensornet.create()
        network_opts = NetworkOptions(handle=handle)
        cholMx = contract('qPg, wgq -> wqP', cholM, xtot, options=network_opts)
        cholMxconj = contract('qPg, wgq -> wqP', cholM.conj(), xconjtot, options=network_opts)
        cutensornet.destroy(handle)
        return cholMx, cholMxconj

def construct_full_Lx_batch(Lx, kpq_mat, unique_qs):
    """
    Construct full Lx from Lx using advanced indexing. Lx: [nwalkers, nk, nisdf]. 
    """
    nk = kpq_mat.shape[1]
    nq = len(unique_qs)
    nwalker = Lx.shape[0]
    nisdf = Lx.shape[-1]  # Lx: (nwalker, nq, nisdf)

    fullLconjx = xp.zeros((nwalker, nk, nk, nisdf), dtype=xp.complex128)
    q_idx = unique_qs[:, None]          # shape (nq, 1)
    ik_idx = xp.arange(nk)[None, :] # shape (1, nk)

    row_idx = kpq_mat[q_idx, ik_idx]
    row_idx = xp.broadcast_to(row_idx, (nq, nk))
    col_idx = xp.broadcast_to(ik_idx, (nq, nk))
    data = Lx[:, :, None, :]  # shape (nw, nq, 1, nisdf)
    data = xp.broadcast_to(data, (nwalker, nq, nk, nisdf))
    fullLconjx[:, row_idx, col_idx, :] = data

    return fullLconjx
        
def apply_VHS_to_phi_batch(cgto, Lx, Lconjx, phi, kpq_mat, kmq_mat, unique_qs, network_opts):
    """
    Apply VHS to phi in batch.
    """
    nwalkers = Lx.shape[0]
    nk = kpq_mat.shape[1]
    nbsf = cgto.shape[2]
    nisdf = Lx.shape[-1]
    outphi = xp.zeros_like(phi)
    # calculate intermediate array memory
    mem_cost = nwalkers * nk * nk * nisdf * nbsf * 16 * 2/ 1024**3
    # max_mem = 80 percent of available memory
    max_mem = 0.3 * xp.cuda.Device().mem_info[0] / 1024**3
    num_nisdf_chunks = max(1, math.ceil(mem_cost / max_mem))
    nisdf_chunk_size = math.ceil(nisdf / num_nisdf_chunks)
    nisdf_left = nisdf
    for i in range(num_nisdf_chunks):
        if nisdf_left == 0:
            break
        nisdf_chunk = min(nisdf_chunk_size, nisdf_left)
        nisdf_left -= nisdf_chunk
        Lx_chunk = Lx[:, :, i * nisdf_chunk_size: i * nisdf_chunk_size + nisdf_chunk]
        Lconjx_chunk = Lconjx[:, :, i * nisdf_chunk_size: i * nisdf_chunk_size + nisdf_chunk]
        full_Lx = construct_full_Lx_batch(Lx_chunk, kpq_mat, unique_qs)
        full_Lconjx = construct_full_Lx_batch(Lconjx_chunk, kmq_mat, unique_qs)
        fullLpLconjx = full_Lx + full_Lconjx
        phi_reshape = phi.reshape(nwalkers, nk, nbsf, nk, -1)
        cgto_slice = cgto[:, i * nisdf_chunk_size: i * nisdf_chunk_size + nisdf_chunk, :]
        out = contract('wKkP, kPp, KPr, wKrQi -> wkpQi', fullLpLconjx, cgto_slice.conj(), cgto_slice, phi_reshape, options=network_opts)
        del full_Lx, full_Lconjx, fullLpLconjx, cgto_slice, phi_reshape
        outphi += out.reshape(nwalkers, nk * nbsf, -1)
        del out
    xp._default_memory_pool.free_all_blocks()
    return outphi

def construct_VHS_batch(cgto, Lx, Lconjx, kpq_mat, kmq_mat, unique_qs, network_opts):
    """
    Apply VHS to phi in batch.
    """
    nwalkers = Lx.shape[0]
    nk = kpq_mat.shape[1]
    nbsf = cgto.shape[2]
    nisdf = Lx.shape[-1]
    outvhs = xp.zeros((nwalkers, nk * nbsf, nk * nbsf), dtype=xp.complex128)
    # calculate intermediate array memory
    mem_cost = nwalkers * nk * nk * nisdf * nbsf * 16 * 2/ 1024**3
    # max_mem = 80 percent of available memory
    max_mem = 0.3 * xp.cuda.Device().mem_info[0] / 1024**3
    num_nisdf_chunks = max(1, math.ceil(mem_cost / max_mem))
    nisdf_chunk_size = math.ceil(nisdf / num_nisdf_chunks)
    nisdf_left = nisdf
    for i in range(num_nisdf_chunks):
        if nisdf_left == 0:
            break
        nisdf_chunk = min(nisdf_chunk_size, nisdf_left)
        nisdf_left -= nisdf_chunk
        Lx_chunk = Lx[:, :, i * nisdf_chunk_size: i * nisdf_chunk_size + nisdf_chunk]
        Lconjx_chunk = Lconjx[:, :, i * nisdf_chunk_size: i * nisdf_chunk_size + nisdf_chunk]
        full_Lx = construct_full_Lx_batch(Lx_chunk, kpq_mat, unique_qs)
        full_Lconjx = construct_full_Lx_batch(Lconjx_chunk, kmq_mat, unique_qs)
        fullLpLconjx = full_Lx + full_Lconjx
        cgto_slice = cgto[:, i * nisdf_chunk_size: i * nisdf_chunk_size + nisdf_chunk, :]
        out = contract('wKkP, kPp, KPr -> wkpKr', fullLpLconjx, cgto_slice.conj(), cgto_slice, options=network_opts)
        del full_Lx, full_Lconjx, fullLpLconjx, cgto_slice
        outvhs += out.reshape(nwalkers, nk * nbsf, nk * nbsf)
        del out
    xp._default_memory_pool.free_all_blocks()
    return outvhs

Phaseless = {"cholesky": PhaselessKptChol, "isdf": PhaselessKptISDF, "cholchunked": PhaselessKptCholChunked}
