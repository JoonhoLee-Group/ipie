# from line_profiler import LineProfiler
from math import ceil, sqrt

import numpy
from numba import jit

from ipie.estimators.local_energy import local_energy_G
from ipie.estimators.kernels import exx_kpt_kernel
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import synchronize
from ipie.config import config
from ipie.utils.backend import get_device_memory

from ipie.systems.generic import Generic
from ipie.hamiltonians.kpt_chunked import KptComplexCholChunked
from ipie.walkers.uhf_walkers import UHFWalkers
from ipie.trial_wavefunction.single_det_kpt import KptSingleDet
from ipie.estimators.local_energy_kpt_sd import kpt_symmchol_ecoul_kernel_uhf, kpt_symmchol_exx_kernel

import time

# from line_profiler import profile

import plum
# Note specialisations occur to because:
# 1. Numba does not allow for mixing types without a warning so need to split
# real and complex components apart when rchol is real. Green's function is
# complex in general.
# Optimize for case when wavefunction is RHF (factor of 2 saving)

@plum.dispatch
def local_energy_kpt_single_det_uhf_chunked(
    system: Generic,
    hamiltonian: KptComplexCholChunked,
    walkers: UHFWalkers,
    trial: KptSingleDet,
):
    """Compute local energy for walker batch (all walkers at once).

    Single determinant UHF case.

    Parameters
    ----------
    system : system object
        System being studied.
    hamiltonian : hamiltonian object
        Hamiltonian being studied.
    walkers : WalkerBatch
        Walkers object.
    trial : trial object
        Trial wavefunction.

    Returns
    -------
    local_energy : np.ndarray
        Total, one-body and two-body energies.
    """
    if config.get_option("use_gpu"):
        return local_energy_kpt_single_det_uhf_batch_chunked_gpu(system, hamiltonian, walkers, trial)
    else:
        return local_energy_kpt_single_det_uhf_chunked_cpu(system, hamiltonian, walkers, trial)

def local_energy_kpt_single_det_uhf_chunked_cpu(
    system: Generic,
    hamiltonian: KptComplexCholChunked,
    walkers: UHFWalkers,
    trial: KptSingleDet,
):
    """Compute local energy for walker batch (all walkers at once).

    Single determinant UHF case.

    Parameters
    ----------
    system : system object
        System being studied.
    hamiltonian : hamiltonian object
        Hamiltonian being studied.
    walkers : WalkerBatch
        Walkers object.
    trial : trial object
        Trial wavefunction.

    Returns
    -------
    local_energy : np.ndarray
        Total, one-body and two-body energies.
    """
    assert hamiltonian.chunked

    nwalkers = walkers.Ghalfa.shape[0]
    nk = hamiltonian.nk
    nalpha = trial.nalpha
    nbeta = trial.nbeta
    nbasis = hamiltonian.nbasis

    ghalfa = walkers.Ghalfa.reshape(nwalkers, nk, nalpha, nk, nbasis)
    ghalfb = walkers.Ghalfb.reshape(nwalkers, nk, nbeta, nk, nbasis)
    ghalfaT = walkers.Ghalfa.transpose(0, 2, 1).reshape(nwalkers, nk, nbasis, nk, nalpha)
    ghalfbT = walkers.Ghalfb.transpose(0, 2, 1).reshape(nwalkers, nk, nbasis, nk, nbeta)

    diagGhalfa = numpy.zeros((nwalkers, nk, nalpha, nbasis), dtype=numpy.complex128)
    diagGhalfb = numpy.zeros((nwalkers, nk, nbeta, nbasis), dtype=numpy.complex128)
    for ik in range(nk):
        diagGhalfa[:, ik, :, :] = ghalfa[:, ik, :, ik, :]
        diagGhalfb[:, ik, :, :] = ghalfb[:, ik, :, ik, :]
    diagGhalfa = diagGhalfa.reshape(nwalkers, nk * nalpha * nbasis)
    diagGhalfb = diagGhalfb.reshape(nwalkers, nk * nbeta * nbasis)
    e1b = diagGhalfa.dot(trial._rH1a.ravel())
    e1b += diagGhalfb.dot(trial._rH1b.ravel())
    e1b /= nk
    e1b += hamiltonian.ecore

    ghalfa = ghalfa.transpose(1, 3, 0, 2, 4).copy() # nk, nk, nw, nalpha, nbasis
    ghalfb = ghalfb.transpose(1, 3, 0, 2, 4).copy() # nk, nk, nw, nbeta, nbasis
    ghalfaTcoul = ghalfaT.transpose(1, 3, 0, 2, 4).copy() # nk, nk, nw, nbasis, nalpha
    ghalfbTcoul = ghalfbT.transpose(1, 3, 0, 2, 4).copy() # nk, nk, nw, nbasis, nbeta
    ghalfaTx = ghalfaT.transpose(1, 3, 2, 4, 0).copy() # nk, nk, nbasis, nalpha, nw
    ghalfbTx = ghalfbT.transpose(1, 3, 2, 4, 0).copy() # nk, nk, nbasis, nbeta, nw

    ghalfa_send = ghalfa.copy()
    ghalfb_send = ghalfb.copy()
    ghalfaTcoul_send = ghalfaTcoul.copy()
    ghalfbTcoul_send = ghalfbTcoul.copy()
    ghalfaTx_send = ghalfaTx.copy()
    ghalfbTx_send = ghalfbTx.copy()

    ghalfa_recv = xp.zeros_like(ghalfa)
    ghalfb_recv = xp.zeros_like(ghalfb)
    ghalfaTcoul_recv = xp.zeros_like(ghalfaTcoul)
    ghalfbTcoul_recv = xp.zeros_like(ghalfbTcoul)
    ghalfaTx_recv = xp.zeros_like(ghalfaTx)
    ghalfbTx_recv = xp.zeros_like(ghalfbTx)

    handler = walkers.mpi_handler
    senders = handler.senders
    receivers = handler.receivers

    rchola_chunk = trial._rchola_chunk
    rcholb_chunk = trial._rcholb_chunk
    rcholbara_chunk = trial._rcholbara_chunk
    rcholbarb_chunk = trial._rcholbarb_chunk

    ecoul_send = kpt_symmchol_ecoul_kernel_uhf(
        rchola_chunk, rcholb_chunk, rcholbara_chunk, rcholbarb_chunk, ghalfa, ghalfb, ghalfaTcoul, ghalfbTcoul, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus
    )

    exx_send = kpt_symmchol_exx_kernel(rchola_chunk, rcholbara_chunk, ghalfa, ghalfaTx, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus) 
    exx_send += kpt_symmchol_exx_kernel(rcholb_chunk, rcholbarb_chunk, ghalfb, ghalfbTx, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus) 

    exx_recv = exx_send.copy()
    ecoul_recv = ecoul_send.copy()

    for _ in range(handler.ssize - 1):
        for isend, sender in enumerate(senders):
            if handler.srank == isend:
                handler.scomm.Send(ghalfa_send, dest=receivers[isend], tag=1)
                handler.scomm.Send(ghalfb_send, dest=receivers[isend], tag=2)
                handler.scomm.Send(ghalfaTcoul_send, dest=receivers[isend], tag=3)
                handler.scomm.Send(ghalfbTcoul_send, dest=receivers[isend], tag=4)
                handler.scomm.Send(ghalfaTx_send, dest=receivers[isend], tag=5)
                handler.scomm.Send(ghalfbTx_send, dest=receivers[isend], tag=6)
                handler.scomm.Send(ecoul_send, dest=receivers[isend], tag=7)
                handler.scomm.Send(exx_send, dest=receivers[isend], tag=8)
            elif handler.srank == receivers[isend]:
                sender = numpy.where(receivers == handler.srank)[0]
                handler.scomm.Recv(ghalfa_recv, source=sender, tag=1)
                handler.scomm.Recv(ghalfb_recv, source=sender, tag=2)
                handler.scomm.Recv(ghalfaTcoul_recv, source=sender, tag=3)
                handler.scomm.Recv(ghalfbTcoul_recv, source=sender, tag=4)
                handler.scomm.Recv(ghalfaTx_recv, source=sender, tag=5)
                handler.scomm.Recv(ghalfbTx_recv, source=sender, tag=6)
                handler.scomm.Recv(ecoul_recv, source=sender, tag=7)
                handler.scomm.Recv(exx_recv, source=sender, tag=8)
        handler.scomm.barrier()

    # prepare sending
        ecoul_send = ecoul_recv.copy()
        ecoul_send += kpt_symmchol_ecoul_kernel_uhf(
        rchola_chunk, rcholb_chunk, rcholbara_chunk, rcholbarb_chunk, ghalfa_recv, ghalfb_recv, ghalfaTcoul_recv, ghalfbTcoul_recv, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus
    )
        exx_send = exx_recv.copy()
        exx_send += kpt_symmchol_exx_kernel(rchola_chunk, rcholbara_chunk, ghalfa_recv, ghalfaTx_recv, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus)
        exx_send += kpt_symmchol_exx_kernel(rcholb_chunk, rcholbarb_chunk, ghalfb_recv, ghalfbTx_recv, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus)
        ghalfa_send = ghalfa_recv.copy()
        ghalfb_send = ghalfb_recv.copy()
        ghalfaTcoul_send = ghalfaTcoul_recv.copy()
        ghalfbTcoul_send = ghalfbTcoul_recv.copy()
        ghalfaTx_send = ghalfaTx_recv.copy()
        ghalfbTx_send = ghalfbTx_recv.copy()


    if len(senders) > 1:
        for isend, sender in enumerate(senders):
            if handler.srank == sender:  # sending 1 xshifted to 0 xshifted_buf
                handler.scomm.Send(ecoul_send, dest=receivers[isend], tag=1)
                handler.scomm.Send(exx_send, dest=receivers[isend], tag=2)
            elif handler.srank == receivers[isend]:
                sender = numpy.where(receivers == handler.srank)[0]
                handler.scomm.Recv(ecoul_recv, source=sender, tag=1)
                handler.scomm.Recv(exx_recv, source=sender, tag=2)
    e2b = ecoul_recv + exx_recv

    energy = xp.zeros((nwalkers, 3), dtype=numpy.complex128)
    energy[:, 0] = e1b + e2b
    energy[:, 1] = e1b
    energy[:, 2] = e2b

    return energy

def kpt_symmchol_ecoul_kernel_batch_uhf_gpu(rchola, rcholb, rcholbara, rcholbarb, Ghalfa, Ghalfb, kpq_mat, Sset, Qplus):
    nwalkers = Ghalfa.shape[2]

    # shape of rchola: (nq, nk, nocc, naux, nbsf) (q, k, gamma, i, p)
    # shape of Ghalf: (nk, nk, nw, nocc, nbsf)
    unique_nq = len(Sset) + len(Qplus)
    nbsf = rchola.shape[4]
    nocc = rchola.shape[2]
    naux = rchola.shape[3]
    nk = rchola.shape[1]
    ecoul = xp.zeros((nwalkers), dtype=numpy.complex128)
    X = xp.zeros((unique_nq, naux, nwalkers), dtype=numpy.complex128)
    Xbar = xp.zeros((unique_nq, naux, nwalkers), dtype=numpy.complex128)
    for iq in range(len(Sset)):
        iq_real = Sset[iq]
        ikpq_vec = kpq_mat[iq_real]
        Ga_kpq = Ghalfa[:, ikpq_vec, :, :, :]
        GaT_kpq = Ghalfa[ikpq_vec]
        Gb_kpq = Ghalfb[:, ikpq_vec, :, :, :]
        GbT_kpq = Ghalfb[ikpq_vec]
        rchola_q = rchola[iq]
        rcholb_q = rcholb[iq]
        rcholbara_q = rcholbara[iq]
        rcholbarb_q = rcholbarb[iq]
        X[iq] = xp.einsum("kixp, kkwip -> xw", rchola_q, Ga_kpq, optimize=True) + xp.einsum("kixp, kkwip -> xw", rcholb_q, Gb_kpq, optimize=True)
        Xbar[iq] = xp.einsum("ksxj, kkwjs -> xw", rcholbara_q, GaT_kpq, optimize=True) + xp.einsum("ksxj, kkwjs -> xw", rcholbarb_q, GbT_kpq, optimize=True)

    for iq in range(len(Sset), unique_nq):
        iq_real = Qplus[iq - len(Sset)]
        ikpq_vec = kpq_mat[iq_real]
        Ga_kpq = Ghalfa[:, ikpq_vec, :, :, :]
        GaT_kpq = Ghalfa[ikpq_vec]
        Gb_kpq = Ghalfb[:, ikpq_vec, :, :, :]
        GbT_kpq = Ghalfb[ikpq_vec]
        rchola_q = rchola[iq]
        rcholb_q = rcholb[iq]
        rcholbara_q = rcholbara[iq]
        rcholbarb_q = rcholbarb[iq]
        X[iq] = xp.sqrt(2) * (xp.einsum("kixp, kkwip -> xw", rchola_q, Ga_kpq, optimize=True) + xp.einsum("kixp, kkwip -> xw", rcholb_q, Gb_kpq, optimize=True))
        Xbar[iq] = xp.sqrt(2) * (xp.einsum("ksxj, kkwjs -> xw", rcholbara_q, GaT_kpq, optimize=True) + xp.einsum("ksxj, kkwjs -> xw", rcholbarb_q, GbT_kpq, optimize=True))
        #TODO: possibly write a kernel for this

    ecoul = xp.einsum("qxw, qxw -> w", X, Xbar, optimize=True)

    return 0.5 * ecoul / nk

def kpt_symmchol_ecoul_kernel_batch_rhf_gpu(rchola, rcholbara, Ghalfa, kpq_mat, Sset, Qplus):
    nwalkers = Ghalfa.shape[2]

    # shape of rchola: (nq, nk, nocc, naux, nbsf) (q, k, gamma, i, p)
    # shape of Ghalf: (nk, nk, nw, nocc, nbsf)
    unique_nq = len(Sset) + len(Qplus)
    nbsf = rchola.shape[4]
    nocc = rchola.shape[2]
    naux = rchola.shape[3]
    nk = rchola.shape[1]
    ecoul = xp.zeros((nwalkers), dtype=numpy.complex128)
    X = xp.zeros((unique_nq, naux, nwalkers), dtype=numpy.complex128)
    Xbar = xp.zeros((unique_nq, naux, nwalkers), dtype=numpy.complex128)
    for iq in range(len(Sset)):
        iq_real = Sset[iq]
        ikpq_vec = kpq_mat[iq_real]
        Ga_kpq = Ghalfa[:, ikpq_vec, :, :, :]
        GaT_kpq = Ghalfa[ikpq_vec]
        rchola_q = rchola[iq]
        rcholbara_q = rcholbara[iq]
        X[iq] = 2. * xp.einsum("kixp, kkwip -> xw", rchola_q, Ga_kpq, optimize=True)
        Xbar[iq] = 2. * xp.einsum("ksxj, kkwjs -> xw", rcholbara_q, GaT_kpq, optimize=True)

    for iq in range(len(Sset), unique_nq):
        iq_real = Qplus[iq - len(Sset)]
        ikpq_vec = kpq_mat[iq_real]
        Ga_kpq = Ghalfa[:, ikpq_vec, :, :, :]
        GaT_kpq = Ghalfa[ikpq_vec]
        rchola_q = rchola[iq]
        rcholbara_q = rcholbara[iq]
        X[iq] = xp.sqrt(2) * 2. * xp.einsum("kixp, kkwip -> xw", rchola_q, Ga_kpq, optimize=True)
        Xbar[iq] = xp.sqrt(2) * 2. * xp.einsum("ksxj, kkwjs -> xw", rcholbara_q, GaT_kpq, optimize=True)

    ecoul = xp.einsum("qxw, qxw -> w", X, Xbar, optimize=True)

    return 0.5 * ecoul / nk

def kpt_symmchol_exx_kernel_batch_gpu(rchola_chunk, rcholbara_chunk, Ghalfa, kpq_mat, Sset, Qplus, max_mem=4.0):
    # shape of rchola: (nq, nk, nocc, naux, nbsf) (q, k, i, gamma, p)
    # shape of rcholbara: (nq, nk, nbsf, naux, nocc) (q, k, p, gamma, i)
    # shape of Ghalf: (nk, nk, nw, nocc, nbsf)
    # shape of GhalfT: (nk, nk, nbsf, nocc, nw)
    # buff size: (nchol_chunk, nocc, nwalkers, nocc)
    unique_q = xp.concatenate((Sset, Qplus))
    kpq_res_q = kpq_mat[unique_q]

    nwalkers = Ghalfa.shape[2]
    nSset = len(Sset)
    nQplus = len(Qplus)
    nocc = rchola_chunk.shape[2]
    nchol = rchola_chunk.shape[3]
    nk = rchola_chunk.shape[1]

    exx = xp.zeros((nwalkers), dtype=numpy.complex128)

    kcube_Sset = xp.arange(len(Sset) * nk * nk)
    kcube_Qplus = xp.arange(len(Qplus) * nk * nk) + len(Sset) * nk * nk

    mem_needed = 16 * nwalkers * nocc * nocc * nchol * 2 * nSset * nk * nk / (1024.0**3.0)
    num_nk_chunks_Sset = max(1, ceil(mem_needed / max_mem))
    nk_chunk_Sset_size = ceil(len(kcube_Sset) / num_nk_chunks_Sset)
    nkcube_left = len(kcube_Sset)
    if len(kcube_Sset) > 0:
        for i in range(num_nk_chunks_Sset):
            nk_chunk = min(nkcube_left, nk_chunk_Sset_size)
            nkcube_left -= nk_chunk
            k_sls = kcube_Sset[i * nk_chunk_Sset_size: i * nk_chunk_Sset_size + nk_chunk]
            exx += exx_kpt_kernel(rchola_chunk, rcholbara_chunk, Ghalfa, k_sls, kpq_res_q)

    mem_needed = 16 * nwalkers * nocc * nocc * nchol * 2 * nQplus * nk * nk / (1024.0**3.0)
    num_nk_chunks_Qplus = max(1, ceil(mem_needed / max_mem))
    nk_chunk_Qplus_size = ceil(len(kcube_Qplus) / num_nk_chunks_Qplus)
    if nk_chunk_Qplus_size > 65535:
        nk_chunk_Qplus_size = 65535
        num_nk_chunks_Qplus = ceil(len(kcube_Qplus) / nk_chunk_Qplus_size)
    nkcube_left = len(kcube_Qplus)
    if len(kcube_Qplus) > 0:
        for i in range(num_nk_chunks_Qplus):
            nk_chunk = min(nkcube_left, nk_chunk_Qplus_size)
            nkcube_left -= nk_chunk
            k_sls = kcube_Qplus[i * nk_chunk_Qplus_size: i * nk_chunk_Qplus_size + nk_chunk]
            exx += 2. * exx_kpt_kernel(rchola_chunk, rcholbara_chunk, Ghalfa, k_sls, kpq_res_q)
    return - 0.5 * exx / nk

def local_energy_kpt_single_det_uhf_batch_chunked_gpu(
    system: Generic,
    hamiltonian: KptComplexCholChunked,
    walker_batch: UHFWalkers,
    trial: KptSingleDet,
    max_mem: float = 4.0
):
    """Compute local energy for walker batch (all walkers at once).

    Single determinant case for k point Cholesky, GPU, chunked integrals.

    Parameters
    ----------
    system : system object
        System being studied.
    hamiltonian : hamiltonian object
        Hamiltonian being studied.
    walker_batch : WalkerBatch
        Walkers object.
    trial : trial object
        Trial wavefunctioni.

    Returns
    -------
    local_energy : np.ndarray
        Total, one-body and two-body energies.
    """
    assert hamiltonian.chunked

    nwalkers = walker_batch.Ghalfa.shape[0]
    nk = hamiltonian.nk
    nalpha = trial.nalpha
    nbeta = trial.nbeta
    nbasis = hamiltonian.nbasis
    if walker_batch.rhf:
        ghalfa = walker_batch.Ghalfa.reshape(nwalkers, nk, nalpha, nk, nbasis)

        diagGhalfa = xp.zeros((nwalkers, nk, nalpha, nbasis), dtype=numpy.complex128)
        for ik in range(nk):
            diagGhalfa[:, ik, :, :] = ghalfa[:, ik, :, ik, :]
        diagGhalfa = diagGhalfa.reshape(nwalkers, nk * nalpha * nbasis)
        e1b = 2.0 * diagGhalfa.dot(trial._rH1a.ravel())
        e1b /= nk
        e1b += hamiltonian.ecore

        ghalfa_send = ghalfa.transpose(1, 3, 0, 2, 4).copy()
        ghalfa_recv = xp.zeros_like(ghalfa_send)

        handler = walker_batch.mpi_handler
        senders = handler.senders
        receivers = handler.receivers

        rchola_chunk = trial._rchola_chunk
        rcholbara_chunk = trial._rcholbara_chunk

        ecoul_send = kpt_symmchol_ecoul_kernel_batch_rhf_gpu(
            rchola_chunk, rcholbara_chunk, ghalfa_send, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus
        )

        exx_send = 2.0 * kpt_symmchol_exx_kernel_batch_gpu(rchola_chunk, rcholbara_chunk, ghalfa_send, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus, max_mem) 
        exx_recv = exx_send.copy()
        ecoul_recv = ecoul_send.copy()

        for _ in range(handler.ssize - 1):
            for isend, sender in enumerate(senders):
                if handler.srank == isend:
                    handler.scomm.Send(ghalfa_send, dest=receivers[isend], tag=1)
                    handler.scomm.Send(ecoul_send, dest=receivers[isend], tag=2)
                    handler.scomm.Send(exx_send, dest=receivers[isend], tag=3)
                elif handler.srank == receivers[isend]:
                    sender = numpy.where(receivers == handler.srank)[0]
                    handler.scomm.Recv(ghalfa_recv, source=sender, tag=1)
                    handler.scomm.Recv(ecoul_recv, source=sender, tag=2)
                    handler.scomm.Recv(exx_recv, source=sender, tag=3)
            handler.scomm.barrier()

        # prepare sending
            ecoul_send = ecoul_recv.copy()
            ecoul_send += kpt_symmchol_ecoul_kernel_batch_rhf_gpu(
            rchola_chunk, rcholbara_chunk, ghalfa_recv, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus
        )
            exx_send = exx_recv.copy()
            exx_send += 2.0 * kpt_symmchol_exx_kernel_batch_gpu(rchola_chunk, rcholbara_chunk, ghalfa_recv, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus, max_mem)
            ghalfa_send = ghalfa_recv.copy()
            
        if len(senders) > 1:
            for isend, sender in enumerate(senders):
                if handler.srank == sender:  # sending 1 xshifted to 0 xshifted_buf
                    handler.scomm.Send(ecoul_send, dest=receivers[isend], tag=1)
                    handler.scomm.Send(exx_send, dest=receivers[isend], tag=2)
                elif handler.srank == receivers[isend]:
                    sender = numpy.where(receivers == handler.srank)[0]
                    handler.scomm.Recv(ecoul_recv, source=sender, tag=1)
                    handler.scomm.Recv(exx_recv, source=sender, tag=2)

        e2b = ecoul_recv + exx_recv

        energy = xp.zeros((nwalkers, 3), dtype=numpy.complex128)
        energy[:, 0] = e1b + e2b
        energy[:, 1] = e1b
        energy[:, 2] = e2b

    else:
        ghalfa = walker_batch.Ghalfa.reshape(nwalkers, nk, nalpha, nk, nbasis)
        ghalfb = walker_batch.Ghalfb.reshape(nwalkers, nk, nbeta, nk, nbasis)

        diagGhalfa = xp.zeros((nwalkers, nk, nalpha, nbasis), dtype=numpy.complex128)
        diagGhalfb = xp.zeros((nwalkers, nk, nbeta, nbasis), dtype=numpy.complex128)
        for ik in range(nk):
            diagGhalfa[:, ik, :, :] = ghalfa[:, ik, :, ik, :]
            diagGhalfb[:, ik, :, :] = ghalfb[:, ik, :, ik, :]
        diagGhalfa = diagGhalfa.reshape(nwalkers, nk * nalpha * nbasis)
        diagGhalfb = diagGhalfb.reshape(nwalkers, nk * nbeta * nbasis)
        e1b = diagGhalfa.dot(trial._rH1a.ravel())
        e1b += diagGhalfb.dot(trial._rH1b.ravel())
        e1b /= nk
        e1b += hamiltonian.ecore

        ghalfa_send = ghalfa.transpose(1, 3, 0, 2, 4).copy()
        ghalfb_send = ghalfb.transpose(1, 3, 0, 2, 4).copy()

        ghalfa_recv = xp.zeros_like(ghalfa_send)
        ghalfb_recv = xp.zeros_like(ghalfb_send)


        handler = walker_batch.mpi_handler
        senders = handler.senders
        receivers = handler.receivers

        rchola_chunk = trial._rchola_chunk
        rcholb_chunk = trial._rcholb_chunk
        rcholbara_chunk = trial._rcholbara_chunk
        rcholbarb_chunk = trial._rcholbarb_chunk

        ecoul_send = kpt_symmchol_ecoul_kernel_batch_uhf_gpu(
            rchola_chunk, rcholb_chunk, rcholbara_chunk, rcholbarb_chunk, ghalfa_send, ghalfb_send, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus
        )

        exx_send = kpt_symmchol_exx_kernel_batch_gpu(rchola_chunk, rcholbara_chunk, ghalfa_send, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus, max_mem) 
        exx_send += kpt_symmchol_exx_kernel_batch_gpu(rcholb_chunk, rcholbarb_chunk, ghalfb_send, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus, max_mem) 
        exx_recv = exx_send.copy()
        ecoul_recv = ecoul_send.copy()

        for _ in range(handler.ssize - 1):
            for isend, sender in enumerate(senders):
                if handler.srank == isend:
                    handler.scomm.Send(ghalfa_send, dest=receivers[isend], tag=1)
                    handler.scomm.Send(ghalfb_send, dest=receivers[isend], tag=2)
                    # handler.scomm.Send(ghalfaTx_send, dest=receivers[isend], tag=3)
                    # handler.scomm.Send(ghalfbTx_send, dest=receivers[isend], tag=4)
                    handler.scomm.Send(ecoul_send, dest=receivers[isend], tag=3)
                    handler.scomm.Send(exx_send, dest=receivers[isend], tag=4)
                elif handler.srank == receivers[isend]:
                    sender = numpy.where(receivers == handler.srank)[0]
                    handler.scomm.Recv(ghalfa_recv, source=sender, tag=1)
                    handler.scomm.Recv(ghalfb_recv, source=sender, tag=2)
                    # handler.scomm.Recv(ghalfaTx_recv, source=sender, tag=3)
                    # handler.scomm.Recv(ghalfbTx_recv, source=sender, tag=4)
                    handler.scomm.Recv(ecoul_recv, source=sender, tag=3)
                    handler.scomm.Recv(exx_recv, source=sender, tag=4)
            handler.scomm.barrier()

        # prepare sending
            ecoul_send = ecoul_recv.copy()
            ecoul_send += kpt_symmchol_ecoul_kernel_batch_uhf_gpu(
            rchola_chunk, rcholb_chunk, rcholbara_chunk, rcholbarb_chunk, ghalfa_recv, ghalfb_recv, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus
        )
            exx_send = exx_recv.copy()
            exx_send += kpt_symmchol_exx_kernel_batch_gpu(rchola_chunk, rcholbara_chunk, ghalfa_recv, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus, max_mem)
            exx_send += kpt_symmchol_exx_kernel_batch_gpu(rcholb_chunk, rcholbarb_chunk, ghalfb_recv, hamiltonian.ikpq_mat, hamiltonian.Sset, hamiltonian.Qplus, max_mem)
            ghalfa_send = ghalfa_recv.copy()
            ghalfb_send = ghalfb_recv.copy()
            # ghalfaTx_send = ghalfaTx_recv.copy()
            # ghalfbTx_send = ghalfbTx_recv.copy()

        if len(senders) > 1:
            for isend, sender in enumerate(senders):
                if handler.srank == sender:  # sending 1 xshifted to 0 xshifted_buf
                    handler.scomm.Send(ecoul_send, dest=receivers[isend], tag=1)
                    handler.scomm.Send(exx_send, dest=receivers[isend], tag=2)
                elif handler.srank == receivers[isend]:
                    sender = numpy.where(receivers == handler.srank)[0]
                    handler.scomm.Recv(ecoul_recv, source=sender, tag=1)
                    handler.scomm.Recv(exx_recv, source=sender, tag=2)

        e2b = ecoul_recv + exx_recv

        energy = xp.zeros((nwalkers, 3), dtype=numpy.complex128)
        energy[:, 0] = e1b + e2b
        energy[:, 1] = e1b
        energy[:, 2] = e2b

    xp._default_memory_pool.free_all_blocks()
    return energy