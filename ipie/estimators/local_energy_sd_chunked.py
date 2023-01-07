
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
#          Joonho Lee <linusjoonho@gmail.com>
#

import time

import numpy
from math import ceil

from ipie.estimators.local_energy_sd import (ecoul_kernel_batch_real_rchol_uhf,
                                             exx_kernel_batch_real_rchol)
from ipie.estimators.kernels import exchange_reduction

from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import to_host, synchronize

# Local energy routies for chunked (distributed) integrals. Distributed here
# means over MPI processes with information typically residing on different
# nodes. Green's funtions are sent round-robin and local energy contributions
# are accumulated.

def local_energy_single_det_uhf_batch_chunked(system, hamiltonian, walker_batch, trial):
    """Compute local energy for walker batch (all walkers at once).

    Single determinant UHF case.

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
    nalpha = walker_batch.Ghalfa.shape[1]
    nbeta = walker_batch.Ghalfb.shape[1]
    nbasis = hamiltonian.nbasis
    nchol = hamiltonian.nchol

    Ghalfa = walker_batch.Ghalfa.reshape(nwalkers, nalpha * nbasis)
    Ghalfb = walker_batch.Ghalfb.reshape(nwalkers, nbeta * nbasis)

    e1b = Ghalfa.dot(trial._rH1a.ravel())
    e1b += Ghalfb.dot(trial._rH1b.ravel())
    e1b += hamiltonian.ecore

    Ghalfa_send = Ghalfa.copy()
    Ghalfb_send = Ghalfb.copy()

    Ghalfa_recv = xp.zeros_like(Ghalfa)
    Ghalfb_recv = xp.zeros_like(Ghalfb)

    handler = walker_batch.mpi_handler
    senders = handler.senders
    receivers = handler.receivers

    rchola_chunk = trial._rchola_chunk
    rcholb_chunk = trial._rcholb_chunk

    Ghalfa = Ghalfa.reshape(nwalkers, nalpha * nbasis)
    Ghalfb = Ghalfb.reshape(nwalkers, nbeta * nbasis)
    ecoul_send = ecoul_kernel_batch_real_rchol_uhf(
        rchola_chunk, rcholb_chunk, Ghalfa, Ghalfb
    )
    Ghalfa = Ghalfa.reshape(nwalkers, nalpha, nbasis)
    Ghalfb = Ghalfb.reshape(nwalkers, nbeta, nbasis)
    exx_send = exx_kernel_batch_real_rchol(rchola_chunk, Ghalfa)
    exx_send += exx_kernel_batch_real_rchol(rcholb_chunk, Ghalfb)

    exx_recv = exx_send.copy()
    ecoul_recv = ecoul_send.copy()

    for icycle in range(handler.ssize - 1):
        for isend, sender in enumerate(senders):
            if handler.srank == isend:
                handler.scomm.Send(Ghalfa_send, dest=receivers[isend], tag=1)
                handler.scomm.Send(Ghalfb_send, dest=receivers[isend], tag=2)
                handler.scomm.Send(ecoul_send, dest=receivers[isend], tag=3)
                handler.scomm.Send(exx_send, dest=receivers[isend], tag=4)
            elif handler.srank == receivers[isend]:
                sender = numpy.where(receivers == handler.srank)[0]
                handler.scomm.Recv(Ghalfa_recv, source=sender, tag=1)
                handler.scomm.Recv(Ghalfb_recv, source=sender, tag=2)
                handler.scomm.Recv(ecoul_recv, source=sender, tag=3)
                handler.scomm.Recv(exx_recv, source=sender, tag=4)
        handler.scomm.barrier()

        # prepare sending
        ecoul_send = ecoul_recv.copy()
        Ghalfa_recv = Ghalfa_recv.reshape(nwalkers, nalpha * nbasis)
        Ghalfb_recv = Ghalfb_recv.reshape(nwalkers, nbeta * nbasis)
        ecoul_send += ecoul_kernel_batch_real_rchol_uhf(
            rchola_chunk, rcholb_chunk, Ghalfa_recv, Ghalfb_recv
        )
        Ghalfa_recv = Ghalfa_recv.reshape(nwalkers, nalpha, nbasis)
        Ghalfb_recv = Ghalfb_recv.reshape(nwalkers, nbeta, nbasis)
        exx_send = exx_recv.copy()
        exx_send += exx_kernel_batch_real_rchol(rchola_chunk, Ghalfa_recv)
        exx_send += exx_kernel_batch_real_rchol(rcholb_chunk, Ghalfb_recv)
        Ghalfa_send = Ghalfa_recv.copy()
        Ghalfb_send = Ghalfb_recv.copy()

    if len(senders) > 1:
        for isend, sender in enumerate(senders):
            if handler.srank == sender:  # sending 1 xshifted to 0 xshifted_buf
                handler.scomm.Send(ecoul_send, dest=receivers[isend], tag=1)
                handler.scomm.Send(exx_send, dest=receivers[isend], tag=2)
            elif handler.srank == receivers[isend]:
                sender = numpy.where(receivers == handler.srank)[0]
                handler.scomm.Recv(ecoul_recv, source=sender, tag=1)
                handler.scomm.Recv(exx_recv, source=sender, tag=2)

    e2b = ecoul_recv - exx_recv

    energy = xp.zeros((nwalkers, 3), dtype=numpy.complex128)
    energy[:, 0] = e1b + e2b
    energy[:, 1] = e1b
    energy[:, 2] = e2b

    return energy


def ecoul_kernel_batch_rchol_uhf_gpu(rchola_chunk, rcholb_chunk, Ghalfa, Ghalfb):
    """Compute coulomb contribution for rchol with UHF trial.

    Parameters
    ----------
    rchola_chunk : :class:`numpy.ndarray`
        Half-rotated cholesky (alpha).
    rcholb_chunk : :class:`numpy.ndarray`
        Half-rotated cholesky (beta).
    Ghalfa : :class:`numpy.ndarray`
        Walker's half-rotated "green's function" shape is nalpha  x nbasis.
    Ghalfb : :class:`numpy.ndarray`
        Walker's half-rotated "green's function" shape is nbeta x nbasis.

    Returns
    -------
    ecoul : :class:`numpy.ndarray`
        coulomb contribution for all walkers.
    """
    if xp.isrealobj(rchola_chunk):
        Xa = rchola_chunk.dot(Ghalfa.real.T) + 1.0j * rchola_chunk.dot(
            Ghalfa.imag.T
        )  # naux x nwalkers
        Xb = rcholb_chunk.dot(Ghalfb.real.T) + 1.0j * rcholb_chunk.dot(
            Ghalfb.imag.T
        )  # naux x nwalkers
    else:
        Xa = rchola_chunk.dot(Ghalfa.T)
        Xb = rcholb_chunk.dot(Ghalfb.T)

    ecoul = xp.einsum("xw,xw->w", Xa, Xa, optimize=True)
    ecoul += xp.einsum("xw,xw->w", Xb, Xb, optimize=True)
    ecoul += 2.0 * xp.einsum("xw,xw->w", Xa, Xb, optimize=True)

    ecoul *= 0.5

    return ecoul


def exx_kernel_batch_rchol_gpu(rchola_chunk, Ghalfa):
    """Compute exchange contribution for complex rchol.

    Parameters
    ----------
    rchol_chunk : :class:`numpy.ndarray`
        Chunk of Half-rotated cholesky.
    Ghalf : :class:`numpy.ndarray`
        Walker's half-rotated "green's function" shape is nalpha  x nbasis

    Returns
    -------
    exx : :class:`numpy.ndarray`
        exchange contribution for all walkers.
    """
    Txij = xp.einsum("xim,wjm->wxji", rchola_chunk, Ghalfa)
    rchola_chunk = rchola_chunk.reshape(nchol, nalpha * nbasis)

    exx = xp.einsum("wxji,wxij->w", Txij, Txij)
    exx *= 0.5
    return exx

def exx_kernel_batch_rchol_gpu_low_mem(rchola_chunk, Ghalfa, buff):
    nwalkers = Ghalfa.shape[0]
    nalpha = Ghalfa.shape[1]
    nbasis = Ghalfa.shape[2]
    nchol = rchola_chunk.shape[0]
    rchola_chunk = rchola_chunk.reshape(nchol, nalpha, nbasis)
    exx = xp.zeros(nwalkers, dtype=numpy.complex128)
    _Ghalfa = Ghalfa.reshape((nwalkers * nalpha, nbasis))
    nchol_chunk_size = buff.shape[0]
    nchol_chunks = ceil(nchol / nchol_chunk_size)
    nchol_left = nchol
    _buff = buff.ravel()
    for i in range(nchol_chunks):
        nchol_chunk = min(nchol_chunk_size, nchol_left)
        chol_sls = slice(i * nchol_chunk_size, i * nchol_chunk_size + nchol_chunk)
        size = nwalkers * nchol_chunk * nalpha * nalpha
        # alpha-alpha
        Txij = _buff[:size].reshape((nchol_chunk * nalpha, nwalkers * nalpha))
        rchol = rchola_chunk[chol_sls].reshape((nchol_chunk * nalpha, nbasis))
        xp.dot(rchol, _Ghalfa.T, out=Txij)
        Txij = Txij.reshape((nchol_chunk, nalpha, nwalkers, nalpha))
        exchange_reduction(Txij, exx)
    return 0.5 * exx


def local_energy_single_det_uhf_batch_chunked_gpu(
    system, hamiltonian, walker_batch, trial, max_mem=2.0
):
    """Compute local energy for walker batch (all walkers at once).

    Single determinant case, GPU, chunked integrals.

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
    nalpha = walker_batch.Ghalfa.shape[1]
    nbeta = walker_batch.Ghalfb.shape[1]
    nbasis = hamiltonian.nbasis
    nchol = hamiltonian.nchol

    Ghalfa = walker_batch.Ghalfa.reshape(nwalkers, nalpha * nbasis)
    Ghalfb = walker_batch.Ghalfb.reshape(nwalkers, nbeta * nbasis)

    e1b = Ghalfa.dot(trial._rH1a.ravel())
    e1b += Ghalfb.dot(trial._rH1b.ravel())
    e1b += hamiltonian.ecore

    Ghalfa_send = Ghalfa.copy()
    Ghalfb_send = Ghalfb.copy()

    Ghalfa_recv = xp.zeros_like(Ghalfa)
    Ghalfb_recv = xp.zeros_like(Ghalfb)

    handler = walker_batch.mpi_handler
    senders = handler.senders
    receivers = handler.receivers

    rchola_chunk = trial._rchola_chunk
    rcholb_chunk = trial._rcholb_chunk

    # buffer for low on GPU memory usage
    max_nchol = max(trial._rchola_chunk.shape[0], trial._rcholb_chunk.shape[0])
    max_nocc = max(nalpha, nbeta)
    mem_needed = 16 * nwalkers * max_nocc * max_nocc * max_nchol / (1024.0**3.0)
    num_chunks = max(1, ceil(mem_needed / max_mem))
    chunk_size = ceil(max_nchol / num_chunks)
    nchol_chunks = ceil(max_nchol / chunk_size)
    buff = xp.zeros(shape=(chunk_size, nwalkers * max_nocc * max_nocc),
            dtype=numpy.complex128)

    Ghalfa = Ghalfa.reshape(nwalkers, nalpha * nbasis)
    Ghalfb = Ghalfb.reshape(nwalkers, nbeta * nbasis)
    ecoul_send = ecoul_kernel_batch_rchol_uhf_gpu(
        rchola_chunk, rcholb_chunk, Ghalfa, Ghalfb
    )
    Ghalfa = Ghalfa.reshape(nwalkers, nalpha, nbasis)
    Ghalfb = Ghalfb.reshape(nwalkers, nbeta, nbasis)
    exx_send = exx_kernel_batch_rchol_gpu_low_mem(rchola_chunk, Ghalfa, buff)
    exx_send += exx_kernel_batch_rchol_gpu_low_mem(rcholb_chunk, Ghalfb, buff)

    exx_recv = exx_send.copy()
    ecoul_recv = ecoul_send.copy()
 
    srank = handler.srank

    sender = numpy.where(receivers == handler.srank)[0]
    scomm = handler.scomm
    for icycle in range(handler.ssize - 1):
        synchronize()
        scomm.Isend(Ghalfa_send, dest=receivers[srank], tag=1)
        scomm.Isend(Ghalfb_send, dest=receivers[srank], tag=2)
        scomm.Isend(ecoul_send, dest=receivers[srank], tag=3)
        scomm.Isend(exx_send, dest=receivers[srank], tag=4)
        req1 = scomm.Irecv(Ghalfa_recv, source=sender, tag=1)
        req2 = scomm.Irecv(Ghalfb_recv, source=sender, tag=2)
        req3 = scomm.Irecv(ecoul_recv, source=sender, tag=3)
        req4 = scomm.Irecv(exx_recv, source=sender, tag=4)
        req1.wait()
        req2.wait()
        req3.wait()
        req4.wait()
        scomm.barrier()

        # prepare sending
        ecoul_send = ecoul_recv.copy()
        Ghalfa_recv = Ghalfa_recv.reshape(nwalkers, nalpha * nbasis)
        Ghalfb_recv = Ghalfb_recv.reshape(nwalkers, nbeta * nbasis)
        ecoul_send += ecoul_kernel_batch_rchol_uhf_gpu(
            rchola_chunk, rcholb_chunk, Ghalfa_recv, Ghalfb_recv
        )
        Ghalfa_recv = Ghalfa_recv.reshape(nwalkers, nalpha, nbasis)
        Ghalfb_recv = Ghalfb_recv.reshape(nwalkers, nbeta, nbasis)
        exx_send = exx_recv.copy()
        exx_send += exx_kernel_batch_rchol_gpu_low_mem(rchola_chunk,
                Ghalfa_recv, buff)
        exx_send += exx_kernel_batch_rchol_gpu_low_mem(rcholb_chunk,
                Ghalfb_recv, buff)
        Ghalfa_send = Ghalfa_recv.copy()
        Ghalfb_send = Ghalfb_recv.copy()

    synchronize()
    scomm.Isend(ecoul_send, dest=receivers[srank], tag=1)
    scomm.Isend(exx_send, dest=receivers[srank], tag=2)
    req1 = scomm.Irecv(ecoul_recv, source=sender, tag=1)
    req2 = scomm.Irecv(exx_recv, source=sender, tag=2)
    req1.wait()
    req2.wait()
    handler.scomm.barrier()

    e2b = ecoul_recv - exx_recv

    energy = xp.zeros((nwalkers, 3), dtype=numpy.complex128)
    energy[:, 0] = e1b + e2b
    energy[:, 1] = e1b
    energy[:, 2] = e2b

    synchronize()

    return energy
