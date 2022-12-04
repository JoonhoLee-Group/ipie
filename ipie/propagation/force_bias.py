
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
# Authors: Joonho Lee <linusjoonho@gmail.com>
#          Fionn Malone <fmalone@google.com>
#

import numpy

from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import synchronize, to_host


def construct_force_bias_batch(hamiltonian, walker_batch, trial, mpi_handler=None):
    """Compute optimal force bias.

    Uses rotated Green's function.

    Parameters
    ----------
    hamiltonian : class
        hamiltonian object.

    walker_batch : class
        walker_batch object.

    trial : class
        Trial wavefunction object.

    Returns
    -------
    xbar : :class:`numpy.ndarray`
        Force bias.
    """

    if walker_batch.name == "SingleDetWalkerBatch" and trial.name == "MultiSlater":
        if hamiltonian.chunked:
            return construct_force_bias_batch_single_det_chunked(
                hamiltonian, walker_batch, trial, mpi_handler
            )
        else:
            return construct_force_bias_batch_single_det(
                hamiltonian, walker_batch, trial
            )
    elif (
        walker_batch.name == "MultiDetTrialWalkerBatch" and trial.name == "MultiSlater"
    ):
        return construct_force_bias_batch_multi_det_trial(
            hamiltonian, walker_batch, trial
        )


def construct_force_bias_batch_multi_det_trial(hamiltonian, walker_batch, trial):
    Ga = walker_batch.Ga.reshape(walker_batch.nwalkers, hamiltonian.nbasis**2)
    Gb = walker_batch.Gb.reshape(walker_batch.nwalkers, hamiltonian.nbasis**2)
    # Cholesky vectors. [M^2, nchol]
    # Why are there so many transposes here?
    if numpy.isrealobj(hamiltonian.chol_vecs):
        vbias_batch = numpy.empty(
            (hamiltonian.nchol, walker_batch.nwalkers), dtype=numpy.complex128
        )
        vbias_batch.real = hamiltonian.chol_vecs.T.dot(Ga.T.real + Gb.T.real)
        vbias_batch.imag = hamiltonian.chol_vecs.T.dot(Ga.T.imag + Gb.T.imag)
        vbias_batch = vbias_batch.T.copy()
        return vbias_batch
    else:
        vbias_batch_tmp = hamiltonian.chol_vecs.T.dot(Ga.T + Gb.T)
        vbias_batch_tmp = vbias_batch_tmp.T.copy()
        return vbias_batch_tmp


def construct_force_bias_batch_single_det(hamiltonian, walker_batch, trial):
    """Compute optimal force bias.

    Uses rotated Green's function.

    Parameters
    ----------
    hamiltonian : class
        hamiltonian object.

    walker_batch : class
        walker_batch object.

    trial : class
        Trial wavefunction object.

    Returns
    -------
    xbar : :class:`numpy.ndarray`
        Force bias.
    """
    if walker_batch.rhf:
        Ghalfa = walker_batch.Ghalfa.reshape(
            walker_batch.nwalkers, walker_batch.nup * hamiltonian.nbasis
        )
        if xp.isrealobj(trial._rchola) and xp.isrealobj(trial._rcholb):
            vbias_batch_real = 2.0 * trial._rchola.dot(Ghalfa.T.real)
            vbias_batch_imag = 2.0 * trial._rchola.dot(Ghalfa.T.imag)
            vbias_batch = xp.empty(
                (walker_batch.nwalkers, hamiltonian.nchol), dtype=Ghalfa.dtype
            )
            vbias_batch.real = vbias_batch_real.T.copy()
            vbias_batch.imag = vbias_batch_imag.T.copy()
            synchronize()
            return vbias_batch
        else:
            vbias_batch_tmp = 2.0 * trial._rchola.dot(Ghalfa.T)
            return vbias_batch
            return vbias_batch_tmp.T

    else:
        if trial.mixed_precision:
            dGhalfa = (
                walker_batch.Ghalfa.reshape(
                    walker_batch.nwalkers, walker_batch.nup * hamiltonian.nbasis
                )
                - trial.psia.T.ravel()
            )
            dGhalfb = (
                walker_batch.Ghalfb.reshape(
                    walker_batch.nwalkers, walker_batch.ndown * hamiltonian.nbasis
                )
                - trial.psib.T.ravel()
            )
            dGhalfa = dGhalfa.astype(numpy.complex64)
            dGhalfb = dGhalfb.astype(numpy.complex64)
            if xp.isrealobj(trial._rchola) and xp.isrealobj(trial._rcholb):
                # single precision
                vbias_batch_real = trial._rchola.dot(
                    dGhalfa.T.real
                ) + trial._rcholb.dot(dGhalfb.T.real)
                vbias_batch_imag = trial._rchola.dot(
                    dGhalfa.T.imag
                ) + trial._rcholb.dot(dGhalfb.T.imag)
                # double precision
                vbias_batch = xp.empty(
                    (walker_batch.nwalkers, hamiltonian.nchol),
                    dtype=walker_batch.Ghalfa.dtype,
                )
                vbias_batch.real = vbias_batch_real.T.copy() + trial._vbias0.real
                vbias_batch.imag = vbias_batch_imag.T.copy() + trial._vbias0.imag
                synchronize()
                return vbias_batch
            else:
                vbias_batch_tmp = trial._rchola.dot(dGhalfa.T) + trial._rcholb.dot(
                    dGhalfb.T
                )
                vbias_batch_tmp += trial._vbias0
                synchronize()
                return vbias_batch_tmp.T
        else:
            Ghalfa = walker_batch.Ghalfa.reshape(
                walker_batch.nwalkers, walker_batch.nup * hamiltonian.nbasis
            )
            Ghalfb = walker_batch.Ghalfb.reshape(
                walker_batch.nwalkers, walker_batch.ndown * hamiltonian.nbasis
            )
            if xp.isrealobj(trial._rchola) and xp.isrealobj(trial._rcholb):
                vbias_batch_real = trial._rchola.dot(Ghalfa.T.real) + trial._rcholb.dot(
                    Ghalfb.T.real
                )
                vbias_batch_imag = trial._rchola.dot(Ghalfa.T.imag) + trial._rcholb.dot(
                    Ghalfb.T.imag
                )
                vbias_batch = xp.empty(
                    (walker_batch.nwalkers, hamiltonian.nchol), dtype=Ghalfa.dtype
                )
                vbias_batch.real = vbias_batch_real.T.copy()
                vbias_batch.imag = vbias_batch_imag.T.copy()
                synchronize()
                return vbias_batch
            else:
                vbias_batch_tmp = trial._rchola.dot(Ghalfa.T) + trial._rcholb.dot(
                    Ghalfb.T
                )
                synchronize()
                return vbias_batch_tmp.T


def construct_force_bias_batch_single_det_chunked(
    hamiltonian, walker_batch, trial, handler
):
    """Compute optimal force bias.

    Uses rotated Green's function.

    Parameters
    ----------
    hamiltonian : class
        hamiltonian object.

    walker_batch : class
        walker_batch object.

    trial : class
        Trial wavefunction object.

    Returns
    -------
    xbar : :class:`numpy.ndarray`
        Force bias.
    """
    assert hamiltonian.chunked
    assert xp.isrealobj(trial._rchola)

    Ghalfa = walker_batch.Ghalfa.reshape(
        walker_batch.nwalkers, walker_batch.nup * hamiltonian.nbasis
    )
    Ghalfb = walker_batch.Ghalfb.reshape(
        walker_batch.nwalkers, walker_batch.ndown * hamiltonian.nbasis
    )

    chol_idxs_chunk = hamiltonian.chol_idxs_chunk

    Ghalfa_recv = xp.zeros_like(Ghalfa)
    Ghalfb_recv = xp.zeros_like(Ghalfb)

    Ghalfa_send = Ghalfa.copy()
    Ghalfb_send = Ghalfb.copy()

    ssize = handler.scomm.size
    srank = handler.scomm.rank

    vbias_batch_real_recv = xp.zeros((hamiltonian.nchol, walker_batch.nwalkers))
    vbias_batch_imag_recv = xp.zeros((hamiltonian.nchol, walker_batch.nwalkers))

    vbias_batch_real_send = xp.zeros((hamiltonian.nchol, walker_batch.nwalkers))
    vbias_batch_imag_send = xp.zeros((hamiltonian.nchol, walker_batch.nwalkers))

    vbias_batch_real_send[chol_idxs_chunk, :] = trial._rchola_chunk.dot(
        Ghalfa.T.real
    ) + trial._rcholb_chunk.dot(Ghalfb.T.real)
    vbias_batch_imag_send[chol_idxs_chunk, :] = trial._rchola_chunk.dot(
        Ghalfa.T.imag
    ) + trial._rcholb_chunk.dot(Ghalfb.T.imag)


    tcomm = 0.0
    senders = handler.senders
    receivers = handler.receivers
    for icycle in range(handler.ssize - 1):
        synchronize()

        handler.scomm.Isend(Ghalfa_send, dest=receivers[srank], tag=1)
        handler.scomm.Isend(Ghalfb_send, dest=receivers[srank], tag=2)
        handler.scomm.Isend(vbias_batch_real_send, dest=receivers[srank], tag=3)
        handler.scomm.Isend(vbias_batch_imag_send, dest=receivers[srank], tag=4)

        sender = numpy.where(receivers == srank)[0]
        req1 = handler.scomm.Irecv(Ghalfa_recv, source=sender, tag=1)
        req2 = handler.scomm.Irecv(Ghalfb_recv, source=sender, tag=2)
        req3 = handler.scomm.Irecv(vbias_batch_real_recv, source=sender, tag=3)
        req4 = handler.scomm.Irecv(vbias_batch_imag_recv, source=sender, tag=4)
        req1.wait()
        req2.wait()
        req3.wait()
        req4.wait()

        handler.scomm.barrier()

        # prepare sending
        vbias_batch_real_send = vbias_batch_real_recv.copy()
        vbias_batch_imag_send = vbias_batch_imag_recv.copy()
        vbias_batch_real_send[chol_idxs_chunk, :] = trial._rchola_chunk.dot(
            Ghalfa_recv.T.real
        ) + trial._rcholb_chunk.dot(Ghalfb_recv.T.real)
        vbias_batch_imag_send[chol_idxs_chunk, :] = trial._rchola_chunk.dot(
            Ghalfa_recv.T.imag
        ) + trial._rcholb_chunk.dot(Ghalfb_recv.T.imag)
        Ghalfa_send = Ghalfa_recv.copy()
        Ghalfb_send = Ghalfb_recv.copy()

    synchronize()
    handler.scomm.Isend(vbias_batch_real_send, dest=receivers[srank], tag=1)
    handler.scomm.Isend(vbias_batch_imag_send, dest=receivers[srank], tag=2)

    sender = numpy.where(receivers == srank)[0]
    req1 = handler.scomm.Irecv(vbias_batch_real_recv, source=sender, tag=1)
    req2 = handler.scomm.Irecv(vbias_batch_imag_recv, source=sender, tag=2)
    req1.wait()
    req2.wait()
    handler.scomm.barrier()

    vbias_batch = xp.empty((walker_batch.nwalkers, hamiltonian.nchol), dtype=Ghalfa.dtype)
    vbias_batch.real = vbias_batch_real_recv.T.copy()
    vbias_batch.imag = vbias_batch_imag_recv.T.copy()
    synchronize()
    return vbias_batch
