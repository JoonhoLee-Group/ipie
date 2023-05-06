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
from ipie.utils.backend import synchronize


def construct_force_bias_batch(hamiltonian, walkers, trial, mpi_handler=None):
    """Compute optimal force bias.

    Uses rotated Green's function.

    Parameters
    ----------
    hamiltonian : class
        hamiltonian object.

    walkers : class
        walkers object.

    trial : class
        Trial wavefunction object.

    Returns
    -------
    xbar : :class:`numpy.ndarray`
        Force bias.
    """

    if walkers.name == "SingleDetWalkerBatch" and trial.name == "MultiSlater":
        if hamiltonian.chunked:
            return construct_force_bias_batch_single_det_chunked(
                hamiltonian, walkers, trial, mpi_handler
            )
        else:
            return construct_force_bias_batch_single_det(hamiltonian, walkers, trial)
    elif walkers.name == "MultiDetTrialWalkerBatch" and trial.name == "MultiSlater":
        return construct_force_bias_batch_multi_det_trial(hamiltonian, walkers, trial)


def construct_force_bias_batch_multi_det_trial(hamiltonian, walkers, trial):
    Ga = walkers.Ga.reshape(walkers.nwalkers, hamiltonian.nbasis**2)
    Gb = walkers.Gb.reshape(walkers.nwalkers, hamiltonian.nbasis**2)
    # Cholesky vectors. [M^2, nchol]
    # Why are there so many transposes here?
    if numpy.isrealobj(hamiltonian.chol):
        vbias_batch = numpy.empty((hamiltonian.nchol, walkers.nwalkers), dtype=numpy.complex128)
        vbias_batch.real = hamiltonian.chol.T.dot(Ga.T.real + Gb.T.real)
        vbias_batch.imag = hamiltonian.chol.T.dot(Ga.T.imag + Gb.T.imag)
        vbias_batch = vbias_batch.T.copy()
        return vbias_batch
    else:
        vbias_batch_tmp = hamiltonian.chol.T.dot(Ga.T + Gb.T)
        vbias_batch_tmp = vbias_batch_tmp.T.copy()
        return vbias_batch_tmp


# only implement real Hamiltonian
def construct_force_bias_batch_single_det(
    hamiltonian: "GenericRealChol", walkers: "UHFWalkers", trial: "SingleDetTrial"
):
    """Compute optimal force bias.

    Uses rotated Green's function.

    Parameters
    ----------
    hamiltonian : class
        hamiltonian object.

    walkers : class
        walkers object.

    trial : class
        Trial wavefunction object.

    Returns
    -------
    xbar : :class:`numpy.ndarray`
        Force bias.
    """
    if walkers.rhf:
        Ghalfa = walkers.Ghalfa.reshape(walkers.nwalkers, walkers.nup * hamiltonian.nbasis)
        vbias_batch_real = 2.0 * trial._rchola.dot(Ghalfa.T.real)
        vbias_batch_imag = 2.0 * trial._rchola.dot(Ghalfa.T.imag)
        vbias_batch = xp.empty((walkers.nwalkers, hamiltonian.nchol), dtype=Ghalfa.dtype)
        vbias_batch.real = vbias_batch_real.T.copy()
        vbias_batch.imag = vbias_batch_imag.T.copy()
        synchronize()

        return vbias_batch

    else:
        Ghalfa = walkers.Ghalfa.reshape(walkers.nwalkers, walkers.nup * hamiltonian.nbasis)
        Ghalfb = walkers.Ghalfb.reshape(walkers.nwalkers, walkers.ndown * hamiltonian.nbasis)
        vbias_batch_real = trial._rchola.dot(Ghalfa.T.real) + trial._rcholb.dot(Ghalfb.T.real)
        vbias_batch_imag = trial._rchola.dot(Ghalfa.T.imag) + trial._rcholb.dot(Ghalfb.T.imag)
        vbias_batch = xp.empty((walkers.nwalkers, hamiltonian.nchol), dtype=Ghalfa.dtype)
        vbias_batch.real = vbias_batch_real.T.copy()
        vbias_batch.imag = vbias_batch_imag.T.copy()
        synchronize()
        return vbias_batch


def construct_force_bias_batch_single_det_chunked(hamiltonian, walkers, trial, handler):
    """Compute optimal force bias.

    Uses rotated Green's function.

    Parameters
    ----------
    hamiltonian : class
        hamiltonian object.

    walkers : class
        walkers object.

    trial : class
        Trial wavefunction object.

    Returns
    -------
    xbar : :class:`numpy.ndarray`
        Force bias.
    """
    assert hamiltonian.chunked
    assert xp.isrealobj(trial._rchola)

    Ghalfa = walkers.Ghalfa.reshape(walkers.nwalkers, walkers.nup * hamiltonian.nbasis)
    Ghalfb = walkers.Ghalfb.reshape(walkers.nwalkers, walkers.ndown * hamiltonian.nbasis)

    chol_idxs_chunk = hamiltonian.chol_idxs_chunk

    Ghalfa_recv = xp.zeros_like(Ghalfa)
    Ghalfb_recv = xp.zeros_like(Ghalfb)

    Ghalfa_send = Ghalfa.copy()
    Ghalfb_send = Ghalfb.copy()

    srank = handler.scomm.rank

    vbias_batch_real_recv = xp.zeros((hamiltonian.nchol, walkers.nwalkers))
    vbias_batch_imag_recv = xp.zeros((hamiltonian.nchol, walkers.nwalkers))

    vbias_batch_real_send = xp.zeros((hamiltonian.nchol, walkers.nwalkers))
    vbias_batch_imag_send = xp.zeros((hamiltonian.nchol, walkers.nwalkers))

    vbias_batch_real_send[chol_idxs_chunk, :] = trial._rchola_chunk.dot(
        Ghalfa.T.real
    ) + trial._rcholb_chunk.dot(Ghalfb.T.real)
    vbias_batch_imag_send[chol_idxs_chunk, :] = trial._rchola_chunk.dot(
        Ghalfa.T.imag
    ) + trial._rcholb_chunk.dot(Ghalfb.T.imag)

    receivers = handler.receivers
    for _ in range(handler.ssize - 1):
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

    vbias_batch = xp.empty((walkers.nwalkers, hamiltonian.nchol), dtype=Ghalfa.dtype)
    vbias_batch.real = vbias_batch_real_recv.T.copy()
    vbias_batch.imag = vbias_batch_imag_recv.T.copy()
    synchronize()
    return vbias_batch
