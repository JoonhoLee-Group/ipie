
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
# Authors: Fionn Malone <fionn.malone@gmail.com>
#          Joonho Lee <linusjoonho@gmail.com>
#

import time
from math import ceil

import numpy
from numba import jit

from ipie.estimators.local_energy import local_energy_G
from ipie.estimators.kernels import exchange_reduction
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import synchronize

# Note specialisations occur to because:
# 1. Numba does not allow for mixing types without a warning so need to split
# real and complex components apart when rchol is real. Green's function is
# complex in general.
# Optimize for case when wavefunction is RHF (factor of 2 saving)


@jit(nopython=True, fastmath=True)
def exx_kernel_batch_real_rchol(rchola, Ghalfa_batch):
    """Compute exchange contribution for real rchol.

    Parameters
    ----------
    rchol : :class:`numpy.ndarray`
        Half-rotated cholesky.
    Ghalf : :class:`numpy.ndarray`
        Walker's half-rotated "green's function" shape is nalpha  x nbasis

    Returns
    -------
    exx : :class:`numpy.ndarray`
        exchange contribution for all walkers.
    """
    # sort out cupy later
    zeros = numpy.zeros
    dot = numpy.dot

    naux = rchola.shape[0]
    nwalkers = Ghalfa_batch.shape[0]
    nocc = Ghalfa_batch.shape[1]
    nbsf = Ghalfa_batch.shape[2]

    T = zeros((nocc, nocc), dtype=numpy.complex128)
    exx = zeros((nwalkers), dtype=numpy.complex128)
    rchol = rchola.reshape((naux, nocc, nbsf))
    for iw in range(nwalkers):
        Greal = Ghalfa_batch[iw].T.real.copy()
        Gimag = Ghalfa_batch[iw].T.imag.copy()
        for jx in range(naux):
            T = rchol[jx].dot(Greal) + 1.0j * rchol[jx].dot(Gimag)
            exx[iw] += dot(T.ravel(), T.T.ravel())
    exx *= 0.5
    return exx


@jit(nopython=True, fastmath=True)
def exx_kernel_batch_complex_rchol(rchola, Ghalfa_batch):
    """Compute exchange contribution for complex rchol.

    Parameters
    ----------
    rchol : :class:`numpy.ndarray`
        Half-rotated cholesky.
    Ghalf : :class:`numpy.ndarray`
        Walker's half-rotated "green's function" shape is nalpha  x nbasis

    Returns
    -------
    exx : :class:`numpy.ndarray`
        exchange contribution for all walkers.
    """
    # sort out cupy later
    zeros = numpy.zeros
    dot = numpy.dot

    naux = rchola.shape[0]
    nwalkers = Ghalfa_batch.shape[0]
    nocc = Ghalfa_batch.shape[1]
    nbsf = Ghalfa_batch.shape[2]

    T = zeros((nocc, nocc), dtype=numpy.complex128)
    exx = zeros((nwalkers), dtype=numpy.complex128)
    for iw in range(nwalkers):
        Ghalfa = Ghalfa_batch[iw]
        for jx in range(naux):
            rcholx = rchola[jx].reshape(nocc, nbsf)
            T = rcholx.dot(Ghalfa.T)
            exx[iw] += dot(T.ravel(), T.T.ravel())
    exx *= 0.5
    return exx


@jit(nopython=True, fastmath=True)
def ecoul_kernel_batch_real_rchol_rhf(rchola, Ghalfa_batch):
    """Compute coulomb contribution for real rchol with RHF trial.

    Parameters
    ----------
    rchol : :class:`numpy.ndarray`
        Half-rotated cholesky.
    Ghalf : :class:`numpy.ndarray`
        Walker's half-rotated "green's function" shape is nalpha  x nbasis

    Returns
    -------
    ecoul : :class:`numpy.ndarray`
        coulomb contribution for all walkers.
    """
    # sort out cupy later
    zeros = numpy.zeros
    dot = numpy.dot
    nwalkers = Ghalfa_batch.shape[0]
    Ghalfa_batch_real = Ghalfa_batch.real.copy()
    Ghalfa_batch_imag = Ghalfa_batch.imag.copy()
    X = rchola.dot(Ghalfa_batch_real.T) + 1.0j * rchola.dot(
        Ghalfa_batch_imag.T
    )  # naux x nwalkers
    ecoul = zeros(nwalkers, dtype=numpy.complex128)
    X = X.T.copy()
    for iw in range(nwalkers):
        ecoul[iw] += 2.0 * dot(X[iw], X[iw])

    return ecoul


@jit(nopython=True, fastmath=True)
def ecoul_kernel_batch_real_rchol_uhf(rchola, rcholb, Ghalfa_batch, Ghalfb_batch):
    """Compute coulomb contribution for real rchol with UHF trial.

    Parameters
    ----------
    rchola : :class:`numpy.ndarray`
        Half-rotated cholesky (alpha).
    rcholb : :class:`numpy.ndarray`
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
    # sort out cupy later
    zeros = numpy.zeros
    dot = numpy.dot
    nwalkers = Ghalfa_batch.shape[0]
    Ghalfa_batch_real = Ghalfa_batch.real.copy()
    Ghalfa_batch_imag = Ghalfa_batch.imag.copy()
    Ghalfb_batch_real = Ghalfb_batch.real.copy()
    Ghalfb_batch_imag = Ghalfb_batch.imag.copy()
    X = rchola.dot(Ghalfa_batch_real.T) + 1.0j * rchola.dot(
        Ghalfa_batch_imag.T
    )  # naux x nwalkers
    X += rcholb.dot(Ghalfb_batch_real.T) + 1.0j * rcholb.dot(
        Ghalfb_batch_imag.T
    )  # naux x nwalkers
    ecoul = zeros(nwalkers, dtype=numpy.complex128)
    X = X.T.copy()
    for iw in range(nwalkers):
        ecoul[iw] += dot(X[iw], X[iw])
    ecoul *= 0.5
    return ecoul


@jit(nopython=True, fastmath=True)
def ecoul_kernel_batch_complex_rchol_rhf(rchola, Ghalfa_batch):
    """Compute coulomb contribution for complex rchol with RHF trial.

    Parameters
    ----------
    rchol : :class:`numpy.ndarray`
        Half-rotated cholesky.
    Ghalf : :class:`numpy.ndarray`
        Walker's half-rotated "green's function" shape is nalpha  x nbasis

    Returns
    -------
    ecoul : :class:`numpy.ndarray`
        coulomb contribution for all walkers.
    """
    # sort out cupy later
    zeros = numpy.zeros
    dot = numpy.dot

    X = rchola.dot(Ghalfa_batch.T)
    ecoul = zeros(nwalkers, dtype=numpy.complex128)
    X = X.T.copy()
    for iw in range(nwalkers):
        ecoul[iw] += 2.0 * dot(X[iw], X[iw])
    return ecoul


@jit(nopython=True, fastmath=True)
def ecoul_kernel_batch_complex_rchol_uhf(rchola, rcholb, Ghalfa_batch, Ghalfb_batch):
    """Compute coulomb contribution for real rchol with UHF trial.

    Parameters
    ----------
    rchola : :class:`numpy.ndarray`
        Half-rotated cholesky (alpha).
    rcholb : :class:`numpy.ndarray`
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
    # sort out cupy later
    zeros = numpy.zeros
    dot = numpy.dot

    X = rchola.dot(Ghalfa_batch.T)
    X += rcholb.dot(Ghalfb_batch.T)
    ecoul = zeros(nwalkers, dtype=numpy.complex128)
    X = X.T.copy()
    for iw in range(nwalkers):
        ecoul[iw] += dot(X[iw], X[iw])
    ecoul *= 0.5
    return ecoul


def local_energy_single_det_batch(system, hamiltonian, walker_batch, trial):
    """Compute local energy for walker batch (all walkers at once).

    Single determinant case.

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
    energy = []
    nwalkers = walker_batch.nwalkers
    for idx in range(nwalkers):
        G = [walker_batch.Ga[idx], walker_batch.Gb[idx]]
        Ghalf = [walker_batch.Ghalfa[idx], walker_batch.Ghalfb[idx]]
        energy += [list(local_energy_G(system, hamiltonian, trial, G, Ghalf))]

    energy = xp.array(energy, dtype=numpy.complex128)
    return energy


def local_energy_single_det_batch_einsum(system, hamiltonian, walker_batch, trial):
    """Compute local energy for walker batch (all walkers at once).

    Use einsum rather than numba kernels.

    Single determinant case.

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
    nalpha = walker_batch.Ghalfa.shape[1]
    nbeta = walker_batch.Ghalfb.shape[1]
    nbasis = walker_batch.Ghalfa.shape[-1]
    nchol = hamiltonian.nchol

    walker_batch.Ghalfa = walker_batch.Ghalfa.reshape(nwalkers, nalpha * nbasis)
    walker_batch.Ghalfb = walker_batch.Ghalfb.reshape(nwalkers, nbeta * nbasis)

    e1b = (
        walker_batch.Ghalfa.dot(trial._rH1a.ravel())
        + walker_batch.Ghalfb.dot(trial._rH1b.ravel())
        + hamiltonian.ecore
    )

    if xp.isrealobj(trial._rchola):
        Xa = trial._rchola.dot(walker_batch.Ghalfa.real.T) + 1.0j * trial._rchola.dot(
            walker_batch.Ghalfa.imag.T
        )  # naux x nwalkers
        Xb = trial._rcholb.dot(walker_batch.Ghalfb.real.T) + 1.0j * trial._rcholb.dot(
            walker_batch.Ghalfb.imag.T
        )  # naux x nwalkers
    else:
        Xa = trial._rchola.dot(walker_batch.Ghalfa.T)
        Xb = trial._rcholb.dot(walker_batch.Ghalfb.T)

    ecoul = xp.einsum("xw,xw->w", Xa, Xa, optimize=True)
    ecoul += xp.einsum("xw,xw->w", Xb, Xb, optimize=True)
    ecoul += 2.0 * xp.einsum("xw,xw->w", Xa, Xb, optimize=True)

    walker_batch.Ghalfa = walker_batch.Ghalfa.reshape(nwalkers, nalpha, nbasis)
    walker_batch.Ghalfb = walker_batch.Ghalfb.reshape(nwalkers, nbeta, nbasis)

    Ta = xp.zeros((nwalkers, nalpha, nalpha), dtype=numpy.complex128)
    Tb = xp.zeros((nwalkers, nbeta, nbeta), dtype=numpy.complex128)

    GhalfaT_batch = walker_batch.Ghalfa.transpose(0, 2, 1).copy()  # nw x nbasis x nocc
    GhalfbT_batch = walker_batch.Ghalfb.transpose(0, 2, 1).copy()  # nw x nbasis x nocc

    exx = xp.zeros(
        nwalkers, dtype=numpy.complex128
    )  # we will iterate over cholesky index to update Ex energy for alpha and beta
    # breakpoint()
    for x in range(nchol):  # write a cython function that calls blas for this.
        rmi_a = trial._rchola[x].reshape((nalpha, nbasis))
        rmi_b = trial._rcholb[x].reshape((nbeta, nbasis))
        if xp.isrealobj(trial._rchola):  # this is actually fasater
            Ta[:, :, :].real = rmi_a.dot(GhalfaT_batch.real).transpose(1, 0, 2)
            Ta[:, :, :].imag = rmi_a.dot(GhalfaT_batch.imag).transpose(1, 0, 2)
            Tb[:, :, :].real = rmi_b.dot(GhalfbT_batch.real).transpose(1, 0, 2)
            Tb[:, :, :].imag = rmi_b.dot(GhalfbT_batch.imag).transpose(1, 0, 2)
        else:
            Ta = rmi_a.dot(GhalfaT_batch).transpose(1, 0, 2)
            Tb = rmi_b.dot(GhalfbT_batch).transpose(1, 0, 2)
        # this James Spencer's change is actually slower
        # Ta = walker_batch.Ghalfa @ rmi_a.T
        # Tb = walker_batch.Ghalfb @ rmi_b.T

        exx += xp.einsum("wij,wji->w", Ta, Ta, optimize=True) + einsum(
            "wij,wji->w", Tb, Tb, optimize=True
        )

    e2b = 0.5 * (ecoul - exx)

    energy = xp.zeros((nwalkers, 3), dtype=numpy.complex128)
    energy[:, 0] = e1b + e2b
    energy[:, 1] = e1b
    energy[:, 2] = e2b

    return energy


def local_energy_single_det_rhf_batch(system, hamiltonian, walker_batch, trial):
    """Compute local energy for walker batch (all walkers at once).

    Single determinant RHF case.

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
    nwalkers = walker_batch.Ghalfa.shape[0]
    nalpha = walker_batch.Ghalfa.shape[1]
    nbasis = hamiltonian.nbasis
    nchol = hamiltonian.nchol

    walker_batch.Ghalfa = walker_batch.Ghalfa.reshape(nwalkers, nalpha * nbasis)

    e1b = 2.0 * walker_batch.Ghalfa.dot(trial._rH1a.ravel()) + hamiltonian.ecore

    if xp.isrealobj(trial._rchola):
        ecoul = ecoul_kernel_batch_real_rchol_rhf(trial._rchola, walker_batch.Ghalfa)
    else:
        ecoul = ecoul_kernel_batch_complex_rchol_rhf(trial._rchola, walker_batch.Ghalfa)

    walker_batch.Ghalfa = walker_batch.Ghalfa.reshape(nwalkers, nalpha, nbasis)

    if xp.isrealobj(trial._rchola):
        exx = 2.0 * exx_kernel_batch_real_rchol(trial._rchola, walker_batch.Ghalfa)
    else:
        exx = 2.0 * exx_kernel_batch_complex_rchol(trial._rchola, walker_batch.Ghalfa)

    e2b = ecoul - exx

    energy = xp.zeros((nwalkers, 3), dtype=numpy.complex128)
    energy[:, 0] = e1b + e2b
    energy[:, 1] = e1b
    energy[:, 2] = e2b

    return energy


def two_body_energy_uhf(trial, walker_batch):
    """Compute two body energy only.

    Single determinant case (UHF). For use in Wick's code.

    TODO FDM: Is this used any more (maybe delete.

    Parameters
    ----------
    trial : trial object
        Trial wavefunctioni.
    walker_batch : WalkerBatch
        Walkers object.

    Returns
    -------
    two_body_energy : np.ndarray
        Coulomb + exchange (no core constant contribution).
    """
    nwalkers = walker_batch.Ghalfa.shape[0]
    nalpha = walker_batch.Ghalfa.shape[1]
    nbeta = walker_batch.Ghalfb.shape[1]
    nbasis = walker_batch.Ghalfa.shape[2]
    Ghalfa = walker_batch.Ghalfa.reshape(nwalkers, nalpha * nbasis)
    Ghalfb = walker_batch.Ghalfb.reshape(nwalkers, nbeta * nbasis)
    if xp.isrealobj(trial._rchola):
        ecoul = ecoul_kernel_batch_real_rchol_uhf(
            trial._rchola, trial._rcholb, Ghalfa, Ghalfb
        )
        exx = exx_kernel_batch_real_rchol(
            trial._rchola, walker_batch.Ghalfa
        ) + exx_kernel_batch_real_rchol(trial._rcholb, walker_batch.Ghalfb)
    else:
        ecoul = ecoul_kernel_batch_complex_rchol_uhf(
            trial._rchola, trial._rcholb, Ghalfa, Ghalfb
        )
        exx = exx_kernel_batch_complex_rchol(
            trial._rchola, walker_batch.Ghalfa
        ) + exx_kernel_batch_complex_rchol(trial._rcholb, walker_batch.Ghalfb)
    return ecoul - exx


def local_energy_single_det_uhf_batch(system, hamiltonian, walker_batch, trial):
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
    nwalkers = walker_batch.Ghalfa.shape[0]
    nalpha = walker_batch.Ghalfa.shape[1]
    nbeta = walker_batch.Ghalfb.shape[1]
    nbasis = hamiltonian.nbasis

    Ghalfa = walker_batch.Ghalfa.reshape(nwalkers, nalpha * nbasis)
    Ghalfb = walker_batch.Ghalfb.reshape(nwalkers, nbeta * nbasis)

    e1b = Ghalfa.dot(trial._rH1a.ravel())
    e1b += Ghalfb.dot(trial._rH1b.ravel())
    e1b += hamiltonian.ecore

    if xp.isrealobj(trial._rchola):
        ecoul = ecoul_kernel_batch_real_rchol_uhf(
            trial._rchola, trial._rcholb, Ghalfa, Ghalfb
        )
    else:
        ecoul = ecoul_kernel_batch_complex_rchol_uhf(
            trial._rchola, trial._rcholb, Ghalfa, Ghalfb
        )

    if xp.isrealobj(trial._rchola):
        exx = exx_kernel_batch_real_rchol(
            trial._rchola, walker_batch.Ghalfa
        ) + exx_kernel_batch_real_rchol(trial._rcholb, walker_batch.Ghalfb)
    else:
        exx = exx_kernel_batch_complex_rchol(
            trial._rchola, walker_batch.Ghalfa
        ) + exx_kernel_batch_complex_rchol(trial._rcholb, walker_batch.Ghalfb)

    e2b = ecoul - exx

    energy = xp.zeros((nwalkers, 3), dtype=numpy.complex128)
    energy[:, 0] = e1b + e2b
    energy[:, 1] = e1b
    energy[:, 2] = e2b

    return energy


def local_energy_single_det_batch_gpu_old(system, hamiltonian, walker_batch, trial):
    """Compute local energy for walker batch (all walkers at once).

    Single determinant UHF GPU case.

    einsum code path very memory intensive will OOM without warning.

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
    nwalkers = walker_batch.Ghalfa.shape[0]
    nalpha = walker_batch.Ghalfa.shape[1]
    nbeta = walker_batch.Ghalfb.shape[1]
    nbasis = walker_batch.Ghalfa.shape[-1]
    nchol = hamiltonian.nchol

    walker_batch.Ghalfa = walker_batch.Ghalfa.reshape(nwalkers, nalpha * nbasis)
    walker_batch.Ghalfb = walker_batch.Ghalfb.reshape(nwalkers, nbeta * nbasis)

    e1b = (
        walker_batch.Ghalfa.dot(trial._rH1a.ravel())
        + walker_batch.Ghalfb.dot(trial._rH1b.ravel())
        + hamiltonian.ecore
    )

    if xp.isrealobj(trial._rchola):
        Xa = trial._rchola.dot(walker_batch.Ghalfa.real.T) + 1.0j * trial._rchola.dot(
            walker_batch.Ghalfa.imag.T
        )  # naux x nwalkers
        Xb = trial._rcholb.dot(walker_batch.Ghalfb.real.T) + 1.0j * trial._rcholb.dot(
            walker_batch.Ghalfb.imag.T
        )  # naux x nwalkers
    else:
        Xa = trial._rchola.dot(walker_batch.Ghalfa.T)
        Xb = trial._rcholb.dot(walker_batch.Ghalfb.T)

    ecoul = xp.einsum("xw,xw->w", Xa, Xa, optimize=True)
    ecoul += xp.einsum("xw,xw->w", Xb, Xb, optimize=True)
    ecoul += 2.0 * xp.einsum("xw,xw->w", Xa, Xb, optimize=True)

    walker_batch.Ghalfa = walker_batch.Ghalfa.reshape(nwalkers, nalpha, nbasis)
    walker_batch.Ghalfb = walker_batch.Ghalfb.reshape(nwalkers, nbeta, nbasis)

    trial._rchola = trial._rchola.reshape(nchol, nalpha, nbasis)
    trial._rcholb = trial._rcholb.reshape(nchol, nbeta, nbasis)

    Txij = xp.einsum("xim,wjm->wxji", trial._rchola, walker_batch.Ghalfa)
    exx = xp.einsum("wxji,wxij->w", Txij, Txij)
    Txij = xp.einsum("xim,wjm->wxji", trial._rcholb, walker_batch.Ghalfb)
    exx += xp.einsum("wxji,wxij->w", Txij, Txij)

    trial._rchola = trial._rchola.reshape(nchol, nalpha * nbasis)
    trial._rcholb = trial._rcholb.reshape(nchol, nbeta * nbasis)

    e2b = 0.5 * (ecoul - exx)

    energy = xp.zeros((nwalkers, 3), dtype=numpy.complex128)
    energy[:, 0] = e1b + e2b
    energy[:, 1] = e1b
    energy[:, 2] = e2b

    synchronize()

    return energy


def local_energy_single_det_batch_gpu(
    system, hamiltonian, walker_batch, trial, max_mem=2.0
):
    """Compute local energy for walker batch (all walkers at once).

    Single determinant UHF GPU case.

    Numba kernel jitted route (qmcpack algorithm)

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
    max_mem : float
        Maximum memory in GB for intermediates. Optional. Default 2GB.
        TODO FDM: Remove following config setup.

    Returns
    -------
    local_energy : np.ndarray
        Total, one-body and two-body energies.
    """

    nwalkers = walker_batch.Ghalfa.shape[0]
    nalpha = walker_batch.Ghalfa.shape[1]
    nbeta = walker_batch.Ghalfb.shape[1]
    nbasis = walker_batch.Ghalfa.shape[-1]
    nchol = hamiltonian.nchol

    Ghalfa = walker_batch.Ghalfa.reshape(nwalkers, nalpha * nbasis)
    Ghalfb = walker_batch.Ghalfb.reshape(nwalkers, nbeta * nbasis)

    e1b = (
        Ghalfa.dot(trial._rH1a.ravel())
        + Ghalfb.dot(trial._rH1b.ravel())
        + hamiltonian.ecore
    )

    if xp.isrealobj(trial._rchola):
        Xa = trial._rchola.dot(Ghalfa.real.T) + 1.0j * trial._rchola.dot(
            Ghalfa.imag.T
        )  # naux x nwalkers
        Xb = trial._rcholb.dot(Ghalfb.real.T) + 1.0j * trial._rcholb.dot(
            Ghalfb.imag.T
        )  # naux x nwalkers
    else:
        Xa = trial._rchola.dot(Ghalfa.T)
        Xb = trial._rcholb.dot(Ghalfb.T)

    ecoul = xp.einsum("xw,xw->w", Xa, Xa, optimize=True)
    ecoul += xp.einsum("xw,xw->w", Xb, Xb, optimize=True)
    ecoul += 2.0 * xp.einsum("xw,xw->w", Xa, Xb, optimize=True)

    max_nocc = max(nalpha, nbeta)
    mem_needed = 16 * nwalkers * max_nocc * max_nocc * nchol / (1024.0**3.0)
    num_chunks = max(1, ceil(mem_needed / max_mem))
    chunk_size = ceil(nchol / num_chunks)
    nchol_chunks = ceil(nchol / chunk_size)

    # Buffer for large intermediate tensor
    buff = xp.zeros(shape=(nwalkers * chunk_size * max_nocc * max_nocc),
            dtype=xp.complex128)
    nchol_chunk_size = chunk_size
    nchol_left = nchol
    exx = xp.zeros(nwalkers, dtype=xp.complex128)
    Ghalfa = walker_batch.Ghalfa.reshape((nwalkers * nalpha, nbasis))
    Ghalfb = walker_batch.Ghalfb.reshape((nwalkers * nbeta, nbasis))
    for i in range(nchol_chunks):
        nchol_chunk = min(nchol_chunk_size, nchol_left)
        chol_sls = slice(i * chunk_size, i * chunk_size + nchol_chunk)
        size = nwalkers * nchol_chunk * nalpha * nalpha
        # alpha-alpha
        Txij = buff[:size].reshape((nchol_chunk * nalpha, nwalkers * nalpha))
        rchol = trial._rchola[chol_sls].reshape((nchol_chunk * nalpha, nbasis))
        xp.dot(rchol, Ghalfa.T, out=Txij)
        Txij = Txij.reshape((nchol_chunk, nalpha, nwalkers, nalpha))
        exchange_reduction(Txij, exx)
        # beta-beta
        size = nwalkers * nchol_chunk * nbeta * nbeta
        Txij = buff[:size].reshape((nchol_chunk * nbeta, nwalkers * nbeta))
        rchol = trial._rcholb[chol_sls].reshape((nchol_chunk * nbeta, nbasis))
        xp.dot(rchol, Ghalfb.T, out=Txij)
        Txij = Txij.reshape((nchol_chunk, nbeta, nwalkers, nbeta))
        exchange_reduction(Txij, exx)
        nchol_left -= chunk_size

    e2b = 0.5 * (ecoul - exx)

    energy = xp.zeros((nwalkers, 3), dtype=numpy.complex128)
    energy[:, 0] = e1b + e2b
    energy[:, 1] = e1b
    energy[:, 2] = e2b

    synchronize()
    return energy
