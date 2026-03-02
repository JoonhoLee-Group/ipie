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

from math import ceil, sqrt

import numpy
from numba import jit
from ipie.utils.backend import arraylib as xp

from ipie.systems.generic import Generic
from ipie.hamiltonians.kpt_hamiltonian import KptComplexChol, KptComplexCholSymm
from ipie.walkers.uhf_walkers import UHFWalkers
from ipie.trial_wavefunction.single_det_kpt import KptSingleDet

# from line_profiler import profile

import plum

# Note specialisations occur to because:
# 1. Numba does not allow for mixing types without a warning so need to split
# real and complex components apart when rchol is real. Green's function is
# complex in general.
# Optimize for case when wavefunction is RHF (factor of 2 saving)


@jit(nopython=True, fastmath=True)
def kpt_chol_ecoul_kernel_rhf(rchola, Ghalfa_batch, kpq_mat, mq_vec):
    """Compute coulomb contribution for complex rchol with RHF trial.

    Parameters
    ----------
    rchol : :class:`numpy.ndarray`
        Half-rotated cholesky.
    Ghalf : :class:`numpy.ndarray`
        Walker's half-rotated "green's function" shape is nalpha  x nbasis
    kpts : :class:`numpy.ndarray`
        all k-points in fractional coordinates.

    Returns
    -------
    ecoul : :class:`numpy.ndarray`
        coulomb contribution for all walkers.
    """
    # sort out cupy later
    zeros = numpy.zeros
    dot = numpy.dot
    nwalkers = Ghalfa_batch.shape[0]

    # shape of rchola: (naux, nk, nocc, nk, nbsf) (gamma, k, i, q, p)
    # shape of Ghalf: (nw, nk, nocc, nk, nbsf)
    naux = rchola.shape[0]
    nk = rchola.shape[1]
    ecoul = zeros(nwalkers, dtype=numpy.complex128)
    GhalfaT = Ghalfa_batch.transpose(0, 3, 4, 1, 2)
    X = zeros((nwalkers, naux, nk), dtype=numpy.complex128)
    for iq in range(nk):
        for ik in range(nk):
            ik_pq = kpq_mat[ik, iq]
            i_mq = mq_vec[iq]
            for iw in range(nwalkers):
                for g in range(naux):
                    X[iw, g, iq] += numpy.trace(
                        dot(rchola[g, ik, :, iq, :], GhalfaT[iw, ik_pq, :, ik, :])
                    )

    for iw in range(nwalkers):
        for q in range(nk):
            i_mq = mq_vec[q]
            ecoul[iw] += 2.0 * dot(X[iw, :, q], X[iw, :, i_mq])
    return ecoul / nk


@jit(nopython=True, fastmath=True)
def kpt_chol_exx_kernel(rchola, Ghalfa_batch, kpq_mat, mq_vec):
    """Compute coulomb contribution for complex rchol with RHF trial.

    Parameters
    ----------
    rchol : :class:`numpy.ndarray`
        Half-rotated cholesky.
    Ghalf : :class:`numpy.ndarray`
        Walker's half-rotated "green's function" shape is nalpha  x nbasis
    kpq_mat : :class:`numpy.ndarray`
        all k + q in fractional coordinates.
    mq_vec : :class:`numpy.ndarray`
        all -q in fractional coordinates.

    Returns
    -------
    ecoul : :class:`numpy.ndarray`
        coulomb contribution for all walkers.
    """
    # sort out cupy later
    zeros = numpy.zeros
    dot = numpy.dot
    nwalkers = Ghalfa_batch.shape[0]

    # shape of rchola: (naux, nk, nocc, nk, nbsf) (gamma, k, i, q, p)
    # shape of Ghalf: (nw, nk, nocc, nk, nbsf)
    naux = rchola.shape[0]
    nocc = rchola.shape[2]
    nk = rchola.shape[1]
    exx = zeros(nwalkers, dtype=numpy.complex128)
    GhalfaT = Ghalfa_batch.transpose(0, 3, 4, 1, 2)

    T1 = zeros((nwalkers, naux, nocc, nocc), dtype=numpy.complex128)
    T2 = zeros((nwalkers, naux, nocc, nocc), dtype=numpy.complex128)
    for iq in range(nk):
        for ik in range(nk):
            for ikprime in range(nk):
                ikpr_pq = kpq_mat[ikprime, iq]
                ik_pq = kpq_mat[ik, iq]
                i_mq = mq_vec[iq]
                for n in range(nwalkers):
                    for g in range(naux):
                        T1[n, g] = dot(rchola[g, ik, :, iq, :], GhalfaT[n, ik_pq, :, ikpr_pq, :])
                        T2[n, g] = dot(
                            rchola[g, ikpr_pq, :, i_mq, :], GhalfaT[n, ikprime, :, ik, :]
                        )
                        exx[n] += -numpy.trace(dot(T1[n, g], T2[n, g]))

    return 0.5 * exx / nk


@jit(nopython=True, fastmath=True)
def kpt_chol_ecoul_kernel_uhf(rchola, rcholb, Ghalfa_batch, Ghalfb_batch, kpq_mat, mq_vec):
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

    # shape of rchola: (naux, nk, nocc, nk, nbsf) (gamma, k, i, q, p)
    # shape of Ghalf: (nw, nk, nocc, nk, nbsf)
    naux = rchola.shape[0]
    nk = rchola.shape[1]
    ecoul = zeros(nwalkers, dtype=numpy.complex128)
    GhalfaT = Ghalfa_batch.transpose(0, 3, 4, 1, 2)
    GhalfbT = Ghalfb_batch.transpose(0, 3, 4, 1, 2)
    X = zeros((nwalkers, naux, nk), dtype=numpy.complex128)
    for iq in range(nk):
        for ik in range(nk):
            ik_pq = kpq_mat[ik, iq]
            i_mq = mq_vec[iq]
            for iw in range(nwalkers):
                for g in range(naux):
                    X[iw, g, iq] += numpy.trace(
                        dot(rchola[g, ik, :, iq, :], GhalfaT[iw, ik_pq, :, ik, :])
                    ) + numpy.trace(dot(rcholb[g, ik, :, iq, :], GhalfbT[iw, ik_pq, :, ik, :]))

    for iw in range(nwalkers):
        for q in range(nk):
            i_mq = mq_vec[q]
            ecoul[iw] += dot(X[iw, :, q], X[iw, :, i_mq])
    return 0.5 * ecoul / nk


@jit(nopython=True, fastmath=True)
def kpt_symmchol_ecoul_kernel_rhf(rchola, rcholbara, Ghalfa, GhalfaT, kpq_mat, Sset, Qplus):
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
    nwalkers = Ghalfa.shape[2]

    # shape of rchola: (nq, nk, nocc, naux, nbsf) (q, k, gamma, i, p)
    # shape of Ghalf: (nk, nk, nw, nocc, nbsf)
    unique_nq = len(Sset) + len(Qplus)
    nbsf = rchola.shape[4]
    nocc = rchola.shape[2]
    naux = rchola.shape[3]
    nk = rchola.shape[1]
    rchola = rchola.transpose(0, 1, 3, 2, 4).copy()
    rcholbara = rcholbara.transpose(0, 1, 3, 2, 4).copy()
    ecoul = zeros(nwalkers, dtype=numpy.complex128)
    X = zeros((unique_nq, nwalkers, naux), dtype=numpy.complex128)
    Xbar = zeros((unique_nq, nwalkers, naux), dtype=numpy.complex128)
    for iq in range(len(Sset)):
        iq_real = Sset[iq]
        Xq = X[iq]
        Xbarq = Xbar[iq]
        for ik in range(nk):
            ik_pq = kpq_mat[iq_real, ik]
            La = rchola[iq, ik].reshape(naux, nocc * nbsf)
            Lbara = rcholbara[iq, ik].reshape(naux, nocc * nbsf)
            for iw in range(nwalkers):
                Ghalfa_k_kpq = Ghalfa[ik, ik_pq, iw].reshape(nocc * nbsf)
                GhalfTa_k_kpq = GhalfaT[ik, ik_pq, iw].reshape(nocc * nbsf)
                Xq[iw] += 2.0 * La @ Ghalfa_k_kpq
                Xbarq[iw] += 2.0 * Lbara @ GhalfTa_k_kpq

    for iq in range(len(Sset), len(Sset) + len(Qplus)):
        iq_real = Qplus[iq - len(Sset)]
        Xq = X[iq]
        Xbarq = Xbar[iq]
        for ik in range(nk):
            ik_pq = kpq_mat[iq_real, ik]
            La = rchola[iq, ik].reshape(naux, nocc * nbsf)
            Lbara = rcholbara[iq, ik].reshape(naux, nocc * nbsf)
            for iw in range(nwalkers):
                Ghalfa_k_kpq = Ghalfa[ik, ik_pq, iw].reshape(nocc * nbsf)
                GhalfTa_k_kpq = GhalfaT[ik, ik_pq, iw].reshape(nocc * nbsf)
                Xq[iw] += 2.0 * sqrt(2) * La @ Ghalfa_k_kpq
                Xbarq[iw] += 2.0 * sqrt(2) * Lbara @ GhalfTa_k_kpq

    X = X.transpose(1, 0, 2).copy()
    Xbar = Xbar.transpose(1, 0, 2).copy()
    X = X.reshape(nwalkers, naux * unique_nq)
    Xbar = Xbar.reshape(nwalkers, naux * unique_nq)
    for iw in range(nwalkers):
        ecoul[iw] = dot(X[iw], Xbar[iw])
    return 0.5 * ecoul / nk


@jit(nopython=True, fastmath=True)  # , parallel=True
def kpt_symmchol_exx_kernel_lowmem(rchola, rcholbara, Ghalfa, GhalfaT, kpq_mat, Sset, Qplus):
    """Compute coulomb contribution for complex rchol with RHF trial.

    Parameters
    ----------
    rchol : :class:`numpy.ndarray`
        Half-rotated cholesky.
    Ghalf : :class:`numpy.ndarray`
        Walker's half-rotated "green's function" shape is nalpha  x nbasis
    kpq_mat : :class:`numpy.ndarray`
        all k + q in fractional coordinates.
    mq_vec : :class:`numpy.ndarray`
        all -q in fractional coordinates.

    Returns
    -------
    ecoul : :class:`numpy.ndarray`
        coulomb contribution for all walkers.
    """
    # sort out cupy later
    zeros = numpy.zeros
    nwalkers = Ghalfa.shape[2]

    # shape of rchola: (nq, nk, naux, nocc, nbsf)
    # shape of Ghalf: (nk, nk, nw, nocc, nbsf)
    naux = rchola.shape[2]
    nocc = rchola.shape[3]
    nk = rchola.shape[1]
    exx = zeros(nwalkers, dtype=numpy.complex128)

    for iq in range(len(Sset)):
        iq_real = Sset[iq]
        for ik in range(nk):
            ik_pq = kpq_mat[iq_real, ik]
            for ikprime in range(nk):
                ikpr_pq = kpq_mat[iq_real, ikprime]
                for iw in range(nwalkers):
                    Ghalf_kpq_kprpq = GhalfaT[ik_pq, ikpr_pq, iw]
                    Ghalf_k_kp = Ghalfa[ik, ikprime, iw]
                    for g in range(naux):
                        Lkqg = rchola[iq, ik, :, g].transpose(1, 0, 2).copy()
                        Lbarkpqg = rcholbara[iq, ikprime, :, g].transpose(1, 0, 2).copy()
                        T1 = Lkqg @ Ghalf_kpq_kprpq
                        T2 = Ghalf_k_kp @ Lbarkpqg
                        for i in range(nocc):
                            for j in range(nocc):
                                exx[iw] -= T1[i, j] * T2[i, j]

    for iq in range(len(Sset), len(Sset) + len(Qplus)):
        iq_real = Qplus[iq - len(Sset)]
        for ik in range(nk):
            Lkq = rchola[iq, ik].transpose(1, 0, 2).copy()
            ik_pq = kpq_mat[iq_real, ik]
            for ikprime in range(nk):
                ikpr_pq = kpq_mat[iq_real, ikprime]
                Lbarkpq = rcholbara[iq, ikprime].transpose(1, 0, 2).copy()
                for iw in range(nwalkers):
                    Ghalf_kpq_kprpq = GhalfaT[ik_pq, ikpr_pq, iw]
                    Ghalf_k_kp = Ghalfa[ik, ikprime, iw]
                    for g in range(naux):
                        T1 = Lkq[g] @ Ghalf_kpq_kprpq
                        T2 = Ghalf_k_kp @ Lbarkpq[g]
                        for i in range(nocc):
                            for j in range(nocc):
                                exx[iw] -= 2.0 * T1[i, j] * T2[i, j]


@jit(nopython=True, fastmath=True)  # , parallel=True
def kpt_symmchol_exx_kernel(rchola, rcholbara, Ghalfa, GhalfaT, kpq_mat, Sset, Qplus):
    """Compute coulomb contribution for complex rchol with RHF trial.

    Parameters
    ----------
    rchol : :class:`numpy.ndarray`
        Half-rotated cholesky.
    Ghalf : :class:`numpy.ndarray`
        Walker's half-rotated "green's function" shape is nalpha  x nbasis
    kpq_mat : :class:`numpy.ndarray`
        all k + q in fractional coordinates.
    mq_vec : :class:`numpy.ndarray`
        all -q in fractional coordinates.

    Returns
    -------
    ecoul : :class:`numpy.ndarray`
        coulomb contribution for all walkers.
    """
    # sort out cupy later
    zeros = numpy.zeros
    nwalkers = Ghalfa.shape[2]

    # shape of rchola: (nq, nk, naux, nocc, nbsf) -> (nq, nk, nocc, naux, nbsf)
    # shape of Ghalf: (nk(nocc), nk(nbsf), nw, nocc, nbsf)
    # shape of GhalfT: (nk(nbsf), nk(nocc), nw, nbsf, nocc) -> (nk, nk, nbsf, nocc, nw)
    naux = rchola.shape[3]
    nocc = rchola.shape[2]
    nk = rchola.shape[1]
    exx = zeros(nwalkers, dtype=numpy.complex128)
    for iq in range(len(Sset)):
        iq_real = Sset[iq]
        for ik in range(nk):
            for ikprime in range(nk):
                ikpr_pq = kpq_mat[iq_real, ikprime]
                ik_pq = kpq_mat[iq_real, ik]
                Lkq = rchola[iq, ik].reshape(naux * nocc, -1)
                Lbarkpq = rcholbara[iq, ikprime].reshape(-1, naux * nocc)
                Ghalf_kpq_kprpq = GhalfaT[ik_pq, ikpr_pq].reshape(-1, nocc * nwalkers)
                Ghalf_k_kp = Ghalfa[ik, ikprime].reshape(nwalkers * nocc, -1)
                T1 = Lkq @ Ghalf_kpq_kprpq  # (naux * nocc, nocc * nwalkers)
                T2 = Ghalf_k_kp @ Lbarkpq  # (nwalkers * nocc, naux * nocc)
                T1 = T1.reshape(naux * nocc * nocc, nwalkers).T.copy()
                T2 = T2.reshape(nwalkers, naux * nocc * nocc).copy()
                for iw in range(nwalkers):
                    exx[iw] += -T1[iw] @ T2[iw]

    for iq in range(len(Sset), len(Sset) + len(Qplus)):
        iq_real = Qplus[iq - len(Sset)]
        for ik in range(nk):
            for ikprime in range(nk):
                ikpr_pq = kpq_mat[iq_real, ikprime]
                ik_pq = kpq_mat[iq_real, ik]
                Lkq = rchola[iq, ik].reshape(naux * nocc, -1)
                Lbarkpq = rcholbara[iq, ikprime].reshape(-1, naux * nocc)
                Ghalf_kpq_kprpq = GhalfaT[ik_pq, ikpr_pq].reshape(-1, nocc * nwalkers)
                Ghalf_k_kp = Ghalfa[ik, ikprime].reshape(nwalkers * nocc, -1)
                T1 = Lkq @ Ghalf_kpq_kprpq  # (naux * nocc, nocc * nwalkers)
                T2 = Ghalf_k_kp @ Lbarkpq  # (nwalkers * nocc, naux * nocc)
                T1 = T1.reshape(naux * nocc * nocc, nwalkers).T.copy()
                T2 = T2.reshape(nwalkers, naux * nocc * nocc).copy()
                for iw in range(nwalkers):
                    exx[iw] += -2.0 * T1[iw] @ T2[iw]

    return 0.5 * exx / nk


@jit(nopython=True, fastmath=True)
def kpt_symmchol_ecoul_kernel_uhf(
    rchola, rcholb, rcholbara, rcholbarb, Ghalfa, Ghalfb, GhalfaT, GhalfbT, kpq_mat, Sset, Qplus
):
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
    nwalkers = Ghalfa.shape[2]

    # shape of rchola: (nq, nk, nocc, naux, nbsf) (q, k, gamma, i, p)
    # shape of Ghalf: (nk, nk, nw, nocc, nbsf)
    unique_nq = len(Sset) + len(Qplus)
    nbsf = rchola.shape[4]
    nocc = rchola.shape[2]
    naux = rchola.shape[3]
    nk = rchola.shape[1]
    rchola = rchola.transpose(0, 1, 3, 2, 4).copy()
    rcholb = rcholb.transpose(0, 1, 3, 2, 4).copy()
    rcholbara = rcholbara.transpose(0, 1, 3, 2, 4).copy()
    rcholbarb = rcholbarb.transpose(0, 1, 3, 2, 4).copy()
    ecoul = zeros(nwalkers, dtype=numpy.complex128)
    X = zeros((unique_nq, nwalkers, naux), dtype=numpy.complex128)
    Xbar = zeros((unique_nq, nwalkers, naux), dtype=numpy.complex128)
    for iq in range(len(Sset)):
        iq_real = Sset[iq]
        Xq = X[iq]
        Xbarq = Xbar[iq]
        for ik in range(nk):
            ik_pq = kpq_mat[iq_real, ik]
            La = rchola[iq, ik].reshape(naux, nocc * nbsf)
            Lb = rcholb[iq, ik].reshape(naux, nocc * nbsf)
            Lbara = rcholbara[iq, ik].reshape(naux, nocc * nbsf)
            Lbarb = rcholbarb[iq, ik].reshape(naux, nocc * nbsf)
            for iw in range(nwalkers):
                Ghalfa_k_kpq = Ghalfa[ik, ik_pq, iw].reshape(nocc * nbsf)
                GhalfTa_k_kpq = GhalfaT[ik, ik_pq, iw].reshape(nocc * nbsf)
                Ghalfb_k_kpq = Ghalfb[ik, ik_pq, iw].reshape(nocc * nbsf)
                GhalfTb_k_kpq = GhalfbT[ik, ik_pq, iw].reshape(nocc * nbsf)
                Xq[iw] += La @ Ghalfa_k_kpq + Lb @ Ghalfb_k_kpq
                Xbarq[iw] += Lbara @ GhalfTa_k_kpq + Lbarb @ GhalfTb_k_kpq

    for iq in range(len(Sset), len(Sset) + len(Qplus)):
        iq_real = Qplus[iq - len(Sset)]
        Xq = X[iq]
        Xbarq = Xbar[iq]
        for ik in range(nk):
            ik_pq = kpq_mat[iq_real, ik]
            La = rchola[iq, ik].reshape(naux, nocc * nbsf)
            Lb = rcholb[iq, ik].reshape(naux, nocc * nbsf)
            Lbara = rcholbara[iq, ik].reshape(naux, nocc * nbsf)
            Lbarb = rcholbarb[iq, ik].reshape(naux, nocc * nbsf)
            for iw in range(nwalkers):
                Ghalfa_k_kpq = Ghalfa[ik, ik_pq, iw].reshape(nocc * nbsf)
                GhalfTa_k_kpq = GhalfaT[ik, ik_pq, iw].reshape(nocc * nbsf)
                Ghalfb_k_kpq = Ghalfb[ik, ik_pq, iw].reshape(nocc * nbsf)
                GhalfTb_k_kpq = GhalfbT[ik, ik_pq, iw].reshape(nocc * nbsf)
                Xq[iw] += sqrt(2) * (La @ Ghalfa_k_kpq + Lb @ Ghalfb_k_kpq)
                Xbarq[iw] += sqrt(2) * (Lbara @ GhalfTa_k_kpq + Lbarb @ GhalfTb_k_kpq)

    X = X.transpose(1, 0, 2).copy()
    Xbar = Xbar.transpose(1, 0, 2).copy()
    X = X.reshape(nwalkers, naux * unique_nq)
    Xbar = Xbar.reshape(nwalkers, naux * unique_nq)
    for iw in range(nwalkers):
        ecoul[iw] = dot(X[iw], Xbar[iw])
    return 0.5 * ecoul / nk


@plum.dispatch
def local_energy_kpt_single_det_uhf(
    system: Generic,
    hamiltonian: KptComplexChol,
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
    nwalkers = walkers.Ghalfa.shape[0]
    nk = hamiltonian.nk
    nalpha = trial.nalpha
    nbeta = trial.nbeta
    nbasis = hamiltonian.nbasis

    ghalfa = walkers.Ghalfa.reshape(nwalkers, nk, nalpha, nk, nbasis)
    ghalfb = walkers.Ghalfb.reshape(nwalkers, nk, nbeta, nk, nbasis)

    diagGhalfa = numpy.zeros((nwalkers, nk, nalpha, nbasis), dtype=numpy.complex128)
    diagGhalfb = numpy.zeros((nwalkers, nk, nbeta, nbasis), dtype=numpy.complex128)
    for ik in range(nk):
        diagGhalfa[:, ik, :, :] = ghalfa[:, ik, :, ik, :]
        diagGhalfb[:, ik, :, :] = ghalfb[:, ik, :, ik, :]
    e1b = numpy.einsum("wkip, kip -> w", diagGhalfa, trial._rH1a)  # Ghalfa.dot(trial._rH1a.ravel())
    e1b += numpy.einsum("wkip, kip -> w", diagGhalfb, trial._rH1b)
    e1b /= nk
    e1b += hamiltonian.ecore

    ecoul = kpt_chol_ecoul_kernel_uhf(
        trial._rchola, trial._rcholb, ghalfa, ghalfb, hamiltonian.ikpq_mat, hamiltonian.imq_vec
    )

    exx = kpt_chol_exx_kernel(
        trial._rchola, ghalfa, hamiltonian.ikpq_mat, hamiltonian.imq_vec
    ) + kpt_chol_exx_kernel(trial._rcholb, ghalfb, hamiltonian.ikpq_mat, hamiltonian.imq_vec)

    e2b = ecoul + exx

    energy = xp.zeros((nwalkers, 3), dtype=numpy.complex128)
    energy[:, 0] = e1b + e2b
    energy[:, 1] = e1b
    energy[:, 2] = e2b

    return energy


@plum.dispatch
def local_energy_kpt_single_det_uhf(
    system: Generic,
    hamiltonian: KptComplexCholSymm,
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
    nwalkers = walkers.Ghalfa.shape[0]
    nk = hamiltonian.nk
    nalpha = trial.nalpha
    nbeta = trial.nbeta
    nbasis = hamiltonian.nbasis

    if walkers.rhf:
        ghalfa = walkers.Ghalfa.reshape(nwalkers, nk, nalpha, nk, nbasis)
        ghalfaT = walkers.Ghalfa.transpose(0, 2, 1).reshape(nwalkers, nk, nbasis, nk, nalpha)

        diagGhalfa = numpy.zeros((nwalkers, nk, nalpha, nbasis), dtype=numpy.complex128)
        for ik in range(nk):
            diagGhalfa[:, ik, :, :] = ghalfa[:, ik, :, ik, :]
        e1b = 2.0 * numpy.einsum(
            "wkip, kip -> w", diagGhalfa, trial._rH1a
        )  # Ghalfa.dot(trial._rH1a.ravel())
        e1b /= nk
        e1b += hamiltonian.ecore

        ghalfa = ghalfa.transpose(1, 3, 0, 2, 4).copy()  # nk, nk, nw, nalpha, nbasis
        ghalfaTcoul = ghalfaT.transpose(1, 3, 0, 2, 4).copy()  # nk, nk, nw, nbasis, nalpha
        ghalfaTx = ghalfaT.transpose(1, 3, 2, 4, 0).copy()  # nk, nk, nbasis, nalpha, nw

        ecoul = kpt_symmchol_ecoul_kernel_rhf(
            trial._rchola,
            trial._rcholbara,
            ghalfa,
            ghalfaTcoul,
            hamiltonian.ikpq_mat,
            hamiltonian.Sset,
            hamiltonian.Qplus,
        )

        exx = 2.0 * kpt_symmchol_exx_kernel(
            trial._rchola,
            trial._rcholbara,
            ghalfa,
            ghalfaTx,
            hamiltonian.ikpq_mat,
            hamiltonian.Sset,
            hamiltonian.Qplus,
        )

        e2b = ecoul + exx

        energy = xp.zeros((nwalkers, 3), dtype=numpy.complex128)
        energy[:, 0] = e1b + e2b
        energy[:, 1] = e1b
        energy[:, 2] = e2b
    else:
        ghalfa = walkers.Ghalfa.reshape(nwalkers, nk, nalpha, nk, nbasis)
        ghalfb = walkers.Ghalfb.reshape(nwalkers, nk, nbeta, nk, nbasis)
        ghalfaT = walkers.Ghalfa.transpose(0, 2, 1).reshape(nwalkers, nk, nbasis, nk, nalpha)
        ghalfbT = walkers.Ghalfb.transpose(0, 2, 1).reshape(nwalkers, nk, nbasis, nk, nbeta)

        diagGhalfa = numpy.zeros((nwalkers, nk, nalpha, nbasis), dtype=numpy.complex128)
        diagGhalfb = numpy.zeros((nwalkers, nk, nbeta, nbasis), dtype=numpy.complex128)
        for ik in range(nk):
            diagGhalfa[:, ik, :, :] = ghalfa[:, ik, :, ik, :]
            diagGhalfb[:, ik, :, :] = ghalfb[:, ik, :, ik, :]
        e1b = numpy.einsum(
            "wkip, kip -> w", diagGhalfa, trial._rH1a
        )  # Ghalfa.dot(trial._rH1a.ravel())
        e1b += numpy.einsum("wkip, kip -> w", diagGhalfb, trial._rH1b)
        e1b /= nk
        e1b += hamiltonian.ecore

        ghalfa = ghalfa.transpose(1, 3, 0, 2, 4).copy()  # nk, nk, nw, nalpha, nbasis
        ghalfb = ghalfb.transpose(1, 3, 0, 2, 4).copy()  # nk, nk, nw, nbeta, nbasis
        ghalfaTcoul = ghalfaT.transpose(1, 3, 0, 2, 4).copy()  # nk, nk, nw, nbasis, nalpha
        ghalfbTcoul = ghalfbT.transpose(1, 3, 0, 2, 4).copy()  # nk, nk, nw, nbasis, nbeta
        ghalfaTx = ghalfaT.transpose(1, 3, 2, 4, 0).copy()  # nk, nk, nbasis, nalpha, nw
        ghalfbTx = ghalfbT.transpose(1, 3, 2, 4, 0).copy()  # nk, nk, nbasis, nbeta, nw

        ecoul = kpt_symmchol_ecoul_kernel_uhf(
            trial._rchola,
            trial._rcholb,
            trial._rcholbara,
            trial._rcholbarb,
            ghalfa,
            ghalfb,
            ghalfaTcoul,
            ghalfbTcoul,
            hamiltonian.ikpq_mat,
            hamiltonian.Sset,
            hamiltonian.Qplus,
        )

        exxa = kpt_symmchol_exx_kernel(
            trial._rchola,
            trial._rcholbara,
            ghalfa,
            ghalfaTx,
            hamiltonian.ikpq_mat,
            hamiltonian.Sset,
            hamiltonian.Qplus,
        )
        exxb = kpt_symmchol_exx_kernel(
            trial._rcholb,
            trial._rcholbarb,
            ghalfb,
            ghalfbTx,
            hamiltonian.ikpq_mat,
            hamiltonian.Sset,
            hamiltonian.Qplus,
        )

        e2b = ecoul + exxa + exxb

        energy = xp.zeros((nwalkers, 3), dtype=numpy.complex128)
        energy[:, 0] = e1b + e2b
        energy[:, 1] = e1b
        energy[:, 2] = e2b

    return energy


