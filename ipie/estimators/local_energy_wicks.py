
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

import numpy
from numba import jit

from ipie.estimators.generic import local_energy_generic_cholesky
from ipie.estimators.local_energy_sd import (
    local_energy_single_det_batch_einsum, local_energy_single_det_uhf_batch,
    two_body_energy_uhf)
from ipie.propagation.overlap import (compute_determinants_batched,
                                      get_overlap_one_det_wicks)
from ipie.utils.linalg import minor_mask, minor_mask4

try:
    from ipie.propagation.wicks_kernels import (get_cofactor_matrix_4_batched,
                                                get_cofactor_matrix_batched,
                                                get_det_matrix_batched)
except ImportError:
    pass

from ipie.estimators.kernels.cpu import wicks as wk


def local_energy_multi_det_trial_wicks_batch(system, ham, walker_batch, trial):
    """Compute local energy for walker batch (all walkers at once).

    Multi determinant case (particle-hole) using Wick's theorem algorithm.

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
    nwalkers = walker_batch.nwalkers
    nbasis = ham.nbasis
    nchol = ham.nchol
    Ga = walker_batch.Ga.reshape((nwalkers, nbasis * nbasis))
    Gb = walker_batch.Gb.reshape((nwalkers, nbasis * nbasis))
    e1bs = Ga.dot(ham.H1[0].ravel()) + Gb.dot(ham.H1[1].ravel()) + ham.ecore

    e2bs = []
    for iwalker in range(nwalkers):
        ovlpa0 = walker_batch.det_ovlpas[iwalker, 0]
        ovlpb0 = walker_batch.det_ovlpbs[iwalker, 0]
        ovlp0 = ovlpa0 * ovlpb0
        ovlp = walker_batch.ovlp[iwalker]

        # useful variables
        G0a = walker_batch.G0a[iwalker]
        G0b = walker_batch.G0b[iwalker]
        G0Ha = walker_batch.Ghalfa[iwalker]
        G0Hb = walker_batch.Ghalfb[iwalker]
        Q0a = walker_batch.Q0a[iwalker]
        Q0b = walker_batch.Q0b[iwalker]
        CIa = walker_batch.CIa[iwalker]
        CIb = walker_batch.CIb[iwalker]
        G0 = [G0a, G0b]

        # contribution 1 (disconnected)
        cont1 = local_energy_generic_cholesky(system, ham, G0)[2]

        # contribution 2 (half-connected, two-leg, one-body-like)
        # First, Coulomb-like term
        P0 = G0[0] + G0[1]
        Xa = ham.chol_vecs.T.dot(
            G0[0].ravel()
        )  # numpy.einsum("m,xm->x", G0[0].ravel(), ham.chol_vecs)
        Xb = ham.chol_vecs.T.dot(
            G0[1].ravel()
        )  # numpy.einsum("m,xm->x", G0[1].ravel(), ham.chol_vecs)

        LXa = numpy.einsum("mx,x->m", ham.chol_vecs, Xa, optimize=True)
        LXb = numpy.einsum("mx,x->m", ham.chol_vecs, Xb, optimize=True)
        LXa = LXa.reshape((nbasis, nbasis))
        LXb = LXb.reshape((nbasis, nbasis))

        # useful intermediate
        QCIGa = Q0a.dot(CIa).dot(G0a)
        QCIGb = Q0b.dot(CIb).dot(G0b)

        cont2_Jaa = numpy.sum(QCIGa * LXa)
        cont2_Jbb = numpy.sum(QCIGb * LXb)
        cont2_Jab = numpy.sum(QCIGb * LXa) + numpy.sum(QCIGa * LXb)
        cont2_J = cont2_Jaa + cont2_Jbb + cont2_Jab
        cont2_J *= ovlp0 / ovlp

        # Second, Exchange-like term
        cont2_Kaa = 0.0 + 0.0j
        cont2_Kbb = 0.0 + 0.0j
        for x in range(nchol):
            Lmn = ham.chol_vecs[:, x].reshape((nbasis, nbasis))
            LGL = Lmn.dot(G0a.T).dot(Lmn)
            cont2_Kaa -= numpy.sum(LGL * QCIGa)

            LGL = Lmn.dot(G0b.T).dot(Lmn)
            cont2_Kbb -= numpy.sum(LGL * QCIGb)

        cont2_Kaa *= ovlp0 / ovlp
        cont2_Kbb *= ovlp0 / ovlp

        cont2_K = cont2_Kaa + cont2_Kbb

        Laa = numpy.einsum(
            "iq,pj,ijx->qpx",
            Q0a,
            G0a,
            ham.chol_vecs.reshape((nbasis, nbasis, nchol)),
            optimize=True,
        )
        Lbb = numpy.einsum(
            "iq,pj,ijx->qpx",
            Q0b,
            G0b,
            ham.chol_vecs.reshape((nbasis, nbasis, nchol)),
            optimize=True,
        )

        cont3 = 0.0 + 0.0j

        for jdet in range(1, trial.ndets):

            nex_a = len(trial.cre_a[jdet])
            nex_b = len(trial.cre_b[jdet])

            ovlpa, ovlpb = get_overlap_one_det_wicks(
                nex_a,
                trial.cre_a[jdet],
                trial.anh_a[jdet],
                G0a,
                nex_b,
                trial.cre_b[jdet],
                trial.anh_b[jdet],
                G0b,
            )
            ovlpa *= trial.phase_a[jdet]
            ovlpb *= trial.phase_b[jdet]

            det_a = numpy.zeros((nex_a, nex_a), dtype=numpy.complex128)
            det_b = numpy.zeros((nex_b, nex_b), dtype=numpy.complex128)

            for iex in range(nex_a):
                det_a[iex, iex] = G0a[trial.cre_a[jdet][iex], trial.anh_a[jdet][iex]]
                for jex in range(iex + 1, nex_a):
                    det_a[iex, jex] = G0a[
                        trial.cre_a[jdet][iex], trial.anh_a[jdet][jex]
                    ]
                    det_a[jex, iex] = G0a[
                        trial.cre_a[jdet][jex], trial.anh_a[jdet][iex]
                    ]
            for iex in range(nex_b):
                det_b[iex, iex] = G0b[trial.cre_b[jdet][iex], trial.anh_b[jdet][iex]]
                for jex in range(iex + 1, nex_b):
                    det_b[iex, jex] = G0b[
                        trial.cre_b[jdet][iex], trial.anh_b[jdet][jex]
                    ]
                    det_b[jex, iex] = G0b[
                        trial.cre_b[jdet][jex], trial.anh_b[jdet][iex]
                    ]

            cphasea = trial.coeffs[jdet].conj() * trial.phase_a[jdet]
            cphaseb = trial.coeffs[jdet].conj() * trial.phase_b[jdet]
            cphaseab = (
                trial.coeffs[jdet].conj() * trial.phase_a[jdet] * trial.phase_b[jdet]
            )

            if nex_a > 0 and nex_b > 0:  # 2-leg opposite spin block
                if nex_a == 1 and nex_b == 1:
                    p = trial.cre_a[jdet][0]
                    q = trial.anh_a[jdet][0]
                    r = trial.cre_b[jdet][0]
                    s = trial.anh_b[jdet][0]
                    cont3 += cphaseab * numpy.dot(Laa[q, p, :], Lbb[s, r, :])
                elif nex_a == 2 and nex_b == 1:
                    p = trial.cre_a[jdet][0]
                    q = trial.anh_a[jdet][0]
                    r = trial.cre_a[jdet][1]
                    s = trial.anh_a[jdet][1]
                    t = trial.cre_b[jdet][0]
                    u = trial.anh_b[jdet][0]
                    cofactor = [
                        cphaseab * G0a[r, s],
                        cphaseab * G0a[r, q],
                        cphaseab * G0a[p, s],
                        cphaseab * G0a[p, q],
                    ]
                    cont3 += numpy.dot(Laa[q, p, :], Lbb[u, t, :]) * cofactor[0]
                    cont3 -= numpy.dot(Laa[s, p, :], Lbb[u, t, :]) * cofactor[1]
                    cont3 -= numpy.dot(Laa[q, r, :], Lbb[u, t, :]) * cofactor[2]
                    cont3 += numpy.dot(Laa[s, r, :], Lbb[u, t, :]) * cofactor[3]
                elif nex_a == 1 and nex_b == 2:
                    p = trial.cre_a[jdet][0]
                    q = trial.anh_a[jdet][0]
                    r = trial.cre_b[jdet][0]
                    s = trial.anh_b[jdet][0]
                    t = trial.cre_b[jdet][1]
                    u = trial.anh_b[jdet][1]
                    cofactor = [
                        cphaseab * G0b[t, u],
                        cphaseab * G0b[t, s],
                        cphaseab * G0b[r, u],
                        cphaseab * G0b[r, s],
                    ]
                    cont3 += numpy.dot(Laa[q, p, :], Lbb[s, r, :]) * cofactor[0]
                    cont3 -= numpy.dot(Laa[q, p, :], Lbb[u, r, :]) * cofactor[1]
                    cont3 -= numpy.dot(Laa[q, p, :], Lbb[s, t, :]) * cofactor[2]
                    cont3 += numpy.dot(Laa[q, p, :], Lbb[u, t, :]) * cofactor[3]
                elif nex_a == 2 and nex_b == 2:
                    p = trial.cre_a[jdet][0]
                    q = trial.anh_a[jdet][0]
                    r = trial.cre_a[jdet][1]
                    s = trial.anh_a[jdet][1]

                    t = trial.cre_b[jdet][0]
                    u = trial.anh_b[jdet][0]
                    v = trial.cre_b[jdet][1]
                    w = trial.anh_b[jdet][1]
                    cofactor = [
                        cphaseab * G0a[r, s] * G0b[v, w],
                        cphaseab * G0a[r, q] * G0b[v, w],
                        cphaseab * G0a[p, s] * G0b[v, w],
                        cphaseab * G0a[p, q] * G0b[v, w],
                        cphaseab * G0a[r, s] * G0b[t, u],
                        cphaseab * G0a[r, q] * G0b[t, u],
                        cphaseab * G0a[p, s] * G0b[t, u],
                        cphaseab * G0a[p, q] * G0b[t, u],
                        cphaseab * G0a[r, s] * G0b[v, u],
                        cphaseab * G0a[r, q] * G0b[v, u],
                        cphaseab * G0a[p, s] * G0b[v, u],
                        cphaseab * G0a[p, q] * G0b[v, u],
                        cphaseab * G0a[r, s] * G0b[t, w],
                        cphaseab * G0a[r, q] * G0b[t, w],
                        cphaseab * G0a[p, s] * G0b[t, w],
                        cphaseab * G0a[p, q] * G0b[t, w],
                    ]
                    cont3 += numpy.dot(Laa[q, p, :], Lbb[u, t, :]) * cofactor[0]
                    cont3 -= numpy.dot(Laa[s, p, :], Lbb[u, t, :]) * cofactor[1]
                    cont3 -= numpy.dot(Laa[q, r, :], Lbb[u, t, :]) * cofactor[2]
                    cont3 += numpy.dot(Laa[s, r, :], Lbb[u, t, :]) * cofactor[3]

                    cont3 += numpy.dot(Laa[q, p, :], Lbb[w, v, :]) * cofactor[4]
                    cont3 -= numpy.dot(Laa[s, p, :], Lbb[w, v, :]) * cofactor[5]
                    cont3 -= numpy.dot(Laa[q, r, :], Lbb[w, v, :]) * cofactor[6]
                    cont3 += numpy.dot(Laa[s, r, :], Lbb[w, v, :]) * cofactor[7]

                    cont3 -= numpy.dot(Laa[q, p, :], Lbb[w, t, :]) * cofactor[8]
                    cont3 += numpy.dot(Laa[s, p, :], Lbb[w, t, :]) * cofactor[9]
                    cont3 += numpy.dot(Laa[q, r, :], Lbb[w, t, :]) * cofactor[10]
                    cont3 -= numpy.dot(Laa[s, r, :], Lbb[w, t, :]) * cofactor[11]

                    cont3 -= numpy.dot(Laa[q, p, :], Lbb[u, v, :]) * cofactor[12]
                    cont3 += numpy.dot(Laa[s, p, :], Lbb[u, v, :]) * cofactor[13]
                    cont3 += numpy.dot(Laa[q, r, :], Lbb[u, v, :]) * cofactor[14]
                    cont3 -= numpy.dot(Laa[s, r, :], Lbb[u, v, :]) * cofactor[15]

                elif nex_a == 3 and nex_b == 1:
                    p = trial.cre_a[jdet][0]
                    q = trial.anh_a[jdet][0]
                    r = trial.cre_a[jdet][1]
                    s = trial.anh_a[jdet][1]
                    t = trial.cre_a[jdet][2]
                    u = trial.anh_a[jdet][2]
                    v = trial.cre_b[jdet][0]
                    w = trial.anh_b[jdet][0]
                    cofactor = [
                        G0a[r, s] * G0a[t, u] - G0a[r, u] * G0a[t, s],
                        G0a[r, q] * G0a[t, u] - G0a[r, u] * G0a[t, q],
                        G0a[r, q] * G0a[t, s] - G0a[r, s] * G0a[t, q],
                        G0a[p, s] * G0a[t, u] - G0a[t, s] * G0a[p, u],
                        G0a[p, q] * G0a[t, u] - G0a[t, q] * G0a[p, u],
                        G0a[p, q] * G0a[t, s] - G0a[t, q] * G0a[p, s],
                        G0a[p, s] * G0a[r, u] - G0a[r, s] * G0a[p, u],
                        G0a[p, q] * G0a[r, u] - G0a[r, q] * G0a[p, u],
                        G0a[p, q] * G0a[r, s] - G0a[r, q] * G0a[p, s],
                    ]
                    cont3 += cphaseab * numpy.dot(Laa[q, p], Lbb[w, v]) * cofactor[0]
                    cont3 -= cphaseab * numpy.dot(Laa[s, p], Lbb[w, v]) * cofactor[1]
                    cont3 += cphaseab * numpy.dot(Laa[u, p], Lbb[w, v]) * cofactor[2]
                    cont3 -= cphaseab * numpy.dot(Laa[q, r], Lbb[w, v]) * cofactor[3]
                    cont3 += cphaseab * numpy.dot(Laa[s, r], Lbb[w, v]) * cofactor[4]
                    cont3 -= cphaseab * numpy.dot(Laa[u, r], Lbb[w, v]) * cofactor[5]
                    cont3 += cphaseab * numpy.dot(Laa[q, t], Lbb[w, v]) * cofactor[6]
                    cont3 -= cphaseab * numpy.dot(Laa[s, t], Lbb[w, v]) * cofactor[7]
                    cont3 += cphaseab * numpy.dot(Laa[u, t], Lbb[w, v]) * cofactor[8]
                elif nex_a == 1 and nex_b == 3:
                    p = trial.cre_b[jdet][0]
                    q = trial.anh_b[jdet][0]
                    r = trial.cre_b[jdet][1]
                    s = trial.anh_b[jdet][1]
                    t = trial.cre_b[jdet][2]
                    u = trial.anh_b[jdet][2]
                    v = trial.cre_a[jdet][0]
                    w = trial.anh_a[jdet][0]
                    cofactor = [
                        G0b[r, s] * G0b[t, u] - G0b[r, u] * G0b[t, s],
                        G0b[r, q] * G0b[t, u] - G0b[r, u] * G0b[t, q],
                        G0b[r, q] * G0b[t, s] - G0b[r, s] * G0b[t, q],
                        G0b[p, s] * G0b[t, u] - G0b[t, s] * G0b[p, u],
                        G0b[p, q] * G0b[t, u] - G0b[t, q] * G0b[p, u],
                        G0b[p, q] * G0b[t, s] - G0b[t, q] * G0b[p, s],
                        G0b[p, s] * G0b[r, u] - G0b[r, s] * G0b[p, u],
                        G0b[p, q] * G0b[r, u] - G0b[r, q] * G0b[p, u],
                        G0b[p, q] * G0b[r, s] - G0b[r, q] * G0b[p, s],
                    ]

                    cont3 += cphaseab * numpy.dot(Lbb[q, p], Laa[w, v]) * cofactor[0]
                    cont3 -= cphaseab * numpy.dot(Lbb[s, p], Laa[w, v]) * cofactor[1]
                    cont3 += cphaseab * numpy.dot(Lbb[u, p], Laa[w, v]) * cofactor[2]
                    cont3 -= cphaseab * numpy.dot(Lbb[q, r], Laa[w, v]) * cofactor[3]
                    cont3 += cphaseab * numpy.dot(Lbb[s, r], Laa[w, v]) * cofactor[4]
                    cont3 -= cphaseab * numpy.dot(Lbb[u, r], Laa[w, v]) * cofactor[5]
                    cont3 += cphaseab * numpy.dot(Lbb[q, t], Laa[w, v]) * cofactor[6]
                    cont3 -= cphaseab * numpy.dot(Lbb[s, t], Laa[w, v]) * cofactor[7]
                    cont3 += cphaseab * numpy.dot(Lbb[u, t], Laa[w, v]) * cofactor[8]

                else:
                    cofactor_a = numpy.zeros(
                        (nex_a - 1, nex_a - 1), dtype=numpy.complex128
                    )
                    cofactor_b = numpy.zeros(
                        (nex_b - 1, nex_b - 1), dtype=numpy.complex128
                    )
                    for iex in range(nex_a):
                        for jex in range(nex_a):
                            p = trial.cre_a[jdet][iex]
                            q = trial.anh_a[jdet][jex]
                            cofactor_a[:, :] = minor_mask(det_a, iex, jex)
                            det_cofactor_a = (-1) ** (iex + jex) * numpy.linalg.det(
                                cofactor_a
                            )
                            for kex in range(nex_b):
                                for lex in range(nex_b):
                                    r = trial.cre_b[jdet][kex]
                                    s = trial.anh_b[jdet][lex]
                                    cofactor_b[:, :] = minor_mask(det_b, kex, lex)
                                    det_cofactor_b = (-1) ** (
                                        kex + lex
                                    ) * numpy.linalg.det(cofactor_b)
                                    const = cphaseab * det_cofactor_a * det_cofactor_b
                                    cont3 += (
                                        numpy.dot(Laa[q, p, :], Lbb[s, r, :]) * const
                                    )

            if nex_a == 2:  # 4-leg same spin block aaaa
                p = trial.cre_a[jdet][0]
                q = trial.anh_a[jdet][0]
                r = trial.cre_a[jdet][1]
                s = trial.anh_a[jdet][1]
                const = cphasea * ovlpb
                cont3 += (
                    numpy.dot(Laa[q, p, :], Laa[s, r, :])
                    - numpy.dot(Laa[q, r, :], Laa[s, p, :])
                ) * const
            elif nex_a > 2:
                cofactor = numpy.zeros((nex_a - 2, nex_a - 2), dtype=numpy.complex128)
                for iex in range(nex_a):
                    for jex in range(nex_a):
                        p = trial.cre_a[jdet][iex]
                        q = trial.anh_a[jdet][jex]
                        for kex in range(iex + 1, nex_a):
                            for lex in range(jex + 1, nex_a):
                                r = trial.cre_a[jdet][kex]
                                s = trial.anh_a[jdet][lex]
                                cofactor[:, :] = minor_mask4(det_a, iex, jex, kex, lex)
                                det_cofactor = (-1) ** (
                                    kex + lex + iex + jex
                                ) * numpy.linalg.det(cofactor)
                                const = cphasea * det_cofactor * ovlpb
                                cont3 += (
                                    numpy.dot(Laa[q, p, :], Laa[s, r, :])
                                    - numpy.dot(Laa[q, r, :], Laa[s, p, :])
                                ) * const

            if nex_b == 2:  # 4-leg same spin block bbbb
                p = trial.cre_b[jdet][0]
                q = trial.anh_b[jdet][0]
                r = trial.cre_b[jdet][1]
                s = trial.anh_b[jdet][1]
                const = cphaseb * ovlpa
                cont3 += (
                    numpy.dot(Lbb[q, p, :], Lbb[s, r, :])
                    - numpy.dot(Lbb[q, r, :], Lbb[s, p, :])
                ) * const

            elif nex_b > 2:
                cofactor = numpy.zeros((nex_b - 2, nex_b - 2), dtype=numpy.complex128)
                for iex in range(nex_b):
                    for jex in range(nex_b):
                        p = trial.cre_b[jdet][iex]
                        q = trial.anh_b[jdet][jex]
                        for kex in range(iex + 1, nex_b):
                            for lex in range(jex + 1, nex_b):
                                r = trial.cre_b[jdet][kex]
                                s = trial.anh_b[jdet][lex]
                                cofactor[:, :] = minor_mask4(det_b, iex, jex, kex, lex)
                                det_cofactor = (-1) ** (
                                    kex + lex + iex + jex
                                ) * numpy.linalg.det(cofactor)
                                const = cphaseb * det_cofactor * ovlpa
                                cont3 += (
                                    numpy.dot(Lbb[q, p, :], Lbb[s, r, :])
                                    - numpy.dot(Lbb[q, r, :], Lbb[s, p, :])
                                ) * const
        cont3 *= ovlp0 / ovlp

        e2bs += [cont1 + cont2_J + cont2_K + cont3]

    e2bs = numpy.array(e2bs, dtype=numpy.complex128)

    etot = e1bs + e2bs

    energy = numpy.zeros((walker_batch.nwalkers, 3), dtype=numpy.complex128)
    energy[:, 0] = etot
    energy[:, 1] = e1bs
    energy[:, 2] = e2bs
    return energy


@jit(nopython=True, fastmath=True)
def build_Laa(Q0a, Q0b, G0a, G0b, chol, Laa, Lbb):
    naux = chol.shape[-1]
    nbsf = Q0a.shape[1]
    nwalkers = G0a.shape[0]
    Lx = chol.transpose((2, 1, 0)).copy()
    for iw in range(nwalkers):
        G0a_real = G0a[iw].real.copy()
        G0a_imag = G0a[iw].imag.copy()
        G0b_real = G0b[iw].real.copy()
        G0b_imag = G0b[iw].imag.copy()
        for x in range(naux):
            T1 = numpy.dot(G0a_real, Lx[x]) + 1j * numpy.dot(G0a_imag, Lx[x])
            Laa[iw, :, :, x] = numpy.dot(T1, Q0a[iw]).T
            # pj,ji->pi,iq->pq
            T1 = numpy.dot(G0b_real, Lx[x]) + 1j * numpy.dot(G0b_imag, Lx[x])
            Lbb[iw, :, :, x] = numpy.dot(T1, Q0b[iw]).T


@jit(nopython=True, fastmath=True)
def build_contributions12(
    rchol_a,
    rchol_b,
    rchol_act_a,
    rchol_act_b,
    theta_a,
    theta_b,
    CI_a,
    CI_b,
    Lvo_a,
    Lvo_b,
):
    """Build contributions one and two for wicks local energy.

    A bit messy but faster to reuse intermediates for reference and contribution
    2 (half-connected, two-leg, one-body-like terms).


    Parameters
    ----------

    Returns
    -------
    cont1 : numpy.array
        Reference determinant contribution (exchange + coulomb) (nwalker,)
    cont2 : numpy.array
        Half-connected, two-leg one-body-like term (nwalker,)
    """
    nchol = rchol_a.shape[0]
    nwalkers = theta_a.shape[0]
    nbasis = theta_a.shape[2]
    cont1_J = numpy.zeros(nwalkers, dtype=numpy.complex128)
    cont2_J = numpy.zeros(nwalkers, dtype=numpy.complex128)
    cont1_K = numpy.zeros(nwalkers, dtype=numpy.complex128)
    cont2_K = numpy.zeros(nwalkers, dtype=numpy.complex128)
    nact = CI_a.shape[1]
    # assert nact == CI_a.shape[1]
    Q = numpy.zeros((nact, nbasis), dtype=numpy.complex128)
    # X = numpy.zeros((2,nchol), dtype=numpy.complex128)
    X = numpy.zeros(nchol, dtype=numpy.complex128)
    F = numpy.zeros(nchol, dtype=numpy.complex128)
    for iw in range(nwalkers):
        # alpha slices
        X.fill(0.0 + 0.0j)
        F.fill(0.0 + 0.0j)
        nocc_a = theta_a.shape[1]
        nocc_act_a = CI_a.shape[2]
        nfrozen_a = nocc_a - nocc_act_a
        # very messy but numba won't let us use lists for the moment
        theta_act_a = theta_a[iw, :, nfrozen_a : nfrozen_a + nact].copy()
        theta_act_real_a = theta_act_a.real.copy()
        theta_act_imag_a = theta_act_a.imag.copy()
        theta_occ_a = theta_a[iw].copy()
        theta_occ_real_a = theta_occ_a.real.copy()
        theta_occ_imag_a = theta_occ_a.imag.copy()
        nocc_b = theta_b.shape[1]
        nocc_act_b = CI_b.shape[2]
        nfrozen_b = nocc_b - nocc_act_b
        theta_act_b = theta_b[iw, :, nfrozen_b : nfrozen_b + nact].copy()
        theta_act_real_b = theta_act_b.real.copy()
        theta_act_imag_b = theta_act_b.imag.copy()
        theta_occ_b = theta_b[iw].copy()
        theta_occ_real_b = theta_occ_b.real.copy()
        theta_occ_imag_b = theta_occ_b.imag.copy()
        for x in range(nchol):
            # alpha contributions
            # T[a,b] = rchol[b,q] theta[a,q]
            T = numpy.dot(
                theta_occ_real_a, rchol_a[x].reshape((nocc_a, nbasis)).T
            ) + 1j * numpy.dot(theta_occ_imag_a, rchol_a[x].reshape((nocc_a, nbasis)).T)
            exx = numpy.dot(T.ravel(), T.T.ravel())
            X[x] += numpy.trace(T)
            cont1_K[iw] += -0.5 * exx
            # add exchange contribution
            # Ttilde[t,r] = theta_{c,t} rchol[c,r]
            # O(X N A_v M)
            Lut_1 = numpy.dot(
                theta_occ_real_a,
                rchol_act_a[x].reshape(nact, nbasis).T,
            ) + 1j * numpy.dot(
                theta_occ_imag_a,
                rchol_act_a[x].reshape(nact, nbasis).T,
            )
            Lut_2 = numpy.dot(T, theta_act_a)
            # # index a gets contracted with CI tensor later so only need the
            # # occupied orbitals in the active space not the full set of occupied.
            Lut = (Lut_1 - Lut_2).T.copy()
            # # (nvir_act x nocc) x (nocc x nocc_act)
            # O(X A_v N A_v)
            A = numpy.dot(Lut, T[nfrozen_a : nfrozen_a + nocc_act_a, :].T.copy())
            Lvo_a[iw, x] = Lut[:, nfrozen_a : nfrozen_a + nocc_act_a]
            # # nvir_act x nocc_act
            cont2_K[iw] -= numpy.sum(A * CI_a[iw])
            F[x] += numpy.sum(Lvo_a[iw, x] * CI_a[iw])
            # # beta contributions
            T = numpy.dot(
                theta_occ_real_b, rchol_b[x].reshape((nocc_b, nbasis)).T
            ) + 1j * numpy.dot(theta_occ_imag_b, rchol_b[x].reshape((nocc_b, nbasis)).T)
            exx = numpy.dot(T.ravel(), T.T.ravel())
            X[x] += numpy.trace(T)
            cont1_K[iw] += -0.5 * exx
            # add exchange contribution
            # Ttilde[t,r] = theta_{c,t} rchol[c,r]
            Lut_1 = numpy.dot(
                theta_occ_real_b,
                rchol_act_b[x].reshape(nact, nbasis).T,
            ) + 1j * numpy.dot(
                theta_occ_imag_b,
                rchol_act_b[x].reshape(nact, nbasis).T,
            )
            Lut_2 = numpy.dot(T, theta_act_b)
            # index a gets contracted with CI tensor later so only need the
            # occupied orbitals in the active space not the full set of occupied.
            Lut = (Lut_1 - Lut_2).T.copy()
            # (nvir_act x nocc) x (nocc x nocc_act)
            A = numpy.dot(Lut, T[nfrozen_b : nfrozen_b + nocc_act_b, :].copy().T)
            Lvo_b[iw, x] = Lut[:, nfrozen_b : nfrozen_b + nocc_act_b]
            # nvir_act x nocc_act
            cont2_K[iw] -= numpy.sum(A * CI_b[iw])
            F[x] += numpy.sum(Lvo_b[iw, x] * CI_b[iw])
        cont1_J[iw] += 0.5 * numpy.dot(X, X)
        cont2_J[iw] += numpy.dot(F, X)

    return cont1_J + cont1_K, cont2_J + cont2_K


def local_energy_multi_det_trial_wicks_batch_opt_chunked(
    system, ham, walker_batch, trial, max_mem=2.0
):
    """Compute local energy for walker batch (all walkers at once).

    Multi determinant case (particle-hole) using Wick's theorem algorithm.

    Optimized algorithm that "batches" over walkers / determinants.

    Chunks over determinants necessary for large expansions / larger system
    sizes.

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
        Max memory used for intermediate buffer in GB. Optional. Default 2 GB.

    Returns
    -------
    local_energy : np.ndarray
        Total, one-body and two-body energies.
    """
    nwalkers = walker_batch.nwalkers
    nbasis = ham.nbasis
    nchol = ham.nchol
    nalpha = system.nup
    nbeta = system.ndown
    Ga = walker_batch.Ga.reshape((nwalkers, nbasis * nbasis))
    Gb = walker_batch.Gb.reshape((nwalkers, nbasis * nbasis))
    e1b = Ga.dot(ham.H1[0].ravel()) + Gb.dot(ham.H1[1].ravel()) + ham.ecore

    ovlpa0 = walker_batch.det_ovlpas[:, 0]
    ovlpb0 = walker_batch.det_ovlpbs[:, 0]
    ovlp0 = ovlpa0 * ovlpb0
    ovlp = walker_batch.ovlp

    # useful variables
    G0a = walker_batch.G0a
    G0b = walker_batch.G0b
    G0Ha = walker_batch.Ghalfa
    G0Hb = walker_batch.Ghalfb
    Q0a = walker_batch.Q0a
    Q0b = walker_batch.Q0b
    CIa = walker_batch.CIa
    CIb = walker_batch.CIb

    Lvo_a = numpy.zeros(
        (nwalkers, nchol, trial.nact, trial.nocc_alpha), dtype=numpy.complex128
    )
    Lvo_b = numpy.zeros(
        (nwalkers, nchol, trial.nact, trial.nocc_beta), dtype=numpy.complex128
    )
    cont1, cont2 = build_contributions12(
        trial._rchola,
        trial._rcholb,
        trial._rchola_act,
        trial._rcholb_act,
        walker_batch.Ghalfa,
        walker_batch.Ghalfb,
        walker_batch.CIa,
        walker_batch.CIb,
        Lvo_a,
        Lvo_b,
    )
    cont2 = (ovlp0 / ovlp) * cont2

    Laa = Lvo_a.transpose((0, 2, 3, 1)).copy()
    Lbb = Lvo_b.transpose((0, 2, 3, 1)).copy()

    dets_a_full, dets_b_full = compute_determinants_batched(
        walker_batch.Ghalfa, walker_batch.Ghalfb, trial
    )
    ndets = len(trial.coeffs)
    cphase_a = trial.coeffs.conj() * trial.phase_a
    cphase_b = trial.coeffs.conj() * trial.phase_b
    ovlpa = dets_a_full * trial.phase_a[None, :]
    ovlpb = dets_b_full * trial.phase_b[None, :]
    c_phasea_ovlpb = cphase_a[None, :] * ovlpb
    c_phaseb_ovlpa = cphase_b[None, :] * ovlpa
    cphase_ab = cphase_a * trial.phase_b
    start = time.time()
    cont3 = numpy.zeros_like(cont2)
    det_sizes_a = max(
        [
            max(
                [
                    len(trial.cre_ex_a_chunk[ichunk][i]) * i * i
                    for i in range(1, trial.max_excite + 1)
                ]
            )
            for ichunk in range(trial.ndet_chunks)
        ]
    )
    det_sizes_b = max(
        max(
            [
                len(trial.cre_ex_b_chunk[ichunk][i]) * i * i
                for i in range(1, trial.max_excite + 1)
            ]
        )
        for ichunk in range(trial.ndet_chunks)
    )
    max_size = max(det_sizes_a, det_sizes_b)
    det_mat_buffer = numpy.zeros((2 * nwalkers * max_size), dtype=numpy.complex128)
    # energy_os = numpy.zeros((nwalkers, ndets), dtype=numpy.complex128)
    # energy_ss = numpy.zeros((nwalkers, ndets), dtype=numpy.complex128)
    for ichunk in range(trial.ndet_chunks):
        ndets_chunk = trial.ndets_per_chunk[ichunk]
        alpha_os_buffer = numpy.zeros(
            (nwalkers, ndets_chunk, nchol), dtype=numpy.complex128
        )
        beta_os_buffer = numpy.zeros(
            (nwalkers, ndets_chunk, nchol), dtype=numpy.complex128
        )
        alpha_ss_buffer = numpy.zeros((nwalkers, ndets_chunk), dtype=numpy.complex128)
        beta_ss_buffer = numpy.zeros((nwalkers, ndets_chunk), dtype=numpy.complex128)
        for iexcit in range(1, trial.max_excite + 1):
            ndets_a = len(trial.cre_ex_a_chunk[ichunk][iexcit])
            det_size = (nwalkers, ndets_a, iexcit, iexcit)
            nelem_det = int(numpy.prod(det_size))
            det_mat_a = det_mat_buffer[:nelem_det].reshape(det_size)
            cof_size = (nwalkers, ndets_a, max(iexcit - 1, 1), max(iexcit - 1, 1))
            cofactor_matrix_a = det_mat_buffer[
                nelem_det : nelem_det + numpy.prod(cof_size)
            ].reshape(cof_size)
            _start = time.time()
            if ndets_a > 0:
                wk.build_det_matrix(
                    trial.cre_ex_a_chunk[ichunk][iexcit],
                    trial.anh_ex_a_chunk[ichunk][iexcit],
                    trial.occ_map_a,
                    trial.nfrozen,
                    walker_batch.Ghalfa,
                    det_mat_a,
                )
                if iexcit == 1:
                    _start = time.time()
                    wk.fill_os_singles(
                        trial.cre_ex_a_chunk[ichunk][iexcit],
                        trial.anh_ex_a_chunk[ichunk][iexcit],
                        trial.occ_map_a,
                        trial.nfrozen,
                        Laa,
                        alpha_os_buffer,
                        trial.slices_alpha_chunk[ichunk][1],
                    )
                elif iexcit == 2:
                    _start = time.time()
                    wk.fill_os_doubles(
                        trial.cre_ex_a_chunk[ichunk][iexcit],
                        trial.anh_ex_a_chunk[ichunk][iexcit],
                        trial.occ_map_a,
                        trial.nfrozen,
                        G0a,
                        Laa,
                        alpha_os_buffer,
                        trial.slices_alpha_chunk[ichunk][2],
                    )
                elif iexcit == 3:
                    _start = time.time()
                    wk.fill_os_triples(
                        trial.cre_ex_a_chunk[ichunk][iexcit],
                        trial.anh_ex_a_chunk[ichunk][iexcit],
                        trial.occ_map_a,
                        trial.nfrozen,
                        G0a,
                        Laa,
                        alpha_os_buffer,
                        trial.slices_alpha_chunk[ichunk][3],
                    )
                else:
                    _start = time.time()
                    wk.fill_os_nfold(
                        trial.cre_ex_a_chunk[ichunk][iexcit],
                        trial.anh_ex_a_chunk[ichunk][iexcit],
                        trial.occ_map_a,
                        det_mat_a,
                        cofactor_matrix_a,
                        Laa,
                        alpha_os_buffer,
                        trial.slices_alpha_chunk[ichunk][iexcit],
                    )
                if iexcit >= 2 and ndets_a > 0:
                    if iexcit == 2:
                        _start = time.time()
                        wk.get_ss_doubles(
                            trial.cre_ex_a_chunk[ichunk][iexcit],
                            trial.anh_ex_a_chunk[ichunk][iexcit],
                            trial.occ_map_a,
                            Laa,
                            alpha_ss_buffer,
                            trial.slices_alpha_chunk[ichunk][iexcit],
                        )
                    else:
                        _start = time.time()
                        wk.get_ss_nfold(
                            trial.cre_ex_a_chunk[ichunk][iexcit],
                            trial.anh_ex_a_chunk[ichunk][iexcit],
                            trial.occ_map_a,
                            det_mat_a,
                            cofactor_matrix_a[
                                :, :, : max(iexcit - 2, 1), : max(iexcit - 2, 1)
                            ],
                            Laa,
                            alpha_ss_buffer,
                            trial.slices_alpha_chunk[ichunk][iexcit],
                        )
            ndets_b = len(trial.cre_ex_b_chunk[ichunk][iexcit])
            det_size = (nwalkers, ndets_b, iexcit, iexcit)
            nelem_det = int(numpy.prod(det_size))
            det_mat_b = det_mat_buffer[:nelem_det].reshape(det_size)
            cof_size = (nwalkers, ndets_b, max(iexcit - 1, 1), max(iexcit - 1, 1))
            cofactor_matrix_b = det_mat_buffer[
                nelem_det : nelem_det + numpy.prod(cof_size)
            ].reshape(cof_size)
            if ndets_b > 0:
                wk.build_det_matrix(
                    trial.cre_ex_b_chunk[ichunk][iexcit],
                    trial.anh_ex_b_chunk[ichunk][iexcit],
                    trial.occ_map_b,
                    trial.nfrozen,
                    walker_batch.Ghalfb,
                    det_mat_b,
                )
                if iexcit == 1:
                    wk.fill_os_singles(
                        trial.cre_ex_b_chunk[ichunk][iexcit],
                        trial.anh_ex_b_chunk[ichunk][iexcit],
                        trial.occ_map_b,
                        trial.nfrozen,
                        Lbb,
                        beta_os_buffer,
                        trial.slices_beta_chunk[ichunk][1],
                    )
                elif iexcit == 2:
                    wk.fill_os_doubles(
                        trial.cre_ex_b_chunk[ichunk][iexcit],
                        trial.anh_ex_b_chunk[ichunk][iexcit],
                        trial.occ_map_b,
                        trial.nfrozen,
                        G0b,
                        Lbb,
                        beta_os_buffer,
                        trial.slices_beta_chunk[ichunk][2],
                    )
                elif iexcit == 3:
                    wk.fill_os_triples(
                        trial.cre_ex_b_chunk[ichunk][iexcit],
                        trial.anh_ex_b_chunk[ichunk][iexcit],
                        trial.occ_map_b,
                        trial.nfrozen,
                        G0b,
                        Lbb,
                        beta_os_buffer,
                        trial.slices_beta_chunk[ichunk][3],
                    )
                else:
                    wk.fill_os_nfold(
                        trial.cre_ex_b_chunk[ichunk][iexcit],
                        trial.anh_ex_b_chunk[ichunk][iexcit],
                        trial.occ_map_b,
                        det_mat_b,
                        cofactor_matrix_b,
                        Lbb,
                        beta_os_buffer,
                        trial.slices_beta_chunk[ichunk][iexcit],
                    )
            if iexcit >= 2 and ndets_b > 0:
                if iexcit == 2:
                    wk.get_ss_doubles(
                        trial.cre_ex_b_chunk[ichunk][iexcit],
                        trial.anh_ex_b_chunk[ichunk][iexcit],
                        trial.occ_map_b,
                        Lbb,
                        beta_ss_buffer,
                        trial.slices_beta_chunk[ichunk][iexcit],
                    )
                else:
                    wk.get_ss_nfold(
                        trial.cre_ex_b_chunk[ichunk][iexcit],
                        trial.anh_ex_b_chunk[ichunk][iexcit],
                        trial.occ_map_b,
                        det_mat_b,
                        cofactor_matrix_b[
                            :, :, : max(iexcit - 2, 1), : max(iexcit - 2, 1)
                        ],
                        Lbb,
                        beta_ss_buffer,
                        trial.slices_beta_chunk[ichunk][iexcit],
                    )

        # orders buffer by determinant index not alpha/beta index
        ma = trial.excit_map_a_chunk[ichunk]
        mb = trial.excit_map_b_chunk[ichunk]
        # 1 for reference determinant
        start = 1 + ichunk * trial.ndets_chunk_max
        # inclusive of endpoint
        end = min(1 + (ichunk + 1) * trial.ndets_chunk_max, trial.ndets)
        energy_os = numpy.einsum(
            "wJx,wJx,J->w",
            alpha_os_buffer[:, ma],
            beta_os_buffer[:, mb],
            cphase_ab[start:end],
            optimize=True,
        )
        energy_ss = numpy.einsum(
            "wJ,wJ->w",
            alpha_ss_buffer[:, ma],
            c_phasea_ovlpb[:, start:end],
            optimize=True,
        )
        energy_ss += numpy.einsum(
            "wJ,wJ->w",
            beta_ss_buffer[:, mb],
            c_phaseb_ovlpa[:, start:end],
            optimize=True,
        )

        cont3 += (energy_os + energy_ss) * (ovlp0 / ovlp)
    e2b = cont1 + cont2 + cont3

    walker_energies = numpy.zeros((nwalkers, 3), dtype=numpy.complex128)
    walker_energies[:, 0] = e1b + e2b
    walker_energies[:, 1] = e1b
    walker_energies[:, 2] = e2b
    return walker_energies


def local_energy_multi_det_trial_wicks_batch_opt(
    system, ham, walker_batch, trial, max_mem=2.0
):
    """Compute local energy for walker batch (all walkers at once).

    Multi determinant case (particle-hole) using Wick's theorem algorithm.

    Optimized algorithm that "batches" over walkers / determinants.

    No chunking over determinants thus very high memory requirements.

    May OOM without warning.

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
        Max memory used for intermediate buffer in GB. Optional. Default 2 GB.

    Returns
    -------
    local_energy : np.ndarray
        Total, one-body and two-body energies.
    """
    nwalkers = walker_batch.nwalkers
    nbasis = ham.nbasis
    nchol = ham.nchol
    nalpha = system.nup
    nbeta = system.ndown
    Ga = walker_batch.Ga.reshape((nwalkers, nbasis * nbasis))
    Gb = walker_batch.Gb.reshape((nwalkers, nbasis * nbasis))
    e1b = Ga.dot(ham.H1[0].ravel()) + Gb.dot(ham.H1[1].ravel()) + ham.ecore

    ovlpa0 = walker_batch.det_ovlpas[:, 0]
    ovlpb0 = walker_batch.det_ovlpbs[:, 0]
    ovlp0 = ovlpa0 * ovlpb0
    ovlp = walker_batch.ovlp

    # useful variables
    G0a = walker_batch.G0a
    G0b = walker_batch.G0b
    G0Ha = walker_batch.Ghalfa
    G0Hb = walker_batch.Ghalfb
    Q0a = walker_batch.Q0a
    Q0b = walker_batch.Q0b
    CIa = walker_batch.CIa
    CIb = walker_batch.CIb

    Lvo_a = numpy.zeros(
        (nwalkers, nchol, trial.nact, trial.nocc_alpha), dtype=numpy.complex128
    )
    Lvo_b = numpy.zeros(
        (nwalkers, nchol, trial.nact, trial.nocc_beta), dtype=numpy.complex128
    )
    cont1, cont2 = build_contributions12(
        trial._rchola,
        trial._rcholb,
        trial._rchola_act,
        trial._rcholb_act,
        walker_batch.Ghalfa,
        walker_batch.Ghalfb,
        walker_batch.CIa,
        walker_batch.CIb,
        Lvo_a,
        Lvo_b,
    )
    cont2 = (ovlp0 / ovlp) * cont2

    Laa = Lvo_a.transpose((0, 2, 3, 1)).copy()
    Lbb = Lvo_b.transpose((0, 2, 3, 1)).copy()

    dets_a_full, dets_b_full = compute_determinants_batched(
        walker_batch.Ghalfa, walker_batch.Ghalfb, trial
    )
    ndets = len(trial.coeffs)
    cphase_a = trial.coeffs.conj() * trial.phase_a
    cphase_b = trial.coeffs.conj() * trial.phase_b
    ovlpa = dets_a_full * trial.phase_a[None, :]
    ovlpb = dets_b_full * trial.phase_b[None, :]
    c_phasea_ovlpb = cphase_a[None, :] * ovlpb
    c_phaseb_ovlpa = cphase_b[None, :] * ovlpa
    cphase_ab = cphase_a * trial.phase_b
    energy_os = numpy.zeros((nwalkers, ndets), dtype=numpy.complex128)
    energy_ss = numpy.zeros((nwalkers, ndets), dtype=numpy.complex128)
    alpha_os_buffer = numpy.zeros((nwalkers, ndets, nchol), dtype=numpy.complex128)
    beta_os_buffer = numpy.zeros((nwalkers, ndets, nchol), dtype=numpy.complex128)
    alpha_ss_buffer = numpy.zeros((nwalkers, ndets), dtype=numpy.complex128)
    beta_ss_buffer = numpy.zeros((nwalkers, ndets), dtype=numpy.complex128)
    start = time.time()
    map_alpha = numpy.concatenate(
        [numpy.array([0], dtype=numpy.int32)] + trial.excit_map_a
    )
    map_beta = numpy.concatenate(
        [numpy.array([0], dtype=numpy.int32)] + trial.excit_map_b
    )
    det_sizes_a = max(
        [len(trial.cre_ex_a[i]) * i * i for i in range(1, trial.max_excite + 1)]
    )
    det_sizes_b = max(
        [len(trial.cre_ex_b[i]) * i * i for i in range(1, trial.max_excite + 1)]
    )
    max_size = max(det_sizes_a, det_sizes_b)
    det_mat_buffer = numpy.zeros((2 * nwalkers * max_size), dtype=numpy.complex128)
    for iexcit in range(1, trial.max_excite + 1):
        ndets_a = len(trial.cre_ex_a[iexcit])
        det_size = (nwalkers, ndets_a, iexcit, iexcit)
        nelem_det = int(numpy.prod(det_size))
        det_mat_a = det_mat_buffer[:nelem_det].reshape(det_size)
        cof_size = (nwalkers, ndets_a, max(iexcit - 1, 1), max(iexcit - 1, 1))
        cofactor_matrix_a = det_mat_buffer[
            nelem_det : nelem_det + numpy.prod(cof_size)
        ].reshape(cof_size)
        _start = time.time()
        if ndets_a > 0:
            wk.build_det_matrix(
                trial.cre_ex_a[iexcit],
                trial.anh_ex_a[iexcit],
                trial.occ_map_a,
                trial.nfrozen,
                walker_batch.Ghalfa,
                det_mat_a,
            )
            if iexcit == 1:
                _start = time.time()
                wk.fill_os_singles(
                    trial.cre_ex_a[iexcit],
                    trial.anh_ex_a[iexcit],
                    trial.occ_map_a,
                    trial.nfrozen,
                    Laa,
                    alpha_os_buffer,
                    trial.slices_alpha[1],
                )
            elif iexcit == 2:
                _start = time.time()
                wk.fill_os_doubles(
                    trial.cre_ex_a[iexcit],
                    trial.anh_ex_a[iexcit],
                    trial.occ_map_a,
                    trial.nfrozen,
                    G0a,
                    Laa,
                    alpha_os_buffer,
                    trial.slices_alpha[2],
                )
            elif iexcit == 3:
                _start = time.time()
                wk.fill_os_triples(
                    trial.cre_ex_a[iexcit],
                    trial.anh_ex_a[iexcit],
                    trial.occ_map_a,
                    trial.nfrozen,
                    G0a,
                    Laa,
                    alpha_os_buffer,
                    trial.slices_alpha[3],
                )
            else:
                _start = time.time()
                wk.fill_os_nfold(
                    trial.cre_ex_a[iexcit],
                    trial.anh_ex_a[iexcit],
                    trial.occ_map_a,
                    det_mat_a,
                    cofactor_matrix_a,
                    Laa,
                    alpha_os_buffer,
                    trial.slices_alpha[iexcit],
                )
            if iexcit >= 2 and ndets_a > 0:
                if iexcit == 2:
                    _start = time.time()
                    wk.get_ss_doubles(
                        trial.cre_ex_a[iexcit],
                        trial.anh_ex_a[iexcit],
                        trial.occ_map_a,
                        Laa,
                        alpha_ss_buffer,
                        trial.slices_alpha[iexcit],
                    )
                else:
                    _start = time.time()
                    wk.get_ss_nfold(
                        trial.cre_ex_a[iexcit],
                        trial.anh_ex_a[iexcit],
                        trial.occ_map_a,
                        det_mat_a,
                        cofactor_matrix_a[
                            :, :, : max(iexcit - 2, 1), : max(iexcit - 2, 1)
                        ],
                        Laa,
                        alpha_ss_buffer,
                        trial.slices_alpha[iexcit],
                    )
        ndets_b = len(trial.cre_ex_b[iexcit])
        det_size = (nwalkers, ndets_b, iexcit, iexcit)
        nelem_det = int(numpy.prod(det_size))
        det_mat_b = det_mat_buffer[:nelem_det].reshape(det_size)
        cof_size = (nwalkers, ndets_b, max(iexcit - 1, 1), max(iexcit - 1, 1))
        cofactor_matrix_b = det_mat_buffer[
            nelem_det : nelem_det + numpy.prod(cof_size)
        ].reshape(cof_size)
        if ndets_b > 0:
            wk.build_det_matrix(
                trial.cre_ex_b[iexcit],
                trial.anh_ex_b[iexcit],
                trial.occ_map_b,
                trial.nfrozen,
                walker_batch.Ghalfb,
                det_mat_b,
            )
            if iexcit == 1:
                wk.fill_os_singles(
                    trial.cre_ex_b[iexcit],
                    trial.anh_ex_b[iexcit],
                    trial.occ_map_b,
                    trial.nfrozen,
                    Lbb,
                    beta_os_buffer,
                    trial.slices_beta[1],
                )
            elif iexcit == 2:
                wk.fill_os_doubles(
                    trial.cre_ex_b[iexcit],
                    trial.anh_ex_b[iexcit],
                    trial.occ_map_b,
                    trial.nfrozen,
                    G0b,
                    Lbb,
                    beta_os_buffer,
                    trial.slices_beta[2],
                )
            elif iexcit == 3:
                wk.fill_os_triples(
                    trial.cre_ex_b[iexcit],
                    trial.anh_ex_b[iexcit],
                    trial.occ_map_b,
                    trial.nfrozen,
                    G0b,
                    Lbb,
                    beta_os_buffer,
                    trial.slices_beta[3],
                )
            else:
                wk.fill_os_nfold(
                    trial.cre_ex_b[iexcit],
                    trial.anh_ex_b[iexcit],
                    trial.occ_map_b,
                    det_mat_b,
                    cofactor_matrix_b,
                    Lbb,
                    beta_os_buffer,
                    trial.slices_beta[iexcit],
                )
        if iexcit >= 2 and ndets_b > 0:
            if iexcit == 2:
                wk.get_ss_doubles(
                    trial.cre_ex_b[iexcit],
                    trial.anh_ex_b[iexcit],
                    trial.occ_map_b,
                    Lbb,
                    beta_ss_buffer,
                    trial.slices_beta[iexcit],
                )
            else:
                wk.get_ss_nfold(
                    trial.cre_ex_b[iexcit],
                    trial.anh_ex_b[iexcit],
                    trial.occ_map_b,
                    det_mat_b,
                    cofactor_matrix_b[:, :, : max(iexcit - 2, 1), : max(iexcit - 2, 1)],
                    Lbb,
                    beta_ss_buffer,
                    trial.slices_beta[iexcit],
                )

    bufferab = numpy.zeros_like(alpha_os_buffer)
    bufferba = numpy.zeros_like(beta_os_buffer)
    bufferbb = numpy.zeros_like(beta_ss_buffer)
    bufferaa = numpy.zeros_like(alpha_ss_buffer)
    bufferab[:, map_alpha, :] = alpha_os_buffer[:, :, :]
    bufferba[:, map_beta, :] = beta_os_buffer[:, :, :]
    bufferbb[:, map_beta] = beta_ss_buffer[:, :]
    bufferaa[:, map_alpha] = alpha_ss_buffer[:, :]
    energy_os = numpy.einsum(
        "wJx,wJx,J->w", bufferab, bufferba, cphase_ab, optimize=True
    )
    energy_ss = numpy.einsum("wJ,wJ->w", bufferaa, c_phasea_ovlpb, optimize=True)
    energy_ss += numpy.einsum("wJ,wJ->w", bufferbb, c_phaseb_ovlpa, optimize=True)

    cont3 = (energy_os + energy_ss) * (ovlp0 / ovlp)
    e2b = cont1 + cont2 + cont3

    walker_energies = numpy.zeros((nwalkers, 3), dtype=numpy.complex128)
    walker_energies[:, 0] = e1b + e2b
    walker_energies[:, 1] = e1b
    walker_energies[:, 2] = e2b
    return walker_energies


# Depracted / unused?
# kernels were jitte with numba for better performance.

def get_same_spin_double_contribution_batched(
    cre, anh, buffer, excit_map, chol_fact, det_sls
):
    p = cre[:, 0]
    q = anh[:, 0]
    r = cre[:, 1]
    s = anh[:, 1]
    contribution = (
        chol_fact[:, q, p] * chol_fact[:, s, r]
        - chol_fact[:, q, r] * chol_fact[:, s, p]
    )
    buffer[:, det_sls] = contribution


def get_same_spin_double_contribution_batched_contr(
    cre, anh, buffer, excit_map, chol_fact, det_sls
):
    p = cre[:, 0]
    q = anh[:, 0]
    r = cre[:, 1]
    s = anh[:, 1]
    contribution = numpy.einsum(
        "wJx,wJx->wJ", chol_fact[:, q, p], chol_fact[:, s, r], optimize=True
    )
    contribution -= numpy.einsum(
        "wJx,wJx->wJ", chol_fact[:, q, r], chol_fact[:, s, p], optimize=True
    )
    buffer[:, det_sls] = contribution


def fill_same_spin_contribution_batched(
    nexcit, cre, anh, dets_mat, cofactor_matrix, chol_fact, buffer, excit_map, det_sls
):
    nwalkers = dets_mat.shape[0]
    ndet_level = dets_mat.shape[1]
    accumulator = numpy.zeros((nwalkers, ndet_level), dtype=numpy.complex128)
    for iex in range(nexcit):
        for jex in range(nexcit):
            p = cre[:, iex]
            q = anh[:, jex]
            for kex in range(iex + 1, nexcit):
                for lex in range(jex + 1, nexcit):
                    r = cre[:, kex]
                    s = anh[:, lex]
                    get_cofactor_matrix_4_batched(
                        nwalkers,
                        ndet_level,
                        nexcit,
                        iex,
                        jex,
                        kex,
                        lex,
                        dets_mat,
                        cofactor_matrix,
                    )
                    det_cofactor = (-1) ** (kex + lex + iex + jex) * numpy.linalg.det(
                        cofactor_matrix
                    )
                    # Nw*Nd
                    dets_contrib = (
                        chol_fact[:, q, p] * chol_fact[:, s, r]
                        - chol_fact[:, q, r] * chol_fact[:, s, p]
                    ) * det_cofactor
                    accumulator += dets_contrib

    buffer[:, det_sls] = accumulator


def fill_same_spin_contribution_batched_contr(
    nexcit, cre, anh, dets_mat, cofactor_matrix, chol_fact, buffer, excit_map, det_sls
):
    nwalkers = dets_mat.shape[0]
    ndet_level = dets_mat.shape[1]
    accumulator = numpy.zeros((nwalkers, ndet_level), dtype=numpy.complex128)
    t_cof = 0.0
    t_det = 0.0
    t_slice = 0.0
    t_eins = 0.0
    for iex in range(nexcit):
        for jex in range(nexcit):
            p = cre[:, iex]
            q = anh[:, jex]
            chol_a = chol_fact[:, q, p]
            for kex in range(iex + 1, nexcit):
                r = cre[:, kex]
                chol_c = chol_fact[:, q, r]
                for lex in range(jex + 1, nexcit):
                    s = anh[:, lex]
                    get_cofactor_matrix_4_batched(
                        nwalkers,
                        ndet_level,
                        nexcit,
                        iex,
                        jex,
                        kex,
                        lex,
                        dets_mat,
                        cofactor_matrix,
                    )
                    det_cofactor = (-1) ** (kex + lex + iex + jex) * numpy.linalg.det(
                        cofactor_matrix
                    )
                    # Nw*Nd
                    chol_b = chol_fact[:, s, r]
                    chol_d = chol_fact[:, s, p]
                    contribution = numpy.einsum(
                        "wJx,wJx->wJ", chol_a, chol_b, optimize=True
                    )
                    contribution -= numpy.einsum(
                        "wJx,wJx->wJ", chol_c, chol_d, optimize=True
                    )
                    # contribution = numpy.einsum('wJx,wJx->wJ', chol_fact[:,q,p], chol_fact[:,s,r], optimize=True)
                    # contribution -= numpy.einsum('wJx,wJx->wJ', chol_fact[:,q,r], chol_fact[:,s,p], optimize=True)
                    # contribution *= det_cofactor
                    accumulator += det_cofactor * contribution

    buffer[:, det_sls] = accumulator


def fill_opp_spin_factors_batched(
    nexcit,
    cre,
    anh,
    excit_map,
    det_matrix,
    cofactor_matrix,
    chol_factor,
    spin_buffer,
    det_sls,
):
    nwalkers = cofactor_matrix.shape[0]
    ndet_level = cofactor_matrix.shape[1]
    accumulator = numpy.zeros((nwalkers, ndet_level), dtype=numpy.complex128)
    for iex in range(nexcit):
        ps = cre[:, iex]
        for jex in range(nexcit):
            qs = anh[:, jex]
            get_cofactor_matrix_batched(
                nwalkers, ndet_level, nexcit, iex, jex, det_matrix, cofactor_matrix
            )
            # Nw x Nd_l
            det_cofactors = (-1) ** (iex + jex) * numpy.linalg.det(cofactor_matrix)
            accumulator += det_cofactors * chol_factor[:, qs, ps]
    spin_buffer[:, det_sls] = accumulator


def fill_opp_spin_factors_batched_chol(
    nexcit,
    cre,
    anh,
    excit_map,
    det_matrix,
    cofactor_matrix,
    chol_factor,
    spin_buffer,
    det_sls,
):
    nwalkers = cofactor_matrix.shape[0]
    ndet_level = cofactor_matrix.shape[1]
    accumulator = numpy.zeros(
        (nwalkers, ndet_level, chol_factor.shape[-1]), dtype=numpy.complex128
    )
    for iex in range(nexcit):
        ps = cre[:, iex]
        for jex in range(nexcit):
            qs = anh[:, jex]
            get_cofactor_matrix_batched(
                nwalkers, ndet_level, nexcit, iex, jex, det_matrix, cofactor_matrix
            )
            # Nw x Nd_l
            det_cofactors = (-1) ** (iex + jex) * numpy.linalg.det(cofactor_matrix)
            accumulator += det_cofactors[:, :, None] * chol_factor[:, qs, ps]
    spin_buffer[:, det_sls] = accumulator


def fill_opp_spin_factors_batched_singles(
    cre, anh, excit_map, chol_factor, spin_buffer, det_sls
):
    p = cre[:, 0]
    q = anh[:, 0]
    start = time.time()
    x = chol_factor[:, q, p]
    start = time.time()
    # tmp = numpy.zeros_like(spin_buffer)
    # tmp_2 = numpy.zeros_like(spin_buffer)
    # tmp[:,excit_map[1],:] = x
    spin_buffer[:, det_sls] = x
    # tmp_2[:,excit_map[1],:] = spin_buffer[:,det_sls]


# @jit(nopython=True, fastmath=True)
# def fill_opp_spin_factors_batched_singles(
# cre,
# anh,
# excit_map,
# chol_factor,
# spin_buffer,
# det_sls):
# ps = cre[:, 0]
# qs = anh[:, 0]
# ndets = ps.shape[0]
# start = det_sls.start
# for idet in range(ndets):
# spin_buffer[:,start+idet] = chol_factor[:, qs[idet], ps[idet]]


def fill_opp_spin_factors_batched_doubles(
    cre, anh, excit_map, G0, chol_factor, spin_buffer, sls
):
    p = cre[:, 0]
    q = anh[:, 0]
    r = cre[:, 1]
    s = anh[:, 1]
    accumulator = chol_factor[:, q, p] * G0[:, r, s]
    accumulator -= chol_factor[:, s, p] * G0[:, r, q]
    accumulator -= chol_factor[:, q, r] * G0[:, p, s]
    accumulator += chol_factor[:, s, r] * G0[:, p, q]
    spin_buffer[:, sls] = accumulator


# @jit(nopython=True, fastmath=True)
# def fill_opp_spin_factors_batched_doubles_chol(
# cre,
# anh,
# excit_map,
# G0,
# chol_factor,
# spin_buffer,
# det_sls):
# start = det_sls.start
# ndets = cre.shape[0]
# for idet in range(ndets):
# p = cre[idet, 0]
# q = anh[idet, 0]
# r = cre[idet, 1]
# s = anh[idet, 1]
# spin_buffer[:, start + idet] = chol_factor[:,q,p] * G0[:,r,s,None]
# spin_buffer[:, start + idet] -= chol_factor[:,s,p] * G0[:,r,q,None]
# spin_buffer[:, start + idet] -= chol_factor[:,q,r] * G0[:,p,s,None]
# spin_buffer[:, start + idet] += chol_factor[:,s,r] * G0[:,p,q,None]


def fill_opp_spin_factors_batched_doubles_chol(
    cre, anh, excit_map, G0, chol_factor, spin_buffer, det_sls
):
    p = cre[:, 0]
    q = anh[:, 0]
    r = cre[:, 1]
    s = anh[:, 1]
    accumulator = chol_factor[:, q, p] * G0[:, r, s, None]
    accumulator -= chol_factor[:, s, p] * G0[:, r, q, None]
    accumulator -= chol_factor[:, q, r] * G0[:, p, s, None]
    accumulator += chol_factor[:, s, r] * G0[:, p, q, None]
    spin_buffer[:, det_sls] = accumulator


def fill_opp_spin_factors_batched_triples(
    cre, anh, excit_map, G0, chol_factor, spin_buffer, det_sls
):
    p = cre[:, 0]
    q = anh[:, 0]
    r = cre[:, 1]
    s = anh[:, 1]
    t = cre[:, 2]
    u = anh[:, 2]
    accumulator = chol_factor[:, q, p] * (
        G0[:, r, s] * G0[:, t, u] - G0[:, r, u] * G0[:, t, s]
    )
    accumulator -= chol_factor[:, s, p] * (
        G0[:, r, q] * G0[:, t, u] - G0[:, r, u] * G0[:, t, q]
    )
    accumulator += chol_factor[:, u, p] * (
        G0[:, r, q] * G0[:, t, s] - G0[:, r, s] * G0[:, t, q]
    )
    accumulator -= chol_factor[:, q, r] * (
        G0[:, p, s] * G0[:, t, u] - G0[:, t, s] * G0[:, p, u]
    )
    accumulator += chol_factor[:, s, r] * (
        G0[:, p, q] * G0[:, t, u] - G0[:, t, q] * G0[:, p, u]
    )
    accumulator -= chol_factor[:, u, r] * (
        G0[:, p, q] * G0[:, t, s] - G0[:, t, q] * G0[:, p, s]
    )
    accumulator += chol_factor[:, q, t] * (
        G0[:, p, s] * G0[:, r, u] - G0[:, r, s] * G0[:, p, u]
    )
    accumulator -= chol_factor[:, s, t] * (
        G0[:, p, q] * G0[:, r, u] - G0[:, r, q] * G0[:, p, u]
    )
    accumulator += chol_factor[:, u, t] * (
        G0[:, p, q] * G0[:, r, s] - G0[:, r, q] * G0[:, p, s]
    )
    spin_buffer[:, det_sls] = accumulator


def fill_opp_spin_factors_batched_triples_chol(
    cre, anh, excit_map, G0, chol_factor, spin_buffer, det_sls
):
    p = cre[:, 0]
    q = anh[:, 0]
    r = cre[:, 1]
    s = anh[:, 1]
    t = cre[:, 2]
    u = anh[:, 2]
    cofac = G0[:, r, s] * G0[:, t, u] - G0[:, r, u] * G0[:, t, s]
    accumulator = chol_factor[:, q, p] * cofac[:, :, None]
    cofac = G0[:, r, q] * G0[:, t, u] - G0[:, r, u] * G0[:, t, q]
    accumulator -= chol_factor[:, s, p] * cofac[:, :, None]
    cofac = G0[:, r, q] * G0[:, t, s] - G0[:, r, s] * G0[:, t, q]
    accumulator += chol_factor[:, u, p] * cofac[:, :, None]
    cofac = G0[:, p, s] * G0[:, t, u] - G0[:, t, s] * G0[:, p, u]
    accumulator -= chol_factor[:, q, r] * cofac[:, :, None]
    cofac = G0[:, p, q] * G0[:, t, u] - G0[:, t, q] * G0[:, p, u]
    accumulator += chol_factor[:, s, r] * cofac[:, :, None]
    cofac = G0[:, p, q] * G0[:, t, s] - G0[:, t, q] * G0[:, p, s]
    accumulator -= chol_factor[:, u, r] * cofac[:, :, None]
    cofac = G0[:, p, s] * G0[:, r, u] - G0[:, r, s] * G0[:, p, u]
    accumulator += chol_factor[:, q, t] * cofac[:, :, None]
    cofac = G0[:, p, q] * G0[:, r, u] - G0[:, r, q] * G0[:, p, u]
    accumulator -= chol_factor[:, s, t] * cofac[:, :, None]
    cofac = G0[:, p, q] * G0[:, r, s] - G0[:, r, q] * G0[:, p, s]
    accumulator += chol_factor[:, u, t] * cofac[:, :, None]
    spin_buffer[:, det_sls] = accumulator


@jit(nopython=True, fastmath=True)
def build_exchange_contribution(chol_vecs, G0a, G0b, QCIGa, QCIGb):
    nchol = chol_vecs.shape[-1]
    nbasis = G0a.shape[-1]
    nwalkers = G0a.shape[0]
    cont2_Kaa = numpy.zeros(nwalkers, dtype=numpy.complex128)
    cont2_Kbb = numpy.zeros(nwalkers, dtype=numpy.complex128)
    for iw in range(nwalkers):
        G0a_real = G0a[iw].real.copy()
        G0a_imag = G0a[iw].imag.copy()
        G0b_real = G0b[iw].real.copy()
        G0b_imag = G0b[iw].imag.copy()
        for x in range(nchol):
            Lmn = chol_vecs[:, x].copy().reshape((nbasis, nbasis))
            LGL = Lmn @ G0a_real.T @ Lmn + 1j * (Lmn @ G0a_imag.T @ Lmn)
            cont2_Kaa[iw] -= numpy.sum(LGL * QCIGa[iw])
            LGL = Lmn @ G0b_real.T @ Lmn + 1j * (Lmn @ G0b_imag.T @ Lmn)
            cont2_Kbb[iw] -= numpy.sum(LGL * QCIGb[iw])

    return cont2_Kaa, cont2_Kbb


@jit(nopython=True, fastmath=True)
def build_exchange_contributions_opt(
    rchol,
    rchol_act,
    theta,
    CI,
    Lvo,
):
    nchol = rchol.shape[0]
    nbasis = theta.shape[-1]
    nocc = theta.shape[1]
    nwalkers = theta.shape[0]
    nact = CI.shape[1]
    nocc_act = CI.shape[2]
    nfrozen = nocc - nocc_act
    cont2_Kaa = numpy.zeros(nwalkers, dtype=numpy.complex128)
    theta_real = theta.real.copy()
    theta_imag = theta.imag.copy()
    Q = numpy.zeros((nact, nbasis), dtype=numpy.complex128)
    nextra = nact - nocc
    for iw in range(nwalkers):
        theta_act_real = theta[iw, :, nfrozen : nfrozen + nact].real.copy()
        theta_act_imag = theta[iw, :, nfrozen : nfrozen + nact].imag.copy()
        theta_occ = theta[iw, nfrozen : nfrozen + nocc_act, :].copy()
        theta_occ_real = theta_occ.real.copy()
        theta_occ_imag = theta_occ.imag.copy()
        for x in range(nchol):
            # T = rchol[b,q] theta_{a,q}
            # index a gets contracted with CI tensor later so only need the
            # occupied orbitals in the active space not the full set of occupied.
            T = numpy.dot(
                theta_occ_real, rchol[x].reshape((nocc, nbasis)).T
            ) + 1j * numpy.dot(theta_occ_imag, rchol[x].reshape((nocc, nbasis)).T)
            # Ttilde[t,r] = theta_{c,t} rchol[c,r]
            Ttilde = numpy.dot(
                theta_act_real.T, rchol[x].reshape((nocc, nbasis))
            ) + 1j * numpy.dot(theta_act_imag.T, rchol[x].reshape((nocc, nbasis)))
            Q = rchol_act[x].reshape((nact, nbasis)) - Ttilde
            A = numpy.dot(Q.T, CI[iw])  # Q_tr I_ta -> A_{ra}
            B = numpy.dot(theta[iw], A)  # theta_{br} A_{ra} -> C_{ba}
            cont2_Kaa[iw] -= numpy.dot(T.ravel(), B.T.ravel())
            Lvo[iw, x, :, :] = numpy.dot(Q, theta_occ.T)

    return cont2_Kaa

