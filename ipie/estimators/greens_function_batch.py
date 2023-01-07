
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
#          Joonho Lee
#          linusjoonho <linusjoonho@gmail.com>
#

import numpy
import scipy.linalg

from ipie.legacy.estimators.greens_function import gab_mod, gab_spin
from ipie.propagation.overlap import (compute_determinants_batched,
                                      get_overlap_one_det_wicks)
from ipie.utils.linalg import minor_mask
from ipie.utils.misc import is_cupy
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import synchronize

try:
    from ipie.propagation.wicks_kernels import (get_cofactor_matrix_batched,
                                                get_det_matrix_batched,
                                                reduce_to_CI_tensor)
except ImportError:
    pass
from numba import jit

from ipie.estimators.kernels.cpu import wicks as wk

def compute_greens_function(walker_batch, trial):
    compute_gf = get_greens_function(trial)
    return compute_gf(walker_batch,trial)

# Later we will add walker kinds as an input too
def get_greens_function(trial):
    """Wrapper to select the compute_greens_function function.

    Parameters
    ----------
    trial : class
        Trial wavefunction object.

    Returns
    -------
    propagator : class or None
        Propagator object.
    """

    if trial.name == "MultiSlater" and trial.ndets == 1:
        if is_cupy(
            trial.psi
        ):  # if even one array is a cupy array we should assume the rest is done with cupy
            compute_greens_function = greens_function_single_det_batch
        else:
            compute_greens_function = greens_function_single_det
    elif trial.name == "MultiSlater" and trial.ndets > 1 and trial.wicks == False:
        compute_greens_function = greens_function_multi_det
    elif (
        trial.name == "MultiSlater"
        and trial.ndets > 1
        and trial.wicks == True
        and not trial.optimized
    ):
        # compute_greens_function = greens_function_multi_det
        compute_greens_function = greens_function_multi_det_wicks
    elif (
        trial.name == "MultiSlater"
        and trial.ndets > 1
        and trial.wicks
        and trial.optimized
    ):
        # compute_greens_function = greens_function_multi_det
        compute_greens_function = greens_function_multi_det_wicks_opt
    else:
        compute_greens_function = None

    return compute_greens_function


def greens_function(walker_batch, trial, build_full=False):
    compute_greens_function = get_greens_function(trial)
    return compute_greens_function(walker_batch, trial, build_full=build_full)


def greens_function_single_det(walker_batch, trial, build_full=False):
    """Compute walker's green's function.

    Parameters
    ----------
    walker_batch : object
        SingleDetWalkerBatch object.
    trial : object
        Trial wavefunction object.
    Returns
    -------
    det : float64 / complex128
        Determinant of overlap matrix.
    """
    nup = walker_batch.nup
    ndown = walker_batch.ndown

    det = []
    for iw in range(walker_batch.nwalkers):
        ovlp = xp.dot(walker_batch.phia[iw].T, trial.psi[:, :nup].conj())
        ovlp_inv = xp.linalg.inv(ovlp)
        walker_batch.Ghalfa[iw] = xp.dot(ovlp_inv, walker_batch.phia[iw].T)
        if not trial.half_rotated or build_full:
            walker_batch.Ga[iw] = xp.dot(
                trial.psi[:, :nup].conj(), walker_batch.Ghalfa[iw]
            )
        sign_a, log_ovlp_a = xp.linalg.slogdet(ovlp)
        sign_b, log_ovlp_b = 1.0, 0.0
        if ndown > 0 and not walker_batch.rhf:
            ovlp = xp.dot(walker_batch.phib[iw].T, trial.psi[:, nup:].conj())
            sign_b, log_ovlp_b = xp.linalg.slogdet(ovlp)
            walker_batch.Ghalfb[iw] = xp.dot(xp.linalg.inv(ovlp), walker_batch.phib[iw].T)
            if not trial.half_rotated or build_full:
                walker_batch.Gb[iw] = xp.dot(
                    trial.psi[:, nup:].conj(), walker_batch.Ghalfb[iw]
                )
            det += [
                sign_a
                * sign_b
                * xp.exp(log_ovlp_a + log_ovlp_b - walker_batch.log_shift[iw])
            ]
        elif ndown > 0 and walker_batch.rhf:
            det += [
                sign_a
                * sign_a
                * xp.exp(log_ovlp_a + log_ovlp_a - walker_batch.log_shift[iw])
            ]
        elif ndown == 0:
            det += [sign_a * xp.exp(log_ovlp_a - walker_batch.log_shift)]

    det = xp.array(det, dtype=numpy.complex128)

    synchronize()

    return det


def greens_function_single_det_batch(walker_batch, trial, build_full=False):
    """Compute walker's green's function using only batched operations.

    Parameters
    ----------
    walker_batch : object
        SingleDetWalkerBatch object.
    trial : object
        Trial wavefunction object.
    Returns
    -------
    ot : float64 / complex128
        Overlap with trial.
    """
    nup = walker_batch.nup
    ndown = walker_batch.ndown

    ovlp_a = xp.einsum("wmi,mj->wij", walker_batch.phia, trial.psia.conj(), optimize=True)
    ovlp_inv_a = xp.linalg.inv(ovlp_a)
    sign_a, log_ovlp_a = xp.linalg.slogdet(ovlp_a)

    walker_batch.Ghalfa = xp.einsum(
        "wij,wmj->wim", ovlp_inv_a, walker_batch.phia, optimize=True
    )
    if not trial.half_rotated or build_full:
        walker_batch.Ga = xp.einsum(
            "mi,win->wmn", trial.psia.conj(), walker_batch.Ghalfa, optimize=True
        )

    if ndown > 0 and not walker_batch.rhf:
        ovlp_b = xp.einsum(
            "wmi,mj->wij", walker_batch.phib, trial.psib.conj(), optimize=True
        )
        ovlp_inv_b = xp.linalg.inv(ovlp_b)

        sign_b, log_ovlp_b = xp.linalg.slogdet(ovlp_b)
        walker_batch.Ghalfb = xp.einsum(
            "wij,wmj->wim", ovlp_inv_b, walker_batch.phib, optimize=True
        )
        if not trial.half_rotated or build_full:
            walker_batch.Gb = xp.einsum(
                "mi,win->wmn", trial.psib.conj(), walker_batch.Ghalfb, optimize=True
            )
        ot = sign_a * sign_b * xp.exp(log_ovlp_a + log_ovlp_b - walker_batch.log_shift)
    elif ndown > 0 and walker_batch.rhf:
        ot = sign_a * sign_a * xp.exp(log_ovlp_a + log_ovlp_a - walker_batch.log_shift)
    elif ndown == 0:
        ot = sign_a * xp.exp(log_ovlp_a - walker_batch.log_shift)

    synchronize()

    return ot


def greens_function_multi_det(walker_batch, trial, build_full=False):
    """Compute walker's green's function.

    Parameters
    ----------
    walker_batch : object
        MultiDetTrialWalkerBatch object.
    trial : object
        Trial wavefunction object.
    Returns
    -------
    det : float64 / complex128
        Determinant of overlap matrix.
    """
    nup = walker_batch.nup
    walker_batch.Ga.fill(0.0)
    walker_batch.Gb.fill(0.0)
    tot_ovlps = numpy.zeros(walker_batch.nwalkers, dtype=numpy.complex128)
    for iw in range(walker_batch.nwalkers):
        for (ix, detix) in enumerate(trial.psi):
            # construct "local" green's functions for each component of psi_T
            Oup = numpy.dot(walker_batch.phia[iw].T, detix[:, :nup].conj())
            # det(A) = det(A^T)
            sign_a, logdet_a = numpy.linalg.slogdet(Oup)
            walker_batch.det_ovlpas[iw, ix] = sign_a * numpy.exp(logdet_a)
            if abs(walker_batch.det_ovlpas[iw, ix]) < 1e-16:
                continue

            Odn = numpy.dot(walker_batch.phib[iw].T, detix[:, nup:].conj())
            sign_b, logdet_b = numpy.linalg.slogdet(Odn)
            walker_batch.det_ovlpbs[iw, ix] = sign_b * numpy.exp(logdet_b)
            ovlp = walker_batch.det_ovlpas[iw, ix] * walker_batch.det_ovlpbs[iw, ix]
            if abs(ovlp) < 1e-16:
                continue

            inv_ovlp = scipy.linalg.inv(Oup)
            walker_batch.Gihalfa[iw, ix, :, :] = numpy.dot(
                inv_ovlp, walker_batch.phia[iw].T
            )
            walker_batch.Gia[iw, ix, :, :] = numpy.dot(
                detix[:, :nup].conj(), walker_batch.Gihalfa[iw, ix, :, :]
            )

            inv_ovlp = scipy.linalg.inv(Odn)
            walker_batch.Gihalfb[iw, ix, :, :] = numpy.dot(
                inv_ovlp, walker_batch.phib[iw].T
            )
            walker_batch.Gib[iw, ix, :, :] = numpy.dot(
                detix[:, nup:].conj(), walker_batch.Gihalfb[iw, ix, :, :]
            )

            tot_ovlps[iw] += trial.coeffs[ix].conj() * ovlp
            walker_batch.det_weights[iw, ix] = trial.coeffs[ix].conj() * ovlp

            walker_batch.Ga[iw] += (
                walker_batch.Gia[iw, ix, :, :] * ovlp * trial.coeffs[ix].conj()
            )
            walker_batch.Gb[iw] += (
                walker_batch.Gib[iw, ix, :, :] * ovlp * trial.coeffs[ix].conj()
            )

        walker_batch.Ga[iw] /= tot_ovlps[iw]
        walker_batch.Gb[iw] /= tot_ovlps[iw]

    return tot_ovlps


def greens_function_multi_det_wicks(walker_batch, trial, build_full=False):
    """Compute walker's green's function using Wick's theorem.

    Parameters
    ----------
    walker_batch : object
        MultiDetTrialWalkerBatch object.
    trial : object
        Trial wavefunction object.
    Returns
    -------
    det : float64 / complex128
        Determinant of overlap matrix.
    """
    tot_ovlps = numpy.zeros(walker_batch.nwalkers, dtype=numpy.complex128)
    nbasis = walker_batch.Ga.shape[-1]

    nup = walker_batch.nup
    ndown = walker_batch.ndown

    walker_batch.Ga.fill(0.0 + 0.0j)
    walker_batch.Gb.fill(0.0 + 0.0j)

    for iw in range(walker_batch.nwalkers):
        phia = walker_batch.phia[iw]  # walker wfn
        phib = walker_batch.phib[iw]  # walker wfn

        Oalpha = numpy.dot(trial.psi0a.conj().T, phia)
        sign_a, logdet_a = numpy.linalg.slogdet(Oalpha)
        logdet_b, sign_b = 0.0, 1.0
        Obeta = numpy.dot(trial.psi0b.conj().T, phib)
        sign_b, logdet_b = numpy.linalg.slogdet(Obeta)

        ovlp0 = sign_a * sign_b * numpy.exp(logdet_a + logdet_b)
        walker_batch.det_ovlpas[iw, 0] = sign_a * numpy.exp(logdet_a)
        walker_batch.det_ovlpbs[iw, 0] = sign_b * numpy.exp(logdet_b)
        ovlpa0 = walker_batch.det_ovlpas[iw, 0]
        ovlpab = walker_batch.det_ovlpbs[iw, 0]

        # G0, G0H = gab_spin(trial.psi0, phi, nup, ndown)
        G0a, G0Ha = gab_mod(trial.psi0a, phia)
        G0b, G0Hb = gab_mod(trial.psi0b, phib)
        walker_batch.G0a[iw] = G0a
        walker_batch.G0b[iw] = G0b
        walker_batch.Ghalfa[iw] = G0Ha
        walker_batch.Ghalfb[iw] = G0Hb
        walker_batch.Q0a[iw] = numpy.eye(nbasis) - walker_batch.G0a[iw]
        walker_batch.Q0b[iw] = numpy.eye(nbasis) - walker_batch.G0b[iw]

        G0a = walker_batch.G0a[iw]
        G0b = walker_batch.G0b[iw]
        Q0a = walker_batch.Q0a[iw]
        Q0b = walker_batch.Q0b[iw]

        ovlp = 0.0 + 0.0j
        ovlp += trial.coeffs[0].conj()

        walker_batch.Ga[iw] += G0a * trial.coeffs[0].conj()
        walker_batch.Gb[iw] += G0b * trial.coeffs[0].conj()

        walker_batch.CIa[iw].fill(0.0 + 0.0j)
        walker_batch.CIb[iw].fill(0.0 + 0.0j)

        for jdet in range(1, trial.ndets):
            nex_a = len(trial.cre_a[jdet])
            nex_b = len(trial.cre_b[jdet])

            ovlp_a, ovlp_b = get_overlap_one_det_wicks(
                nex_a,
                trial.cre_a[jdet],
                trial.anh_a[jdet],
                G0a,
                nex_b,
                trial.cre_b[jdet],
                trial.anh_b[jdet],
                G0b,
            )

            walker_batch.det_ovlpas[iw, jdet] = (
                ovlp_a * trial.phase_a[jdet]
            )  # phase included
            walker_batch.det_ovlpbs[iw, jdet] = (
                ovlp_b * trial.phase_b[jdet]
            )  # phase included
            ovlpa = walker_batch.det_ovlpas[iw, jdet]
            ovlpb = walker_batch.det_ovlpbs[iw, jdet]

            ovlp += trial.coeffs[jdet].conj() * ovlpa * ovlpb

            c_phasea_ovlpb = trial.coeffs[jdet].conj() * trial.phase_a[jdet] * ovlpb
            c_phaseb_ovlpa = trial.coeffs[jdet].conj() * trial.phase_b[jdet] * ovlpa

            # contribution 1 (disconnected diagrams)
            walker_batch.Ga[iw] += trial.coeffs[jdet].conj() * G0a * ovlpa * ovlpb
            walker_batch.Gb[iw] += trial.coeffs[jdet].conj() * G0b * ovlpa * ovlpb
            # intermediates for contribution 2 (connected diagrams)
            if nex_a == 1:
                walker_batch.CIa[
                    iw, trial.anh_a[jdet][0], trial.cre_a[jdet][0]
                ] += c_phasea_ovlpb
            elif nex_a == 2:
                p = trial.cre_a[jdet][0]
                q = trial.anh_a[jdet][0]
                r = trial.cre_a[jdet][1]
                s = trial.anh_a[jdet][1]
                walker_batch.CIa[iw, q, p] += c_phasea_ovlpb * G0a[r, s]
                walker_batch.CIa[iw, s, r] += c_phasea_ovlpb * G0a[p, q]
                walker_batch.CIa[iw, q, r] -= c_phasea_ovlpb * G0a[p, s]
                walker_batch.CIa[iw, s, p] -= c_phasea_ovlpb * G0a[r, q]
            elif nex_a == 3:
                p = trial.cre_a[jdet][0]
                q = trial.anh_a[jdet][0]
                r = trial.cre_a[jdet][1]
                s = trial.anh_a[jdet][1]
                t = trial.cre_a[jdet][2]
                u = trial.anh_a[jdet][2]

                walker_batch.CIa[iw, q, p] += c_phasea_ovlpb * (
                    G0a[r, s] * G0a[t, u] - G0a[r, u] * G0a[t, s]
                )  # 0 0
                walker_batch.CIa[iw, s, p] -= c_phasea_ovlpb * (
                    G0a[r, q] * G0a[t, u] - G0a[r, u] * G0a[t, q]
                )  # 0 1
                walker_batch.CIa[iw, u, p] += c_phasea_ovlpb * (
                    G0a[r, q] * G0a[t, s] - G0a[r, s] * G0a[t, q]
                )  # 0 2

                walker_batch.CIa[iw, q, r] -= c_phasea_ovlpb * (
                    G0a[p, s] * G0a[t, u] - G0a[p, u] * G0a[t, s]
                )  # 1 0
                walker_batch.CIa[iw, s, r] += c_phasea_ovlpb * (
                    G0a[p, q] * G0a[t, u] - G0a[p, u] * G0a[t, q]
                )  # 1 1
                walker_batch.CIa[iw, u, r] -= c_phasea_ovlpb * (
                    G0a[p, q] * G0a[t, s] - G0a[p, s] * G0a[t, q]
                )  # 1 2

                walker_batch.CIa[iw, q, t] += c_phasea_ovlpb * (
                    G0a[p, s] * G0a[r, u] - G0a[p, u] * G0a[r, s]
                )  # 2 0
                walker_batch.CIa[iw, s, t] -= c_phasea_ovlpb * (
                    G0a[p, q] * G0a[r, u] - G0a[p, u] * G0a[r, q]
                )  # 2 1
                walker_batch.CIa[iw, u, t] += c_phasea_ovlpb * (
                    G0a[p, q] * G0a[r, s] - G0a[p, s] * G0a[r, q]
                )  # 2 2

            elif nex_a > 3:
                det_a = numpy.zeros((nex_a, nex_a), dtype=numpy.complex128)
                for iex in range(nex_a):
                    det_a[iex, iex] = G0a[
                        trial.cre_a[jdet][iex], trial.anh_a[jdet][iex]
                    ]
                    for jex in range(iex + 1, nex_a):
                        det_a[iex, jex] = G0a[
                            trial.cre_a[jdet][iex], trial.anh_a[jdet][jex]
                        ]
                        det_a[jex, iex] = G0a[
                            trial.cre_a[jdet][jex], trial.anh_a[jdet][iex]
                        ]
                cofactor = numpy.zeros((nex_a - 1, nex_a - 1), dtype=numpy.complex128)
                for iex in range(nex_a):
                    p = trial.cre_a[jdet][iex]
                    for jex in range(nex_a):
                        q = trial.anh_a[jdet][jex]
                        cofactor[:, :] = minor_mask(det_a, iex, jex)
                        walker_batch.CIa[iw, q, p] += (
                            c_phasea_ovlpb
                            * (-1) ** (iex + jex)
                            * numpy.linalg.det(cofactor)
                        )

            if nex_b == 1:
                walker_batch.CIb[
                    iw, trial.anh_b[jdet][0], trial.cre_b[jdet][0]
                ] += c_phaseb_ovlpa
            elif nex_b == 2:
                p = trial.cre_b[jdet][0]
                q = trial.anh_b[jdet][0]
                r = trial.cre_b[jdet][1]
                s = trial.anh_b[jdet][1]
                walker_batch.CIb[iw, q, p] += c_phaseb_ovlpa * G0b[r, s]
                walker_batch.CIb[iw, s, r] += c_phaseb_ovlpa * G0b[p, q]
                walker_batch.CIb[iw, q, r] -= c_phaseb_ovlpa * G0b[p, s]
                walker_batch.CIb[iw, s, p] -= c_phaseb_ovlpa * G0b[r, q]
            elif nex_b == 3:
                p = trial.cre_b[jdet][0]
                q = trial.anh_b[jdet][0]
                r = trial.cre_b[jdet][1]
                s = trial.anh_b[jdet][1]
                t = trial.cre_b[jdet][2]
                u = trial.anh_b[jdet][2]

                walker_batch.CIb[iw, q, p] += c_phaseb_ovlpa * (
                    G0b[r, s] * G0b[t, u] - G0b[r, u] * G0b[t, s]
                )  # 0 0
                walker_batch.CIb[iw, s, p] -= c_phaseb_ovlpa * (
                    G0b[r, q] * G0b[t, u] - G0b[r, u] * G0b[t, q]
                )  # 0 1
                walker_batch.CIb[iw, u, p] += c_phaseb_ovlpa * (
                    G0b[r, q] * G0b[t, s] - G0b[r, s] * G0b[t, q]
                )  # 0 2

                walker_batch.CIb[iw, q, r] -= c_phaseb_ovlpa * (
                    G0b[p, s] * G0b[t, u] - G0b[p, u] * G0b[t, s]
                )  # 1 0
                walker_batch.CIb[iw, s, r] += c_phaseb_ovlpa * (
                    G0b[p, q] * G0b[t, u] - G0b[p, u] * G0b[t, q]
                )  # 1 1
                walker_batch.CIb[iw, u, r] -= c_phaseb_ovlpa * (
                    G0b[p, q] * G0b[t, s] - G0b[p, s] * G0b[t, q]
                )  # 1 2

                walker_batch.CIb[iw, q, t] += c_phaseb_ovlpa * (
                    G0b[p, s] * G0b[r, u] - G0b[p, u] * G0b[r, s]
                )  # 2 0
                walker_batch.CIb[iw, s, t] -= c_phaseb_ovlpa * (
                    G0b[p, q] * G0b[r, u] - G0b[p, u] * G0b[r, q]
                )  # 2 1
                walker_batch.CIb[iw, u, t] += c_phaseb_ovlpa * (
                    G0b[p, q] * G0b[r, s] - G0b[p, s] * G0b[r, q]
                )  # 2 2

            elif nex_b > 3:
                det_b = numpy.zeros((nex_b, nex_b), dtype=numpy.complex128)
                for iex in range(nex_b):
                    det_b[iex, iex] = G0b[
                        trial.cre_b[jdet][iex], trial.anh_b[jdet][iex]
                    ]
                    for jex in range(iex + 1, nex_b):
                        det_b[iex, jex] = G0b[
                            trial.cre_b[jdet][iex], trial.anh_b[jdet][jex]
                        ]
                        det_b[jex, iex] = G0b[
                            trial.cre_b[jdet][jex], trial.anh_b[jdet][iex]
                        ]
                cofactor = numpy.zeros((nex_b - 1, nex_b - 1), dtype=numpy.complex128)
                for iex in range(nex_b):
                    p = trial.cre_b[jdet][iex]
                    for jex in range(nex_b):
                        q = trial.anh_b[jdet][jex]
                        cofactor[:, :] = minor_mask(det_b, iex, jex)
                        walker_batch.CIb[iw, q, p] += (
                            c_phaseb_ovlpa
                            * (-1) ** (iex + jex)
                            * numpy.linalg.det(cofactor)
                        )

        # contribution 2 (connected diagrams)
        walker_batch.Ga[iw] += Q0a.dot(walker_batch.CIa[iw]).dot(G0a)
        walker_batch.Gb[iw] += Q0b.dot(walker_batch.CIb[iw]).dot(G0b)

        # multiplying everything by reference overlap
        ovlp *= ovlp0
        walker_batch.Ga[iw] *= ovlp0
        walker_batch.Gb[iw] *= ovlp0

        walker_batch.Ga[iw] /= ovlp
        walker_batch.Gb[iw] /= ovlp

        tot_ovlps[iw] = ovlp

    return tot_ovlps


def build_CI_single_excitation(walker_batch, trial, c_phasea_ovlpb, c_phaseb_ovlpa):
    """Single excitation contributions to CI intermediate for wicks.

    Parameters
    ----------
    walker_batch : object
        MultiDetTrialWalkerBatch object.
    trial : object
        Trial wavefunction object.
    c_phasea_ovlpb : np.ndarray
        ci coeffs and phases of alpha dets * overlaps of beta determinants.
    c_phaseb_ovlpa : np.ndarray
        ci coeffs and phases of beta dets * overlaps of alpha determinants.
    Returns
    -------
    None, modifies walker_batch.CIa, and walker_batch.CIb inplace.
    """
    ndets_a = len(trial.cre_ex_a[1])
    nwalkers = walker_batch.G0a.shape[0]
    if trial.cre_ex_a[1].shape[0] == 0:
        pass
    else:
        ps = trial.cre_ex_a[1][:, 0]
        qs = trial.anh_ex_a[1][:, 0]
        reduce_to_CI_tensor(
            nwalkers,
            ndets_a,
            qs,
            ps,
            walker_batch.CIa,
            c_phasea_ovlpb[:, trial.excit_map_a[1]],
        )
    ndets_b = len(trial.cre_ex_b[1])
    if trial.cre_ex_b[1].shape[0] == 0:
        pass
    else:
        ps = trial.cre_ex_b[1][:, 0]
        qs = trial.anh_ex_b[1][:, 0]
        reduce_to_CI_tensor(
            nwalkers,
            ndets_b,
            qs,
            ps,
            walker_batch.CIb,
            c_phaseb_ovlpa[:, trial.excit_map_b[1]],
        )


def build_CI_single_excitation_opt(walker_batch, trial, c_phasea_ovlpb, c_phaseb_ovlpa):
    """Single excitation contributions to CI intermediate for wicks.

    Optimized using numba.

    Parameters
    ----------
    walker_batch : object
        MultiDetTrialWalkerBatch object.
    trial : object
        Trial wavefunction object.
    c_phasea_ovlpb : np.ndarray
        ci coeffs and phases of alpha dets * overlaps of beta determinants.
    c_phaseb_ovlpa : np.ndarray
        ci coeffs and phases of beta dets * overlaps of alpha determinants.
    Returns
    -------
    None, modifies walker_batch.CIa, and walker_batch.CIb inplace.
    """
    ndets_a = len(trial.cre_ex_a[1])
    nwalkers = walker_batch.G0a.shape[0]
    if trial.cre_ex_a[1].shape[0] == 0:
        pass
    else:
        phases = c_phasea_ovlpb[:, trial.excit_map_a[1]]
        wk.reduce_CI_singles(
            trial.cre_ex_a[1],
            trial.anh_ex_a[1],
            trial.occ_map_a,
            phases,
            walker_batch.CIa,
        )
    ndets_b = len(trial.cre_ex_b[1])
    if trial.cre_ex_b[1].shape[0] == 0:
        pass
    else:
        phases = c_phaseb_ovlpa[:, trial.excit_map_b[1]]
        ps = trial.cre_ex_b[1][:, 0]
        qs = trial.anh_ex_b[1][:, 0]
        wk.reduce_CI_singles(
            trial.cre_ex_b[1],
            trial.anh_ex_b[1],
            trial.occ_map_b,
            phases,
            walker_batch.CIb,
        )


def build_CI_double_excitation(walker_batch, trial, c_phasea_ovlpb, c_phaseb_ovlpa):
    """Double excitation contributions to CI intermediate for wicks.

    Parameters
    ----------
    walker_batch : object
        MultiDetTrialWalkerBatch object.
    trial : object
        Trial wavefunction object.
    c_phasea_ovlpb : np.ndarray
        ci coeffs and phases of alpha dets * overlaps of beta determinants.
    c_phaseb_ovlpa : np.ndarray
        ci coeffs and phases of beta dets * overlaps of alpha determinants.
    Returns
    -------
    None, modifies walker_batch.CIa, and walker_batch.CIb inplace.
    """
    ndets_a = len(trial.cre_ex_a[2])
    nwalkers = walker_batch.G0a.shape[0]
    if ndets_a == 0:
        pass
    else:
        ps = trial.cre_ex_a[2][:, 0]
        qs = trial.anh_ex_a[2][:, 0]
        rs = trial.cre_ex_a[2][:, 1]
        ss = trial.anh_ex_a[2][:, 1]
        phases = c_phasea_ovlpb[:, trial.excit_map_a[2]]
        rhs = phases * walker_batch.G0a[:, rs, ss]
        reduce_to_CI_tensor(
            nwalkers,
            ndets_a,
            qs,
            ps,
            walker_batch.CIa,
            rhs,
        )
        rhs = phases * walker_batch.G0a[:, ps, qs]
        reduce_to_CI_tensor(
            nwalkers,
            ndets_a,
            ss,
            rs,
            walker_batch.CIa,
            rhs,
        )
        rhs = -phases * walker_batch.G0a[:, ps, ss]
        reduce_to_CI_tensor(
            nwalkers,
            ndets_a,
            qs,
            rs,
            walker_batch.CIa,
            rhs,
        )
        rhs = -phases * walker_batch.G0a[:, rs, qs]
        reduce_to_CI_tensor(
            nwalkers,
            ndets_a,
            ss,
            ps,
            walker_batch.CIa,
            rhs,
        )
    ndets_b = len(trial.cre_ex_b[2])
    if ndets_b == 0:
        pass
    else:
        ps = trial.cre_ex_b[2][:, 0]
        qs = trial.anh_ex_b[2][:, 0]
        rs = trial.cre_ex_b[2][:, 1]
        ss = trial.anh_ex_b[2][:, 1]
        phases = c_phaseb_ovlpa[:, trial.excit_map_b[2]]
        rhs = phases * walker_batch.G0b[:, rs, ss]
        reduce_to_CI_tensor(
            nwalkers,
            ndets_b,
            qs,
            ps,
            walker_batch.CIb,
            rhs,
        )
        rhs = phases * walker_batch.G0b[:, ps, qs]
        reduce_to_CI_tensor(
            nwalkers,
            ndets_b,
            ss,
            rs,
            walker_batch.CIb,
            rhs,
        )
        rhs = -phases * walker_batch.G0b[:, ps, ss]
        reduce_to_CI_tensor(
            nwalkers,
            ndets_b,
            qs,
            rs,
            walker_batch.CIb,
            rhs,
        )
        rhs = -phases * walker_batch.G0b[:, rs, qs]
        reduce_to_CI_tensor(
            nwalkers,
            ndets_b,
            ss,
            ps,
            walker_batch.CIb,
            rhs,
        )


def build_CI_double_excitation_opt(walker_batch, trial, c_phasea_ovlpb, c_phaseb_ovlpa):
    """Double excitation contributions to CI intermediate for wicks.

    Optimized using numba.

    Parameters
    ----------
    walker_batch : object
        MultiDetTrialWalkerBatch object.
    trial : object
        Trial wavefunction object.
    c_phasea_ovlpb : np.ndarray
        ci coeffs and phases of alpha dets * overlaps of beta determinants.
    c_phaseb_ovlpa : np.ndarray
        ci coeffs and phases of beta dets * overlaps of alpha determinants.
    Returns
    -------
    None, modifies walker_batch.CIa, and walker_batch.CIb inplace.
    """
    ndets_a = len(trial.cre_ex_a[2])
    nwalkers = walker_batch.G0a.shape[0]
    if trial.cre_ex_a[2].shape[0] == 0:
        pass
    else:
        phases = c_phasea_ovlpb[:, trial.excit_map_a[2]]
        wk.reduce_CI_doubles(
            trial.cre_ex_a[2],
            trial.anh_ex_a[2],
            trial.occ_map_a,
            trial.nfrozen,
            phases,
            walker_batch.Ghalfa,
            walker_batch.CIa,
        )
    ndets_b = len(trial.cre_ex_b[1])
    if trial.cre_ex_b[2].shape[0] == 0:
        pass
    else:
        phases = c_phaseb_ovlpa[:, trial.excit_map_b[2]]
        wk.reduce_CI_doubles(
            trial.cre_ex_b[2],
            trial.anh_ex_b[2],
            trial.occ_map_b,
            trial.nfrozen,
            phases,
            walker_batch.Ghalfb,
            walker_batch.CIb,
        )


def build_CI_triple_excitation_opt(walker_batch, trial, c_phasea_ovlpb, c_phaseb_ovlpa):
    """Triple excitation contributions to CI intermediate for wicks.

    Optimized using numba.

    Parameters
    ----------
    walker_batch : object
        MultiDetTrialWalkerBatch object.
    trial : object
        Trial wavefunction object.
    c_phasea_ovlpb : np.ndarray
        ci coeffs and phases of alpha dets * overlaps of beta determinants.
    c_phaseb_ovlpa : np.ndarray
        ci coeffs and phases of beta dets * overlaps of alpha determinants.
    Returns
    -------
    None, modifies walker_batch.CIa, and walker_batch.CIb inplace.
    """
    ndets_a = len(trial.cre_ex_a[3])
    nwalkers = walker_batch.G0a.shape[0]
    if trial.cre_ex_a[3].shape[0] == 0:
        pass
    else:
        phases = c_phasea_ovlpb[:, trial.excit_map_a[3]]
        wk.reduce_CI_triples(
            trial.cre_ex_a[3],
            trial.anh_ex_a[3],
            trial.occ_map_a,
            trial.nfrozen,
            phases,
            walker_batch.Ghalfa,
            walker_batch.CIa,
        )
    ndets_b = len(trial.cre_ex_b[3])
    if trial.cre_ex_b[3].shape[0] == 0:
        pass
    else:
        phases = c_phaseb_ovlpa[:, trial.excit_map_b[3]]
        wk.reduce_CI_triples(
            trial.cre_ex_b[3],
            trial.anh_ex_b[3],
            trial.occ_map_b,
            trial.nfrozen,
            phases,
            walker_batch.Ghalfb,
            walker_batch.CIb,
        )


def build_CI_triple_excitation(walker_batch, trial, c_phasea_ovlpb, c_phaseb_ovlpa):
    """Triple excitation contributions to CI intermediate for wicks.

    Parameters
    ----------
    walker_batch : object
        MultiDetTrialWalkerBatch object.
    trial : object
        Trial wavefunction object.
    c_phasea_ovlpb : np.ndarray
        ci coeffs and phases of alpha dets * overlaps of beta determinants.
    c_phaseb_ovlpa : np.ndarray
        ci coeffs and phases of beta dets * overlaps of alpha determinants.
    Returns
    -------
    None, modifies walker_batch.CIa, and walker_batch.CIb inplace.
    """
    ndets_a = len(trial.cre_ex_a[3])
    nwalkers = walker_batch.G0a.shape[0]
    if trial.cre_ex_a[3].shape[0] == 0:
        pass
    else:
        ps = trial.cre_ex_a[3][:, 0]
        qs = trial.anh_ex_a[3][:, 0]
        rs = trial.cre_ex_a[3][:, 1]
        ss = trial.anh_ex_a[3][:, 1]
        ts = trial.cre_ex_a[3][:, 2]
        us = trial.anh_ex_a[3][:, 2]
        phases = c_phasea_ovlpb[:, trial.excit_map_a[3]]
        rhs = phases * (
            walker_batch.G0a[:, rs, ss] * walker_batch.G0a[:, ts, us]
            - walker_batch.G0a[:, rs, us] * walker_batch.G0a[:, ts, ss]
        )  # 0 0
        reduce_to_CI_tensor(
            nwalkers,
            ndets_a,
            qs,
            ps,
            walker_batch.CIa,
            rhs,
        )
        rhs = -phases * (
            walker_batch.G0a[:, rs, qs] * walker_batch.G0a[:, ts, us]
            - walker_batch.G0a[:, rs, us] * walker_batch.G0a[:, ts, qs]
        )  # 0 1
        reduce_to_CI_tensor(
            nwalkers,
            ndets_a,
            ss,
            ps,
            walker_batch.CIa,
            rhs,
        )
        rhs = phases * (
            walker_batch.G0a[:, rs, qs] * walker_batch.G0a[:, ts, ss]
            - walker_batch.G0a[:, rs, ss] * walker_batch.G0a[:, ts, qs]
        )  # 0 2
        reduce_to_CI_tensor(
            nwalkers,
            ndets_a,
            us,
            ps,
            walker_batch.CIa,
            rhs,
        )

        rhs = -phases * (
            walker_batch.G0a[:, ps, ss] * walker_batch.G0a[:, ts, us]
            - walker_batch.G0a[:, ps, us] * walker_batch.G0a[:, ts, ss]
        )  # 1 0
        reduce_to_CI_tensor(
            nwalkers,
            ndets_a,
            qs,
            rs,
            walker_batch.CIa,
            rhs,
        )
        rhs = phases * (
            walker_batch.G0a[:, ps, qs] * walker_batch.G0a[:, ts, us]
            - walker_batch.G0a[:, ps, us] * walker_batch.G0a[:, ts, qs]
        )  # 1 1
        reduce_to_CI_tensor(
            nwalkers,
            ndets_a,
            ss,
            rs,
            walker_batch.CIa,
            rhs,
        )
        rhs = -phases * (
            walker_batch.G0a[:, ps, qs] * walker_batch.G0a[:, ts, ss]
            - walker_batch.G0a[:, ps, ss] * walker_batch.G0a[:, ts, qs]
        )  # 1 2
        reduce_to_CI_tensor(
            nwalkers,
            ndets_a,
            us,
            rs,
            walker_batch.CIa,
            rhs,
        )

        rhs = phases * (
            walker_batch.G0a[:, ps, ss] * walker_batch.G0a[:, rs, us]
            - walker_batch.G0a[:, ps, us] * walker_batch.G0a[:, rs, ss]
        )  # 2 0
        reduce_to_CI_tensor(
            nwalkers,
            ndets_a,
            qs,
            ts,
            walker_batch.CIa,
            rhs,
        )
        rhs = -phases * (
            walker_batch.G0a[:, ps, qs] * walker_batch.G0a[:, rs, us]
            - walker_batch.G0a[:, ps, us] * walker_batch.G0a[:, rs, qs]
        )  # 2 1
        reduce_to_CI_tensor(
            nwalkers,
            ndets_a,
            ss,
            ts,
            walker_batch.CIa,
            rhs,
        )
        rhs = phases * (
            walker_batch.G0a[:, ps, qs] * walker_batch.G0a[:, rs, ss]
            - walker_batch.G0a[:, ps, ss] * walker_batch.G0a[:, rs, qs]
        )  # 2 2
        reduce_to_CI_tensor(
            nwalkers,
            ndets_a,
            us,
            ts,
            walker_batch.CIa,
            rhs,
        )
    ndets_b = len(trial.cre_ex_b[3])
    if trial.cre_ex_b[3].shape[0] == 0:
        pass
    else:
        ps = trial.cre_ex_b[3][:, 0]
        qs = trial.anh_ex_b[3][:, 0]
        rs = trial.cre_ex_b[3][:, 1]
        ss = trial.anh_ex_b[3][:, 1]
        ts = trial.cre_ex_b[3][:, 2]
        us = trial.anh_ex_b[3][:, 2]
        phases = c_phaseb_ovlpa[:, trial.excit_map_b[3]]
        rhs = phases * (
            walker_batch.G0b[:, rs, ss] * walker_batch.G0b[:, ts, us]
            - walker_batch.G0b[:, rs, us] * walker_batch.G0b[:, ts, ss]
        )  # 0 0
        reduce_to_CI_tensor(
            nwalkers,
            ndets_b,
            qs,
            ps,
            walker_batch.CIb,
            rhs,
        )
        rhs = -phases * (
            walker_batch.G0b[:, rs, qs] * walker_batch.G0b[:, ts, us]
            - walker_batch.G0b[:, rs, us] * walker_batch.G0b[:, ts, qs]
        )  # 0 1
        reduce_to_CI_tensor(
            nwalkers,
            ndets_b,
            ss,
            ps,
            walker_batch.CIb,
            rhs,
        )
        rhs = phases * (
            walker_batch.G0b[:, rs, qs] * walker_batch.G0b[:, ts, ss]
            - walker_batch.G0b[:, rs, ss] * walker_batch.G0b[:, ts, qs]
        )  # 0 2
        reduce_to_CI_tensor(
            nwalkers,
            ndets_b,
            us,
            ps,
            walker_batch.CIb,
            rhs,
        )

        rhs = -phases * (
            walker_batch.G0b[:, ps, ss] * walker_batch.G0b[:, ts, us]
            - walker_batch.G0b[:, ps, us] * walker_batch.G0b[:, ts, ss]
        )  # 1 0
        reduce_to_CI_tensor(
            nwalkers,
            ndets_b,
            qs,
            rs,
            walker_batch.CIb,
            rhs,
        )
        rhs = phases * (
            walker_batch.G0b[:, ps, qs] * walker_batch.G0b[:, ts, us]
            - walker_batch.G0b[:, ps, us] * walker_batch.G0b[:, ts, qs]
        )  # 1 1
        reduce_to_CI_tensor(
            nwalkers,
            ndets_b,
            ss,
            rs,
            walker_batch.CIb,
            rhs,
        )
        rhs = -phases * (
            walker_batch.G0b[:, ps, qs] * walker_batch.G0b[:, ts, ss]
            - walker_batch.G0b[:, ps, ss] * walker_batch.G0b[:, ts, qs]
        )  # 1 2
        reduce_to_CI_tensor(
            nwalkers,
            ndets_b,
            us,
            rs,
            walker_batch.CIb,
            rhs,
        )

        rhs = phases * (
            walker_batch.G0b[:, ps, ss] * walker_batch.G0b[:, rs, us]
            - walker_batch.G0b[:, ps, us] * walker_batch.G0b[:, rs, ss]
        )  # 2 0
        reduce_to_CI_tensor(
            nwalkers,
            ndets_b,
            qs,
            ts,
            walker_batch.CIb,
            rhs,
        )
        rhs = -phases * (
            walker_batch.G0b[:, ps, qs] * walker_batch.G0b[:, rs, us]
            - walker_batch.G0b[:, ps, us] * walker_batch.G0b[:, rs, qs]
        )  # 2 1
        reduce_to_CI_tensor(
            nwalkers,
            ndets_b,
            ss,
            ts,
            walker_batch.CIb,
            rhs,
        )
        rhs = phases * (
            walker_batch.G0b[:, ps, qs] * walker_batch.G0b[:, rs, ss]
            - walker_batch.G0b[:, ps, ss] * walker_batch.G0b[:, rs, qs]
        )  # 2 2
        reduce_to_CI_tensor(
            nwalkers,
            ndets_b,
            us,
            ts,
            walker_batch.CIb,
            rhs,
        )


def build_CI_nfold_excitation(
    nexcit, walker_batch, trial, c_phasea_ovlpb, c_phaseb_ovlpa
):
    """N-fold excitation contributions to CI intermediate for wicks.

    Parameters
    ----------
    nexcit : int
        Excitation level
    walker_batch : object
        MultiDetTrialWalkerBatch object.
    trial : object
        Trial wavefunction object.
    c_phasea_ovlpb : np.ndarray
        ci coeffs and phases of alpha dets * overlaps of beta determinants.
    c_phaseb_ovlpa : np.ndarray
        ci coeffs and phases of beta dets * overlaps of alpha determinants.
    Returns
    -------
    None, modifies walker_batch.CIa, and walker_batch.CIb inplace.
    """
    ndets_a = len(trial.cre_ex_a[nexcit])
    nwalkers = walker_batch.G0a.shape[0]
    if ndets_a == 0:
        dets_a = None
    else:
        det_mat = numpy.zeros(
            (nwalkers, ndets_a, nexcit, nexcit), dtype=numpy.complex128
        )
        get_det_matrix_batched(
            nexcit,
            trial.cre_ex_a[nexcit],
            trial.anh_ex_a[nexcit],
            walker_batch.G0a,
            det_mat,
        )
        cofactor_matrix = numpy.zeros(
            (nwalkers, ndets_a, nexcit - 1, nexcit - 1), dtype=numpy.complex128
        )
        phases = c_phasea_ovlpb[:, trial.excit_map_a[nexcit]]
        for iex in range(nexcit):
            ps = trial.cre_ex_a[nexcit][:, iex]
            for jex in range(nexcit):
                qs = trial.anh_ex_a[nexcit][:, jex]
                get_cofactor_matrix_batched(
                    nwalkers, ndets_a, nexcit, iex, jex, det_mat, cofactor_matrix
                )
                dets_a = numpy.linalg.det(cofactor_matrix)
                rhs = (-1) ** (iex + jex) * dets_a * phases
                reduce_to_CI_tensor(
                    nwalkers,
                    ndets_a,
                    qs,
                    ps,
                    walker_batch.CIa,
                    rhs,
                )
    ndets_b = len(trial.cre_ex_b[nexcit])
    if ndets_b == 0:
        dets_b = None
    else:
        det_mat = numpy.zeros(
            (nwalkers, ndets_b, nexcit, nexcit), dtype=numpy.complex128
        )
        get_det_matrix_batched(
            nexcit,
            trial.cre_ex_b[nexcit],
            trial.anh_ex_b[nexcit],
            walker_batch.G0b,
            det_mat,
        )
        cofactor_matrix = numpy.zeros(
            (nwalkers, ndets_b, nexcit - 1, nexcit - 1), dtype=numpy.complex128
        )
        phases = c_phaseb_ovlpa[:, trial.excit_map_b[nexcit]]
        for iex in range(nexcit):
            ps = trial.cre_ex_b[nexcit][:, iex]
            for jex in range(nexcit):
                qs = trial.anh_ex_b[nexcit][:, jex]
                get_cofactor_matrix_batched(
                    nwalkers, ndets_b, nexcit, iex, jex, det_mat, cofactor_matrix
                )
                dets_b = numpy.linalg.det(cofactor_matrix)
                rhs = (-1) ** (iex + jex) * dets_b * phases
                reduce_to_CI_tensor(
                    nwalkers,
                    ndets_b,
                    qs,
                    ps,
                    walker_batch.CIb,
                    rhs,
                )
    return dets_a, dets_b


def build_CI_nfold_excitation_opt(
    nexcit, walker_batch, trial, c_phasea_ovlpb, c_phaseb_ovlpa
):
    """N-fold excitation contributions to CI intermediate for wicks.

    Optimized using numba.

    Parameters
    ----------
    nexcit : int
        Excitation level
    walker_batch : object
        MultiDetTrialWalkerBatch object.
    trial : object
        Trial wavefunction object.
    c_phasea_ovlpb : np.ndarray
        ci coeffs and phases of alpha dets * overlaps of beta determinants.
    c_phaseb_ovlpa : np.ndarray
        ci coeffs and phases of beta dets * overlaps of alpha determinants.
    Returns
    -------
    None, modifies walker_batch.CIa, and walker_batch.CIb inplace.
    """
    ndets_a = len(trial.cre_ex_a[nexcit])
    nwalkers = walker_batch.G0a.shape[0]
    if ndets_a == 0:
        pass
    else:
        phases = c_phasea_ovlpb[:, trial.excit_map_a[nexcit]]
        det_mat = numpy.zeros(
            (nwalkers, ndets_a, nexcit, nexcit), dtype=numpy.complex128
        )
        wk.build_det_matrix(
            trial.cre_ex_a[nexcit],
            trial.anh_ex_a[nexcit],
            trial.occ_map_a,
            trial.nfrozen,
            walker_batch.Ghalfa,
            det_mat,
        )
        cof_mat = numpy.zeros(
            (nwalkers, ndets_a, nexcit - 1, nexcit - 1), dtype=numpy.complex128
        )
        wk.reduce_CI_nfold(
            trial.cre_ex_a[nexcit],
            trial.anh_ex_a[nexcit],
            trial.occ_map_a,
            trial.nfrozen,
            phases,
            det_mat,
            cof_mat,
            walker_batch.CIa,
        )
    ndets_b = len(trial.cre_ex_b[nexcit])
    if ndets_b == 0:
        pass
    else:
        phases = c_phaseb_ovlpa[:, trial.excit_map_b[nexcit]]
        det_mat = numpy.zeros(
            (nwalkers, ndets_b, nexcit, nexcit), dtype=numpy.complex128
        )
        wk.build_det_matrix(
            trial.cre_ex_b[nexcit],
            trial.anh_ex_b[nexcit],
            trial.occ_map_b,
            trial.nfrozen,
            walker_batch.Ghalfb,
            det_mat,
        )
        cof_mat = numpy.zeros(
            (nwalkers, ndets_b, nexcit - 1, nexcit - 1), dtype=numpy.complex128
        )
        wk.reduce_CI_nfold(
            trial.cre_ex_b[nexcit],
            trial.anh_ex_b[nexcit],
            trial.occ_map_b,
            trial.nfrozen,
            phases,
            det_mat,
            cof_mat,
            walker_batch.CIb,
        )


@jit(nopython=True, fastmath=True)
def contract_CI(Q0_act, CI, Ghalf, G):
    """numba kernel to contract Q, CI and Ghalf to form G

    Parameters
    ----------
    Q0_act : numpy.ndarray
        1-G.
    CI : numpy.ndarray
        Intermediate tensor.
    Ghalf : numpy.ndarray
        Walker half rotated Green's function
    G: numpy.ndarray
        Walker Green's function
    Returns
    -------
    None, modifies G in place
    """
    nwalkers = Ghalf.shape[0]
    for iw in range(nwalkers):
        G[iw] += numpy.dot(Q0_act[iw], numpy.dot(CI[iw], Ghalf[iw]))


def greens_function_multi_det_wicks_opt(walker_batch, trial, build_full=False):
    """Compute walker's green's function using Wick's theorem.

    Parameters
    ----------
    walker_batch : object
        MultiDetTrialWalkerBatch object.
    trial : object
        Trial wavefunction object.
    Returns
    -------
    det : float64 / complex128
        Determinant of overlap matrix.
    """
    import time

    tot_ovlps = numpy.zeros(walker_batch.nwalkers, dtype=numpy.complex128)
    nbasis = walker_batch.Ga.shape[-1]

    na = walker_batch.nup
    nb = walker_batch.ndown

    walker_batch.Ga.fill(0.0 + 0.0j)
    walker_batch.Gb.fill(0.0 + 0.0j)

    # Build reference Green's functions and overlaps
    start = time.time()
    # Note abuse of naming convention this is really theta for the reference
    # determinant.
    G0a = numpy.zeros((walker_batch.nwalkers, nbasis, nbasis), dtype=numpy.complex128)
    G0b = numpy.zeros((walker_batch.nwalkers, nbasis, nbasis), dtype=numpy.complex128)
    ovlps0 = numpy.zeros((walker_batch.nwalkers), dtype=numpy.complex128)
    signs_a = numpy.zeros_like(ovlps0)
    signs_b = numpy.zeros_like(ovlps0)
    logdets_a = numpy.zeros_like(ovlps0)
    logdets_b = numpy.zeros_like(ovlps0)
    for iw in range(walker_batch.nwalkers):
        ovlp = numpy.dot(walker_batch.phia[iw].T, trial.psi[0, :, :na].conj())
        ovlp_inv = numpy.linalg.inv(ovlp)
        walker_batch.Ghalfa[iw] = numpy.dot(ovlp_inv, walker_batch.phia[iw].T)
        G0a[iw] = numpy.dot(trial.psi[0, :, :na].conj(), walker_batch.Ghalfa[iw])
        sign_a, log_ovlp_a = numpy.linalg.slogdet(ovlp)
        sign_b, log_ovlp_b = 1.0, 0.0
        ovlp = numpy.dot(walker_batch.phib[iw].T, trial.psi[0, :, na:].conj())
        sign_b, log_ovlp_b = numpy.linalg.slogdet(ovlp)
        walker_batch.Ghalfb[iw] = numpy.dot(
            numpy.linalg.inv(ovlp), walker_batch.phib[iw].T
        )
        G0b[iw] = numpy.dot(trial.psi[0, :, na:].conj(), walker_batch.Ghalfb[iw])
        signs_a[iw] = sign_a
        signs_b[iw] = sign_b
        logdets_a[iw] = log_ovlp_a
        logdets_b[iw] = log_ovlp_b
    ovlps0 = signs_a * signs_b * numpy.exp(logdets_a + logdets_b)
    walker_batch.G0a = G0a
    walker_batch.G0b = G0b
    walker_batch.Q0a = numpy.eye(nbasis)[None, :] - G0a
    walker_batch.Q0b = numpy.eye(nbasis)[None, :] - G0b
    walker_batch.CIa.fill(0.0 + 0.0j)
    walker_batch.CIb.fill(0.0 + 0.0j)
    dets_a_full, dets_b_full = compute_determinants_batched(
        walker_batch.Ghalfa, walker_batch.Ghalfb, trial
    )

    walker_batch.det_ovlpas = dets_a_full * trial.phase_a[None, :]  # phase included
    walker_batch.det_ovlpbs = dets_b_full * trial.phase_b[None, :]  # phase included
    ovlpa = walker_batch.det_ovlpas
    ovlpb = walker_batch.det_ovlpbs

    start = time.time()
    c_phasea_ovlpb = numpy.einsum(
        "wJ,J->wJ", ovlpb, trial.phase_a * trial.coeffs.conj(), optimize=True
    )
    c_phaseb_ovlpa = numpy.einsum(
        "wJ,J->wJ", ovlpa, trial.phase_b * trial.coeffs.conj(), optimize=True
    )
    # contribution 1 (disconnected diagrams)
    ovlps = numpy.einsum("wJ,J->w", ovlpa * ovlpb, trial.coeffs.conj(), optimize=True)
    walker_batch.Ga += numpy.einsum("w,wpq->wpq", ovlps, G0a, optimize=True)
    walker_batch.Gb += numpy.einsum("w,wpq->wpq", ovlps, G0b, optimize=True)
    # intermediates for contribution 2 (connected diagrams)
    build_CI_single_excitation_opt(walker_batch, trial, c_phasea_ovlpb, c_phaseb_ovlpa)
    build_CI_double_excitation_opt(walker_batch, trial, c_phasea_ovlpb, c_phaseb_ovlpa)
    if trial.max_excite >= 3:
        build_CI_triple_excitation_opt(
            walker_batch, trial, c_phasea_ovlpb, c_phaseb_ovlpa
        )
    for iexcit in range(4, trial.max_excite + 1):
        start = time.time()
        build_CI_nfold_excitation_opt(
            iexcit, walker_batch, trial, c_phasea_ovlpb, c_phaseb_ovlpa
        )
    # contribution 2 (connected diagrams)
    # Frozen orbitals not in original active space calculation but reincluded in
    # AFQMC
    act_orb = trial.act_orb_alpha
    contract_CI(
        walker_batch.Q0a[:, :, act_orb].copy(),
        walker_batch.CIa,
        walker_batch.Ghalfa[:, act_orb].copy(),
        walker_batch.Ga,
    )
    act_orb = trial.act_orb_beta
    contract_CI(
        walker_batch.Q0b[:, :, act_orb].copy(),
        walker_batch.CIb,
        walker_batch.Ghalfb[:, act_orb].copy(),
        walker_batch.Gb,
    )
    # multiplying everything by reference overlap
    ovlps *= ovlps0
    walker_batch.Ga *= (ovlps0 / ovlps)[:, None, None]
    walker_batch.Gb *= (ovlps0 / ovlps)[:, None, None]
    walker_batch.det_ovlpas[:, 0] = signs_a * numpy.exp(logdets_a)
    walker_batch.det_ovlpbs[:, 0] = signs_b * numpy.exp(logdets_b)
    return ovlps
