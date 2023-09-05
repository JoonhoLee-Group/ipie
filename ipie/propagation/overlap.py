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
#

import itertools

import numpy
import scipy.linalg

from ipie.estimators.kernels.cpu import wicks as wk
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import synchronize


def calc_overlap_single_det_uhf(walkers: "UHFWalkers", trial: "SingleDet"):
    """Caculate overlap with single det trial wavefunction.

    Parameters
    ----------
    walkers : object
        WalkerBatch object (this stores some intermediates for the particular trial wfn).
    trial : object
        Trial wavefunction object.

    Returns
    -------
    ot : float / complex
        Overlap.
    """
    ndown = walkers.ndown
    ovlp_a = xp.einsum("wmi,mj->wij", walkers.phia, trial.psi0a.conj(), optimize=True)
    sign_a, log_ovlp_a = xp.linalg.slogdet(ovlp_a)

    if ndown > 0 and not walkers.rhf:
        ovlp_b = xp.einsum("wmi,mj->wij", walkers.phib, trial.psi0b.conj(), optimize=True)
        sign_b, log_ovlp_b = xp.linalg.slogdet(ovlp_b)
        ot = sign_a * sign_b * xp.exp(log_ovlp_a + log_ovlp_b - walkers.log_shift)
    elif ndown > 0 and walkers.rhf:
        ot = sign_a * sign_a * xp.exp(log_ovlp_a + log_ovlp_a - walkers.log_shift)
    elif ndown == 0:
        ot = sign_a * xp.exp(log_ovlp_a - walkers.log_shift)

    synchronize()

    return ot


def calc_overlap_single_det_ghf(walkers: "GHFWalkers", trial: "SingleDet"):
    ovlp = xp.einsum("wmi,mj->wij", walkers.phi, trial.psi0.conj(), optimize=True)
    sign, log_ovlp = xp.linalg.slogdet(ovlp)

    ot = sign * xp.exp(log_ovlp - walkers.log_shift)

    return ot


# overlap for a given determinant
# note that the phase is not included
def get_overlap_one_det_wicks(nex_a, cre_a, anh_a, G0a, nex_b, cre_b, anh_b, G0b):
    ovlp_a = 0.0 + 0.0j
    ovlp_b = 0.0 + 0.0j
    if nex_a == 1:
        p = cre_a[0]
        q = anh_a[0]
        ovlp_a = G0a[p, q]
    elif nex_a == 2:
        p = cre_a[0]
        q = anh_a[0]
        r = cre_a[1]
        s = anh_a[1]
        ovlp_a = G0a[p, q] * G0a[r, s] - G0a[p, s] * G0a[r, q]
    elif nex_a == 3:
        p = cre_a[0]
        q = anh_a[0]
        r = cre_a[1]
        s = anh_a[1]
        t = cre_a[2]
        u = anh_a[2]
        ovlp_a = G0a[p, q] * (G0a[r, s] * G0a[t, u] - G0a[r, u] * G0a[t, s])
        ovlp_a -= G0a[p, s] * (G0a[r, q] * G0a[t, u] - G0a[r, u] * G0a[t, q])
        ovlp_a += G0a[p, u] * (G0a[r, q] * G0a[t, s] - G0a[r, s] * G0a[t, q])
    else:
        det_a = numpy.zeros((nex_a, nex_a), dtype=numpy.complex128)
        for iex in range(nex_a):
            p = cre_a[iex]
            q = anh_a[iex]
            det_a[iex, iex] = G0a[p, q]
            for jex in range(iex + 1, nex_a):
                r = cre_a[jex]
                s = anh_a[jex]
                det_a[iex, jex] = G0a[p, s]
                det_a[jex, iex] = G0a[r, q]
        ovlp_a = numpy.linalg.det(det_a)

    if nex_b == 1:
        p = cre_b[0]
        q = anh_b[0]
        ovlp_b = G0b[p, q]
    elif nex_b == 2:
        p = cre_b[0]
        q = anh_b[0]
        r = cre_b[1]
        s = anh_b[1]
        ovlp_b = G0b[p, q] * G0b[r, s] - G0b[p, s] * G0b[r, q]
    elif nex_b == 3:
        p = cre_b[0]
        q = anh_b[0]
        r = cre_b[1]
        s = anh_b[1]
        t = cre_b[2]
        u = anh_b[2]
        ovlp_b = G0b[p, q] * (G0b[r, s] * G0b[t, u] - G0b[r, u] * G0b[t, s])
        ovlp_b -= G0b[p, s] * (G0b[r, q] * G0b[t, u] - G0b[r, u] * G0b[t, q])
        ovlp_b += G0b[p, u] * (G0b[r, q] * G0b[t, s] - G0b[r, s] * G0b[t, q])
    else:
        det_b = numpy.zeros((nex_b, nex_b), dtype=numpy.complex128)
        for iex in range(nex_b):
            p = cre_b[iex]
            q = anh_b[iex]
            det_b[iex, iex] = G0b[p, q]
            for jex in range(iex + 1, nex_b):
                r = cre_b[jex]
                s = anh_b[jex]
                det_b[iex, jex] = G0b[p, s]
                det_b[jex, iex] = G0b[r, q]
        ovlp_b = numpy.linalg.det(det_b)

    return ovlp_a, ovlp_b


def calc_overlap_multi_det_wicks(walker_batch, trial):
    """Calculate overlap with multidet trial wavefunction using Wick's Theorem.

    Parameters
    ----------
    walker_batch : object
        WalkerBatch object (this stores some intermediates for the particular trial wfn).
    trial : object
        Trial wavefunction object.

    Returns
    -------
    ovlps : float / complex
        Overlap.
    """
    psi0a = trial.psi0a  # reference det
    psi0b = trial.psi0b  # reference det

    ovlps = []
    for iw in range(walker_batch.nwalkers):
        phia = walker_batch.phia[iw]
        Oalpha = numpy.dot(psi0a.conj().T, phia)
        sign_a, logdet_a = numpy.linalg.slogdet(Oalpha)
        logdet_b, sign_b = 0.0, 1.0
        theta_a = scipy.linalg.inv(Oalpha.T) @ phia.T
        greens_a = psi0a.conj() @ theta_a

        phib = walker_batch.phib[iw]
        Obeta = numpy.dot(psi0b.conj().T, phib)
        sign_b, logdet_b = numpy.linalg.slogdet(Obeta)
        theta_b = scipy.linalg.inv(Obeta.T) @ phib.T
        greens_b = psi0b.conj() @ theta_b

        ovlp0 = sign_a * sign_b * numpy.exp(logdet_a + logdet_b)

        ovlp = 0.0 + 0.0j
        ovlp += trial.coeffs[0].conj()
        for jdet in range(1, trial.num_dets):
            nex_a = len(trial.anh_a[jdet])
            nex_b = len(trial.anh_b[jdet])
            ovlp_a, ovlp_b = get_overlap_one_det_wicks(
                nex_a,
                trial.cre_a[jdet],
                trial.anh_a[jdet],
                greens_a,
                nex_b,
                trial.cre_b[jdet],
                trial.anh_b[jdet],
                greens_b,
            )

            tmp = (
                trial.coeffs[jdet].conj()
                * ovlp_a
                * ovlp_b
                * trial.phase_a[jdet]
                * trial.phase_b[jdet]
            )
            ovlp += tmp
        ovlp *= ovlp0

        ovlps += [ovlp]

    ovlps = numpy.array(ovlps, dtype=numpy.complex128)

    return ovlps


def get_dets_single_excitation_batched(G0wa, G0wb, trial):
    """Compute alpha and beta overlaps at single excitation level.

    Parameters
    ----------
    G0wa : numpy.ndarray
        Alpha reference Green's function (all walkers).
    G0wb : numpy.ndarray
        Bewa reference Green's function (all walkers).
    trial : MultiSlater
        Trial wavefunction instance.

    Returns
    -------
    dets_a : numpy.ndarray
        Alpha overlaps shape (nw, ndets_excit1_alpha)
    dets_b : numpy.ndarray
        Beta overlaps shape (nw, ndets_excit1_beta)
    """
    ndets_a = trial.cre_ex_a[1].shape[0]
    ndets_b = trial.cre_ex_b[1].shape[0]
    if ndets_a == 0:
        dets_a = None
    else:
        # arrays of length ndet_per_single_excitation
        ps, qs = trial.cre_ex_a[1][:, 0], trial.anh_ex_a[1][:, 0]
        dets_a = G0wa[:, ps, qs]
    if ndets_b == 0:
        dets_b = None
    else:
        ps, qs = trial.cre_ex_b[1][:, 0], trial.anh_ex_b[1][:, 0]
        dets_b = G0wb[:, ps, qs]
    return dets_a, dets_b


def get_dets_single_excitation_batched_opt(G0wa, G0wb, trial):
    """Compute alpha and beta overlaps at single excitation level.

    Parameters
    ----------
    G0wa : numpy.ndarray
        Alpha reference Green's function (all walkers).
    G0wb : numpy.ndarray
        Bewa reference Green's function (all walkers).
    trial : MultiSlater
        Trial wavefunction instance.

    Returns
    -------
    dets_a : numpy.ndarray
        Alpha overlaps shape (nw, ndets_excit1_alpha)
    dets_b : numpy.ndarray
        Beta overlaps shape (nw, ndets_excit1_beta)
    """
    ndets_a = trial.cre_ex_a[1].shape[0]
    ndets_b = trial.cre_ex_b[1].shape[0]
    nwalkers = G0wa.shape[0]
    if ndets_a == 0:
        dets_a = None
    else:
        # arrays of length ndet_per_single_excitation
        dets_a = numpy.zeros((nwalkers, ndets_a), dtype=numpy.complex128)
        wk.get_dets_singles(
            trial.cre_ex_a[1],
            trial.anh_ex_a[1],
            trial.occ_map_a,
            trial.nfrozen,
            G0wa,
            dets_a,
        )
    if ndets_b == 0:
        dets_b = None
    else:
        dets_b = numpy.zeros((nwalkers, ndets_b), dtype=numpy.complex128)
        wk.get_dets_singles(
            trial.cre_ex_b[1],
            trial.anh_ex_b[1],
            trial.occ_map_b,
            trial.nfrozen,
            G0wb,
            dets_b,
        )
    return dets_a, dets_b


def get_dets_double_excitation_batched(G0wa, G0wb, trial):
    """Compute alpha and beta overlaps at double excitation level.

    Parameters
    ----------
    G0wa : numpy.ndarray
        Alpha reference Green's function (all walkers).
    G0wb : numpy.ndarray
        Bewa reference Green's function (all walkers).
    trial : MultiSlater
        Trial wavefunction instance.

    Returns
    -------
    dets_a : numpy.ndarray
        Alpha overlaps shape (nw, ndets_excit2_alpha)
    dets_b : numpy.ndarray
        Beta overlaps shape (nw, ndets_excit2_beta)
    """
    if trial.cre_ex_a[2].shape[0] == 0:
        dets_a = None
    else:
        ps, qs = trial.cre_ex_a[2][:, 0], trial.anh_ex_a[2][:, 0]
        rs, ss = trial.cre_ex_a[2][:, 1], trial.anh_ex_a[2][:, 1]
        dets_a = G0wa[:, ps, qs] * G0wa[:, rs, ss] - G0wa[:, ps, ss] * G0wa[:, rs, qs]
    if trial.cre_ex_b[2].shape[0] == 0:
        dets_b = None
    else:
        ps, qs = trial.cre_ex_b[2][:, 0], trial.anh_ex_b[2][:, 0]
        rs, ss = trial.cre_ex_b[2][:, 1], trial.anh_ex_b[2][:, 1]
        dets_b = G0wb[:, ps, qs] * G0wb[:, rs, ss] - G0wb[:, ps, ss] * G0wb[:, rs, qs]
    return dets_a, dets_b


def get_dets_triple_excitation_batched(G0wa, G0wb, trial):
    """Compute alpha and beta overlaps at triple excitation level.

    Parameters
    ----------
    G0wa : numpy.ndarray
        Alpha reference Green's function (all walkers).
    G0wb : numpy.ndarray
        Bewa reference Green's function (all walkers).
    trial : MultiSlater
        Trial wavefunction instance.

    Returns
    -------
    dets_a : numpy.ndarray
        Alpha overlaps shape (nw, ndets_excit3_alpha)
    dets_b : numpy.ndarray
        Beta overlaps shape (nw, ndets_excit3_beta)
    """
    if trial.cre_ex_a[3].shape[0] == 0:
        dets_a = None
    else:
        ps, qs = trial.cre_ex_a[3][:, 0], trial.anh_ex_a[3][:, 0]
        rs, ss = trial.cre_ex_a[3][:, 1], trial.anh_ex_a[3][:, 1]
        ts, us = trial.cre_ex_a[3][:, 2], trial.anh_ex_a[3][:, 2]
        dets_a = G0wa[:, ps, qs] * (
            G0wa[:, rs, ss] * G0wa[:, ts, us] - G0wa[:, rs, us] * G0wa[:, ts, ss]
        )
        dets_a -= G0wa[:, ps, ss] * (
            G0wa[:, rs, qs] * G0wa[:, ts, us] - G0wa[:, rs, us] * G0wa[:, ts, qs]
        )
        dets_a += G0wa[:, ps, us] * (
            G0wa[:, rs, qs] * G0wa[:, ts, ss] - G0wa[:, rs, ss] * G0wa[:, ts, qs]
        )
    if trial.cre_ex_b[3].shape[0] == 0:
        dets_b = None
    else:
        ps, qs = trial.cre_ex_b[3][:, 0], trial.anh_ex_b[3][:, 0]
        rs, ss = trial.cre_ex_b[3][:, 1], trial.anh_ex_b[3][:, 1]
        ts, us = trial.cre_ex_b[3][:, 2], trial.anh_ex_b[3][:, 2]
        dets_b = G0wb[:, ps, qs] * (
            G0wb[:, rs, ss] * G0wb[:, ts, us] - G0wb[:, rs, us] * G0wb[:, ts, ss]
        )
        dets_b -= G0wb[:, ps, ss] * (
            G0wb[:, rs, qs] * G0wb[:, ts, us] - G0wb[:, rs, us] * G0wb[:, ts, qs]
        )
        dets_b += G0wb[:, ps, us] * (
            G0wb[:, rs, qs] * G0wb[:, ts, ss] - G0wb[:, rs, ss] * G0wb[:, ts, qs]
        )
    return dets_a, dets_b


def get_dets_nfold_excitation_batched(nexcit, G0wa, G0wb, trial):
    """Compute alpha and beta overlaps at arbitrary excitation level.

    Parameters
    ----------
    G0wa : numpy.ndarray
        Alpha reference Green's function (all walkers).
    G0wb : numpy.ndarray
        Bewa reference Green's function (all walkers).
    trial : MultiSlater
        Trial wavefunction instance.

    Returns
    -------
    dets_a : numpy.ndarray
        Alpha overlaps shape (nw, ndets_excitn_alpha)
    dets_b : numpy.ndarray
        Beta overlaps shape (nw, ndets_excitn_beta)
    """
    ndets_a = len(trial.cre_ex_a[nexcit])
    nwalkers = G0wa.shape[0]
    indices = numpy.indices((nexcit, nexcit))
    if ndets_a == 0:
        dets_a = None
    else:
        det_mat = numpy.zeros((nwalkers, ndets_a, nexcit, nexcit), dtype=numpy.complex128)
        ps = trial.cre_ex_a[nexcit]
        qs = trial.anh_ex_a[nexcit]
        psqs = numpy.array([list(itertools.product(p, q)) for (p, q) in zip(ps, qs)])
        _shape = (nwalkers, ndets_a, nexcit, nexcit)
        det_mat[:, :, indices[0], indices[1]] = G0wa[:, psqs[:, :, 0], psqs[:, :, 1]].reshape(
            _shape
        )
        dets_a = numpy.linalg.det(det_mat)
    ndets_b = len(trial.cre_ex_b[nexcit])
    if ndets_b == 0:
        dets_b = None
    else:
        det_mat = numpy.zeros((nwalkers, ndets_b, nexcit, nexcit), dtype=numpy.complex128)
        ps = trial.cre_ex_b[nexcit]
        qs = trial.anh_ex_b[nexcit]
        psqs = numpy.array([list(itertools.product(p, q)) for (p, q) in zip(ps, qs)])
        _shape = (nwalkers, ndets_b, nexcit, nexcit)
        det_mat[:, :, indices[0], indices[1]] = G0wb[:, psqs[:, :, 0], psqs[:, :, 1]].reshape(
            _shape
        )
        dets_b = numpy.linalg.det(det_mat)
    return dets_a, dets_b


def get_dets_double_excitation_batched_opt(G0wa, G0wb, trial):
    """Compute alpha and beta overlaps at double excitation level.

    Parameters
    ----------
    G0wa : numpy.ndarray
        Alpha reference Green's function (all walkers).
    G0wb : numpy.ndarray
        Bewa reference Green's function (all walkers).
    trial : MultiSlater
        Trial wavefunction instance.

    Returns
    -------
    dets_a : numpy.ndarray
        Alpha overlaps shape (nw, ndets_excit2_alpha)
    dets_b : numpy.ndarray
        Beta overlaps shape (nw, ndets_excit2_beta)
    """
    ndets_a = trial.cre_ex_a[2].shape[0]
    ndets_b = trial.cre_ex_b[2].shape[0]
    nwalkers = G0wa.shape[0]
    if ndets_a == 0:
        dets_a = None
    else:
        # arrays of length ndet_per_single_excitation
        dets_a = numpy.zeros((nwalkers, ndets_a), dtype=numpy.complex128)
        wk.get_dets_doubles(
            trial.cre_ex_a[2],
            trial.anh_ex_a[2],
            trial.occ_map_a,
            trial.nfrozen,
            G0wa,
            dets_a,
        )
    if ndets_b == 0:
        dets_b = None
    else:
        dets_b = numpy.zeros((nwalkers, ndets_b), dtype=numpy.complex128)
        wk.get_dets_doubles(
            trial.cre_ex_b[2],
            trial.anh_ex_b[2],
            trial.occ_map_b,
            trial.nfrozen,
            G0wb,
            dets_b,
        )
    return dets_a, dets_b


def get_dets_triple_excitation_batched_opt(G0wa, G0wb, trial):
    """Compute alpha and beta overlaps at triple excitation level.

    Parameters
    ----------
    G0wa : numpy.ndarray
        Alpha reference Green's function (all walkers).
    G0wb : numpy.ndarray
        Bewa reference Green's function (all walkers).
    trial : MultiSlater
        Trial wavefunction instance.

    Returns
    -------
    dets_a : numpy.ndarray
        Alpha overlaps shape (nw, ndets_excit3_alpha)
    dets_b : numpy.ndarray
        Beta overlaps shape (nw, ndets_excit3_beta)
    """
    ndets_a = trial.cre_ex_a[3].shape[0]
    ndets_b = trial.cre_ex_b[3].shape[0]
    nwalkers = G0wa.shape[0]
    if ndets_a == 0:
        dets_a = None
    else:
        # arrays of length ndet_per_single_excitation
        dets_a = numpy.zeros((nwalkers, ndets_a), dtype=numpy.complex128)
        wk.get_dets_triples(
            trial.cre_ex_a[3],
            trial.anh_ex_a[3],
            trial.occ_map_a,
            trial.nfrozen,
            G0wa,
            dets_a,
        )
    if ndets_b == 0:
        dets_b = None
    else:
        dets_b = numpy.zeros((nwalkers, ndets_b), dtype=numpy.complex128)
        wk.get_dets_triples(
            trial.cre_ex_b[3],
            trial.anh_ex_b[3],
            trial.occ_map_b,
            trial.nfrozen,
            G0wb,
            dets_b,
        )
    return dets_a, dets_b


def get_dets_nfold_excitation_batched_opt(nexcit, G0wa, G0wb, trial):
    """Compute alpha and beta overlaps at arbitrary excitation level.

    Parameters
    ----------
    G0wa : numpy.ndarray
        Alpha reference Green's function (all walkers).
    G0wb : numpy.ndarray
        Bewa reference Green's function (all walkers).
    trial : MultiSlater
        Trial wavefunction instance.

    Returns
    -------
    dets_a : numpy.ndarray
        Alpha overlaps shape (nw, ndets_excitn_alpha)
    dets_b : numpy.ndarray
        Beta overlaps shape (nw, ndets_excitn_beta)
    """
    ndets_a = len(trial.cre_ex_a[nexcit])
    nwalkers = G0wa.shape[0]
    ndets_a = trial.cre_ex_a[nexcit].shape[0]
    ndets_b = trial.cre_ex_b[nexcit].shape[0]
    nwalkers = G0wa.shape[0]
    if ndets_a == 0:
        dets_a = None
    else:
        # arrays of length ndet_per_single_excitation
        dets_a = numpy.zeros((nwalkers, ndets_a), dtype=numpy.complex128)
        wk.get_dets_nfold(
            trial.cre_ex_a[nexcit],
            trial.anh_ex_a[nexcit],
            trial.occ_map_a,
            trial.nfrozen,
            G0wa,
            dets_a,
        )
    if ndets_b == 0:
        dets_b = None
    else:
        dets_b = numpy.zeros((nwalkers, ndets_b), dtype=numpy.complex128)
        wk.get_dets_nfold(
            trial.cre_ex_b[nexcit],
            trial.anh_ex_b[nexcit],
            trial.occ_map_b,
            trial.nfrozen,
            G0wb,
            dets_b,
        )
    return dets_a, dets_b


def compute_determinants_batched(G0a, G0b, trial):
    nwalker = G0a.shape[0]
    ndets = len(trial.coeffs)
    dets_a_full = numpy.ones((nwalker, ndets), dtype=numpy.complex128)
    dets_b_full = numpy.ones((nwalker, ndets), dtype=numpy.complex128)
    # Use low level excitation optimizations
    # TODO: Optimization Use one buffer + one remapping at the end.
    dets_a, dets_b = get_dets_single_excitation_batched_opt(G0a, G0b, trial)
    dets_a_full[:, trial.excit_map_a[1]] = dets_a
    dets_b_full[:, trial.excit_map_b[1]] = dets_b
    if trial.max_excite < 2:
        return dets_a_full, dets_b_full
    dets_a, dets_b = get_dets_double_excitation_batched_opt(G0a, G0b, trial)
    dets_a_full[:, trial.excit_map_a[2]] = dets_a
    dets_b_full[:, trial.excit_map_b[2]] = dets_b
    if trial.max_excite < 3:
        return dets_a_full, dets_b_full
    dets_a, dets_b = get_dets_triple_excitation_batched_opt(G0a, G0b, trial)
    dets_a_full[:, trial.excit_map_a[3]] = dets_a
    dets_b_full[:, trial.excit_map_b[3]] = dets_b
    for iexcit in range(4, trial.max_excite + 1):
        dets_a, dets_b = get_dets_nfold_excitation_batched_opt(iexcit, G0a, G0b, trial)
        dets_a_full[:, trial.excit_map_a[iexcit]] = dets_a
        dets_b_full[:, trial.excit_map_b[iexcit]] = dets_b

    return dets_a_full, dets_b_full


def calc_overlap_multi_det_wicks_opt(walker_batch, trial):
    """Calculate overlap with multidet trial wavefunction using Wick's Theorem.

    Parameters
    ----------
    walker_batch : object
        WalkerBatch object (this stores some intermediates for the particular trial wfn).
    trial : object
        Trial wavefunction object.

    Returns
    -------
    ovlps : float / complex
        Overlap.
    """
    ovlps = []
    ovlp_mats_a = numpy.einsum("wmi,mj->wji", walker_batch.phia, trial.psi0a.conj(), optimize=True)
    signs_a, logdets_a = numpy.linalg.slogdet(ovlp_mats_a)
    ovlp_mats_b = numpy.einsum("wmi,mj->wji", walker_batch.phib, trial.psi0b.conj(), optimize=True)
    signs_b, logdets_b = numpy.linalg.slogdet(ovlp_mats_b)
    ovlps0 = signs_a * signs_b * numpy.exp(logdets_a + logdets_b)
    inv_ovlps_a = numpy.linalg.inv(ovlp_mats_a)
    theta_a = numpy.einsum("wmi,wij->wjm", walker_batch.phia, inv_ovlps_a, optimize=True)
    inv_ovlps_b = numpy.linalg.inv(ovlp_mats_b)
    theta_b = numpy.einsum("wmi,wij->wjm", walker_batch.phib, inv_ovlps_b, optimize=True)
    # Use low level excitation optimizations
    ovlps = numpy.array(ovlps, dtype=numpy.complex128)
    dets_a_full, dets_b_full = compute_determinants_batched(theta_a, theta_b, trial)

    dets_full = ovlps0[:, None] * dets_a_full * dets_b_full
    # This could be precomputed?
    det_factors = trial.coeffs.conj() * trial.phase_a * trial.phase_b
    ovlps = numpy.dot(dets_full, det_factors)
    # ovlps = numpy.einsum(
    # 'w,J,wJ,wJ,J,J->w',
    # ovlps0,
    # trial.coeffs.conj(),
    # dets_a_full,
    # dets_b_full,
    # trial.phase_a,
    # trial.phase_b,
    # optimize=True)

    return ovlps


def calc_overlap_multi_det(walker_batch, trial):
    """Caculate overlap with multidet trial wavefunction.

    Parameters
    ----------
    walker_batch : object
        WalkerBatch object (this stores some intermediates for the particular trial wfn).
    trial : object
        Trial wavefunction object.

    Returns
    -------
    ovlp : float / complex
        Overlap.
    """
    nup = walker_batch.nup
    for iw in range(walker_batch.nwalkers):
        for i, det in enumerate(trial.psi):
            Oup = numpy.dot(det[:, :nup].conj().T, walker_batch.phia[iw])
            Odn = numpy.dot(det[:, nup:].conj().T, walker_batch.phib[iw])
            sign_a, logdet_a = numpy.linalg.slogdet(Oup)
            sign_b, logdet_b = numpy.linalg.slogdet(Odn)
            walker_batch.det_ovlpas[iw, i] = sign_a * numpy.exp(logdet_a)
            walker_batch.det_ovlpbs[iw, i] = sign_b * numpy.exp(logdet_b)
    return numpy.einsum(
        "wi,wi,i->w",
        walker_batch.det_ovlpas,
        walker_batch.det_ovlpbs,
        trial.coeffs.conj(),
        optimize=True,
    )


### Legacy overlap functions useful for testing.
def get_det_matrix_batched(nex, cre, anh, G0, det_matrix):
    nwalker = G0.shape[0]
    ndet = cre.shape[0]

    for iw in range(nwalker):
        for idet in range(ndet):
            for iex in range(nex):
                p = cre[idet][iex]
                q = anh[idet][iex]
                det_matrix[iw, idet, iex, iex] = G0[iw, p, q]
                for jex in range(iex + 1, nex):
                    r = cre[idet][jex]
                    s = anh[idet][jex]
                    det_matrix[iw, idet, iex, jex] = G0[iw, p, s]
                    det_matrix[iw, idet, jex, iex] = G0[iw, r, q]
    return det_matrix


def reduce_to_CI_tensor(nwalker, ndet_level, ps, qs, tensor, rhs):
    for iw in range(nwalker):
        for idet in range(ndet_level):
            # += not supported in cython for complex types.
            tensor[iw, ps[idet], qs[idet]] = tensor[iw, ps[idet], qs[idet]] + rhs[iw, idet]


def get_cofactor_matrix_batched(nwalker, ndet, nexcit, row, col, det_matrix, cofactor):
    if nexcit - 1 <= 0:
        for iw in range(nwalker):
            for idet in range(ndet):
                cofactor[iw, idet, 0, 0] = 1.0

    for iw in range(nwalker):
        for idet in range(ndet):
            for i in range(nexcit):
                ishift = 0
                jshift = 0
                if i > row:
                    ishift = 1
                if i == nexcit - 1 and row == nexcit - 1:
                    continue
                for j in range(nexcit):
                    if j > col:
                        jshift = 1
                    if j == nexcit - 1 and col == nexcit - 1:
                        continue
                    cofactor[iw, idet, i - ishift, j - jshift] = det_matrix[iw, idet, i, j]


def get_cofactor_matrix_4_batched(
    nwalker, ndet, nexcit, row_1, col_1, row_2, col_2, det_matrix, cofactor
):
    ncols = det_matrix.shape[3]
    if ncols - 2 <= 0:
        for iw in range(nwalker):
            for idet in range(ndet):
                cofactor[iw, idet, 0, 0] = 1.0
        return

    for iw in range(nwalker):
        for idet in range(ndet):
            for i in range(nexcit):
                ishift_1 = 0
                jshift_1 = 0
                ishift_2 = 0
                jshift_2 = 0
                if i > row_1:
                    ishift_1 = 1
                if i > row_2:
                    ishift_2 = 1
                if i == nexcit - 2 and (row_1 == nexcit - 2):
                    continue
                if i == nexcit - 1 and (row_2 == nexcit - 1):
                    continue
                for j in range(nexcit):
                    if j > col_1:
                        jshift_1 = 1
                    if j > col_2:
                        jshift_2 = 1
                    if j == nexcit - 2 and (col_1 == nexcit - 2):
                        continue
                    if j == nexcit - 1 and (col_2 == nexcit - 1):
                        continue
                    # if col_1 == nexcit-2 or col_2 == nexcit-2:
                    # continue
                    # print(i, j, i - (ishift_1+ishift_2), j -
                    # (jshift_1+jshift_2), nexcit-2, row_1, row_2)
                    cofactor[
                        iw,
                        idet,
                        max(i - (ishift_1 + ishift_2), 0),
                        max(j - (jshift_1 + jshift_2), 0),
                    ] = det_matrix[iw, idet, i, j]
