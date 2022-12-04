
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
# Author: Fionn Malone <fionn.malone@gmail.com>
#

import numpy
from numba import jit


# element wise
@jit(nopython=True, fastmath=True)
def dot_real_cplx(
    A,
    B_real,
    B_cplx,
):
    """Element wise multiplication of a real number with a complex one.

    C = A * B

    Numba complains if types aren't matched so split it up.

    Parameters
    ----------
    A : float
        Real number / array.
    B : complex
        Complex number / array.

    Returns
    -------
    C : complex
        result
    """

    return A * B_real + 1j * (A * B_cplx)


# Overlap

# Note mapping arrays account for occupied indices not matching compressed
# format which may arise when the reference determinant does not follow aufbau
# principle (i.e. not doubly occupied up to the fermi level.
# e.g. D0a = [0, 1, 4], mapping = [0, 1, 0, 0, 2]
# mapping[orb] is then used to address arrays of dimension nocc * nmo and
# similar (half rotated Green's functio) and avoid out of bounds errors.

@jit(nopython=True, fastmath=True)
def get_dets_singles(cre, anh, mapping, offset, G0, dets):
    """Get overlap from singly excited Slater-Determinants.

    Parameters
    ----------
    cre : np.ndarray
        Array containing orbitals excitations of occupied.
    anh : np.ndarray
        Array containing orbitals excitations to virtuals.
    mapping : np.ndarray
        Map original (occupied) orbital to index in compressed form.
    offset : int
        Offset for frozen core.
    G0 : np.ndarray
        (Half rotated) batched Green's function.
    dets : np.ndarray
        Output array of determinants <D_I|phi>.

    Returns
    -------
    None
    """
    qs = anh[:, 0] + offset
    ndets = qs.shape[0]
    nwalkers = G0.shape[0]
    for idet in range(ndets):
        p = mapping[cre[idet, 0]] + offset
        dets[:, idet] = G0[:, p, qs[idet]]


@jit(nopython=True, fastmath=True)
def get_dets_doubles(cre, anh, mapping, offset, G0, dets):
    """Get overlap from double excited Slater-Determinants.

    Parameters
    ----------
    cre : np.ndarray
        Array containing orbitals excitations of occupied.
    anh : np.ndarray
        Array containing orbitals excitations to virtuals.
    mapping : np.ndarray
        Map original (occupied) orbital to index in compressed form.
    offset : int
        Offset for frozen core.
    G0 : np.ndarray
        (Half rotated) batched Green's function.
    dets : np.ndarray
        Output array of determinants <D_I|phi>.

    Returns
    -------
    None
    """
    qs = anh[:, 0] + offset
    ss = anh[:, 1] + offset
    ndets = qs.shape[0]
    nwalkers = G0.shape[0]
    for iw in range(nwalkers):
        for idet in range(ndets):
            p = mapping[cre[idet, 0]] + offset
            r = mapping[cre[idet, 1]] + offset
            dets[iw, idet] = (
                G0[iw, p, qs[idet]] * G0[iw, r, ss[idet]]
                - G0[iw, p, ss[idet]] * G0[iw, r, qs[idet]]
            )


@jit(nopython=True, fastmath=True)
def get_dets_triples(
    cre,
    anh,
    mapping,
    offset,
    G0,
    dets,
):
    """Get overlap from triply excited Slater-Determinants.

    Parameters
    ----------
    cre : np.ndarray
        Array containing orbitals excitations of occupied.
    anh : np.ndarray
        Array containing orbitals excitations to virtuals.
    mapping : np.ndarray
        Map original (occupied) orbital to index in compressed form.
    offset : int
        Offset for frozen core.
    G0 : np.ndarray
        (Half rotated) batched Green's function.
    dets : np.ndarray
        Output array of determinants <D_I|phi>.

    Returns
    -------
    None
    """
    ndets = len(cre)
    nwalkers = G0.shape[0]
    for iw in range(nwalkers):
        for idet in range(ndets):
            ps, qs = mapping[cre[idet, 0]] + offset, anh[idet, 0] + offset
            rs, ss = mapping[cre[idet, 1]] + offset, anh[idet, 1] + offset
            ts, us = mapping[cre[idet, 2]] + offset, anh[idet, 2] + offset
            dets[iw, idet] = G0[iw, ps, qs] * (
                G0[iw, rs, ss] * G0[iw, ts, us] - G0[iw, rs, us] * G0[iw, ts, ss]
            )
            dets[iw, idet] -= G0[iw, ps, ss] * (
                G0[iw, rs, qs] * G0[iw, ts, us] - G0[iw, rs, us] * G0[iw, ts, qs]
            )
            dets[iw, idet] += G0[iw, ps, us] * (
                G0[iw, rs, qs] * G0[iw, ts, ss] - G0[iw, rs, ss] * G0[iw, ts, qs]
            )


@jit(nopython=True, fastmath=True)
def get_dets_nfold(cre, anh, mapping, offset, G0, dets):
    """Get overlap from n-fold excited Slater-Determinants.

    Parameters
    ----------
    cre : np.ndarray
        Array containing orbitals excitations of occupied.
    anh : np.ndarray
        Array containing orbitals excitations to virtuals.
    mapping : np.ndarray
        Map original (occupied) orbital to index in compressed form.
    offset : int
        Offset for frozen core.
    G0 : np.ndarray
        (Half rotated) batched Green's function.
    dets : np.ndarray
        Output array of determinants <D_I|phi>.

    Returns
    -------
    None
    """
    ndets = len(cre)
    nwalkers = G0.shape[0]
    nex = cre.shape[-1]
    det = numpy.zeros((nex, nex), dtype=numpy.complex128)
    for iw in range(nwalkers):
        for idet in range(ndets):
            for iex in range(nex):
                p = mapping[cre[idet, iex]] + offset
                q = anh[idet, iex] + offset
                det[iex, iex] = G0[iw, p, q]
                for jex in range(iex + 1, nex):
                    r = mapping[cre[idet, jex]] + offset
                    s = anh[idet, jex] + offset
                    det[iex, jex] = G0[iw, p, s]
                    det[jex, iex] = G0[iw, r, q]
            dets[iw, idet] = numpy.linalg.det(det)


@jit(nopython=True, fastmath=True)
def build_det_matrix(cre, anh, mapping, offset, G0, det_mat):
    """Build matrix of determinants for n-fold excitations.

    Parameters
    ----------
    cre : np.ndarray
        Array containing orbitals excitations of occupied.
    anh : np.ndarray
        Array containing orbitals excitations to virtuals.
    mapping : np.ndarray
        Map original (occupied) orbital to index in compressed form.
    offset : int
        Offset for frozen core.
    G0 : np.ndarray
        (Half rotated) batched Green's function.
    det_matrix : np.ndarray
        Output array of determinants <D_I|phi>.

    Returns
    -------
    None
    """
    nwalker = det_mat.shape[0]
    ndet = det_mat.shape[1]
    if ndet == 0:
        return
    nex = det_mat.shape[2]
    for iw in range(nwalker):
        for idet in range(ndet):
            for iex in range(nex):
                p = mapping[cre[idet, iex]] + offset
                q = anh[idet, iex] + offset
                det_mat[iw, idet, iex, iex] = G0[iw, p, q]
                for jex in range(iex + 1, nex):
                    r = mapping[cre[idet, jex]] + offset
                    s = anh[idet, jex] + offset
                    det_mat[iw, idet, iex, jex] = G0[iw, p, s]
                    det_mat[iw, idet, jex, iex] = G0[iw, r, q]


# Green's function

@jit(nopython=False, fastmath=False)
def reduce_CI_singles(cre, anh, mapping, phases, CI):
    """Reduction to CI intermediate for singles.

    Parameters
    ----------
    cre : np.ndarray
        Array containing orbitals excitations of occupied.
    anh : np.ndarray
        Array containing orbitals excitations to virtuals.
    mapping : np.ndarray
        Map original (occupied) orbital to index in compressed form.
    phases : np.ndarray
        Phase factors.
    CI : np.ndarray
        Output array for CI intermediate.

    Returns
    -------
    None
    """
    ps = cre[:, 0]
    qs = anh[:, 0]
    ndets = len(cre)
    nwalkers = phases.shape[0]
    for iw in range(nwalkers):
        for idet in range(ndets):
            p = mapping[ps[idet]]
            q = qs[idet]
            CI[iw, q, p] += phases[iw, idet]


@jit(nopython=True, fastmath=True)
def reduce_CI_doubles(cre, anh, mapping, offset, phases, G0, CI):
    """Reduction to CI intermediate for doubles.

    Parameters
    ----------
    cre : np.ndarray
        Array containing orbitals excitations of occupied.
    anh : np.ndarray
        Array containing orbitals excitations to virtuals.
    mapping : np.ndarray
        Map original (occupied) orbital to index in compressed form.
    offset : int
        Offset for frozen core.
    phases : np.ndarray
        Phase factors.
    CI : np.ndarray
        Output array for CI intermediate.

    Returns
    -------
    None
    """
    ps = cre[:, 0]
    qs = anh[:, 0]
    rs = cre[:, 1]
    ss = anh[:, 1]
    ndets = len(cre)
    nwalkers = G0.shape[0]
    for iw in range(nwalkers):
        for idet in range(ndets):
            p = mapping[ps[idet]]
            q = qs[idet]
            r = mapping[rs[idet]]
            s = ss[idet]
            po = mapping[ps[idet]] + offset
            qo = qs[idet] + offset
            ro = mapping[rs[idet]] + offset
            so = ss[idet] + offset
            CI[iw, q, p] += phases[iw, idet] * G0[iw, ro, so]
            CI[iw, s, r] += phases[iw, idet] * G0[iw, po, qo]
            CI[iw, q, r] -= phases[iw, idet] * G0[iw, po, so]
            CI[iw, s, p] -= phases[iw, idet] * G0[iw, ro, qo]


@jit(nopython=True, fastmath=True)
def reduce_CI_triples(cre, anh, mapping, offset, phases, G0, CI):
    """Reduction to CI intermediate for triples.

    Parameters
    ----------
    cre : np.ndarray
        Array containing orbitals excitations of occupied.
    anh : np.ndarray
        Array containing orbitals excitations to virtuals.
    mapping : np.ndarray
        Map original (occupied) orbital to index in compressed form.
    offset : int
        Offset for frozen core.
    phases : np.ndarray
        Phase factors.
    CI : np.ndarray
        Output array for CI intermediate.

    Returns
    -------
    None
    """
    ps = cre[:, 0]
    qs = anh[:, 0]
    rs = cre[:, 1]
    ss = anh[:, 1]
    ts = cre[:, 2]
    us = anh[:, 2]
    ndets = len(cre)
    nwalkers = G0.shape[0]
    for iw in range(nwalkers):
        for idet in range(ndets):
            p = mapping[ps[idet]]
            q = qs[idet]
            r = mapping[rs[idet]]
            s = ss[idet]
            t = mapping[ts[idet]]
            u = us[idet]
            po = mapping[ps[idet]] + offset
            qo = qs[idet] + offset
            ro = mapping[rs[idet]] + offset
            so = ss[idet] + offset
            to = mapping[ts[idet]] + offset
            uo = us[idet] + offset
            CI[iw, q, p] += phases[iw, idet] * (
                G0[iw, ro, so] * G0[iw, to, uo] - G0[iw, ro, uo] * G0[iw, to, so]
            )  # 0 0
            CI[iw, s, p] -= phases[iw, idet] * (
                G0[iw, ro, qo] * G0[iw, to, uo] - G0[iw, ro, uo] * G0[iw, to, qo]
            )  # 0 1
            CI[iw, u, p] += phases[iw, idet] * (
                G0[iw, ro, qo] * G0[iw, to, so] - G0[iw, ro, so] * G0[iw, to, qo]
            )  # 0 2

            CI[iw, q, r] -= phases[iw, idet] * (
                G0[iw, po, so] * G0[iw, to, uo] - G0[iw, po, uo] * G0[iw, to, so]
            )  # 1 0
            CI[iw, s, r] += phases[iw, idet] * (
                G0[iw, po, qo] * G0[iw, to, uo] - G0[iw, po, uo] * G0[iw, to, qo]
            )  # 1 1
            CI[iw, u, r] -= phases[iw, idet] * (
                G0[iw, po, qo] * G0[iw, to, so] - G0[iw, po, so] * G0[iw, to, qo]
            )  # 1 2

            CI[iw, q, t] += phases[iw, idet] * (
                G0[iw, po, so] * G0[iw, ro, uo] - G0[iw, po, uo] * G0[iw, ro, so]
            )  # 2 0
            CI[iw, s, t] -= phases[iw, idet] * (
                G0[iw, po, qo] * G0[iw, ro, uo] - G0[iw, po, uo] * G0[iw, ro, qo]
            )  # 2 1
            CI[iw, u, t] += phases[iw, idet] * (
                G0[iw, po, qo] * G0[iw, ro, so] - G0[iw, po, so] * G0[iw, ro, qo]
            )  # 2 2


@jit(nopython=True, fastmath=True)
def _reduce_nfold_cofactor_contribution(
    ps, qs, mapping, sign, phases, cofactor_matrix, CI
):
    """Reduction to CI intermediate from cofactor contributions.

    Parameters
    ----------
    ps : np.ndarray
        Array containing orbitals excitations of occupied.
    qs : np.ndarray
        Array containing orbitals excitations to virtuals.
    mapping : np.ndarray
        Map original (occupied) orbital to index in compressed form.
    signs : int
        Phase factor arrising from excitation level.
    cofactor_matrix : np.ndarray
        Cofactor matrix previously constructed.
    CI : np.ndarray
        Output array for CI intermediate.

    Returns
    -------
    None
    """
    nwalkers = cofactor_matrix.shape[0]
    ndets = cofactor_matrix.shape[1]
    for iw in range(nwalkers):
        for idet in range(ndets):
            p = mapping[ps[idet]]
            q = qs[idet]
            det = numpy.linalg.det(cofactor_matrix[iw, idet])
            rhs = sign * det * phases[iw, idet]
            CI[iw, q, p] += rhs


@jit(nopython=True, fastmath=True)
def reduce_CI_nfold(cre, anh, mapping, offset, phases, det_mat, cof_mat, CI):
    """Reduction to CI intermediate for n-fold excitations.

    Parameters
    ----------
    cre : np.ndarray
        Array containing orbitals excitations of occupied.
    anh : np.ndarray
        Array containing orbitals excitations to virtuals.
    mapping : np.ndarray
        Map original (occupied) orbital to index in compressed form.
    offset : int
        Offset for frozen core.
    phases : np.ndarray
        Phase factors.
    det_mat: np.ndarray
        Array of determinants <D_I|phi>.
    cof_mat: np.ndarray
        Cofactor matrix previously constructed.
    CI : np.ndarray
        Output array for CI intermediate.

    Returns
    -------
    None
    """
    ndets = len(cre)
    nwalkers = CI.shape[0]
    nexcit = det_mat.shape[-1]
    for iex in range(nexcit):
        p = cre[:, iex]
        for jex in range(nexcit):
            q = anh[:, jex]
            # TODO FDM: effectively looping over wavefunction twice here, for
            # CPU may be better to squash building and reduction.
            build_cofactor_matrix(iex, jex, det_mat, cof_mat)
            sign = (-1 + 0.0j) ** (iex + jex)
            _reduce_nfold_cofactor_contribution(
                p, q, mapping, sign, phases, cof_mat, CI
            )


# Energy evaluation

@jit(nopython=True, fastmath=True)
def fill_os_singles(cre, anh, mapping, offset, chol_factor, spin_buffer, det_sls):
    """Fill opposite spin (os) contributions from singles.

    Parameters
    ----------
    cre : np.ndarray
        Array containing orbitals excitations of occupied.
    anh : np.ndarray
        Array containing orbitals excitations to virtuals.
    mapping : np.ndarray
        Map original (occupied) orbital to index in compressed form.
    offset : int
        Offset for frozen core.
    chol_factor : np.ndarray
        Lxqp intermediate constructed elsewhere.
    spin_buffer : np.ndarray
        Buffer for holding contribution.
    det_sls : np.ndarray
        Index slice for this exctitation level's contributions.

    Returns
    -------
    None
    """
    ps = cre[:, 0]
    qs = anh[:, 0]
    ndets = ps.shape[0]
    start = det_sls.start
    for idet in range(ndets):
        spin_buffer[:, start + idet] = chol_factor[:, qs[idet], mapping[ps[idet]]]


@jit(nopython=True, fastmath=True)
def fill_os_doubles(cre, anh, mapping, offset, G0, chol_factor, spin_buffer, det_sls):
    """Fill opposite spin (os) contributions from doubles.

    Parameters
    ----------
    cre : np.ndarray
        Array containing orbitals excitations of occupied.
    anh : np.ndarray
        Array containing orbitals excitations to virtuals.
    mapping : np.ndarray
        Map original (occupied) orbital to index in compressed form.
    G0 : np.ndarray
        Half-rotated reference Green's function.
    offset : int
        Offset for frozen core.
    chol_factor : np.ndarray
        Lxqp intermediate constructed elsewhere.
    spin_buffer : np.ndarray
        Buffer for holding contribution.
    det_sls : np.ndarray
        Index slice for this exctitation level's contributions.

    Returns
    -------
    None
    """
    start = det_sls.start
    ndets = cre.shape[0]
    nwalkers = G0.shape[0]
    for iw in range(nwalkers):
        G0_real = G0[iw].real.copy()
        G0_imag = G0[iw].imag.copy()
        for idet in range(ndets):
            p = mapping[cre[idet, 0]]
            q = anh[idet, 0]
            r = mapping[cre[idet, 1]]
            s = anh[idet, 1]
            po = cre[idet, 0] + offset
            qo = anh[idet, 0] + offset
            ro = cre[idet, 1] + offset
            so = anh[idet, 1] + offset
            spin_buffer[iw, start + idet, :] = (
                dot_real_cplx(
                    chol_factor[iw, q, p, :], G0_real[ro, so], G0_imag[ro, so]
                )
                - dot_real_cplx(
                    chol_factor[iw, s, p, :], G0_real[ro, qo], G0_imag[ro, qo]
                )
                - dot_real_cplx(
                    chol_factor[iw, q, r, :], G0_real[po, so], G0_imag[po, so]
                )
                + dot_real_cplx(
                    chol_factor[iw, s, r, :], G0_real[po, qo], G0_imag[po, qo]
                )
            )


@jit(nopython=True, fastmath=True)
def fill_os_triples(cre, anh, mapping, offset, G0w, chol_factor, spin_buffer, det_sls):
    """Fill opposite spin (os) contributions from triples.

    Parameters
    ----------
    cre : np.ndarray
        Array containing orbitals excitations of occupied.
    anh : np.ndarray
        Array containing orbitals excitations to virtuals.
    mapping : np.ndarray
        Map original (occupied) orbital to index in compressed form.
    offset : int
        Offset for frozen core.
    G0 : np.ndarray
        Half-rotated reference Green's function.
    chol_factor : np.ndarray
        Lxqp intermediate constructed elsewhere.
    spin_buffer : np.ndarray
        Buffer for holding contribution.
    det_sls : np.ndarray
        Index slice for this exctitation level's contributions.

    Returns
    -------
    None
    """
    start = det_sls.start
    ndets = cre.shape[0]
    nwalkers = G0w.shape[0]
    for iw in range(nwalkers):
        G0 = G0w[iw]
        for idet in range(ndets):
            p = mapping[cre[idet, 0]]
            q = anh[idet, 0]
            r = mapping[cre[idet, 1]]
            s = anh[idet, 1]
            t = mapping[cre[idet, 2]]
            u = anh[idet, 2]
            po = cre[idet, 0] + offset
            qo = anh[idet, 0] + offset
            ro = cre[idet, 1] + offset
            so = anh[idet, 1] + offset
            to = cre[idet, 2] + offset
            uo = anh[idet, 2] + offset
            cofac = G0[ro, so] * G0[to, uo] - G0[ro, uo] * G0[to, so]
            spin_buffer[iw, start + idet, :] = dot_real_cplx(
                chol_factor[iw, q, p], cofac.real, cofac.imag
            )
            cofac = G0[ro, qo] * G0[to, uo] - G0[ro, uo] * G0[to, qo]
            spin_buffer[iw, start + idet, :] -= dot_real_cplx(
                chol_factor[iw, s, p], cofac.real, cofac.imag
            )
            cofac = G0[ro, qo] * G0[to, so] - G0[ro, so] * G0[to, qo]
            spin_buffer[iw, start + idet, :] += dot_real_cplx(
                chol_factor[iw, u, p], cofac.real, cofac.imag
            )
            cofac = G0[po, so] * G0[to, uo] - G0[to, so] * G0[po, uo]
            spin_buffer[iw, start + idet, :] -= dot_real_cplx(
                chol_factor[iw, q, r], cofac.real, cofac.imag
            )
            cofac = G0[po, qo] * G0[to, uo] - G0[to, qo] * G0[po, uo]
            spin_buffer[iw, start + idet, :] += dot_real_cplx(
                chol_factor[iw, s, r], cofac.real, cofac.imag
            )
            cofac = G0[po, qo] * G0[to, so] - G0[to, qo] * G0[po, so]
            spin_buffer[iw, start + idet, :] -= dot_real_cplx(
                chol_factor[iw, u, r], cofac.real, cofac.imag
            )
            cofac = G0[po, so] * G0[ro, uo] - G0[ro, so] * G0[po, uo]
            spin_buffer[iw, start + idet, :] += dot_real_cplx(
                chol_factor[iw, q, t], cofac.real, cofac.imag
            )
            cofac = G0[po, qo] * G0[ro, uo] - G0[ro, qo] * G0[po, uo]
            spin_buffer[iw, start + idet, :] -= dot_real_cplx(
                chol_factor[iw, s, t], cofac.real, cofac.imag
            )
            cofac = G0[po, qo] * G0[ro, so] - G0[ro, qo] * G0[po, so]
            spin_buffer[iw, start + idet, :] += dot_real_cplx(
                chol_factor[iw, u, t], cofac.real, cofac.imag
            )


@jit(nopython=True, fastmath=True)
def get_ss_doubles(cre, anh, mapping, chol_fact, buffer, det_sls):
    """Fill same spin (ss) contributions from doubles.

    Parameters
    ----------
    cre : np.ndarray
        Array containing orbitals excitations of occupied.
    anh : np.ndarray
        Array containing orbitals excitations to virtuals.
    mapping : np.ndarray
        Map original (occupied) orbital to index in compressed form.
    chol_fact : np.ndarray
        Lxqp intermediate constructed elsewhere.
    buffer : np.ndarray
        Buffer for holding contribution.
    det_sls : np.ndarray
        Index slice for this exctitation level's contributions.

    Returns
    -------
    None
    """
    start = det_sls.start
    ndets = cre.shape[0]
    nwalkers = chol_fact.shape[0]
    for iw in range(nwalkers):
        for idet in range(ndets):
            p = mapping[cre[idet, 0]]
            q = anh[idet, 0]
            r = mapping[cre[idet, 1]]
            s = anh[idet, 1]
            buffer[iw, start + idet] += (
                numpy.dot(chol_fact[iw, q, p], chol_fact[iw, s, r]) + 0j
            )
            buffer[iw, start + idet] -= (
                numpy.dot(chol_fact[iw, q, r], chol_fact[iw, s, p]) + 0j
            )


@jit(nopython=True, fastmath=True)
def build_cofactor_matrix(row, col, det_matrix, cofactor):
    """Build cofactor matrix.

    Parameters
    ----------
    row : int
        Row to delete when building cofactor.
    col : int
        Column to delete when building cofactor.
    det_matrix : np.ndarray
        Precomputed array of determinants <D_I|phi> for given excitation level.
    cofactor : np.ndarray
        Cofactor matrix.

    Returns
    -------
    None
    """
    nwalker = det_matrix.shape[0]
    ndet = det_matrix.shape[1]
    nexcit = det_matrix.shape[2]
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
                    cofactor[iw, idet, i - ishift, j - jshift] = det_matrix[
                        iw, idet, i, j
                    ]


@jit(nopython=True, fastmath=True)
def build_cofactor_matrix_4(row_1, col_1, row_2, col_2, det_matrix, cofactor):
    """Build cofactor matrix with 2 rows/cols deleted.

    Parameters
    ----------
    row_1 : int
        Row to delete when building cofactor.
    col_1 : int
        Column to delete when building cofactor.
    row_2 : int
        Row to delete when building cofactor.
    col_2 : int
        Column to delete when building cofactor.
    det_matrix : np.ndarray
        Precomputed array of determinants <D_I|phi> for given excitation level.
    cofactor : np.ndarray
        Cofactor matrix.

    Returns
    -------
    None
    """
    nwalker = det_matrix.shape[0]
    ndet = det_matrix.shape[1]
    nexcit = det_matrix.shape[2]
    if nexcit - 2 <= 0:
        for iw in range(nwalker):
            for idet in range(ndet):
                cofactor[iw, idet, 0, 0] = 1.0 + 0j
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
                    ii = max(i - (ishift_1 + ishift_2), 0)
                    jj = max(j - (jshift_1 + jshift_2), 0)
                    cofactor[iw, idet, ii, jj] = det_matrix[iw, idet, i, j]


@jit(nopython=True, fastmath=True)
def reduce_os_spin_factor(
    ps, qs, mapping, phase, cof_mat, chol_factor, spin_buffer, det_sls
):
    """Reduce opposite spin (os) contributions into spin_buffer.

    Parameters
    ----------
    ps : np.ndarray
        Array containing orbitals excitations of occupied.
    qs : np.ndarray
        Array containing orbitals excitations to virtuals.
    mapping : np.ndarray
        Map original (occupied) orbital to index in compressed form.
    phases : np.ndarray
        Phase factors.
    cof_mat: np.ndarray
        Cofactor matrix previously constructed.
    chol_fact : np.ndarray
        Lxqp intermediate constructed elsewhere.
    spin_buffer : np.ndarray
        Buffer for holding contribution.
    det_sls : np.ndarray
        Index slice for this exctitation level's contributions.

    Returns
    -------
    None
    """
    nwalker = chol_factor.shape[0]
    ndet = cof_mat.shape[1]
    start = det_sls.start
    # assert ndet == det_sls.end - det_sls.start
    for iw in range(nwalker):
        for idet in range(ndet):
            det_cofactor = phase * numpy.linalg.det(cof_mat[iw, idet])
            p = mapping[ps[idet]]
            spin_buffer[iw, start + idet] += dot_real_cplx(
                chol_factor[iw, qs[idet], p],
                det_cofactor.real,
                det_cofactor.imag,
            )


@jit(nopython=True, fastmath=True)
def fill_os_nfold(
    cre, anh, mapping, det_matrix, cof_mat, chol_factor, spin_buffer, det_sls
):
    """Fill opposite spin (os) n-fold contributions into spin_buffer.

    Parameters
    ----------
    cre : np.ndarray
        Array containing orbitals excitations of occupied.
    anh : np.ndarray
        Array containing orbitals excitations to virtuals.
    mapping : np.ndarray
        Map original (occupied) orbital to index in compressed form.
    det_matrix : np.ndarray
        Array of determinants <D_I|phi> for n-fold excitation.
    cof_mat: np.ndarray
        Cofactor matrix buffer.
    chol_factor : np.ndarray
        Lxqp intermediate constructed elsewhere.
    spin_buffer : np.ndarray
        Buffer for holding contribution.
    det_sls : np.ndarray
        Index slice for this exctitation level's contributions.

    Returns
    -------
    None
    """
    nwalkers = cof_mat.shape[0]
    ndet = cof_mat.shape[1]
    nexcit = det_matrix.shape[-1]
    for iex in range(nexcit):
        ps = cre[:, iex]
        for jex in range(nexcit):
            qs = anh[:, jex]
            build_cofactor_matrix(iex, jex, det_matrix, cof_mat)
            # nwalkers x ndet
            phase = (-1.0 + 0.0j) ** (iex + jex)
            reduce_os_spin_factor(
                ps, qs, mapping, phase, cof_mat, chol_factor, spin_buffer, det_sls
            )


@jit(nopython=True, fastmath=True)
def reduce_ss_spin_factor(
    ps, qs, rs, ss, mapping, phase, cof_mat, chol_factor, spin_buffer, det_sls
):
    """Reduce same-spin (ss) n-fold contributions into spin_buffer.

    Parameters
    ----------
    ps : np.ndarray
        Array containing orbitals excitations of occupied.
    qs : np.ndarray
        Array containing orbitals excitations to virtuals.
    rs : np.ndarray
        Array containing orbitals excitations of occupied.
    ss : np.ndarray
        Array containing orbitals excitations to virtuals.
    mapping : np.ndarray
        Map original (occupied) orbital to index in compressed form.
    phases : np.ndarray
        Phase factors.
    cof_mat: np.ndarray
        Cofactor matrix buffer.
    chol_factor : np.ndarray
        Lxqp intermediate constructed elsewhere.
    spin_buffer : np.ndarray
        Buffer for holding contribution.
    det_sls : np.ndarray
        Index slice for this exctitation level's contributions.

    Returns
    -------
    None
    """
    nwalker = chol_factor.shape[0]
    ndet = cof_mat.shape[1]
    start = det_sls.start
    for iw in range(nwalker):
        for idet in range(ndet):
            det_cofactor = phase * numpy.linalg.det(cof_mat[iw, idet])
            p = mapping[ps[idet]]
            r = mapping[rs[idet]]
            chol_a = chol_factor[iw, ss[idet], r]
            chol_b = chol_factor[iw, qs[idet], p]
            cont_ab = numpy.dot(chol_a, chol_b)
            spin_buffer[iw, start + idet] += dot_real_cplx(
                cont_ab,
                det_cofactor.real,
                det_cofactor.imag,
            )
            chol_c = chol_factor[iw, qs[idet], r]
            chol_d = chol_factor[iw, ss[idet], p]
            cont_cd = numpy.dot(chol_c, chol_d)
            spin_buffer[iw, start + idet] -= dot_real_cplx(
                cont_cd,
                det_cofactor.real,
                det_cofactor.imag,
            )


@jit(nopython=True, fastmath=True)
def get_ss_nfold(cre, anh, mapping, dets_mat, cof_mat, chol_fact, buffer, det_sls):
    """Build same-spin (ss) n-fold contributions.

    Parameters
    ----------
    cre : np.ndarray
        Array containing orbitals excitations of occupied.
    anh : np.ndarray
        Array containing orbitals excitations to virtuals.
    mapping : np.ndarray
        Map original (occupied) orbital to index in compressed form.
    det_matrix : np.ndarray
        Output array of determinants <D_I|phi>.
    cof_mat: np.ndarray
        Cofactor matrix buffer.
    chol_factor : np.ndarray
        Lxqp intermediate constructed elsewhere.
    buffer : np.ndarray
        Buffer for holding contribution.
    det_sls : np.ndarray
        Index slice for this exctitation level's contributions.

    Returns
    -------
    None
    """
    nwalkers = dets_mat.shape[0]
    ndet_level = dets_mat.shape[1]
    nexcit = dets_mat.shape[-1]
    for iex in range(nexcit):
        for jex in range(nexcit):
            ps = cre[:, iex]
            qs = anh[:, jex]
            for kex in range(iex + 1, nexcit):
                rs = cre[:, kex]
                for lex in range(jex + 1, nexcit):
                    ss = anh[:, lex]
                    build_cofactor_matrix_4(iex, jex, kex, lex, dets_mat, cof_mat)
                    phase = (-1.0 + 0.0j) ** (kex + lex + iex + jex)
                    reduce_ss_spin_factor(
                        ps,
                        qs,
                        rs,
                        ss,
                        mapping,
                        phase,
                        cof_mat,
                        chol_fact,
                        buffer,
                        det_sls,
                    )
