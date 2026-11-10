# Copyright 2026 The ipie Developers. All Rights Reserved.
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

import math
import time
from types import SimpleNamespace

import numpy
import scipy.linalg
from numba import njit, prange

from ipie.propagation.hirsch_base import HirschBase
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import synchronize, to_host
from ipie.walkers.ghf_walkers import GHFWalkers

_HUBBARD_GPU_MEMORY_FRACTION = 0.70

# Every site update below multiplies the tracked overlap by the determinant ratio r_x only. The
# Hirsch constant aux_wfac[x] (1 for the spin decoupling, field dependent for the charge one)
# enters the sampling probability and the weight, never the overlap <psi_T|phi>.


@njit(cache=True, inline="always")
def sherman_morrison_from_q_cpu(ainv, u, q, delta_xi, g):
    """ainv <- ainv - (ainv u) delta_xi q / (1 + delta_xi g), with q = phi_row ainv, g = q u."""
    nocc = ainv.shape[0]
    scale = delta_xi / (1.0 + delta_xi * g)
    for i in range(nocc):
        val = 0.0 + 0.0j
        for k in range(nocc):
            val += ainv[i, k] * u[k]
        scaled_val = val * scale
        for j in range(nocc):
            ainv[i, j] -= scaled_val * q[j]


@njit(cache=True)
def hubbard_two_body_uhf_cpu(
    phia,
    phib,
    inva,
    invb,
    weight,
    ovlp,
    psi0a_conj,
    psi0b_conj,
    delta,
    aux_wfac,
    random_fields,
    rhf,
):
    """Discrete Hirsch sweep over all sites for UHF walkers (RHF: down spin mirrors up)."""
    nwalkers = phia.shape[0]
    nbasis = phia.shape[1]
    nup = phia.shape[2]
    ndown = phib.shape[2]
    for iw in range(nwalkers):
        if abs(weight[iw]) == 0.0:
            continue
        q_up = numpy.empty(nup, dtype=numpy.complex128)
        q_down = numpy.empty(ndown, dtype=numpy.complex128)
        for site in range(nbasis):
            for j in range(nup):
                q_up[j] = 0.0 + 0.0j
            for k in range(nup):
                phi_k = phia[iw, site, k]
                for j in range(nup):
                    q_up[j] += inva[iw, k, j] * phi_k
            gup = 0.0 + 0.0j
            for j in range(nup):
                gup += psi0a_conj[site, j] * q_up[j]

            gdown = 0.0 + 0.0j
            if ndown > 0 and not rhf:
                for j in range(ndown):
                    q_down[j] = 0.0 + 0.0j
                for k in range(ndown):
                    phi_k = phib[iw, site, k]
                    for j in range(ndown):
                        q_down[j] += invb[iw, k, j] * phi_k
                for j in range(ndown):
                    gdown += psi0b_conj[site, j] * q_down[j]
            elif ndown > 0 and rhf:
                gdown = gup

            r0 = (1.0 + delta[0, 0] * gup) * (1.0 + delta[0, 1] * gdown)
            r1 = (1.0 + delta[1, 0] * gup) * (1.0 + delta[1, 1] * gdown)
            p0_real = max((0.5 * r0 * aux_wfac[0]).real, 0.0)
            p1_real = max((0.5 * r1 * aux_wfac[1]).real, 0.0)
            norm = p0_real + p1_real
            if norm <= 0.0:
                weight[iw] = 0.0
                break

            xi = 0 if random_fields[site, iw] < p0_real / norm else 1
            weight[iw] *= norm
            ovlp[iw] *= r0 if xi == 0 else r1

            delta_up = delta[xi, 0]
            scale_up = 1.0 + delta_up
            for j in range(nup):
                phia[iw, site, j] *= scale_up
            sherman_morrison_from_q_cpu(inva[iw], psi0a_conj[site], q_up, delta_up, gup)

            if ndown > 0 and not rhf:
                delta_down = delta[xi, 1]
                scale_down = 1.0 + delta_down
                for j in range(ndown):
                    phib[iw, site, j] *= scale_down
                sherman_morrison_from_q_cpu(invb[iw], psi0b_conj[site], q_down, delta_down, gdown)


@njit(parallel=True, cache=True)
def hubbard_kinetic_ghf_cpu(phi, inv_ovlp, ovlp, weight, log_shift, exp_h1, psi0_h):
    """phi <- exp_h1 phi; rebuild inv_ovlp and the overlap; constrained weight factor.

    One walker per numba thread; the BLAS/LAPACK calls inside must run single threaded.
    """
    for iw in prange(phi.shape[0]):
        p = numpy.dot(exp_h1, phi[iw])
        phi[iw] = p
        a = numpy.dot(psi0_h, p)
        sign, logdet = numpy.linalg.slogdet(a)
        inv_ovlp[iw] = numpy.linalg.inv(a)
        ovlp_new = sign * numpy.exp(logdet - log_shift[iw])
        ratio = ovlp_new / ovlp[iw]
        weight_factor = ratio.real if ratio.real > 0.0 else 0.0
        weight[iw] *= weight_factor
        if weight_factor > 0.0:
            ovlp[iw] = ovlp_new


@njit(parallel=True, cache=True, fastmath={"reassoc", "contract"})
def hubbard_two_body_ghf_cpu(
    phi, inv_ovlp, weight, ovlp, psi0_conj, psi0_conj_pairs, delta, aux_wfac, random_fields
):
    """Discrete Hirsch sweep over all sites for GHF walkers and a single-determinant GHF trial.

    Site s acts on the spin orbitals up = s and down = s + nbasis. With the 2 x nocc rows
    u = phi[(up, down)] inv_ovlp, the 2 x 2 block g = u psi0_conj[(up, down)].T gives the
    overlap ratio r_x = det(1 + D_x g), D_x = diag(delta[x]). The field x is sampled from
    max(Re(r_x aux_wfac[x] / 2), 0), the weight gets the normalization, the overlap gets r_x,
    and inv_ovlp the rank-2 Woodbury update
    inv_ovlp -= (inv_ovlp psi0_conj_pairs[s]) (1 + D_x g)^{-1} D_x u.
    One walker per numba thread, so its inverse overlap stays in cache across sites.
    """
    nbasis = phi.shape[1] // 2
    nocc = phi.shape[2]
    for iw in prange(phi.shape[0]):
        if weight[iw] != 0.0:
            pw = phi[iw]
            aw = inv_ovlp[iw]
            rows = numpy.empty((2, nocc), dtype=numpy.complex128)
            y0 = numpy.empty(nocc, dtype=numpy.complex128)
            y1 = numpy.empty(nocc, dtype=numpy.complex128)
            for site in range(nbasis):
                up = site
                down = site + nbasis
                rows[0] = pw[up]
                rows[1] = pw[down]
                u = numpy.dot(rows, aw)
                v = numpy.dot(aw, psi0_conj_pairs[site])
                guu = 0j
                gud = 0j
                gdu = 0j
                gdd = 0j
                for c in range(nocc):
                    guu += u[0, c] * psi0_conj[up, c]
                    gud += u[0, c] * psi0_conj[down, c]
                    gdu += u[1, c] * psi0_conj[up, c]
                    gdd += u[1, c] * psi0_conj[down, c]
                r0 = (1.0 + delta[0, 0] * guu) * (1.0 + delta[0, 1] * gdd) - delta[0, 0] * delta[
                    0, 1
                ] * gud * gdu
                r1 = (1.0 + delta[1, 0] * guu) * (1.0 + delta[1, 1] * gdd) - delta[1, 0] * delta[
                    1, 1
                ] * gud * gdu
                pr0 = max((0.5 * r0 * aux_wfac[0]).real, 0.0)
                pr1 = max((0.5 * r1 * aux_wfac[1]).real, 0.0)
                norm = pr0 + pr1
                if norm <= 0.0:
                    weight[iw] = 0.0
                    break
                xi = 0 if random_fields[site, iw] < pr0 / norm else 1
                weight[iw] *= norm
                ovlp[iw] *= r0 if xi == 0 else r1
                du = delta[xi, 0]
                dd = delta[xi, 1]
                c00 = 1.0 + du * guu
                c01 = du * gud
                c10 = dd * gdu
                c11 = 1.0 + dd * gdd
                det = c00 * c11 - c01 * c10
                for c in range(nocc):
                    a0 = du * u[0, c]
                    a1 = dd * u[1, c]
                    y0[c] = (c11 * a0 - c01 * a1) / det
                    y1[c] = (c00 * a1 - c10 * a0) / det
                for a in range(nocc):
                    va0 = v[a, 0]
                    va1 = v[a, 1]
                    for c in range(nocc):
                        aw[a, c] -= va0 * y0[c] + va1 * y1[c]
                for c in range(nocc):
                    pw[up, c] *= 1.0 + du
                    pw[down, c] *= 1.0 + dd


@njit(parallel=True, cache=True)
def hubbard_two_body_noci_ghf_cpu(
    phi, inv, rel_ovlp, cstar, weight, ovlp, detsc, delta, aux_wfac, random_fields
):
    """Discrete Hirsch sweep over all sites for GHF walkers and a NOCI-GHF trial.

    phi (w, 2nb, m), inv (K, w, m, m) per-determinant inverse overlaps, rel_ovlp (K, w)
    determinant overlaps relative to a per-walker reference, detsc (K, 2nb, m) conjugated
    determinants. The site ratio is the overlap-weighted average
    r(x) = sum_k c_k^* O_k R_k(x) / sum_k c_k^* O_k of the single-determinant ratios
    R_k(x) = det(1 + D_x g_k). Each determinant's inverse overlap gets the rank-2 Woodbury
    update for the up and down rows of the site; its 2 x 2 capacitance matrix is 1 + D_x g_k.
    One walker per numba thread.
    """
    nwalkers = phi.shape[0]
    nbasis = phi.shape[1] // 2
    nocc = phi.shape[2]
    ndet = inv.shape[0]
    for iw in prange(nwalkers):
        if abs(weight[iw]) == 0.0:
            continue
        qup = numpy.empty((ndet, nocc), dtype=numpy.complex128)
        qdn = numpy.empty((ndet, nocc), dtype=numpy.complex128)
        auu = numpy.empty((ndet, nocc), dtype=numpy.complex128)
        aud = numpy.empty((ndet, nocc), dtype=numpy.complex128)
        guu = numpy.empty(ndet, dtype=numpy.complex128)
        gud = numpy.empty(ndet, dtype=numpy.complex128)
        gdu = numpy.empty(ndet, dtype=numpy.complex128)
        gdd = numpy.empty(ndet, dtype=numpy.complex128)
        for site in range(nbasis):
            up = site
            dn = site + nbasis
            num0 = 0.0 + 0.0j
            num1 = 0.0 + 0.0j
            psum = 0.0 + 0.0j
            for k in range(ndet):
                # One pass over inv[k, iw]: the row projections q = phi_row inv and the
                # field-independent columns a = inv u (u = conjugated determinant rows).
                for j in range(nocc):
                    qup[k, j] = 0.0 + 0.0j
                    qdn[k, j] = 0.0 + 0.0j
                for i in range(nocc):
                    pu = phi[iw, up, i]
                    pd = phi[iw, dn, i]
                    su = 0.0 + 0.0j
                    sd = 0.0 + 0.0j
                    for j in range(nocc):
                        a = inv[k, iw, i, j]
                        qup[k, j] += pu * a
                        qdn[k, j] += pd * a
                        su += a * detsc[k, up, j]
                        sd += a * detsc[k, dn, j]
                    auu[k, i] = su
                    aud[k, i] = sd
                g_uu = 0.0 + 0.0j
                g_ud = 0.0 + 0.0j
                g_du = 0.0 + 0.0j
                g_dd = 0.0 + 0.0j
                for j in range(nocc):
                    tu = detsc[k, up, j]
                    td = detsc[k, dn, j]
                    g_uu += qup[k, j] * tu
                    g_ud += qup[k, j] * td
                    g_du += qdn[k, j] * tu
                    g_dd += qdn[k, j] * td
                guu[k] = g_uu
                gud[k] = g_ud
                gdu[k] = g_du
                gdd[k] = g_dd
                r0 = (1.0 + delta[0, 0] * g_uu) * (1.0 + delta[0, 1] * g_dd) - delta[0, 0] * delta[
                    0, 1
                ] * g_ud * g_du
                r1 = (1.0 + delta[1, 0] * g_uu) * (1.0 + delta[1, 1] * g_dd) - delta[1, 0] * delta[
                    1, 1
                ] * g_ud * g_du
                pk = cstar[k] * rel_ovlp[k, iw]
                psum += pk
                num0 += pk * r0
                num1 += pk * r1
            ratio0 = num0 / psum
            ratio1 = num1 / psum
            p0_real = max((0.5 * ratio0 * aux_wfac[0]).real, 0.0)
            p1_real = max((0.5 * ratio1 * aux_wfac[1]).real, 0.0)
            norm = p0_real + p1_real
            if norm <= 0.0:
                weight[iw] = 0.0
                break
            xi = 0 if random_fields[site, iw] < p0_real / norm else 1
            weight[iw] *= norm
            ovlp[iw] *= ratio0 if xi == 0 else ratio1

            d0 = delta[xi, 0]
            d1 = delta[xi, 1]
            for k in range(ndet):
                b00 = 1.0 + d0 * guu[k]
                b01 = d0 * gud[k]
                b10 = d1 * gdu[k]
                b11 = 1.0 + d1 * gdd[k]
                rk = b00 * b11 - b01 * b10
                rel_ovlp[k, iw] = rel_ovlp[k, iw] * rk
                ib00 = b11 / rk
                ib01 = -b01 / rk
                ib10 = -b10 / rk
                ib11 = b00 / rk
                for i in range(nocc):
                    e0 = (auu[k, i] * ib00 + aud[k, i] * ib10) * d0
                    e1 = (auu[k, i] * ib01 + aud[k, i] * ib11) * d1
                    for j in range(nocc):
                        inv[k, iw, i, j] -= e0 * qup[k, j] + e1 * qdn[k, j]
            for j in range(nocc):
                phi[iw, up, j] *= 1.0 + d0
                phi[iw, dn, j] *= 1.0 + d1


class HubbardSingleSite(HirschBase):
    """Discrete Hubbard propagator with adaptive CPU, einsum, and CUDA paths."""

    def _use_numba_ghf(self, walkers, trial):
        from ipie.trial_wavefunction.noci_ghf import NOCIGHF

        return (
            isinstance(walkers, GHFWalkers)
            and not isinstance(trial, NOCIGHF)
            and not hasattr(xp, "RawKernel")
        )

    def _ghf_numba_arrays(self, walkers, trial):
        """Contiguous complex trial and one-body arrays for the GHF numba kernels (cached)."""
        # Hold the source arrays so the identity check cannot match a recycled object.
        key = (trial.psi0, self.expH1)
        cached = getattr(self, "_ghf_numba_key", None)
        if cached is not None and cached[0] is key[0] and cached[1] is key[1]:
            return
        nbasis = walkers.nbasis
        psi0_conj = numpy.ascontiguousarray(numpy.conjugate(trial.psi0), dtype=numpy.complex128)
        self._ghf_psi0_conj = psi0_conj
        self._ghf_psi0_h = numpy.ascontiguousarray(psi0_conj.T)
        self._ghf_psi0_conj_pairs = numpy.ascontiguousarray(
            numpy.stack([psi0_conj[[s, s + nbasis]].T for s in range(nbasis)])
        )
        exp_h1 = self.expH1
        if exp_h1.ndim == 3:
            exp_h1 = scipy.linalg.block_diag(exp_h1[0], exp_h1[1])
        self._ghf_exp_h1 = numpy.ascontiguousarray(exp_h1, dtype=numpy.complex128)
        self._ghf_numba_key = key

    def _contiguous_ghf_walkers(self, walkers):
        walkers.phi = numpy.ascontiguousarray(walkers.phi, dtype=numpy.complex128)
        walkers.inv_ovlp = numpy.ascontiguousarray(walkers.inv_ovlp, dtype=numpy.complex128)
        walkers.ovlp = numpy.ascontiguousarray(walkers.ovlp, dtype=numpy.complex128)

    def _numba_ghf_kinetic(self, walkers, trial):
        self._ghf_numba_arrays(walkers, trial)
        self._contiguous_ghf_walkers(walkers)
        hubbard_kinetic_ghf_cpu(
            walkers.phi,
            walkers.inv_ovlp,
            walkers.ovlp,
            walkers.weight,
            numpy.asarray(walkers.log_shift, dtype=numpy.float64),
            self._ghf_exp_h1,
            self._ghf_psi0_h,
        )

    def _numba_ghf_two_body(self, walkers, hamiltonian, trial, random_fields):
        self._ghf_numba_arrays(walkers, trial)
        self._contiguous_ghf_walkers(walkers)
        hubbard_two_body_ghf_cpu(
            walkers.phi,
            walkers.inv_ovlp,
            walkers.weight,
            walkers.ovlp,
            self._ghf_psi0_conj,
            self._ghf_psi0_conj_pairs,
            numpy.asarray(self.delta, dtype=numpy.complex128),
            numpy.asarray(self.aux_wfac, dtype=numpy.complex128),
            numpy.asarray(random_fields, dtype=numpy.float64),
        )

    def kinetic_importance_sampling(self, walkers, trial):
        # For a multi-determinant (NOCI) GHF trial the constraint overlap ratio
        # must use the full trial overlap, not the reference-determinant inverse
        # overlap that _calc_overlap_from_inverse would give.
        from ipie.trial_wavefunction.noci_ghf import NOCIGHF

        if self._use_numba_ghf(walkers, trial):
            start_time = time.time()
            self._numba_ghf_kinetic(walkers, trial)
            self.timer.tovlp += time.time() - start_time
            return
        if not isinstance(trial, NOCIGHF):
            return super().kinetic_importance_sampling(walkers, trial)
        start_time = time.time()
        self.propagate_walkers_one_body(walkers)
        ovlp_new = trial.calc_overlap(walkers)
        ratio = ovlp_new / walkers.ovlp
        phase = xp.angle(ratio)
        weight_factor = xp.where(xp.abs(phase) < 0.5 * math.pi, ratio.real, 0.0)
        walkers.weight *= weight_factor
        walkers.ovlp = xp.where(weight_factor > 0.0, ovlp_new, walkers.ovlp)
        synchronize()
        self.timer.tovlp += time.time() - start_time

    def _batched_sherman_morrison(self, ainv, u, vt):
        au = xp.einsum("wij,j->wi", ainv, u, optimize=True)
        vta = xp.einsum("wi,wij->wj", vt, ainv, optimize=True)
        denom = 1.0 + xp.einsum("wi,i->w", vta, u, optimize=True)
        update = xp.einsum("wi,wj->wij", au, vta, optimize=True) / denom[:, None, None]
        return ainv - update

    def _site_greens_diagonal(self, inv_ovlp, phi, psi0_site, site):
        q = xp.einsum("wij,wi->wj", inv_ovlp, phi[:, site, :], optimize=True)
        return xp.einsum("j,wj->w", psi0_site.conj(), q, optimize=True)

    def _ghf_site_greens_block(self, walkers, trial, site):
        nbasis = walkers.nbasis
        up = site
        down = site + nbasis
        phi_up = walkers.phi[:, up, :]
        phi_down = walkers.phi[:, down, :]
        psi_up = trial.psi0[up, :].conj()
        psi_down = trial.psi0[down, :].conj()

        ghalf_up = xp.einsum("wi,wij->wj", phi_up, walkers.inv_ovlp, optimize=True)
        ghalf_down = xp.einsum("wi,wij->wj", phi_down, walkers.inv_ovlp, optimize=True)
        guu = xp.einsum("wi,i->w", ghalf_up, psi_up, optimize=True)
        gud = xp.einsum("wi,i->w", ghalf_up, psi_down, optimize=True)
        gdu = xp.einsum("wi,i->w", ghalf_down, psi_up, optimize=True)
        gdd = xp.einsum("wi,i->w", ghalf_down, psi_down, optimize=True)
        return guu, gud, gdu, gdd

    def _legacy_two_body(self, walkers, hamiltonian, trial, random_fields=None):
        for iw in range(walkers.nwalkers):
            if abs(to_host(walkers.weight[iw])) == 0:
                continue
            for i in range(hamiltonian.nbasis):
                uup = walkers.phia[iw, i, :]
                q_up = xp.dot(walkers.inv_ovlp_a[iw].T, uup)
                gup = xp.dot(trial.psi0a[i, :].conj(), q_up)
                gdown = 0.0
                if walkers.ndown > 0 and not walkers.rhf:
                    udown = walkers.phib[iw, i, :]
                    q_down = xp.dot(walkers.inv_ovlp_b[iw].T, udown)
                    gdown = xp.dot(trial.psi0b[i, :].conj(), q_down)
                elif walkers.ndown > 0 and walkers.rhf:
                    gdown = gup

                r1 = (1.0 + self.delta[0, 0] * gup) * (1.0 + self.delta[0, 1] * gdown)
                r2 = (1.0 + self.delta[1, 0] * gup) * (1.0 + self.delta[1, 1] * gdown)
                ratios = xp.array([r1, r2], dtype=xp.complex128)
                probs = 0.5 * ratios * self.aux_wfac
                phaseless_ratio = xp.maximum(probs.real, xp.array([0.0, 0.0]))
                norm = phaseless_ratio[0] + phaseless_ratio[1]
                norm_scalar = float(to_host(norm))
                rnd = (
                    float(to_host(random_fields[i, iw]))
                    if random_fields is not None
                    else float(to_host(xp.random.random()))
                )
                if norm_scalar <= 0.0:
                    walkers.weight[iw] = 0.0
                    break
                walkers.weight[iw] *= norm
                p0 = float(to_host(phaseless_ratio[0] / norm))
                xi = 0 if rnd < p0 else 1

                vtup = walkers.phia[iw, i, :] * self.delta[xi, 0]
                walkers.phia[iw, i, :] = walkers.phia[iw, i, :] + vtup
                if walkers.ndown > 0 and not walkers.rhf:
                    vtdown = walkers.phib[iw, i, :] * self.delta[xi, 1]
                    walkers.phib[iw, i, :] = walkers.phib[iw, i, :] + vtdown
                else:
                    vtdown = None
                walkers.ovlp[iw] = walkers.ovlp[iw] * ratios[xi]
                walkers.inv_ovlp_a[iw] = self._sherman_morrison(
                    walkers.inv_ovlp_a[iw], trial.psi0a[i, :].conj(), vtup
                )
                if walkers.ndown > 0 and not walkers.rhf:
                    walkers.inv_ovlp_b[iw] = self._sherman_morrison(
                        walkers.inv_ovlp_b[iw], trial.psi0b[i, :].conj(), vtdown
                    )

    def _cpu_numba_two_body(self, walkers, hamiltonian, trial, random_fields):
        hubbard_two_body_uhf_cpu(
            walkers.phia,
            walkers.phib,
            walkers.inv_ovlp_a,
            walkers.inv_ovlp_b,
            walkers.weight,
            walkers.ovlp,
            numpy.ascontiguousarray(numpy.conjugate(trial.psi0a), dtype=numpy.complex128),
            numpy.ascontiguousarray(numpy.conjugate(trial.psi0b), dtype=numpy.complex128),
            numpy.asarray(self.delta, dtype=numpy.complex128),
            numpy.asarray(self.aux_wfac, dtype=numpy.complex128),
            numpy.asarray(random_fields, dtype=numpy.float64),
            bool(walkers.rhf),
        )

    def _einsum_two_body(self, walkers, hamiltonian, trial, random_fields):
        zero = xp.asarray(0.0)
        for i in range(hamiltonian.nbasis):
            gup = self._site_greens_diagonal(walkers.inv_ovlp_a, walkers.phia, trial.psi0a[i], i)
            if walkers.ndown > 0 and not walkers.rhf:
                gdown = self._site_greens_diagonal(
                    walkers.inv_ovlp_b, walkers.phib, trial.psi0b[i], i
                )
            elif walkers.ndown > 0 and walkers.rhf:
                gdown = gup
            else:
                gdown = zero

            r1 = (1.0 + self.delta[0, 0] * gup) * (1.0 + self.delta[0, 1] * gdown)
            r2 = (1.0 + self.delta[1, 0] * gup) * (1.0 + self.delta[1, 1] * gdown)
            ratios = xp.stack([r1, r2], axis=1)
            probs = 0.5 * ratios * self.aux_wfac[None, :]
            phaseless_ratio = xp.maximum(probs.real, 0.0)
            norm = xp.sum(phaseless_ratio, axis=1)
            live = (norm > 0.0) & (xp.abs(walkers.weight) > 0.0)

            norm_safe = xp.where(live, norm, 1.0)
            p0 = xp.where(live, phaseless_ratio[:, 0] / norm_safe, 1.0)
            xi = (random_fields[i] >= p0).astype(numpy.int32)
            selected = ratios[xp.arange(walkers.nwalkers), xi]

            walkers.weight *= xp.where(live, norm, 0.0)
            walkers.ovlp[...] = xp.where(live, walkers.ovlp * selected, walkers.ovlp)

            vtup = walkers.phia[:, i, :] * self.delta[xi, 0][:, None]
            vtup = xp.where(live[:, None], vtup, 0.0)
            walkers.phia[:, i, :] += vtup
            walkers.inv_ovlp_a[...] = self._batched_sherman_morrison(
                walkers.inv_ovlp_a, trial.psi0a[i, :].conj(), vtup
            )

            if walkers.ndown > 0 and not walkers.rhf:
                vtdown = walkers.phib[:, i, :] * self.delta[xi, 1][:, None]
                vtdown = xp.where(live[:, None], vtdown, 0.0)
                walkers.phib[:, i, :] += vtdown
                walkers.inv_ovlp_b[...] = self._batched_sherman_morrison(
                    walkers.inv_ovlp_b, trial.psi0b[i, :].conj(), vtdown
                )

    def _ghf_two_body(self, walkers, hamiltonian, trial, random_fields):
        for i in range(hamiltonian.nbasis):
            guu, gud, gdu, gdd = self._ghf_site_greens_block(walkers, trial, i)

            r1 = (1.0 + self.delta[0, 0] * guu) * (1.0 + self.delta[0, 1] * gdd) - self.delta[
                0, 0
            ] * self.delta[0, 1] * gud * gdu
            r2 = (1.0 + self.delta[1, 0] * guu) * (1.0 + self.delta[1, 1] * gdd) - self.delta[
                1, 0
            ] * self.delta[1, 1] * gud * gdu
            ratios = xp.stack([r1, r2], axis=1)
            probs = 0.5 * ratios * self.aux_wfac[None, :]
            phaseless_ratio = xp.maximum(probs.real, 0.0)
            norm = xp.sum(phaseless_ratio, axis=1)
            live = (norm > 0.0) & (xp.abs(walkers.weight) > 0.0)

            norm_safe = xp.where(live, norm, 1.0)
            p0 = xp.where(live, phaseless_ratio[:, 0] / norm_safe, 1.0)
            xi = (random_fields[i] >= p0).astype(numpy.int32)
            selected = ratios[xp.arange(walkers.nwalkers), xi]

            walkers.weight *= xp.where(live, norm, 0.0)
            walkers.ovlp[...] = xp.where(live, walkers.ovlp * selected, walkers.ovlp)

            up = i
            down = i + walkers.nbasis
            delta_up = self.delta[xi, 0]
            delta_down = self.delta[xi, 1]
            vtup = walkers.phi[:, up, :] * delta_up[:, None]
            vtdown = walkers.phi[:, down, :] * delta_down[:, None]
            vtup = xp.where(live[:, None], vtup, 0.0)
            vtdown = xp.where(live[:, None], vtdown, 0.0)

            walkers.phi[:, up, :] += vtup
            walkers.inv_ovlp[...] = self._batched_sherman_morrison(
                walkers.inv_ovlp, trial.psi0[up, :].conj(), vtup
            )
            walkers.phi[:, down, :] += vtdown
            walkers.inv_ovlp[...] = self._batched_sherman_morrison(
                walkers.inv_ovlp, trial.psi0[down, :].conj(), vtdown
            )

    def _noci_ghf_transients(self, walkers, trial):
        """Per-determinant inverse overlaps and overlaps relative to a per-walker reference.

        inv_ovlp[k, w] = (Phi_k^dag phi_w)^{-1} and rel_ovlp[k, w] = <Phi_k|phi_w> / max_k |...|.
        """
        detsc = xp.asarray(trial.dets_conj)  # (K, 2nb, nocc)
        cstar = xp.asarray(trial.coeffs).conj()  # (K,)
        omat = xp.einsum("kji,wjm->kwim", detsc, walkers.phi, optimize=True)  # (K, w, nocc, nocc)
        signs, logdet = xp.linalg.slogdet(omat)
        inv_ovlp = xp.linalg.inv(omat)
        ref = xp.max(logdet.real, axis=0)  # (w,)
        rel_ovlp = signs * xp.exp(logdet - ref[None, :])  # (K, w)
        return detsc, cstar, inv_ovlp, rel_ovlp

    def _noci_ghf_two_body(self, walkers, hamiltonian, trial, random_fields):
        """Discrete-Hirsch two body for a multi-determinant (NOCI) GHF trial.

        Per walker we keep K transient inverse overlaps inv_ovlp[k] = (Phi_k^dag phi)^{-1}
        and relative overlaps rel_ovlp[k] ∝ <Phi_k|phi>. The single-site importance
        ratio becomes the overlap-weighted average over determinants
            ratio(xi) = sum_k P_k R_k(xi),   P_k = c_k^* O_k / sum_k c_k^* O_k,
        where R_k(xi) is the usual single-determinant GHF site ratio for Phi_k.
        With K=1 this reduces exactly to ``_ghf_two_body``. The sweep runs in
        ``hubbard_two_body_noci_ghf_cpu``; NOCIGHF overlaps are computed with numpy, so this
        trial is CPU only.
        """
        if hasattr(xp, "RawKernel"):
            raise NotImplementedError("NOCIGHF trials are only supported on CPU.")
        detsc, cstar, inv_ovlp, rel_ovlp = self._noci_ghf_transients(walkers, trial)
        walkers.phi = numpy.ascontiguousarray(walkers.phi, dtype=numpy.complex128)
        walkers.ovlp = numpy.ascontiguousarray(walkers.ovlp, dtype=numpy.complex128)
        hubbard_two_body_noci_ghf_cpu(
            walkers.phi,
            numpy.ascontiguousarray(inv_ovlp),
            numpy.ascontiguousarray(rel_ovlp),
            numpy.ascontiguousarray(cstar),
            walkers.weight,
            walkers.ovlp,
            numpy.ascontiguousarray(detsc),
            numpy.asarray(self.delta, dtype=numpy.complex128),
            numpy.asarray(self.aux_wfac, dtype=numpy.complex128),
            numpy.asarray(random_fields, dtype=numpy.float64),
        )

    def _noci_ghf_two_body_einsum(self, walkers, hamiltonian, trial, random_fields):
        """Batched-einsum reference implementation of the NOCI-GHF site sweep."""
        nb = walkers.nbasis
        nw = walkers.nwalkers
        detsc, cstar, inv_ovlp, rel_ovlp = self._noci_ghf_transients(walkers, trial)

        def _ksm(inv, u, vt):
            # batched Sherman-Morrison over (k, w): inv (K,w,n,n), u (K,n), vt (w,n)
            au = xp.einsum("kwij,kj->kwi", inv, u, optimize=True)
            vta = xp.einsum("wi,kwij->kwj", vt, inv, optimize=True)
            denom = 1.0 + xp.einsum("kwi,ki->kw", vta, u, optimize=True)
            upd = xp.einsum("kwi,kwj->kwij", au, vta, optimize=True) / denom[:, :, None, None]
            return inv - upd

        for i in range(nb):
            up = i
            down = i + nb
            psi_up = detsc[:, up, :]  # (K, nocc)
            psi_down = detsc[:, down, :]
            ghalf_up = xp.einsum("wi,kwij->kwj", walkers.phi[:, up, :], inv_ovlp, optimize=True)
            ghalf_down = xp.einsum("wi,kwij->kwj", walkers.phi[:, down, :], inv_ovlp, optimize=True)
            guu = xp.einsum("kwj,kj->kw", ghalf_up, psi_up, optimize=True)
            gud = xp.einsum("kwj,kj->kw", ghalf_up, psi_down, optimize=True)
            gdu = xp.einsum("kwj,kj->kw", ghalf_down, psi_up, optimize=True)
            gdd = xp.einsum("kwj,kj->kw", ghalf_down, psi_down, optimize=True)

            # per-det single-site ratios for the two HS fields
            R = xp.stack(
                [
                    (1.0 + self.delta[xi, 0] * guu) * (1.0 + self.delta[xi, 1] * gdd)
                    - self.delta[xi, 0] * self.delta[xi, 1] * gud * gdu
                    for xi in range(2)
                ],
                axis=0,
            )  # (2, K, w)

            # determinant weights P_k = c_k^* O_k / sum_k c_k^* O_k
            pk = cstar[:, None] * rel_ovlp  # (K, w)
            psum = xp.sum(pk, axis=0)  # (w,)
            ratios = xp.einsum("kw,xkw->wx", pk, R, optimize=True) / psum[:, None]  # (w, 2)

            probs = 0.5 * ratios * self.aux_wfac[None, :]
            phaseless_ratio = xp.maximum(probs.real, 0.0)
            norm = xp.sum(phaseless_ratio, axis=1)
            live = (norm > 0.0) & (xp.abs(walkers.weight) > 0.0)

            norm_safe = xp.where(live, norm, 1.0)
            p0 = xp.where(live, phaseless_ratio[:, 0] / norm_safe, 1.0)
            xi = (random_fields[i] >= p0).astype(numpy.int32)
            selected = ratios[xp.arange(nw), xi]

            walkers.weight *= xp.where(live, norm, 0.0)
            walkers.ovlp[...] = xp.where(live, walkers.ovlp * selected, walkers.ovlp)

            # update relative overlaps: O_k -> O_k * R_k(xi)
            r_sel = R[xi, :, xp.arange(nw)].T  # (K, w)
            rel_ovlp = xp.where(live[None, :], rel_ovlp * r_sel, rel_ovlp)

            # update walker orbitals (shared) and all dets' inverse overlaps
            delta_up = self.delta[xi, 0]
            delta_down = self.delta[xi, 1]
            vtup = xp.where(live[:, None], walkers.phi[:, up, :] * delta_up[:, None], 0.0)
            vtdown = xp.where(live[:, None], walkers.phi[:, down, :] * delta_down[:, None], 0.0)
            walkers.phi[:, up, :] += vtup
            inv_ovlp = _ksm(inv_ovlp, psi_up, vtup)
            walkers.phi[:, down, :] += vtdown
            inv_ovlp = _ksm(inv_ovlp, psi_down, vtdown)

    def _ensure_buffers(self, walkers):
        shape = (walkers.nwalkers, walkers.nup, walkers.ndown)
        if getattr(self, "_buffer_shape", None) == shape:
            return
        self._buffer_shape = shape
        self._vtup = xp.empty((walkers.nwalkers, walkers.nup), dtype=xp.complex128)
        self._vtdown = xp.empty((walkers.nwalkers, walkers.ndown), dtype=xp.complex128)
        self._xi = xp.empty(walkers.nwalkers, dtype=numpy.int32)
        self._live = xp.empty(walkers.nwalkers, dtype=numpy.int8)

    def _complex_trial_orbitals(self, trial):
        # Hold the source arrays so the identity check cannot match a recycled object.
        cached = getattr(self, "_trial_buffer_key", None)
        if cached is None or cached[0] is not trial.psi0a or cached[1] is not trial.psi0b:
            self._trial_buffer_key = (trial.psi0a, trial.psi0b)
            self._psi0a_complex = xp.ascontiguousarray(xp.asarray(trial.psi0a, dtype=xp.complex128))
            self._psi0b_complex = xp.ascontiguousarray(xp.asarray(trial.psi0b, dtype=xp.complex128))
        return self._psi0a_complex, self._psi0b_complex

    def _ensure_cuda_arrays_contiguous(self, walkers):
        walkers.phia = xp.ascontiguousarray(walkers.phia)
        if walkers.phib is not None:
            walkers.phib = xp.ascontiguousarray(walkers.phib)
        walkers.inv_ovlp_a = xp.ascontiguousarray(walkers.inv_ovlp_a)
        if walkers.inv_ovlp_b is not None:
            walkers.inv_ovlp_b = xp.ascontiguousarray(walkers.inv_ovlp_b)
        walkers.weight = xp.ascontiguousarray(walkers.weight)
        walkers.ovlp = xp.ascontiguousarray(walkers.ovlp)

    def _sm_thread_count(self, nocc):
        threads = 1
        while threads < nocc:
            threads *= 2
        return min(max(threads, 32), 256)

    def _site_thread_count(self, nocc):
        if nocc >= 48:
            return 256
        if nocc >= 16:
            return 128
        return 0

    def _large_apply_kernel(self, kernels):
        block = (16, 16)
        shared_mem = (block[0] + block[1] + 1) * numpy.dtype(numpy.complex128).itemsize
        return kernels["sherman_morrison_apply_large"], shared_mem

    def _launch_sherman_morrison(self, kernel, inv, psi_site, vt, nwalkers, nocc):
        threads = self._sm_thread_count(nocc)
        shared_mem = (2 * nocc + 1) * numpy.dtype(numpy.complex128).itemsize
        kernel((nwalkers,), (threads,), (inv, psi_site, vt, nwalkers, nocc), shared_mem=shared_mem)

    def _launch_sherman_morrison_cublas_large(
        self, apply_kernel, inv, psi_site, vt, nwalkers, nocc
    ):
        # The apply kernel indexes au, vta as (nwalkers, nocc) with row stride nocc, so they are
        # built per spin rather than as slices of a max(nup, ndown)-wide buffer.
        u = xp.conj(psi_site)
        au = xp.ascontiguousarray(xp.matmul(inv, u))
        vta = xp.ascontiguousarray(xp.matmul(vt[:, None, :], inv)[:, 0, :])
        denom = xp.ascontiguousarray(1.0 + xp.einsum("wi,i->w", vta, u, optimize=True))
        block = (16, 16)
        grid = ((nocc + block[0] - 1) // block[0], (nocc + block[1] - 1) // block[1], nwalkers)
        apply_kernel(
            grid,
            block,
            (inv, au, vta, denom, nwalkers, nocc),
            shared_mem=getattr(self, "_large_apply_shared_mem", 0),
        )

    def _launch_sherman_morrison_auto(self, kernels, inv, psi_site, vt, nwalkers, nocc):
        shared_mem = (2 * nocc + 1) * numpy.dtype(numpy.complex128).itemsize
        if shared_mem <= 48 * 1024:
            self._launch_sherman_morrison(
                kernels["sherman_morrison"], inv, psi_site, vt, nwalkers, nocc
            )
        else:
            apply_kernel, apply_shared_mem = self._large_apply_kernel(kernels)
            self._large_apply_shared_mem = apply_shared_mem
            self._launch_sherman_morrison_cublas_large(
                apply_kernel, inv, psi_site, vt, nwalkers, nocc
            )

    def _cuda_two_body(self, walkers, hamiltonian, trial, random_fields):
        from ipie.propagation.kernels.gpu.hubbard import get_hubbard_single_site_kernels

        self._ensure_buffers(walkers)
        psi0a, psi0b = self._complex_trial_orbitals(trial)
        kernels = get_hubbard_single_site_kernels()
        site_kernel = kernels["site_update"]
        site_kernel_parallel = kernels["site_update_parallel"]
        threads = 128
        blocks = ((walkers.nwalkers + threads - 1) // threads,)
        parallel_site_threads = self._site_thread_count(walkers.nup)
        use_parallel_site = parallel_site_threads > 0

        for i in range(hamiltonian.nbasis):
            site_args = (
                walkers.phia,
                walkers.phib,
                walkers.inv_ovlp_a,
                walkers.inv_ovlp_b,
                psi0a,
                psi0b,
                self.delta,
                self.aux_wfac,
                random_fields[i],
                walkers.weight,
                walkers.ovlp,
                self._vtup,
                self._vtdown,
                self._xi,
                self._live,
                i,
                walkers.nwalkers,
                hamiltonian.nbasis,
                walkers.nup,
                walkers.ndown,
                int(walkers.rhf),
            )
            if use_parallel_site:
                site_kernel_parallel(
                    (walkers.nwalkers,),
                    (parallel_site_threads,),
                    site_args,
                    shared_mem=4 * parallel_site_threads * numpy.dtype(numpy.float64).itemsize,
                )
            else:
                site_kernel(blocks, (threads,), site_args)
            self._launch_sherman_morrison_auto(
                kernels, walkers.inv_ovlp_a, psi0a[i, :], self._vtup, walkers.nwalkers, walkers.nup
            )
            self._launch_sherman_morrison_auto(
                kernels,
                walkers.inv_ovlp_b,
                psi0b[i, :],
                self._vtdown,
                walkers.nwalkers,
                walkers.ndown,
            )

    def _is_nvidia_gpu(self):
        if not hasattr(xp, "RawKernel"):
            return False
        props = xp.cuda.runtime.getDeviceProperties(xp.cuda.Device().id)
        name = props.get("name", b"")
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="ignore")
        return "NVIDIA" in name.upper() or "CUDA" in name.upper()

    def _estimate_path_bytes(self, walkers, hamiltonian, path):
        complex_bytes = numpy.dtype(numpy.complex128).itemsize
        float_bytes = numpy.dtype(numpy.float64).itemsize
        int_bytes = numpy.dtype(numpy.int32).itemsize
        nbasis = hamiltonian.nbasis
        nocc = max(walkers.nup, walkers.ndown)
        per_walker = (
            2 * nbasis * nocc * complex_bytes
            + 2 * nocc * nocc * complex_bytes
            + 2 * nocc * complex_bytes
            + 2 * complex_bytes
            + float_bytes
            + int_bytes
        )
        if path == "einsum":
            per_walker += 2 * nocc * nocc * complex_bytes
        static = 2 * nbasis * nocc * complex_bytes + nbasis * walkers.nwalkers * float_bytes
        return static, per_walker

    def _safe_chunk_size(self, walkers, hamiltonian, path):
        if not hasattr(xp, "cuda"):
            return walkers.nwalkers
        free_bytes, _ = xp.cuda.Device().mem_info
        static, per_walker = self._estimate_path_bytes(walkers, hamiltonian, path)
        available = int(free_bytes * _HUBBARD_GPU_MEMORY_FRACTION) - static
        if available <= per_walker:
            return 1
        return max(1, min(walkers.nwalkers, available // per_walker))

    def _choose_path(self, walkers, hamiltonian):
        if isinstance(walkers, GHFWalkers):
            if not hasattr(xp, "RawKernel"):
                return "ghf_numba"
            return "ghf_einsum"
        if not hasattr(xp, "RawKernel"):
            return "cpu_numba"
        if not self._is_nvidia_gpu() or walkers.rhf or walkers.ndown == 0:
            return "einsum"
        einsum_chunk = self._safe_chunk_size(walkers, hamiltonian, "einsum")
        cuda_chunk = self._safe_chunk_size(walkers, hamiltonian, "cuda")
        if einsum_chunk < walkers.nwalkers and cuda_chunk >= walkers.nwalkers:
            return "cuda"
        if hamiltonian.nbasis <= 484:
            return "cuda"
        return "einsum"

    def _walker_view(self, walkers, start, stop):
        return SimpleNamespace(
            nwalkers=stop - start,
            nup=walkers.nup,
            ndown=walkers.ndown,
            nbasis=walkers.nbasis,
            rhf=walkers.rhf,
            weight=walkers.weight[start:stop],
            ovlp=walkers.ovlp[start:stop],
            phia=walkers.phia[start:stop],
            phib=walkers.phib[start:stop],
            inv_ovlp_a=walkers.inv_ovlp_a[start:stop],
            inv_ovlp_b=walkers.inv_ovlp_b[start:stop],
        )

    def _run_gpu_path_chunked(self, path, walkers, hamiltonian, trial, random_fields):
        if path == "cuda":
            self._ensure_cuda_arrays_contiguous(walkers)
        chunk_size = self._safe_chunk_size(walkers, hamiltonian, path)
        start = 0
        while start < walkers.nwalkers:
            stop = min(walkers.nwalkers, start + chunk_size)
            chunk = self._walker_view(walkers, start, stop)
            chunk_fields = random_fields[:, start:stop]
            if path == "cuda":
                self._cuda_two_body(chunk, hamiltonian, trial, chunk_fields)
            else:
                self._einsum_two_body(chunk, hamiltonian, trial, chunk_fields)
            start = stop

    def propagate_walkers_two_body(self, walkers, hamiltonian, trial):
        start_time = time.time()
        from ipie.trial_wavefunction.noci_ghf import NOCIGHF

        path = self._choose_path(walkers, hamiltonian)
        if isinstance(trial, NOCIGHF):
            path = "noci_ghf"
        if not hasattr(xp, "RawKernel"):
            random_fields = numpy.random.random((hamiltonian.nbasis, walkers.nwalkers))
        else:
            random_fields = xp.random.random(
                (hamiltonian.nbasis, walkers.nwalkers), dtype=xp.float64
            )

        if path == "cpu_numba":
            self._cpu_numba_two_body(walkers, hamiltonian, trial, random_fields)
        elif path == "cuda":
            self._run_gpu_path_chunked("cuda", walkers, hamiltonian, trial, random_fields)
        elif path == "einsum":
            self._run_gpu_path_chunked("einsum", walkers, hamiltonian, trial, random_fields)
        elif path == "ghf_einsum":
            self._ghf_two_body(walkers, hamiltonian, trial, random_fields)
        elif path == "ghf_numba":
            self._numba_ghf_two_body(walkers, hamiltonian, trial, random_fields)
        elif path == "noci_ghf":
            self._noci_ghf_two_body(walkers, hamiltonian, trial, random_fields)
        else:
            raise ValueError(f"Unknown Hubbard propagation path {path}")

        synchronize()
        self.timer.tgf += time.time() - start_time


class HubbardFB(HirschBase):
    """Discrete Hubbard propagator using an all-site dynamical force bias update.

    Each field is drawn from P(x) proportional to exp(x b) with the real bias b = Re(gamma f),
    f = <n_up - n_down> (spin) or <n_up + n_down - 1> (charge) from the mixed Green's function.
    The importance weight (1/2) / P(x) = norm exp(-x b) keeps the sampling unbiased for any b.
    """

    def propagate_walkers_two_body(self, walkers, hamiltonian, trial):
        start_time = time.time()

        trial.calc_greens_function(walkers, build_full=True)
        nia = xp.einsum("wii->wi", walkers.Ga)
        nib = xp.einsum("wii->wi", walkers.Gb)

        if self.spin_decomp:
            fb_term = nia - nib
        else:
            fb_term = nia + nib - 1.0
        bias = (self.gamma * fb_term).real

        pp = 0.5 * xp.exp(bias)
        pm = 0.5 * xp.exp(-bias)
        norm = pp + pm

        fields = xp.zeros_like(pp, dtype=numpy.int32)
        live = norm > 0.0
        rnd = xp.random.random(norm.shape)
        choose_one = rnd >= xp.where(live, pp / norm, 1.0)
        fields[live] = choose_one[live].astype(numpy.int32)

        fb_fac = xp.ones(walkers.nwalkers, dtype=numpy.float64)
        wfac = xp.ones(walkers.nwalkers, dtype=xp.complex128)
        alive = xp.all(live, axis=1)

        for i in range(walkers.nbasis):
            xi = fields[:, i]
            contrib0 = norm[:, i] * xp.exp(-bias[:, i])
            contrib1 = norm[:, i] * xp.exp(bias[:, i])
            fb_fac *= xp.where(xi == 0, contrib0, contrib1)
            wfac *= self.aux_wfac[xi]
            walkers.phia[:, i, :] *= self.auxf[xi, 0][:, None]
            if walkers.ndown > 0 and not walkers.rhf:
                walkers.phib[:, i, :] *= self.auxf[xi, 1][:, None]

        ovlp_new = trial.calc_overlap(walkers)
        ratio = wfac * ovlp_new / walkers.ovlp
        weight_factor = xp.where(alive, xp.maximum((fb_fac * ratio).real, 0.0), 0.0)
        walkers.weight *= weight_factor
        walkers.ovlp = ovlp_new

        synchronize()
        self.timer.tfbias += time.time() - start_time
