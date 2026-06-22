import math
import time
from types import SimpleNamespace

import numpy

from ipie.propagation.hirsch_base import HirschBase
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import synchronize, to_host
from ipie.walkers.ghf_walkers import GHFWalkers

try:
    from numba import njit
except ImportError:  # pragma: no cover - optional acceleration
    njit = None


_NUMBA_CPU_KERNEL = None
_HUBBARD_GPU_MEMORY_FRACTION = 0.70


def _make_numba_cpu_kernel():
    if njit is None:
        return None

    @njit(cache=True)
    def _kernel(
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

                prob0 = 0.5 * (1.0 + delta[0, 0] * gup) * (1.0 + delta[0, 1] * gdown) * aux_wfac[0]
                prob1 = 0.5 * (1.0 + delta[1, 0] * gup) * (1.0 + delta[1, 1] * gdown) * aux_wfac[1]
                p0_real = prob0.real if prob0.real > 0.0 else 0.0
                p1_real = prob1.real if prob1.real > 0.0 else 0.0
                norm = p0_real + p1_real
                if norm <= 0.0:
                    weight[iw] = 0.0
                    break

                p0 = p0_real / norm
                xi = 0 if random_fields[site, iw] < p0 else 1
                selected = prob0 if xi == 0 else prob1
                weight[iw] *= norm
                ovlp[iw] = 2.0 * ovlp[iw] * selected

                delta_up = delta[xi, 0]
                scale_up = 1.0 + delta_up
                for j in range(nup):
                    phia[iw, site, j] *= scale_up
                _numba_sherman_morrison_from_q(
                    inva[iw],
                    psi0a_conj[site],
                    q_up,
                    delta_up,
                    gup,
                )

                if ndown > 0 and not rhf:
                    delta_down = delta[xi, 1]
                    scale_down = 1.0 + delta_down
                    for j in range(ndown):
                        phib[iw, site, j] *= scale_down
                    _numba_sherman_morrison_from_q(
                        invb[iw],
                        psi0b_conj[site],
                        q_down,
                        delta_down,
                        gdown,
                    )

    @njit(cache=True, inline="always")
    def _numba_sherman_morrison_from_q(ainv, u, q, delta_xi, g):
        nocc = ainv.shape[0]
        scale = delta_xi / (1.0 + delta_xi * g)
        for i in range(nocc):
            val = 0.0 + 0.0j
            for k in range(nocc):
                val += ainv[i, k] * u[k]
            scaled_val = val * scale
            for j in range(nocc):
                ainv[i, j] -= scaled_val * q[j]

    return _kernel


def _numba_cpu_kernel():
    global _NUMBA_CPU_KERNEL
    if _NUMBA_CPU_KERNEL is None:
        _NUMBA_CPU_KERNEL = _make_numba_cpu_kernel()
    return _NUMBA_CPU_KERNEL


class HubbardSingleSite(HirschBase):
    """Discrete Hubbard propagator with adaptive CPU, einsum, and CUDA paths."""

    def kinetic_importance_sampling(self, walkers, trial):
        # For a multi-determinant (NOCI) GHF trial the constraint overlap ratio
        # must use the full trial overlap, not the reference-determinant inverse
        # overlap that _calc_overlap_from_inverse would give.
        from ipie.trial_wavefunction.noci_ghf import NOCIGHF

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
                probs = 0.5 * xp.array([r1, r2], dtype=xp.complex128) * self.aux_wfac
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
                walkers.ovlp[iw] = 2.0 * walkers.ovlp[iw] * probs[xi]
                walkers.inv_ovlp_a[iw] = self._sherman_morrison(
                    walkers.inv_ovlp_a[iw], trial.psi0a[i, :].conj(), vtup
                )
                if walkers.ndown > 0 and not walkers.rhf:
                    walkers.inv_ovlp_b[iw] = self._sherman_morrison(
                        walkers.inv_ovlp_b[iw], trial.psi0b[i, :].conj(), vtdown
                    )

    def _cpu_numba_two_body(self, walkers, hamiltonian, trial, random_fields):
        kernel = _numba_cpu_kernel()
        if kernel is None:
            self._legacy_two_body(walkers, hamiltonian, trial, random_fields=random_fields)
            return
        kernel(
            walkers.phia,
            walkers.phib,
            walkers.inv_ovlp_a,
            walkers.inv_ovlp_b,
            walkers.weight,
            walkers.ovlp,
            numpy.ascontiguousarray(
                numpy.conjugate(trial.psi0a), dtype=numpy.complex128
            ),
            numpy.ascontiguousarray(
                numpy.conjugate(trial.psi0b), dtype=numpy.complex128
            ),
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
            probs = 0.5 * xp.stack([r1, r2], axis=1) * self.aux_wfac[None, :]
            phaseless_ratio = xp.maximum(probs.real, 0.0)
            norm = xp.sum(phaseless_ratio, axis=1)
            live = (norm > 0.0) & (xp.abs(walkers.weight) > 0.0)

            norm_safe = xp.where(live, norm, 1.0)
            p0 = xp.where(live, phaseless_ratio[:, 0] / norm_safe, 1.0)
            xi = (random_fields[i] >= p0).astype(numpy.int32)
            selected = probs[xp.arange(walkers.nwalkers), xi]

            walkers.weight *= xp.where(live, norm, 0.0)
            walkers.ovlp[...] = xp.where(live, 2.0 * walkers.ovlp * selected, walkers.ovlp)

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

            r1 = (
                (1.0 + self.delta[0, 0] * guu) * (1.0 + self.delta[0, 1] * gdd)
                - self.delta[0, 0] * self.delta[0, 1] * gud * gdu
            )
            r2 = (
                (1.0 + self.delta[1, 0] * guu) * (1.0 + self.delta[1, 1] * gdd)
                - self.delta[1, 0] * self.delta[1, 1] * gud * gdu
            )
            probs = 0.5 * xp.stack([r1, r2], axis=1) * self.aux_wfac[None, :]
            phaseless_ratio = xp.maximum(probs.real, 0.0)
            norm = xp.sum(phaseless_ratio, axis=1)
            live = (norm > 0.0) & (xp.abs(walkers.weight) > 0.0)

            norm_safe = xp.where(live, norm, 1.0)
            p0 = xp.where(live, phaseless_ratio[:, 0] / norm_safe, 1.0)
            xi = (random_fields[i] >= p0).astype(numpy.int32)
            selected = probs[xp.arange(walkers.nwalkers), xi]

            walkers.weight *= xp.where(live, norm, 0.0)
            walkers.ovlp[...] = xp.where(live, 2.0 * walkers.ovlp * selected, walkers.ovlp)

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

    def _noci_ghf_two_body(self, walkers, hamiltonian, trial, random_fields):
        """Discrete-Hirsch two body for a multi-determinant (NOCI) GHF trial.

        Per walker we keep K transient inverse overlaps inv_ovlp[k] = (Phi_k^dag phi)^{-1}
        and relative overlaps relO[k] ∝ <Phi_k|phi>.  The single-site importance
        ratio becomes the overlap-weighted average over determinants
            ratio(xi) = sum_k P_k R_k(xi),   P_k = c_k^* O_k / sum_k c_k^* O_k,
        where R_k(xi) is the usual single-determinant GHF site ratio for Phi_k.
        With K=1 this reduces exactly to ``_ghf_two_body``.
        """
        nb = walkers.nbasis
        nw = walkers.nwalkers
        K = trial._num_dets
        detsc = xp.asarray(trial.dets_conj)                     # (K, 2nb, nocc)
        cstar = xp.asarray(trial.coeffs).conj()                 # (K,)

        # recompute transient per-det inverse overlaps and relative overlaps,
        # fully batched over the determinant axis k.
        omat = xp.einsum("kji,wjm->kwim", detsc, walkers.phi, optimize=True)  # (K,w,nocc,nocc)
        signs, logdet = xp.linalg.slogdet(omat)                 # (K,w)
        inv_ovlp = xp.linalg.inv(omat)                          # (K,w,nocc,nocc)
        ref = xp.max(logdet.real, axis=0)                       # (w,)
        relO = signs * xp.exp(logdet - ref[None, :])            # (K, w)

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
            psi_up = detsc[:, up, :]                             # (K, nocc)
            psi_down = detsc[:, down, :]
            ghalf_up = xp.einsum("wi,kwij->kwj", walkers.phi[:, up, :], inv_ovlp, optimize=True)
            ghalf_down = xp.einsum("wi,kwij->kwj", walkers.phi[:, down, :], inv_ovlp, optimize=True)
            guu = xp.einsum("kwj,kj->kw", ghalf_up, psi_up, optimize=True)
            gud = xp.einsum("kwj,kj->kw", ghalf_up, psi_down, optimize=True)
            gdu = xp.einsum("kwj,kj->kw", ghalf_down, psi_up, optimize=True)
            gdd = xp.einsum("kwj,kj->kw", ghalf_down, psi_down, optimize=True)

            # per-det single-site ratios for the two HS fields
            R = xp.stack([
                (1.0 + self.delta[xi, 0] * guu) * (1.0 + self.delta[xi, 1] * gdd)
                - self.delta[xi, 0] * self.delta[xi, 1] * gud * gdu
                for xi in range(2)
            ], axis=0)                                          # (2, K, w)

            # determinant weights P_k = c_k^* O_k / sum_k c_k^* O_k
            pk = cstar[:, None] * relO                          # (K, w)
            Psum = xp.sum(pk, axis=0)                            # (w,)
            ratio = xp.einsum("kw,xkw->xw", pk, R, optimize=True) / Psum[None, :]  # (2, w)

            probs = 0.5 * xp.transpose(ratio, (1, 0)) * self.aux_wfac[None, :]     # (w, 2)
            phaseless_ratio = xp.maximum(probs.real, 0.0)
            norm = xp.sum(phaseless_ratio, axis=1)
            live = (norm > 0.0) & (xp.abs(walkers.weight) > 0.0)

            norm_safe = xp.where(live, norm, 1.0)
            p0 = xp.where(live, phaseless_ratio[:, 0] / norm_safe, 1.0)
            xi = (random_fields[i] >= p0).astype(numpy.int32)
            selected = probs[xp.arange(nw), xi]

            walkers.weight *= xp.where(live, norm, 0.0)
            walkers.ovlp[...] = xp.where(live, 2.0 * walkers.ovlp * selected, walkers.ovlp)

            # update relative overlaps: O_k -> O_k * R_k(xi)
            Rsel = R[xi, :, xp.arange(nw)].T                    # (K, w)
            relO = xp.where(live[None, :], relO * Rsel, relO)

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
        max_occ = max(walkers.nup, walkers.ndown)
        self._large_sm_au = xp.empty((walkers.nwalkers, max_occ), dtype=xp.complex128)
        self._large_sm_vta = xp.empty((walkers.nwalkers, max_occ), dtype=xp.complex128)
        self._large_sm_denom = xp.empty(walkers.nwalkers, dtype=xp.complex128)

    def _complex_trial_orbitals(self, trial):
        key = (id(trial.psi0a), id(trial.psi0b), trial.psi0a.shape, trial.psi0b.shape)
        if getattr(self, "_trial_buffer_key", None) != key:
            self._trial_buffer_key = key
            self._psi0a_complex = xp.ascontiguousarray(
                xp.asarray(trial.psi0a, dtype=xp.complex128)
            )
            self._psi0b_complex = xp.ascontiguousarray(
                xp.asarray(trial.psi0b, dtype=xp.complex128)
            )
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

    def _launch_sherman_morrison_cublas_large(self, apply_kernel, inv, psi_site, vt, nwalkers, nocc):
        u = xp.conj(psi_site)
        self._large_sm_au[:, :nocc] = xp.matmul(inv, u)
        self._large_sm_vta[:, :nocc] = xp.matmul(vt[:, None, :], inv)[:, 0, :]
        self._large_sm_denom[:nwalkers] = 1.0 + xp.einsum(
            "wi,i->w", self._large_sm_vta[:, :nocc], u, optimize=True
        )
        block = (16, 16)
        grid = ((nocc + block[0] - 1) // block[0], (nocc + block[1] - 1) // block[1], nwalkers)
        apply_kernel(
            grid,
            block,
            (inv, self._large_sm_au, self._large_sm_vta, self._large_sm_denom, nwalkers, nocc),
            shared_mem=getattr(self, "_large_apply_shared_mem", 0),
        )

    def _launch_sherman_morrison_auto(self, kernels, inv, psi_site, vt, nwalkers, nocc):
        shared_mem = (2 * nocc + 1) * numpy.dtype(numpy.complex128).itemsize
        if shared_mem <= 48 * 1024:
            self._launch_sherman_morrison(kernels["sherman_morrison"], inv, psi_site, vt, nwalkers, nocc)
        else:
            apply_kernel, apply_shared_mem = self._large_apply_kernel(kernels)
            self._large_apply_shared_mem = apply_shared_mem
            self._launch_sherman_morrison_cublas_large(apply_kernel, inv, psi_site, vt, nwalkers, nocc)

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
                walkers.phia, walkers.phib, walkers.inv_ovlp_a, walkers.inv_ovlp_b,
                psi0a, psi0b, self.delta, self.aux_wfac, random_fields[i], walkers.weight,
                walkers.ovlp, self._vtup, self._vtdown, self._xi, self._live, i,
                walkers.nwalkers, hamiltonian.nbasis, walkers.nup, walkers.ndown, int(walkers.rhf),
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
            self._launch_sherman_morrison_auto(kernels, walkers.inv_ovlp_a, psi0a[i, :], self._vtup, walkers.nwalkers, walkers.nup)
            self._launch_sherman_morrison_auto(kernels, walkers.inv_ovlp_b, psi0b[i, :], self._vtdown, walkers.nwalkers, walkers.ndown)

    def _is_nvidia_gpu(self):
        if not hasattr(xp, "RawKernel"):
            return False
        try:
            props = xp.cuda.runtime.getDeviceProperties(xp.cuda.Device().id)
            name = props.get("name", b"")
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="ignore")
            return "NVIDIA" in name.upper() or "CUDA" in name.upper()
        except Exception:
            return True

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
        try:
            free_bytes, _ = xp.cuda.Device().mem_info
        except Exception:
            return walkers.nwalkers
        static, per_walker = self._estimate_path_bytes(walkers, hamiltonian, path)
        available = int(free_bytes * _HUBBARD_GPU_MEMORY_FRACTION) - static
        if available <= per_walker:
            return 1
        return max(1, min(walkers.nwalkers, available // per_walker))

    def _choose_path(self, walkers, hamiltonian):
        if isinstance(walkers, GHFWalkers):
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
            random_fields = xp.random.random((hamiltonian.nbasis, walkers.nwalkers), dtype=xp.float64)

        try:
            if path == "cpu_legacy":
                self._legacy_two_body(walkers, hamiltonian, trial, random_fields=random_fields)
            elif path == "cpu_numba":
                self._cpu_numba_two_body(walkers, hamiltonian, trial, random_fields)
            elif path == "cuda":
                self._run_gpu_path_chunked("cuda", walkers, hamiltonian, trial, random_fields)
            elif path == "einsum":
                self._run_gpu_path_chunked("einsum", walkers, hamiltonian, trial, random_fields)
            elif path == "ghf_einsum":
                self._ghf_two_body(walkers, hamiltonian, trial, random_fields)
            elif path == "noci_ghf":
                self._noci_ghf_two_body(walkers, hamiltonian, trial, random_fields)
            else:
                raise ValueError(f"Unknown Hubbard propagation path {path}")
        except Exception as exc:
            is_oom = hasattr(xp, "cuda") and isinstance(exc, xp.cuda.memory.OutOfMemoryError)
            if not is_oom or path not in ("cuda", "einsum"):
                raise
            xp.get_default_memory_pool().free_all_blocks()
            fallback = "einsum" if path == "cuda" else "cuda"
            self._run_gpu_path_chunked(fallback, walkers, hamiltonian, trial, random_fields)

        synchronize()
        self.timer.tgf += time.time() - start_time


class HubbardFB(HirschBase):
    """Discrete Hubbard propagator using an all-site dynamical force bias update."""

    def propagate_walkers_two_body(self, walkers, hamiltonian, trial):
        start_time = time.time()

        trial.calc_greens_function(walkers, build_full=True)
        nia = xp.einsum("wii->wi", walkers.Ga)
        nib = xp.einsum("wii->wi", walkers.Gb)

        if self.spin_decomp:
            fb_term = nia - nib
        else:
            fb_term = nia + nib - 1.0

        pp = 0.5 * xp.exp(self.gamma * fb_term).real
        pm = 0.5 * xp.exp(-self.gamma * fb_term).real
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
            contrib0 = 0.5 * norm[:, i] * xp.exp(-self.gamma * fb_term[:, i]).real
            contrib1 = 0.5 * norm[:, i] * xp.exp(self.gamma * fb_term[:, i]).real
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
