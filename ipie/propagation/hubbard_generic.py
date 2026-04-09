import time

import numpy

from ipie.propagation.hirsch_base import HirschBase
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import synchronize, to_host


class HubbardSingleSite(HirschBase):
    """Discrete Hubbard propagator using legacy site-by-site updates."""

    def propagate_walkers_two_body(self, walkers, hamiltonian, trial):
        start_time = time.time()

        # Follow the legacy discrete single-site update order exactly:
        # walker-by-walker, site-by-site, with inverse-overlap updates after
        # each chosen auxiliary field.
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
                rnd = float(to_host(xp.random.random()))
                if self.debug_hubbard and iw == self.debug_iw and i < self.debug_max_sites:
                    self._debug(
                        "site_pre",
                        iw=iw,
                        site=i,
                        gii_up=to_host(gup),
                        gii_dn=to_host(gdown),
                        probs=to_host(probs),
                        phaseless_ratio=to_host(phaseless_ratio),
                        norm=norm_scalar,
                        rand=rnd,
                        weight=to_host(walkers.weight[iw]),
                        ot=to_host(walkers.ovlp[iw]),
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
                if self.debug_hubbard and iw == self.debug_iw and i < self.debug_max_sites:
                    self._debug(
                        "site_post",
                        iw=iw,
                        site=i,
                        xi=xi,
                        weight=to_host(walkers.weight[iw]),
                        ot=to_host(walkers.ovlp[iw]),
                        vtup=to_host(vtup),
                        vtdown=None if vtdown is None else to_host(vtdown),
                    )

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
