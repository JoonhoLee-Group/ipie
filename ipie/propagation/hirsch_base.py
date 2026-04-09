import os
import time

from abc import abstractmethod

import numpy
import plum
import scipy.linalg
import math

from ipie.hamiltonians.hubbard import Hubbard
from ipie.propagation.continuous_base import ContinuousBase
from ipie.propagation.operations import propagate_one_body
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import cast_to_device, synchronize, to_host


@plum.dispatch
def construct_one_body_propagator(hamiltonian: Hubbard, dt: float):
    """Construct the half-step one-body propagator for Hirsch CPMC."""
    return xp.array(
        [
            scipy.linalg.expm(-0.5 * dt * hamiltonian.T[0]),
            scipy.linalg.expm(-0.5 * dt * hamiltonian.T[1]),
        ]
    )


def construct_hirsch_auxiliaries(hamiltonian, dt, spin_decomp=True):
    """Construct discrete Hirsch HS factors."""
    if spin_decomp:
        gamma = numpy.arccosh(numpy.exp(0.5 * dt * hamiltonian.U))
        auxf = numpy.array(
            [
                [numpy.exp(gamma), numpy.exp(-gamma)],
                [numpy.exp(-gamma), numpy.exp(gamma)],
            ],
            dtype=numpy.complex128,
        )
        aux_wfac = numpy.array([1.0, 1.0], dtype=numpy.complex128)
    else:
        gamma = numpy.arccosh(numpy.exp(-0.5 * dt * hamiltonian.U + 0j))
        auxf = numpy.array(
            [
                [numpy.exp(gamma), numpy.exp(gamma)],
                [numpy.exp(-gamma), numpy.exp(-gamma)],
            ],
            dtype=numpy.complex128,
        )
        aux_wfac = numpy.exp(0.5 * dt * hamiltonian.U) * numpy.array(
            [numpy.exp(-gamma), numpy.exp(gamma)],
            dtype=numpy.complex128,
        )
    auxf *= numpy.exp(-0.5 * dt * hamiltonian.U)
    delta = auxf - 1.0
    return gamma, xp.array(auxf), xp.array(aux_wfac), xp.array(delta)


class HirschBase(ContinuousBase):
    """Batched Hirsch constrained-path propagator for Hubbard Hamiltonians."""

    def __init__(
        self,
        time_step,
        ene_bound_const=2.0,
        fb_bound=1.0,
        spin_decomp=True,
        verbose=False,
    ):
        super().__init__(time_step, verbose=verbose)
        self.spin_decomp = spin_decomp
        self.ene_bound_const = ene_bound_const
        self.fb_bound = fb_bound
        self.mpi_handler = None
        self.expH1 = None
        self.gamma = None
        self.auxf = None
        self.aux_wfac = None
        self.delta = None
        self.debug_hubbard = os.environ.get("IPIE_DEBUG_HUBBARD", "0") == "1"
        self.debug_iw = int(os.environ.get("IPIE_DEBUG_WALKER", "0"))
        self.debug_max_sites = int(os.environ.get("IPIE_DEBUG_MAX_SITES", "8"))

    def _debug(self, tag, **kwargs):
        if not self.debug_hubbard:
            return
        details = " ".join(f"{key}={value}" for key, value in kwargs.items())
        print(f"[new:{tag}] {details}")

    def build(self, hamiltonian, trial=None, walkers=None, mpi_handler=None, verbose=False):
        self.mpi_handler = mpi_handler
        self.expH1 = construct_one_body_propagator(hamiltonian, self.dt)
        self.gamma, self.auxf, self.aux_wfac, self.delta = construct_hirsch_auxiliaries(
            hamiltonian, self.dt, spin_decomp=self.spin_decomp
        )
        if verbose:
            print("# Finished setting up Hirsch constrained propagator.")

    def _sherman_morrison(self, ainv, u, vt):
        return ainv - (ainv @ xp.outer(u, vt) @ ainv) / (1.0 + vt @ ainv @ u)

    def _calc_overlap_from_inverse(self, walkers):
        sign_a, logdet_a = xp.linalg.slogdet(walkers.inv_ovlp_a)
        sign_b = xp.ones(walkers.nwalkers, dtype=xp.complex128)
        logdet_b = xp.zeros(walkers.nwalkers, dtype=xp.float64)
        if walkers.ndown > 0:
            sign_b, logdet_b = xp.linalg.slogdet(walkers.inv_ovlp_b)
        det = sign_a * sign_b * xp.exp(logdet_a + logdet_b - walkers.log_shift)
        return 1.0 / det

    def propagate_walkers_one_body(self, walkers):
        start_time = time.time()
        walkers.phia = propagate_one_body(walkers.phia, self.expH1[0])
        if walkers.ndown > 0 and not walkers.rhf:
            walkers.phib = propagate_one_body(walkers.phib, self.expH1[1])
        synchronize()
        self.timer.tgemm += time.time() - start_time

    def kinetic_importance_sampling(self, walkers, trial):
        start_time = time.time()
        self.propagate_walkers_one_body(walkers)
        walkers.inverse_overlap(trial)
        ovlp_new = self._calc_overlap_from_inverse(walkers)
        ratio = ovlp_new / walkers.ovlp
        phase = xp.angle(ratio)
        if self.debug_hubbard and self.debug_iw < walkers.nwalkers:
            iw = self.debug_iw
            self._debug(
                "kinetic",
                iw=iw,
                weight=to_host(walkers.weight[iw]),
                ot_old=to_host(walkers.ovlp[iw]),
                ot_new=to_host(ovlp_new[iw]),
                ratio=to_host(ratio[iw]),
                phase=to_host(phase[iw]),
            )
        weight_factor = xp.where(xp.abs(phase) < 0.5 * math.pi, ratio.real, 0.0)
        walkers.weight *= weight_factor
        walkers.ovlp = xp.where(weight_factor > 0.0, ovlp_new, walkers.ovlp)
        synchronize()
        self.timer.tovlp += time.time() - start_time

    def _calculate_overlap_ratio(self, walkers, i):
        gup = walkers.Ga[:, i, i]
        gdown = walkers.Gb[:, i, i]
        r1 = (1.0 + self.delta[0, 0] * gup) * (1.0 + self.delta[0, 1] * gdown)
        r2 = (1.0 + self.delta[1, 0] * gup) * (1.0 + self.delta[1, 1] * gdown)
        probs = 0.5 * xp.stack([r1, r2], axis=1)
        probs *= self.aux_wfac[None, :]
        return probs

    def _sample_fields_single_site(self, probs):
        phaseless_ratio = xp.maximum(probs.real, 0.0)
        norm = xp.sum(phaseless_ratio, axis=1)
        chosen = xp.zeros(norm.shape[0], dtype=numpy.int32)
        live = norm > 0.0
        if xp.any(live):
            rnd = xp.random.random(int(xp.sum(live)))
            probs0 = phaseless_ratio[live, 0] / norm[live]
            chosen_live = (rnd >= probs0).astype(numpy.int32)
            chosen[live] = chosen_live
        return chosen, norm, live

    def _apply_single_site_update(self, walkers, i, chosen, live):
        if not xp.any(live):
            return
        factors_up = self.auxf[chosen[live], 0]
        walkers.phia[live, i, :] *= factors_up[:, None]
        if walkers.ndown > 0 and not walkers.rhf:
            factors_dn = self.auxf[chosen[live], 1]
            walkers.phib[live, i, :] *= factors_dn[:, None]

    @abstractmethod
    def propagate_walkers_two_body(self, walkers, hamiltonian, trial):
        pass

    def propagate_walkers(self, walkers, hamiltonian, trial, eshift):
        synchronize()
        if walkers.ovlp is None or len(walkers.ovlp) != walkers.nwalkers:
            walkers.ovlp = trial.calc_overlap(walkers)

        self.kinetic_importance_sampling(walkers, trial)
        self.propagate_walkers_two_body(walkers, hamiltonian, trial)
        self.kinetic_importance_sampling(walkers, trial)

        walkers.weight *= numpy.exp(self.dt * eshift)

    def cast_to_cupy(self, verbose=False):
        cast_to_device(self, verbose=verbose)
