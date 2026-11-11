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

import time

from abc import abstractmethod

import numpy
import plum
import scipy.linalg
import math

from ipie.hamiltonians.hubbard import Hubbard, HubbardSpinOrbital
from ipie.propagation.continuous_base import ContinuousBase
from ipie.propagation.operations import propagate_one_body
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import cast_to_device, synchronize
from ipie.walkers.ghf_walkers import GHFWalkers
from ipie.walkers.uhf_walkers import UHFWalkers


@plum.dispatch
def construct_one_body_propagator(hamiltonian: Hubbard, dt: float):
    """Construct the half-step one-body propagator for Hirsch CPMC."""
    return xp.array(
        [
            scipy.linalg.expm(-0.5 * dt * hamiltonian.T[0]),
            scipy.linalg.expm(-0.5 * dt * hamiltonian.T[1]),
        ]
    )


@plum.dispatch
def construct_one_body_propagator(hamiltonian: HubbardSpinOrbital, dt: float):
    """Half-step one-body propagator on the spin orbitals, shape (2 nbasis, 2 nbasis)."""
    return xp.array(scipy.linalg.expm(-0.5 * dt * hamiltonian.H1))


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
        spin_decomp=None,
        verbose=False,
    ):
        # spin_decomp=None picks the real decoupling at build: spin channel for U >= 0, charge
        # channel for U < 0.
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

    def build(self, hamiltonian, trial=None, walkers=None, mpi_handler=None, verbose=False):
        if walkers is not None:
            if isinstance(hamiltonian, HubbardSpinOrbital) and not isinstance(walkers, GHFWalkers):
                raise TypeError("HubbardSpinOrbital couples the spins and needs GHF walkers.")
            if walkers.rhf:
                raise ValueError("The Hirsch propagators do not support RHF walkers.")
            if isinstance(walkers, UHFWalkers) and walkers.ndown == 0:
                raise ValueError("The Hirsch propagators need ndown > 0 for UHF walkers.")
        self.mpi_handler = mpi_handler
        self.expH1 = construct_one_body_propagator(hamiltonian, self.dt)
        if self.spin_decomp is None:
            self.spin_decomp = bool(numpy.real(hamiltonian.U) >= 0.0)
        if self.spin_decomp and numpy.real(hamiltonian.U) < 0.0:
            raise ValueError("The spin decoupling needs U >= 0; use spin_decomp=False for U < 0.")
        self.gamma, self.auxf, self.aux_wfac, self.delta = construct_hirsch_auxiliaries(
            hamiltonian, self.dt, spin_decomp=self.spin_decomp
        )
        if verbose:
            print("# Finished setting up Hirsch constrained propagator.")

    def _sherman_morrison(self, ainv, u, vt):
        return ainv - (ainv @ xp.outer(u, vt) @ ainv) / (1.0 + vt @ ainv @ u)

    def _calc_overlap_from_inverse(self, walkers):
        # det(A^{-1}) exp(log_shift) = 1 / (det(A) exp(-log_shift)), the convention of
        # greens_function_single_det.
        if isinstance(walkers, GHFWalkers):
            sign, logdet = xp.linalg.slogdet(walkers.inv_ovlp)
            det = sign * xp.exp(logdet + walkers.log_shift)
            return 1.0 / det

        sign_a, logdet_a = xp.linalg.slogdet(walkers.inv_ovlp_a)
        sign_b = xp.ones(walkers.nwalkers, dtype=xp.complex128)
        logdet_b = xp.zeros(walkers.nwalkers, dtype=xp.float64)
        if walkers.ndown > 0:
            sign_b, logdet_b = xp.linalg.slogdet(walkers.inv_ovlp_b)
        det = sign_a * sign_b * xp.exp(logdet_a + logdet_b + walkers.log_shift)
        return 1.0 / det

    def propagate_walkers_one_body(self, walkers):
        start_time = time.time()
        if isinstance(walkers, GHFWalkers) and self.expH1.ndim == 2:
            walkers.phi = propagate_one_body(walkers.phi, self.expH1)
        elif isinstance(walkers, GHFWalkers):
            nbasis = walkers.nbasis
            walkers.phi[:, :nbasis, :] = propagate_one_body(
                walkers.phi[:, :nbasis, :], self.expH1[0]
            )
            walkers.phi[:, nbasis:, :] = propagate_one_body(
                walkers.phi[:, nbasis:, :], self.expH1[1]
            )
        else:
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
        weight_factor = xp.where(xp.abs(phase) < 0.5 * math.pi, ratio.real, 0.0)
        walkers.weight *= weight_factor
        walkers.ovlp = xp.where(weight_factor > 0.0, ovlp_new, walkers.ovlp)
        synchronize()
        self.timer.tovlp += time.time() - start_time

    @abstractmethod
    def propagate_walkers_two_body(self, walkers, hamiltonian, trial):
        pass

    def propagate_walkers(self, walkers, hamiltonian, trial, eshift):
        synchronize()
        if walkers.ovlp is None or len(walkers.ovlp) != walkers.nwalkers:
            walkers.ovlp = trial.calc_overlap(walkers)
        ovlp = walkers.ovlp.copy()

        self.kinetic_importance_sampling(walkers, trial)
        self.propagate_walkers_two_body(walkers, hamiltonian, trial)
        self.kinetic_importance_sampling(walkers, trial)
        ovlp_ratio = walkers.ovlp / ovlp
        walkers.hybrid_energy = (-(xp.log(ovlp_ratio)) / self.dt).real

        walkers.weight *= numpy.exp(self.dt * eshift)

    def cast_to_cupy(self, verbose=False):
        cast_to_device(self, verbose=verbose)
