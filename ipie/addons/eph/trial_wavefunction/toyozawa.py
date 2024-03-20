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

import numpy as np
from typing import Tuple

from ipie.addons.eph.walkers.eph_walkers import EPhWalkers
from ipie.addons.eph.trial_wavefunction.coherent_state import CoherentStateTrial
from ipie.addons.eph.trial_wavefunction.variational.toyozawa_variational import circ_perm
from ipie.utils.backend import arraylib as xp
from ipie.estimators.greens_function_single_det import gab_mod_ovlp


class ToyozawaTrial(CoherentStateTrial):
    r"""The Toyozawa trial

    .. math::
        |\Psi(\kappa)\rangle = \sum_n e^{i \kappa n} \sum_{n_1} \alpha_{n_1}^{\kappa}
        a_{n_1}^{\dagger} \exp(-\sum_{n_2} (\beta^\kappa_{n_2 - n} b_{n_2}^{\dagger}
        - \beta^{\kappa^*}_{n_2 - n} b_{n_2}))|0\rangle

    developed by `Toyozawa <https://doi.org/10.1143/PTP.26.29>`_ is translationally
    invariant and reliable offers a good approximation to the polaron ground state
    for most parameter regimes of the Holstein Model. Here, :math:`\alpha,\beta`are
    varaitional parameters, and :math:`|0\rangle` is the total vacuum state.
    For a 1D Holstein chain this reduces to a superposition of cyclically `CoherentState`
    type trials.
    More details may be found in `Zhao et al. <https://doi.org/10.1063/1.474667>`_.

    Attributes
    ----------
    perms : :class:`np.ndarray`
        Rows of this matrix corresponds to cyclic permutations of `range(nsites)`
    nperms : :class:`int`
        Number of permutations in `perms`
    """

    def __init__(
        self,
        wavefunction: np.ndarray,
        w0: float,
        num_elec: Tuple[int, int],
        num_basis: int,
        verbose=False,
    ):
        super().__init__(wavefunction, w0, num_elec, num_basis, verbose=verbose)
        self.perms = circ_perm(np.arange(self.nbasis))
        self.nperms = self.perms.shape[0]

    def calc_energy(self, ham, zero_th=1e-12):
        r"""Computes the variational energy of the trial, i.e.

        .. math::
            E_T = \frac{\langle\Psi_T|\hat{H}|\Psi_T\rangle}{\langle\Psi_T|\Psi_T\rangle}.

        As the Toyozawa trial wavefunction is a superposition of coherent state trials
        the evaluation of :math:`E_T` a naive implementation would scale quadratically
        with the number of sites. Here, we exploit the translational symmetry of the
        wavefunction to obtain linear scaling.

        Parameters
        ----------
        ham:
            Hamiltonian

        Returns
        -------
        etrial : :class:`float`
            Trial energy
        """
        num_energy = 0.0
        denom = 0.0
        beta0 = self.beta_shift * np.sqrt(0.5 * ham.m * ham.w0)
        for perm in self.perms:
            psia_i = self.psia[perm, :]
            beta_i = beta0[perm]

            if self.ndown > 0:
                psib_i = self.psib[perm, :]
                ov = (
                    np.linalg.det(self.psia.T.dot(psia_i))
                    * np.linalg.det(self.psib.T.dot(psib_i))
                    * np.prod(np.exp(-0.5 * (beta0**2 + beta_i**2) + beta0 * beta_i))
                )
            else:
                ov = np.linalg.det(self.psia.T.dot(psia_i)) * np.prod(
                    np.exp(-0.5 * (beta0**2 + beta_i**2) + beta0 * beta_i)
                )

            if ov < zero_th:
                continue

            Ga_i, _, _ = gab_mod_ovlp(self.psia, psia_i)
            if self.ndown > 0:
                Gb_i, _, _ = gab_mod_ovlp(self.psib, psib_i)
            else:
                Gb_i = np.zeros_like(Ga_i)
            G_i = [Ga_i, Gb_i]

            kinetic = np.sum(ham.T[0] * G_i[0] + ham.T[1] * G_i[1])
            e_ph = ham.w0 * np.sum(beta0 * beta_i)
            rho = ham.g_tensor * (G_i[0] + G_i[1])
            e_eph = np.sum(np.dot(rho, beta0 + beta_i))
            num_energy += (kinetic + e_ph + e_eph) * ov
            denom += ov

        etrial = num_energy / denom
        return etrial

    def calc_overlap_perm(self, walkers: EPhWalkers) -> np.ndarray:
        r"""Computes the product of electron and phonon overlaps for each
        permutation :math:`\sigma`,

        .. math::
            \langle \psi_T(\sigma(r))|\psi(\tau)\rangle
            \langle \phi(\sigma(\beta))|X_{\mathrm{w}(\tau)}\rangle.

        Parameters
        ----------
        walkers : :class:`EPhWalkers`
            EPhWalkers object

        Returns
        -------
        ovlp_perm : :class:`np.ndarray`
            Product of electron and phonon overlap for each permutation
        """
        ph_ovlp_perm = self.calc_phonon_overlap_perms(walkers)
        el_ovlp_perm = self.calc_electronic_overlap_perms(walkers)
        walkers.ovlp_perm = el_ovlp_perm * ph_ovlp_perm
        return walkers.ovlp_perm

    def calc_overlap(self, walkers: EPhWalkers) -> np.ndarray:
        r"""Sums product of electronic and phonon overlap for each permutation
        over all permutations,

        .. math::
            \sum_\tau \langle \psi_T(\sigma(r))|\psi(\tau)\rangle
            \langle \phi(\sigma(\beta))|X_{\mathrm{w}(\tau)}\rangle.

        Used when evaluating local energy and when updating
        weight.

        Parameters
        ----------
        walkers : :class:`EPhWalkers`
            EPhWalkers object

        Returns
        -------
        ovlp: :class:`np.ndarray`
            Sum of product of electron and phonon overlap
        """
        ovlp_perm = self.calc_overlap_perm(walkers)
        ovlp = np.sum(ovlp_perm, axis=1)
        return ovlp

    def calc_phonon_overlap_perms(self, walkers: EPhWalkers) -> np.ndarray:
        r"""Updates the walker phonon overlap with each permutation :math:`\tau`,
        i.e. :math:`\langle\phi(\tau(\beta))|X_{\mathrm{w}}\rangle` and stores
        it in `walkers.ph_ovlp`.

        Parameters
        ----------
        walkers : :class:`EPhWalkers`
            EPhWalkers object

        Returns
        -------
        ph_ovlp_perm : :class:`np.ndarray`
            Overlap of walker with permuted coherent states
        """
        for ip, perm in enumerate(self.perms):
            ph_ov = np.exp(
                -(0.5 * self.m * self.w0) * (walkers.phonon_disp - self.beta_shift[perm]) ** 2
            )
            walkers.ph_ovlp[:, ip] = np.prod(ph_ov, axis=1)
        return walkers.ph_ovlp

    def calc_phonon_overlap(self, walkers: EPhWalkers) -> np.ndarray:
        r"""Sums walker phonon overlaps with permuted coherent states over all
        permutations,

        .. math::
            \sum_\tau \langle \phi(\tau(\beta)) | X_{\mathrm{w}} \rangle

        to get total phonon overlap. This is only used to correct
        for the importance sampling in propagate_phonons.

        Parameters
        ----------
        walkers : :class:`EPhWalkers`
            EPhWalkers object

        Returns
        -------
        ph_ovlp : :class:`np.ndarray`
            Total walker phonon overlap
        """
        ph_ovlp_perm = self.calc_phonon_overlap_perms(walkers)
        ph_ovlp = np.sum(ph_ovlp_perm, axis=1)
        return ph_ovlp

    def calc_phonon_gradient(self, walkers: EPhWalkers) -> np.ndarray:
        r"""Computes the phonon gradient,

        .. math::
            \sum_\sigma \frac{\nabla_X \langle \phi(\sigma(\beta)) | X(\tau) \rangle}
            {\rangle \phi(\sigma(\beta)) | X(\tau) \rangle}
            = \sum_\sigma -m \omega \frac{(X(\tau) - \sigma(\beta)) * \langle\phi(\sigma(\beta))|X(\tau)\rangle}
            {\sum_\simga \langle\phi(\sigma(\beta))|X(\tau)\rangle}.

        This is only used when calculating the drift term for the importance
        sampling DMC part of the algorithm.

        Parameters
        ----------
        walkers : :class:`EPhWalkers`
            EPhWalkers object

        Returns
        -------
        grad : :class:`np.ndarray`
            Phonon gradient
        """
        grad = np.zeros_like(walkers.phonon_disp, dtype=np.complex128)
        for ovlp, perm in zip(walkers.ph_ovlp.T, self.perms):
            grad += np.einsum("ni,n->ni", (walkers.phonon_disp - self.beta_shift[perm]), ovlp)
        grad *= -self.m * self.w0
        grad = np.einsum("ni,n->ni", grad, 1 / np.sum(walkers.ph_ovlp, axis=1))
        return grad

    def calc_phonon_laplacian(self, walkers: EPhWalkers, ovlps: np.ndarray) -> np.ndarray:
        r"""Computes the phonon Laplacian, which weights coherent state laplacians
        by overlaps :math:`o(\sigma, r, X, \tau)` passed to this function,

        .. math::
            \sum_\sigma \frac{\nabla_X \langle \phi(\sigma(\beta)) | X(\tau) \rangle}
            {\rangle \phi(\sigma(\beta)) | X(\tau) \rangle}
            = \frac{\sum_sigma ((\sum_i (m \omega (X_i(\tau) - \sigma(\beta)_i))^2) - N m \omega) o(\sigma, r, X, \tau)}
            {\sum_\sigma o(\sigma, r, X, \tau)}.

        Parameters
        ----------
        walkers : :class:`EPhWalkers`
            EPhWalkers object
        ovlps : :class:`np.ndarray`
            Overlaps weighting contributions from permuted coherent states

        Returns
        -------
        laplacian : :class:`np.ndarray`
            Phonon Laplacian
        """
        laplacian = np.zeros(walkers.nwalkers, dtype=np.complex128)
        for ovlp, perm in zip(ovlps.T, self.perms):
            arg = (walkers.phonon_disp - self.beta_shift[perm]) * self.m * self.w0
            arg2 = arg**2
            laplacian += (np.sum(arg2, axis=1) - self.nsites * self.m * self.w0) * ovlp
        laplacian /= np.sum(ovlps, axis=1)
        return laplacian

    def calc_phonon_laplacian_importance(self, walkers: EPhWalkers) -> np.ndarray:
        r"""Computes phonon Laplacian via `calc_phonon_laplacian` with weighting
        by pure phonon overlap. This is only utilized in the importance sampling
        of the DMC procedure.

        Parameters
        ----------
        walkers : :class:`EPhWalkers`
            EPhWalkers object

        Returns
        -------
        ph_lapl : :class:`np.ndarray`
            Phonon Laplacian weigthed by phonon overlaps
        """
        return self.calc_phonon_laplacian(walkers, walkers.ph_ovlp)

    def calc_phonon_laplacian_locenergy(self, walkers: EPhWalkers) -> np.ndarray:
        """Computes phonon Laplacian using total overlap weights as required in
        local energy evaluation.

        Parameters
        ----------
        walkers : :class:`EPhWalkers`
            EPhWalkers object

        Returns
        -------
        ph_lapl : :class:`np.ndarray`
            Phonon Laplacian weigthed by total overlaps
        """
        return self.calc_phonon_laplacian(walkers, walkers.ovlp_perm)

    def calc_electronic_overlap_perms(self, walkers: EPhWalkers) -> np.ndarray:
        r"""Calculates the electronic overlap of each walker with each permuted
        Slater determinant :math:`|\Phi_T(\tau(r_i))\rangle` of the trial,

        .. math::
            \langle \Phi_T(\tau(r_i))|\psi_w\rangle = \mathrm{det(U^{\dagger}V)},

        where :math:`U,V` parametrized the two Slater determinants.

        Parameters
        ----------
        walkers : :class:`EPhWalkers`
            EPhWalkers object

        Returns
        -------
        el_ovlp_perm : :class:`np.ndarray`
            Electronic overlap of each permuted Slater Determiant with walkers
        """
        for ip, perm in enumerate(self.perms):
            ovlp_a = xp.einsum(
                "wmi,mj->wij", walkers.phia, self.psia[perm, :].conj(), optimize=True
            )
            sign_a, log_ovlp_a = xp.linalg.slogdet(ovlp_a)

            if self.ndown > 0:
                ovlp_b = xp.einsum(
                    "wmi,mj->wij", walkers.phib, self.psib[perm, :].conj(), optimize=True
                )
                sign_b, log_ovlp_b = xp.linalg.slogdet(ovlp_b)
                ot = sign_a * sign_b * xp.exp(log_ovlp_a + log_ovlp_b - walkers.log_shift)
            else:
                ot = sign_a * xp.exp(log_ovlp_a - walkers.log_shift)

            walkers.el_ovlp[:, ip] = ot
        return walkers.el_ovlp

    def calc_electronic_overlap(self, walkers: EPhWalkers) -> np.ndarray:
        """Sums walkers.el_ovlp over permutations to obtain total electronic
        overlap of trial with walkers.

        Parameters
        ----------
        walkers : :class:`EPhWalkers`
            EPhWalkers object

        Returns
        -------
        el_ovlp : :class:`np.ndarray`
            Electronic overlap of trial with walkers
        """
        el_ovlp_perms = self.calc_electronic_overlap_perms(walkers)
        el_ovlp = np.sum(el_ovlp_perms, axis=1)
        return el_ovlp

    def calc_greens_function(self, walkers: EPhWalkers, build_full=True) -> np.ndarray:
        r"""Calculates Greens functions by

        .. math::
            G^{\Phi \Psi}_{p\alpha, q\beta}
            = \frac{\sum_{\tau} \delta_{\alpha\beta}(U_\alpha(V^{\dagger}_\alhpa(\tau) U_\alpha) V^{\dagger}_\alpha(\tau)) \langle\Phi_T(\tau(r_i))|\psi_w\rangle}
            {\sum_{\tau} \langle\Phi_T(\tau(r_i))|\psi_w\rangle}

        Parameters
        ----------
        walkers : :class:`EPhWalkers`
            EPhWalkers object

        Returns
        -------
        G : :class:`list`
            List of Greens functions for :math:`\alpha,\beta` spin spaces.
        """
        Ga = np.zeros((walkers.nwalkers, self.nsites, self.nsites), dtype=np.complex128)
        Gb = np.zeros_like(Ga)

        for ovlp, perm in zip(walkers.ovlp_perm.T, self.perms):
            inv_Oa = xp.linalg.inv(
                xp.einsum("ie,nif->nef", self.psia[perm, :], walkers.phia.conj())
            )
            Ga += xp.einsum("nie,nef,jf,n->nji", walkers.phia, inv_Oa, self.psia[perm].conj(), ovlp)

            if self.ndown > 0:
                inv_Ob = xp.linalg.inv(
                    xp.einsum("ie,nif->nef", self.psib[perm, :], walkers.phib.conj())
                )
                Gb += xp.einsum(
                    "nie,nef,jf,n->nji", walkers.phib, inv_Ob, self.psib[perm].conj(), ovlp
                )

        Ga = np.einsum("nij,n->nij", Ga, 1 / walkers.ovlp)
        if self.ndown > 0:
            Gb = np.einsum("nij,n->nij", Gb, 1 / walkers.ovlp)

        return [Ga, Gb]
