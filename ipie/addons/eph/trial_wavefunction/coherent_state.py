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

from ipie.addons.eph.trial_wavefunction.eph_trial_base import EPhTrialWavefunctionBase
from ipie.utils.backend import arraylib as xp
from ipie.estimators.greens_function_single_det import gab_mod_ovlp


class CoherentStateTrial(EPhTrialWavefunctionBase):
    r"""Coherent state trial of the form

    .. math::
        |\Phi\rangle \otimes |\beta\rangle,

    where :math:`|\Phi\rangle` corresponds to the electronic wave function and
    :math:`|\beta\rangle` to the bosonic wave function. This latter is a
    coherent state, i.e. a vacuum state displaced by :math:`\beta`.

    Parameters
    ----------
    wavefunction :
        Concatenation of trial determinants of up and down spin spaces and beta
        specifying the coherent state displacement.
    w0 :
        Phonon frequency
    num_elec :
        Tuple specifying number of up and down spins
    num_basis :
        Number of sites of Holstein chain.
    verbose :
        Print level
    """

    def __init__(self, wavefunction, w0, num_elec, num_basis, verbose=False):
        super().__init__(wavefunction, num_elec, num_basis, verbose=verbose)
        self.num_elec = num_elec
        self.nup, self.ndown = self.num_elec
        self.w0 = w0
        self.m = 1 / w0
        self.nsites = self.nbasis

        self.beta_shift = np.squeeze(wavefunction[:, 0])
        self.psia = wavefunction[:, 1 : self.nup + 1]
        self.psib = wavefunction[:, self.nup + 1 : self.nup + self.ndown + 1]

    def calc_energy(self, ham) -> float:
        r"""Computes the variational energy of the trial,

        .. math::
            E = \frac{\langle \Psi_T |\hat{H}|\Psi_T\rangle}{\langle \Psi_T |\Psi_T\rangle}.

        Parameters
        ----------
        ham : :class:`HolsteinModel`
            Holstein model

        Returns
        -------
        etrial : :class:`float`
            Variational trial energy
        """
        Ga, _, _ = gab_mod_ovlp(self.psia, self.psia)
        if self.ndown > 0:
            Gb, _, _ = gab_mod_ovlp(self.psib, self.psib)
        else:
            Gb = np.zeros_like(Ga)
        G = [Ga, Gb]

        kinetic = np.sum(ham.T[0] * G[0] + ham.T[1] * G[1])

        e_ph = ham.w0 * np.sum(self.beta_shift**2)
        rho = ham.g_tensor * (G[0] + G[1])
        e_eph = np.sum(np.dot(rho, 2 * self.beta_shift))

        etrial = kinetic + e_ph + e_eph
        return etrial

    def calc_overlap(self, walkers) -> np.ndarray:
        r"""Computes the product of electron and phonon overlaps,

        .. math::
            \langle \Psi_T(r)|\Phi(\tau)\rangle \langle \phi(\beta)|X_{\mathrm{w}(\tau)}\rangle.

        Parameters
        ----------
        walkers : :class:`EPhWalkers`
            EPhWalkers object

        Returns
        -------
        ovlp : :class:`np.ndarray`
            Product of electron and phonon overlap
        """
        ph_ovlp = self.calc_phonon_overlap(walkers)
        el_ovlp = self.calc_electronic_overlap(walkers)
        ovlp = el_ovlp * ph_ovlp
        return ovlp

    def calc_phonon_overlap(self, walkers) -> np.ndarray:
        r"""Computes phonon overlap, :math:`\langle \phi(\beta)|X_{\mathrm{w}(\tau)}\rangle`.

        Parameters
        ----------
        walkers : :class:`EPhWalkers`
            EPhWalkers object

        Returns
        -------
        ph_ovlp : :class:`np.ndarray`
            Overlap of walekr position states with coherent trial state.
        """
        ph_ovlp = np.exp(-(self.m * self.w0 / 2) * (walkers.phonon_disp - self.beta_shift) ** 2)
        walkers.ph_ovlp = np.prod(ph_ovlp, axis=1)
        return walkers.ph_ovlp

    def calc_phonon_gradient(self, walkers) -> np.ndarray:
        r"""Computes the gradient of phonon overlaps,

        .. math::
            \frac{\nabla_X \langle\phi(\beta)|X_\mathrm{w}(\tau)\rangle}
            {\langle\phi(\beta)|X_\mathrm{w}(\tau)\rangle} = -m \omega_0 (X(\tau) - \beta).

        Parameters
        ----------
        walkers : :class:`EPhWalkers`
            EPhWalkers object

        Returns
        -------
        grad : :class:`np.ndarray`
            Gradient of phonon overlap
        """
        grad = walkers.phonon_disp - self.beta_shift
        grad *= -self.m * self.w0
        return grad

    def calc_phonon_laplacian(self, walkers) -> np.ndarray:
        r"""Computes the Laplacian of phonon overlaps,

        .. math::
            \frac{\nabla^2_X \langle\phi(\beta)|X_\mathrm{w}(\tau)\rangle}
            {\langle\phi(\beta)|X_\mathrm{w}(\tau)\rangle}
            = - N m \omega_0 + \sum_i ((X_i(\tau) - \beta_i) m \omega_0)^2

        Parameters
        ----------
        walkers : :class:`EPhWalkers`
            EPhWalkers object

        Returns
        -------
        laplacian: :class:`np.ndarray`
            Laplacian of phonon overlap
        """
        arg = (walkers.phonon_disp - self.beta_shift) * self.m * self.w0
        arg2 = arg**2
        laplacian = np.sum(arg2, axis=1) - self.nsites * self.m * self.w0
        return laplacian

    def calc_phonon_laplacian_importance(self, walkers) -> np.ndarray:
        return self.calc_phonon_laplacian(walkers)

    def calc_phonon_laplacian_locenergy(self, walkers) -> np.ndarray:
        return self.calc_phonon_laplacian(walkers)

    def calc_electronic_overlap(self, walkers) -> np.ndarray:
        r"""Computes electronic overlap,

        .. math::
            \langle \Psi_T(r)|\Phi(\tau)\rangle = \prod_\sigma \mathrm{det}(V^{\dagger}_{\sigma}U_\sigam)

        Parameters
        ----------
        walkers : class
            EphWalkers class object

        Returns
        -------
        walker.el_ovlp : np.ndarray
            Electronic overlap
        """
        ovlp_a = xp.einsum("wmi,mj->wij", walkers.phia, self.psia.conj(), optimize=True)
        sign_a, log_ovlp_a = xp.linalg.slogdet(ovlp_a)

        if self.ndown > 0:
            ovlp_b = xp.einsum("wmi,mj->wij", walkers.phib, self.psib.conj(), optimize=True)
            sign_b, log_ovlp_b = xp.linalg.slogdet(ovlp_b)
            ot = sign_a * sign_b * xp.exp(log_ovlp_a + log_ovlp_b - walkers.log_shift)
        else:
            ot = sign_a * xp.exp(log_ovlp_a - walkers.log_shift)

        walkers.el_ovlp = ot

        return walkers.el_ovlp

    def calc_greens_function(self, walkers) -> np.ndarray:
        """Computes Greens function.

        Parameters
        ----------
        walkers : class
            EphWalkers class object

        Returns
        -------
        walkers.G : list
            Greens function for each spin space
        """
        inv_Oa = xp.linalg.inv(
            xp.einsum("ie,nif->nef", self.psia, walkers.phia.conj(), optimize=True)
        )
        Ga = xp.einsum("nie,nef,jf->nji", walkers.phia, inv_Oa, self.psia.conj(), optimize=True)

        if self.ndown > 0:
            inv_Ob = xp.linalg.inv(
                xp.einsum("ie,nif->nef", self.psib, walkers.phib.conj(), optimize=True)
            )
            Gb = xp.einsum("nie,nef,jf->nji", walkers.phib, inv_Ob, self.psib.conj(), optimize=True)

        return [Ga, Gb]
