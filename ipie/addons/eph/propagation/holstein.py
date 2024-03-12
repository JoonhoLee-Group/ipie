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

import numpy
import time
import scipy.linalg

from ipie.addons.eph.hamiltonians.holstein import HolsteinModel
from ipie.addons.eph.trial_wavefunction.eph_trial_base import EPhTrialWavefunctionBase
from ipie.addons.eph.walkers.eph_walkers import EPhWalkers

from ipie.utils.backend import synchronize
from ipie.propagation.operations import propagate_one_body
from ipie.propagation.continuous_base import PropagatorTimer


def construct_one_body_propagator(
    hamiltonian: HolsteinModel, dt: float
) -> Sequence[np.ndarray, np.ndarray]:
    """Exponentiates the electronic hopping term to apply it later as
    part of the trotterized algorithm.

    Parameters
    ----------
    hamiltonian :
        Hamiltonian caryying the one-body term as hamiltonian.T
    dt :
        Time step

    Returns
    -------
    expH1 :
        
    """
    H1 = hamiltonian.T
    expH1 = numpy.array(
        [scipy.linalg.expm(-0.5 * dt * H1[0]), scipy.linalg.expm(-0.5 * dt * H1[1])]
    )
    return expH1


class HolsteinPropagatorFree:
    r"""Propagates walkers by trotterization,
    .. math::
        \mathrm{e}^{-\Delta \tau \hat{H}} \approx \mathrm{e}^{-\Delta \tau \hat{H}_{\mathrm{ph}} / 2}
        \mathrm{e}^{-\Delta \tau \hat{H}_{\mathrm{el}} / 2} \mathrm{e}^{-\Delta \tau \hat{H}_{\mathrm{el-ph}}}
        \mathrm{e}^{-\Delta \tau \hat{H}_{\mathrm{el}} / 2} \mathrm{e}^{-\Delta \tau \hat{H}_{\mathrm{ph}} / 2},

    where propagation under :math:`\hat{H}_{\mathrm{ph}}` employs a generic
    Diffucion MC procedure (notably without importance sampling). Propagation by
    :math:`\hat{H}_{\mathrm{el}}` consists of a simple mat-vec. As
    :math:`\hat{H}_{\mathrm{el-ph}}` is diagonal in bosonic position space we
    can straightforwardly exponentiate the displacements and perform another
    mat-vec with this diagonal matrix apllied to electronic degrees of freedom.

    Parameters
    ----------
    time_step :
        Time step
    verbose :
        Print level
    """

    def __init__(self, time_step: float, verbose: bool = False):
        self.dt = time_step
        self.verbose = verbose
        self.timer = PropagatorTimer()

        self.sqrt_dt = self.dt**0.5
        self.dt_ph = 0.5 * self.dt
        self.mpi_handler = None

    def build(
        self,
        hamiltonian: HolsteinModel,
        trial: EPhTrialWavefunctionBase = None,
        walkers: EPhWalkers = None,
        mpi_handler=None,
    ) -> None:
        """Necessary step before running the AFQMC procedure.
        Sets required attributes.

        Parameters
        ----------
        hamiltonian :
            Holstein model
        trial :
            Trial object
        walkers :
            Walkers object
        mpi_handler :
            MPIHandler specifying rank and size
        """
        self.expH1 = construct_one_body_propagator(hamiltonian, self.dt)
        self.const = hamiltonian.g * numpy.sqrt(2.0 * hamiltonian.m * hamiltonian.w0) * self.dt
        self.w0 = hamiltonian.w0
        self.m = hamiltonian.m
        self.scale = numpy.sqrt(self.dt_ph / self.m)
        self.nsites = hamiltonian.nsites

    def propagate_phonons(
        self, walkers: EPhWalkers, hamiltonian: HolsteinModel, trial: EPhTrialWavefunctionBase
    ) -> None:
        r"""Propagates phonon displacements by adjusting weigths according to
        bosonic on-site energies and sampling the momentum contribution, again
        by trotterizing the phonon propagator.

        .. math::
            \mathrm{e}^{-\Delta \tau \hat{H}_{\mathrm{ph}} / 2} \approx
            \mathrm{e}^{\Delta \tau N \omega / 4}
            \mathrm{e}^{-\Delta \tau \sum_i m \omega \hat{X}_i^2 / 8}
            \mathrm{e}^{-\Delta \tau \sum_i \hat{P}_i^2 / (4 \omega)}
            \mathrm{e}^{-\Delta \tau \sum_i m \omega \hat{X}_i^2 / 8}

        One can obtain the sampling prescription by insertion of resolutions of
        identity, :math:`\int dX |X\rangle \langleX|, and performin the resulting
        Fourier transformation.

        Parameters
        ----------
        walkers :
            Walkers class
        """
        start_time = time.time()

        pot = 0.25 * self.m * self.w0**2 * numpy.sum(walkers.phonon_disp**2, axis=1)
        pot = numpy.real(pot)
        walkers.weight *= numpy.exp(-self.dt_ph * pot)

        N = numpy.random.normal(loc=0.0, scale=self.scale, size=(walkers.nwalkers, self.nsites))
        walkers.phonon_disp = walkers.phonon_disp + N

        pot = 0.25 * self.m * self.w0**2 * numpy.sum(walkers.phonon_disp**2, axis=1)
        pot = numpy.real(pot)
        walkers.weight *= numpy.exp(-self.dt_ph * pot)

        # Does not matter for estimators but helps with population control
        walkers.weight *= numpy.exp(self.dt_ph * self.nsites * self.w0 / 2)

        synchronize()
        self.timer.tgemm += time.time() - start_time

    def propagate_electron(
        self, walkers: EPhWalkers, hamiltonian: HolsteinModel, trial: EPhTrialWavefunctionBase
    ) -> None:
        r"""Propagates electronic degrees of freedom via

        .. math::
            \mathrm{e}^{-\Delta \tau (\hat{H}_{\mathrm{el}} \otimes \hat{I}_{\mathrm{ph}} + \hat{H}_{\mathrm{el-ph}})}
            \approx \mathrm{e}^{-\Delta \tau \hat{H}_{\mathrm{el}} / 2}
            \mathrm{e}^{-\Delta \tau \hat{H}_{\mathrm{el-ph}}}
            \mathrm{e}^{-\Delta \tau \hat{H}_{\mathrm{el}} / 2}.

        This acts on walkers of the form :math:`|\phi(\tau)\rangle \otimes |X(\tau)\rangle`.


        Parameters
        ----------
        walkers :
            Walkers class
        trial :
            Trial class
        """
        start_time = time.time()
        synchronize()
        self.timer.tgf += time.time() - start_time

        expEph = numpy.exp(self.const * walkers.phonon_disp)

        walkers.phia = propagate_one_body(walkers.phia, self.expH1[0])
        walkers.phia = numpy.einsum("ni,nie->nie", expEph, walkers.phia)
        walkers.phia = propagate_one_body(walkers.phia, self.expH1[0])

        if walkers.ndown > 0:
            walkers.phib = propagate_one_body(walkers.phib, self.expH1[1])
            walkers.phib = numpy.einsum("ni,nie->nie", expEph, walkers.phib)
            walkers.phib = propagate_one_body(walkers.phib, self.expH1[1])

    def propagate_walkers(
        self,
        walkers: EPhWalkers,
        hamiltonian: HolsteinModel,
        trial: EPhTrialWavefunctionBase,
        eshift: float = None,
    ) -> None:
        r"""Propagates walkers by trotterized propagator.

        Parameters
        ----------
        walkers :
            EPhWalkers object
        hamiltonian :
            HolsteinModel object
        trial :
            EPhTrialWavefunctionBase object
        eshift :
            Only purpose is compatibility with AFQMC object, irrelevant for
            propagation
        """
        synchronize()
        start_time = time.time()
        ovlp = trial.calc_overlap(walkers)
        walkers.ovlp = ovlp
        synchronize()
        self.timer.tgf += time.time() - start_time

        # Update Walkers
        # a) DMC for phonon degrees of freedom
        self.propagate_phonons(walkers, hamiltonian, trial)

        # b) One-body propagation for electrons
        self.propagate_electron(walkers, hamiltonian, trial)

        # c) DMC for phonon degrees of freedom
        self.propagate_phonons(walkers, hamiltonian, trial)

        # Update weights (and later do phaseless for multi-electron)
        start_time = time.time()
        ovlp_new = trial.calc_overlap(walkers)
        walkers.ovlp = ovlp_new
        synchronize()
        self.timer.tovlp += time.time() - start_time

        start_time = time.time()
        self.update_weight(walkers, ovlp, ovlp_new)
        synchronize()
        self.timer.tupdate += time.time() - start_time

    def update_weight(self, walkers, ovlp, ovlp_new) -> None:
        walkers.weight *= ovlp_new / ovlp


class HolsteinPropagator(HolsteinPropagatorFree):
    r"""Propagates walkers by trotterization, employing importance sampling for
    the bosonic degrees of freedom. This results in a different weigth update,
    and the additional displacement update by the drift term,

    .. math::
        D = \frac{\nabla_X \langle \Psi_\mathrm{T} | \psi(\tau), X(\tau)\rangle}
        {\langle \Psi_\mathrm{T} | \psi(\tau), X(\tau)\rangle},

    such that the revised displacement update reads

    .. math::
        X(\tau+\Delta\tau) = X(\tau)
        + \mathcal{N}(\mu=0, \sigma = \sqrt{\frac{\Delta\tau}{m}})
        + \frac{\Delta\tau}{m} D.

    Parameters
    ----------
    time_step :
        Time step
    verbose :
        Print level
    """

    def __init__(self, time_step, verbose=False):
        super().__init__(time_step, verbose=verbose)

    def propagate_phonons(
        self, walkers: EPhWalkers, hamiltonian: HolsteinModel, trial: EPhTrialWavefunctionBase
    ) -> None:
        """Propagates phonons via Diffusion MC including drift term."""
        start_time = time.time()

        # No ZPE in pot -> cancels with ZPE of etrial,
        # wouldn't affect estimators anyways
        ph_ovlp_old = trial.calc_phonon_overlap(walkers)

        pot = 0.5 * hamiltonian.m * hamiltonian.w0**2 * numpy.sum(walkers.phonon_disp**2, axis=1)
        pot -= 0.5 * trial.calc_phonon_laplacian_importance(walkers) / hamiltonian.m
        pot = numpy.real(pot)
        walkers.weight *= numpy.exp(-self.dt_ph * pot / 2)

        N = numpy.random.normal(loc=0.0, scale=self.scale, size=(walkers.nwalkers, self.nsites))
        drift = trial.calc_phonon_gradient(walkers)
        walkers.phonon_disp = walkers.phonon_disp + N + self.dt_ph * drift / hamiltonian.m

        ph_ovlp_new = trial.calc_phonon_overlap(walkers)

        pot = 0.5 * hamiltonian.m * hamiltonian.w0**2 * numpy.sum(walkers.phonon_disp**2, axis=1)
        pot -= 0.5 * trial.calc_phonon_laplacian_importance(walkers) / hamiltonian.m
        pot = numpy.real(pot)
        walkers.weight *= numpy.exp(-self.dt_ph * pot / 2)

        walkers.weight *= ph_ovlp_old / ph_ovlp_new
        walkers.weight *= numpy.exp(self.dt_ph * trial.energy)

        synchronize()
        self.timer.tgemm += time.time() - start_time
