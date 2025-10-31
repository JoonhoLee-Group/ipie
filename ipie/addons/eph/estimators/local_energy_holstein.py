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

from ipie.addons.eph.hamiltonians.holstein import HolsteinModel
from ipie.addons.eph.trial_wavefunction.eph_trial_base import EPhTrialWavefunctionBase
from ipie.addons.eph.walkers.eph_walkers import EPhWalkers

from ipie.systems.generic import Generic
from ipie.utils.backend import arraylib as xp


def local_energy_holstein(
    system: Generic,
    hamiltonian: HolsteinModel,
    walkers: EPhWalkers,
    trial: EPhTrialWavefunctionBase,
) -> np.ndarray:
    r"""Computes the local energy for the Holstein model via

    .. math::
        \frac{\langle \Psi_\mathrm{T} | \hat{H} | \Phi_\mathrm{w}\rangle}
        {\langle \Psi_\mathrm{T} | \Phi_\mathrm{w}\rangle},

    where :math:`| \Phi_\mathrm{w}\rangle = \sum_{\tau \in cyclic perms}
    |\phi(\tau(r))\rangle \otimes |\beta(\tau(X))\rangle`. In this ansatz for
    the walkers :math:`|\beta\rangle` is a coherent state, which corresonds to a
    by :math:`\beta` displaced vacuum state.

    Parameters
    ----------
    system : :class:`Generic`
        Generic object carrying information on number and spin of electrons
    hamiltonian : :class:`HolsteinModel`
        HolsteinModel object
    walkers : :class:`EPhWalkers`
        EPhWalkers object
    trial : :class:`EPhTrialWavefunctionBase`
        EPhTrialWavefunctionBase object
    """

    energy = xp.zeros((walkers.nwalkers, 4), dtype=xp.complex128)

    gf = trial.calc_greens_function(walkers)
    walkers.Ga, walkers.Gb = gf[0], gf[1]

    energy[:, 1] = np.sum(hamiltonian.T[0] * gf[0], axis=(1, 2))
    if system.ndown > 0:
        energy[:, 1] += np.sum(hamiltonian.T[1] * gf[1], axis=(1, 2))

    energy[:, 2] = np.sum(np.diagonal(gf[0], axis1=1, axis2=2) * walkers.phonon_disp, axis=1)
    if system.ndown > 0:
        energy[:, 2] += np.sum(np.diagonal(gf[1], axis1=1, axis2=2) * walkers.phonon_disp, axis=1)
    energy[:, 2] *= hamiltonian.const

    energy[:, 3] = 0.5 * hamiltonian.m * hamiltonian.w0**2 * np.sum(walkers.phonon_disp**2, axis=1)
    energy[:, 3] -= 0.5 * hamiltonian.nsites * hamiltonian.w0
    energy[:, 3] -= 0.5 * trial.calc_phonon_laplacian_locenergy(walkers) / hamiltonian.m

    energy[:, 0] = np.sum(energy[:, 1:], axis=1)

    return energy
