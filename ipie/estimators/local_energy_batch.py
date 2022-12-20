
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
# Authors: Joonho Lee
#          Fionn Malone <fionn.malone@gmail.com>
#

import time

import numpy
from numba import jit

from ipie.estimators.local_energy import local_energy_G
from ipie.estimators.local_energy_sd import (
    local_energy_single_det_batch, local_energy_single_det_batch_einsum,
    local_energy_single_det_batch_gpu, local_energy_single_det_rhf_batch,
    local_energy_single_det_uhf_batch)
from ipie.estimators.local_energy_sd_chunked import (
    local_energy_single_det_uhf_batch_chunked,
    local_energy_single_det_uhf_batch_chunked_gpu)
from ipie.estimators.local_energy_wicks import (
    local_energy_multi_det_trial_wicks_batch,
    local_energy_multi_det_trial_wicks_batch_opt,
    local_energy_multi_det_trial_wicks_batch_opt_chunked)
from ipie.utils.misc import is_cupy


# TODO: should pass hamiltonian here and make it work for all possible types
# this is a generic local_energy handler. So many possible combinations of local energy strategies...
def local_energy_batch(system, hamiltonian, walker_batch, trial):
    """Compute local energy for walker batch (all walkers at once).

    Parameters
    ----------
    system : system object
        System being studied.
    hamiltonian : hamiltonian object
        Hamiltonian being studied.
    walker_batch : WalkerBatch
        Walkers object.
    trial : trial object
        Trial wavefunctioni.

    Returns
    -------
    local_energy : np.ndarray
        Total, one-body and two-body energies.
    """

    if walker_batch.name == "SingleDetWalkerBatch":
        if is_cupy(walker_batch.phia):
            if hamiltonian.chunked:
                return local_energy_single_det_uhf_batch_chunked_gpu(
                    system, hamiltonian, walker_batch, trial
                )
            else:
                return local_energy_single_det_batch_gpu(
                    system, hamiltonian, walker_batch, trial
                )
        elif walker_batch.rhf:
            return local_energy_single_det_rhf_batch(
                system, hamiltonian, walker_batch, trial
            )
        else:
            if hamiltonian.chunked:
                return local_energy_single_det_uhf_batch_chunked(
                    system, hamiltonian, walker_batch, trial
                )
            else:
                return local_energy_single_det_uhf_batch(
                    system, hamiltonian, walker_batch, trial
                )
                # \TODO switch to this
                # return local_energy_single_det_uhf_batch(system, hamiltonian, walker_batch, trial)
    elif walker_batch.name == "MultiDetTrialWalkerBatch" and trial.wicks == False:
        return local_energy_multi_det_trial_batch(
            system, hamiltonian, walker_batch, trial
        )
    elif (
        trial.name == "MultiSlater"
        and trial.ndets > 1
        and trial.wicks == True
        and not trial.optimized
    ):
        # return local_energy_multi_det_trial_batch(system, hamiltonian, walker_batch, trial)
        return local_energy_multi_det_trial_wicks_batch(
            system, hamiltonian, walker_batch, trial
        )
    elif (
        trial.name == "MultiSlater"
        and trial.ndets > 1
        and trial.wicks == True
        and trial.optimized == True
        and trial.ndet_chunks > 1
    ):
        # return local_energy_multi_det_trial_batch(system, hamiltonian, walker_batch, trial)
        return local_energy_multi_det_trial_wicks_batch_opt_chunked(
            system, hamiltonian, walker_batch, trial
        )
    elif (
        trial.name == "MultiSlater"
        and trial.ndets > 1
        and trial.wicks == True
        and trial.optimized == True
    ):
        # return local_energy_multi_det_trial_batch(system, hamiltonian, walker_batch, trial)
        return local_energy_multi_det_trial_wicks_batch_opt(
            system, hamiltonian, walker_batch, trial
        )


def local_energy_multi_det_trial_batch(system, hamiltonian, walker_batch, trial):
    """Compute local energy for walker batch (all walkers at once) with MSD.

    Naive O(Ndet) algorithm, no optimizations.

    Parameters
    ----------
    system : system object
        System being studied.
    hamiltonian : hamiltonian object
        Hamiltonian being studied.
    walker_batch : WalkerBatch
        Walkers object.
    trial : trial object
        Trial wavefunctioni.

    Returns
    -------
    local_energy : np.ndarray
        Total, one-body and two-body energies.
    """
    energy = []
    ndets = trial.ndets
    nwalkers = walker_batch.nwalkers
    # ndets x nwalkers
    for iwalker, (w, Ga, Gb, Ghalfa, Ghalfb) in enumerate(
        zip(
            walker_batch.det_weights,
            walker_batch.Gia,
            walker_batch.Gib,
            walker_batch.Gihalfa,
            walker_batch.Gihalfb,
        )
    ):
        denom = 0.0 + 0.0j
        numer0 = 0.0 + 0.0j
        numer1 = 0.0 + 0.0j
        numer2 = 0.0 + 0.0j
        for idet in range(ndets):
            # construct "local" green's functions for each component of A
            G = [Ga[idet], Gb[idet]]
            Ghalf = [Ghalfa[idet], Ghalfb[idet]]
            # return (e1b+e2b+ham.ecore, e1b+ham.ecore, e2b)
            e = list(local_energy_G(system, hamiltonian, trial, G, Ghalf=None))
            numer0 += w[idet] * e[0]
            numer1 += w[idet] * e[1]
            numer2 += w[idet] * e[2]
            denom += w[idet]
        # return (e1b+e2b+ham.ecore, e1b+ham.ecore, e2b)
        energy += [list([numer0 / denom, numer1 / denom, numer2 / denom])]

    energy = numpy.array(energy, dtype=numpy.complex128)
    return energy
