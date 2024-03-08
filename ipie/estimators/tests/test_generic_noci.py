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
# Authors: Fionn Malone <fionn.malone@gmail.com>
#          Joonho Lee
#

import numpy as np
import pytest

from ipie.estimators.energy import local_energy
from ipie.estimators.local_energy_wicks import local_energy_multi_det_trial_wicks_batch_opt_chunked
from ipie.hamiltonians.generic import Generic as HamGeneric
from ipie.systems.generic import Generic
from ipie.trial_wavefunction.noci import NOCI
from ipie.trial_wavefunction.particle_hole import ParticleHole
from ipie.trial_wavefunction.single_det import SingleDet
from ipie.utils.testing import generate_hamiltonian, get_random_nomsd, get_random_phmsd_opt
from ipie.walkers.walkers_dispatch import UHFWalkersTrial


@pytest.mark.unit
def test_greens_function_noci():
    np.random.seed(7)
    nmo = 12
    nelec = (7, 7)
    nwalkers = 10
    nsteps = 100
    ndets = 11
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    system = Generic(nelec=nelec)
    hamiltonian = HamGeneric(h1e=np.array([h1e, h1e]), chol=chol.reshape((-1, nmo * nmo)).T.copy())
    # Test PH type wavefunction.
    coeffs, wfn, init = get_random_nomsd(
        system.nup, system.ndown, hamiltonian.nbasis, ndet=ndets, init=True
    )
    trial = NOCI((coeffs, wfn), nelec, nmo)
    trial.build()
    trial.half_rotate(hamiltonian)
    g = trial.build_one_rdm()
    assert np.isclose(g[0].trace(), nelec[0])
    assert np.isclose(g[1].trace(), nelec[0])
    walkers = UHFWalkersTrial(trial, init, system.nup, system.ndown, hamiltonian.nbasis, nwalkers)
    walkers.build(trial)
    assert walkers.Gia.shape == (ndets, nwalkers, nmo, nmo)
    assert walkers.Ghalfa.shape == (ndets, nwalkers, nelec[0], nmo)
    assert walkers.Ga.shape == (nwalkers, nmo, nmo)
    assert walkers.Gb.shape == (nwalkers, nmo, nmo)
    trial.calc_force_bias(hamiltonian, walkers)
    ovlp_gf = trial.calc_greens_function(walkers)
    ovlp_ov = trial.calc_overlap(walkers)
    assert np.allclose(ovlp_gf, ovlp_ov)


@pytest.mark.unit
def test_local_energy_noci():
    np.random.seed(7)
    nmo = 12
    nelec = (7, 7)
    nwalkers = 10
    nsteps = 100
    ndets = 11
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    system = Generic(nelec=nelec)
    hamiltonian = HamGeneric(h1e=np.array([h1e, h1e]), chol=chol.reshape((-1, nmo * nmo)).T.copy())
    # Test PH type wavefunction.
    coeffs, wfn, init = get_random_nomsd(
        system.nup, system.ndown, hamiltonian.nbasis, ndet=ndets, init=True, cplx=False
    )
    wfn0 = wfn[0].copy()
    for i in range(len(wfn)):
        wfn[i] = wfn0.copy()
        coeffs[i] = coeffs[0]
    trial = NOCI((coeffs, wfn), nelec, nmo)
    trial.build()
    trial.half_rotate(hamiltonian)
    g = trial.build_one_rdm()
    walkers = UHFWalkersTrial(trial, init, system.nup, system.ndown, hamiltonian.nbasis, nwalkers)
    walkers.build(trial)
    energy = local_energy(system, hamiltonian, walkers, trial)
    trial_sd = SingleDet(wfn[0], nelec, nmo)
    walkers_sd = UHFWalkersTrial(trial_sd, init, system.nup, system.ndown, hamiltonian.nbasis, nwalkers)
    walkers.build(trial)
    energy_sd = local_energy(system, hamiltonian, walkers, trial)
    assert np.allclose(energy, energy_sd)
    coeffs, wfn, init = get_random_nomsd(
        system.nup, system.ndown, hamiltonian.nbasis, ndet=ndets, init=True, cplx=False
    )
    trial = NOCI((coeffs, wfn), nelec, nmo)
    trial.build()
    trial.half_rotate(hamiltonian)
    g = trial.build_one_rdm()
    walkers = UHFWalkersTrial(trial, init, system.nup, system.ndown, hamiltonian.nbasis, nwalkers)
    walkers.build(trial)
    energy = local_energy(system, hamiltonian, walkers, trial)
    assert not np.allclose(energy, energy_sd)
    # Test against PHMSD
    wfn, init = get_random_phmsd_opt(system.nup, system.ndown, hamiltonian.nbasis, ndet=11, init=True)
    trial_phmsd = ParticleHole(
        wfn,
        nelec,
        nmo,
    )
    trial_phmsd.build()
    trial_phmsd.half_rotate(hamiltonian)
    noci = np.zeros((ndets, nmo, sum(nelec)))
    for idet, (occa, occb) in enumerate(zip(wfn[1], wfn[2])):
        for iorb, occ in enumerate(occa):
            noci[idet, occ, iorb] = 1.0
        for iorb, occ in enumerate(occb):
            noci[idet, occ, nelec[0] + iorb] = 1.0

    trial = NOCI((wfn[0], noci), nelec, nmo)
    trial.build()
    trial.half_rotate(hamiltonian)
    walkers = UHFWalkersTrial(trial, init, system.nup, system.ndown, hamiltonian.nbasis, nwalkers)
    walkers.build(trial)
    ovlp_noci = trial.calc_overlap(walkers)
    trial.calc_greens_function(walkers)
    fb_noci = trial.calc_force_bias(hamiltonian, walkers)
    energy = local_energy(system, hamiltonian, walkers, trial)
    walkers_phmsd = UHFWalkersTrial(
        trial_phmsd, init, system.nup, system.ndown, hamiltonian.nbasis, nwalkers
    )
    walkers_phmsd.build(trial_phmsd)
    trial_phmsd.calc_greens_function(walkers_phmsd)
    assert np.allclose(ovlp_noci, trial_phmsd.calc_overlap(walkers_phmsd))
    assert np.allclose(walkers_phmsd.Ga, walkers.Ga)
    assert np.allclose(walkers_phmsd.Gb, walkers.Gb)
    assert np.allclose(fb_noci, trial_phmsd.calc_force_bias(hamiltonian, walkers))
    e_phmsd = local_energy_multi_det_trial_wicks_batch_opt_chunked(
        system, hamiltonian, walkers_phmsd, trial_phmsd
    )
    assert np.allclose(energy, e_phmsd)
