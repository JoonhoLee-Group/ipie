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

import itertools

import numpy
import pytest

from ipie.estimators.greens_function_multi_det import (
    greens_function_multi_det,
    greens_function_multi_det_wicks,
    greens_function_multi_det_wicks_opt,
)
from ipie.estimators.local_energy_batch import local_energy_multi_det_trial_batch
from ipie.estimators.local_energy_wicks import (
    local_energy_multi_det_trial_wicks_batch,
    local_energy_multi_det_trial_wicks_batch_opt,
)
from ipie.hamiltonians.generic import Generic as HamGeneric
from ipie.legacy.hamiltonians._generic import Generic as LegacyHamGeneric
from ipie.legacy.estimators.ci import simple_fci
from ipie.legacy.estimators.local_energy import local_energy_generic_cholesky
from ipie.propagation.overlap import calc_overlap_multi_det_wicks
from ipie.systems.generic import Generic
from ipie.trial_wavefunction.noci import NOCI
from ipie.trial_wavefunction.particle_hole import (
    ParticleHoleNaive,
    ParticleHoleWicks,
    ParticleHoleWicksSlow,
)
from ipie.utils.linalg import reortho
from ipie.utils.misc import dotdict
from ipie.utils.testing import generate_hamiltonian, get_random_wavefunction
from ipie.walkers.walkers_dispatch import UHFWalkersTrial


@pytest.mark.unit
def test_walker_overlap_nomsd():
    system = dotdict({"nup": 5, "ndown": 5, "nelec": (5, 5), "ne": 10})
    ham = dotdict({"nbasis": 10})
    numpy.random.seed(7)

    ndets = 5
    a = numpy.random.rand(ndets * ham.nbasis * (system.nup + system.ndown))
    b = numpy.random.rand(ndets * ham.nbasis * (system.nup + system.ndown))
    wfn = (a + 1j * b).reshape((ndets, ham.nbasis, system.nup + system.ndown))
    coeffs = numpy.random.randn(ndets) + 1.0j * numpy.random.randn(ndets)

    nwalkers = 5
    # Test NOMSD type wavefunction.
    nelec = (system.nup, system.ndown)
    trial = NOCI((coeffs, wfn), nelec, ham.nbasis)

    walkers = UHFWalkersTrial(trial, wfn[0], system.nup, system.ndown, ham.nbasis, nwalkers, 25)
    walkers.build(trial)
    walkers.ovlp = trial.calc_overlap(walkers)

    def calc_ovlp(a, b):
        return numpy.linalg.det(numpy.dot(a.conj().T, b))

    ovlp = numpy.array([0.0 + 0j for iw in range(nwalkers)])
    na = system.nup
    pa = trial.psi[0, :, :na]
    pb = trial.psi[0, :, na:]
    for iw in range(nwalkers):
        for i, d in enumerate(trial.psi):
            ovlp[iw] += coeffs[i].conj() * calc_ovlp(d[:, :na], pa) * calc_ovlp(d[:, na:], pb)
    assert ovlp.real == pytest.approx(walkers.ovlp.real)
    assert ovlp.imag == pytest.approx(walkers.ovlp.imag)


@pytest.mark.unit
def test_walker_overlap_phmsd():
    system = dotdict({"nup": 5, "ndown": 5, "nelec": (5, 5), "ne": 10})
    ham = dotdict({"nbasis": 10})
    numpy.random.seed(7)
    # Test PH type wavefunction.
    orbs = numpy.arange(ham.nbasis)
    oa = [c for c in itertools.combinations(orbs, system.nup)]
    ob = [c for c in itertools.combinations(orbs, system.ndown)]
    oa, ob = zip(*itertools.product(oa, ob))
    oa = oa[:5]
    ob = ob[:5]
    coeffs = (
        numpy.array([0.9, 0.01, 0.01, 0.02, 0.04], dtype=numpy.complex128)
        + numpy.random.randn(5) * 1.0j
    )
    nwalkers = 10
    wfn = (coeffs, oa, ob)
    a = numpy.random.rand(ham.nbasis * (system.nup + system.ndown))
    b = numpy.random.rand(ham.nbasis * (system.nup + system.ndown))
    init = (a + 1j * b).reshape((ham.nbasis, system.nup + system.ndown))
    nelec = (system.nup, system.ndown)
    trial = ParticleHoleNaive(wfn, nelec, ham.nbasis)

    walkers = UHFWalkersTrial(trial, init, system.nup, system.ndown, ham.nbasis, nwalkers, 25)
    walkers.build(trial)
    walkers.ovlp = trial.calc_greens_function(walkers)

    I = numpy.eye(ham.nbasis)
    ovlp_sum = numpy.zeros(nwalkers, dtype=numpy.complex128)
    for iw in range(nwalkers):
        for idet, (c, occa, occb) in enumerate(zip(coeffs, oa, ob)):
            psia = I[:, occa]
            psib = I[:, occb]
            sa = numpy.dot(psia.conj().T, init[:, : system.nup])
            sb = numpy.dot(psib.conj().T, init[:, system.nup :])
            isa = numpy.linalg.inv(sa)
            isb = numpy.linalg.inv(sb)
            ga = numpy.dot(init[:, : system.nup], numpy.dot(isa, psia.conj().T)).T
            gb = numpy.dot(init[:, system.nup :], numpy.dot(isb, psib.conj().T)).T
            ovlp = numpy.linalg.det(sa) * numpy.linalg.det(sb)
            ovlp_sum[iw] += c.conj() * ovlp
            walk_ovlp = walkers.det_ovlpas[iw, idet] * walkers.det_ovlpbs[iw, idet]
            assert ovlp == pytest.approx(walk_ovlp)
            assert numpy.linalg.norm(ga - walkers.Gia[iw, idet]) == pytest.approx(0)
            assert numpy.linalg.norm(gb - walkers.Gib[iw, idet]) == pytest.approx(0)

    trial = ParticleHoleWicksSlow(wfn, nelec, ham.nbasis)
    ovlp_wicks = calc_overlap_multi_det_wicks(walkers, trial)

    assert ovlp_sum == pytest.approx(walkers.ovlp)
    assert ovlp_wicks == pytest.approx(walkers.ovlp)


@pytest.mark.unit
def test_walker_energy():
    numpy.random.seed(7)
    nelec = (2, 2)
    nmo = 5
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    system = Generic(nelec=nelec)
    ham = HamGeneric(
        h1e=numpy.array([h1e, h1e]),
        chol=chol.reshape((-1, nmo * nmo)).T.copy(),
        ecore=enuc,
    )

    legacy_ham = LegacyHamGeneric(
        h1e=numpy.array([h1e, h1e]),
        chol=chol.reshape((-1, nmo * nmo)).T.copy(),
        ecore=enuc,
    )

    (e0, ev), (d, oa, ob) = simple_fci(system, ham, gen_dets=True)
    na = system.nup
    init = get_random_wavefunction(nelec, nmo)
    init[:, :na], R = reortho(init[:, :na])
    init[:, na:], R = reortho(init[:, na:])
    trial = ParticleHoleWicksSlow(
        (ev[:, 0], oa, ob),
        nelec,
        nmo,
    )
    trial.calculate_energy(system, ham)
    trial.half_rotate(ham)
    trial_slow = ParticleHoleNaive((ev[:, 0], oa, ob), nelec, nmo)
    trial_slow.calculate_energy(system, ham)
    trial_slow.half_rotate(ham)
    trial_opt = ParticleHoleWicks(
        (ev[:, 0], oa, ob),
        nelec,
        nmo,
    )
    trial_opt.calculate_energy(system, ham)
    trial_opt.half_rotate(ham)

    nwalkers = 10
    ndets = len(oa)
    # walker = MultiDetTrialWalkerBatch(system, ham, trial_slow, nwalkers, init)
    # walker_opt = MultiDetTrialWalkerBatch(system, ham, trial_opt, nwalkers, init)

    walkers0 = UHFWalkersTrial(trial, init, system.nup, system.ndown, ham.nbasis, nwalkers, 25)
    walkers0.build(trial_slow)
    walkers0.ovlp = trial_slow.calc_greens_function(walkers0)

    walkers = UHFWalkersTrial(trial_slow, init, system.nup, system.ndown, ham.nbasis, nwalkers, 25)
    walkers.build(trial_slow)
    walkers.ovlp = trial_slow.calc_greens_function(walkers)

    walkers_opt = UHFWalkersTrial(
        trial_opt, init, system.nup, system.ndown, ham.nbasis, nwalkers, 25
    )
    walkers_opt.build(trial_opt)
    walkers_opt.ovlp = trial_opt.calc_greens_function(walkers_opt)

    nume = 0
    deno = 0
    energies = numpy.zeros(nwalkers, dtype=numpy.complex128)
    for iw in range(nwalkers):
        for i in range(trial.num_dets):
            psia = trial_slow.psi[i, :, :na]
            psib = trial_slow.psi[i, :, na:]
            oa = numpy.dot(psia.conj().T, init[:, :na])
            ob = numpy.dot(psib.conj().T, init[:, na:])
            isa = numpy.linalg.inv(oa)
            isb = numpy.linalg.inv(ob)
            ovlp = numpy.linalg.det(oa) * numpy.linalg.det(ob)
            ga = numpy.dot(init[:, : system.nup], numpy.dot(isa, psia.conj().T)).T
            gb = numpy.dot(init[:, system.nup :], numpy.dot(isb, psib.conj().T)).T
            e = local_energy_generic_cholesky(system, legacy_ham, numpy.array([ga, gb]))[0]
            nume += trial.coeffs[i].conj() * ovlp * e
            deno += trial.coeffs[i].conj() * ovlp
        energies[iw] = nume / deno

    greens_function_multi_det_wicks(
        walkers0, trial
    )  # compute green's function using Wick's theorem
    greens_function_multi_det_wicks_opt(
        walkers_opt, trial_opt
    )  # compute green's function using Wick's theorem
    e_wicks = local_energy_multi_det_trial_wicks_batch(system, ham, walkers0, trial)
    e_wicks_opt = local_energy_multi_det_trial_wicks_batch_opt(system, ham, walkers_opt, trial_opt)
    greens_function_multi_det(walkers, trial_slow)
    e_simple = local_energy_multi_det_trial_batch(system, ham, walkers, trial_slow)

    assert e_simple[:, 0] == pytest.approx(energies)
    assert e_wicks_opt[:, 0] == pytest.approx(e_wicks[:, 0])
    assert e_wicks[:, 0] == pytest.approx(energies)

    # e = local_energy_batch(system, ham, walkers, trial, iw=0)
    # assert e[:,0] == pytest.approx(energies[0])


if __name__ == "__main__":
    test_walker_overlap_nomsd()
    test_walker_overlap_phmsd()
    test_walker_energy()
