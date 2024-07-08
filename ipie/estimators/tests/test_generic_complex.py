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

import numpy
import pytest

from ipie.estimators.energy import local_energy

from ipie.utils.testing import (
        build_test_case_handlers, 
        build_test_case_handlers_ghf
        )
from ipie.systems.generic import Generic
from ipie.hamiltonians.generic import Generic as HamGeneric
from ipie.utils.misc import dotdict


@pytest.mark.unit
def test_local_energy_single_det_vs_real():
    numpy.random.seed(7)
    nmo = 10
    nelec = (6, 5)
    nwalkers = 8
    nsteps = 25
    qmc = dotdict(
        {
            "dt": 0.005,
            "nstblz": 5,
            "nwalkers": nwalkers,
            "hybrid": True,
            "num_steps": nsteps,
        }
    )

    test_handler = build_test_case_handlers(
        nelec,
        nmo,
        num_dets=1,
        options=qmc,
        seed=7,
        complex_integrals=False,
        complex_trial=False,
        trial_type="single_det",
    )

    ham = test_handler.hamiltonian
    walkers = test_handler.walkers
    system = Generic(nelec)
    trial = test_handler.trial

    chol = ham.chol
    cx_chol = numpy.array(chol, dtype=numpy.complex128)
    cx_ham = HamGeneric(
        numpy.array(ham.H1, dtype=numpy.complex128), cx_chol, ham.ecore, verbose=False
    )

    energy = local_energy(system, ham, walkers, trial)
    trial.half_rotate(cx_ham)
    cx_energy = local_energy(system, cx_ham, walkers, trial)
    numpy.testing.assert_allclose(energy, cx_energy, atol=1e-10)


@pytest.mark.unit
def test_local_energy_single_det_ghf_vs_real():
    numpy.random.seed(7)
    nmo = 10
    nelec = (6, 5)
    nwalkers = 8
    nsteps = 25
    qmc = dotdict(
        {
            "dt": 0.005,
            "nstblz": 5,
            "nwalkers": nwalkers,
            "hybrid": True,
            "num_steps": nsteps,
        }
    )

    test_handler = build_test_case_handlers_ghf(
        nelec,
        nmo,
        num_dets=1,
        options=qmc,
        seed=7,
        complex_integrals=False,
        complex_trial=False,
        trial_type="single_det_ghf",
    )

    ham = test_handler.hamiltonian
    walkers = test_handler.walkers
    system = Generic(nelec)
    trial = test_handler.trial

    chol = ham.chol
    cx_chol = numpy.array(chol, dtype=numpy.complex128)
    cx_ham = HamGeneric(
        numpy.array(ham.H1, dtype=numpy.complex128), cx_chol, ham.ecore, verbose=False
    )

    energy = local_energy(system, ham, walkers, trial)
    cx_energy = local_energy(system, cx_ham, walkers, trial)
    numpy.testing.assert_allclose(energy, cx_energy, atol=1e-10)


@pytest.mark.unit
def test_local_energy_single_det_vs_eri():
    numpy.random.seed(7)
    nmo = 10
    nelec = (6, 5)
    nwalkers = 1
    nsteps = 25
    qmc = dotdict(
        {
            "dt": 0.005,
            "nstblz": 5,
            "nwalkers": nwalkers,
            "hybrid": True,
            "num_steps": nsteps,
        }
    )

    test_handler = build_test_case_handlers(
        nelec,
        nmo,
        num_dets=1,
        options=qmc,
        seed=7,
        complex_integrals=True,
        complex_trial=True,
        trial_type="single_det",
        choltol=1e-10,
    )

    ham = test_handler.hamiltonian
    walkers = test_handler.walkers
    system = Generic(nelec)
    trial = test_handler.trial

    walkers.ovlp = trial.calc_greens_function(walkers, build_full=True)

    energy = local_energy(system, ham, walkers, trial)
    etot = energy[:, 0]
    e1 = energy[:, 1]
    e2 = energy[:, 2]

    # Testing 1-body term.
    h1e = ham.H1[0]
    G = walkers.Ga + walkers.Gb
    e1ref = numpy.einsum("ij,wij->w", h1e, G)
    numpy.testing.assert_allclose(e1, e1ref, atol=1e-10)

    # Testing 2-body term.
    nbasis = ham.nbasis
    eri = ham.eri.reshape(nbasis, nbasis, nbasis, nbasis)
    chol = ham.chol.copy()
    nchol = chol.shape[1]
    chol = chol.reshape(nbasis, nbasis, nchol)

    # Check if chol and eri are consistent.
    eri_chol = numpy.einsum("mnx,slx->mnls", chol, chol.conj())
    numpy.testing.assert_allclose(eri, eri_chol, atol=1e-10)

    ecoul = 0.5 * numpy.einsum("ijkl,wij,wkl->w", eri, G, G)
    exx = -0.5 * numpy.einsum("ijkl,wil,wkj->w", eri, walkers.Ga, walkers.Ga)
    exx -= 0.5 * numpy.einsum("ijkl,wil,wkj->w", eri, walkers.Gb, walkers.Gb)
    e2ref = ecoul + exx
    numpy.testing.assert_allclose(e2, e2ref, atol=1e-10)


@pytest.mark.unit
def test_local_energy_single_det_ghf_vs_eri():
    numpy.random.seed(7)
    nmo = 10
    nelec = (6, 5)
    nwalkers = 1
    nsteps = 25
    qmc = dotdict(
        {
            "dt": 0.005,
            "nstblz": 5,
            "nwalkers": nwalkers,
            "hybrid": True,
            "num_steps": nsteps,
        }
    )

    test_handler = build_test_case_handlers_ghf(
        nelec,
        nmo,
        num_dets=1,
        options=qmc,
        seed=7,
        complex_integrals=True,
        complex_trial=True,
        trial_type="single_det_ghf",
        choltol=1e-10,
    )

    ham = test_handler.hamiltonian
    walkers = test_handler.walkers
    system = Generic(nelec)
    trial = test_handler.trial

    walkers.ovlp = trial.calc_greens_function(walkers, build_full=True)

    energy = local_energy(system, ham, walkers, trial)
    etot = energy[:, 0]
    e1 = energy[:, 1]
    e2 = energy[:, 2]

    # Testing 1-body term.
    h1e = ham.H1[0]
    G = walkers.Ga + walkers.Gb
    e1ref = numpy.einsum("ij,wij->w", h1e, G)
    numpy.testing.assert_allclose(e1, e1ref, atol=1e-10)

    # Testing 2-body term.
    nbasis = ham.nbasis
    eri = ham.eri.reshape(nbasis, nbasis, nbasis, nbasis)
    chol = ham.chol.copy()
    nchol = chol.shape[1]
    chol = chol.reshape(nbasis, nbasis, nchol)

    # Check if chol and eri are consistent.
    eri_chol = numpy.einsum("mnx,slx->mnls", chol, chol.conj())
    numpy.testing.assert_allclose(eri, eri_chol, atol=1e-10)

    Gab = walkers.G[:, :nbasis, nbasis:]
    Gba = walkers.G[:, nbasis:, :nbasis]
    ecoul = 0.5 * numpy.einsum("ijkl,wij,wkl->w", eri, G, G)
    exx = numpy.einsum("ijkl,wil,wkj->w", eri, walkers.Ga, walkers.Ga)
    exx += numpy.einsum("ijkl,wil,wkj->w", eri, walkers.Gb, walkers.Gb)
    exx += numpy.einsum("ijkl,wil,wkj->w", eri, Gab, Gba)
    exx += numpy.einsum("ijkl,wil,wkj->w", eri, Gba, Gab)
    exx *= -0.5
    e2ref = ecoul + exx
    numpy.testing.assert_allclose(e2, e2ref, atol=1e-10)


if __name__ == "__main__":
    test_local_energy_single_det_vs_real()
    test_local_energy_single_det_ghf_vs_real()
    test_local_energy_single_det_vs_eri()
    test_local_energy_single_det_ghf_vs_eri()
