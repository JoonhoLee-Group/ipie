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

import numpy
import pytest

from ipie.hamiltonians.hubbard import Hubbard
from ipie.propagation.hubbard_generic import HubbardSingleSite
from ipie.trial_wavefunction.noci_ghf import NOCIGHF
from ipie.trial_wavefunction.single_det_ghf import SingleDetGHF
from ipie.utils.mpi import MPIHandler
from ipie.walkers.ghf_walkers import GHFWalkers


def random_complex(rng, shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def build_random_noci_case(nbasis=5, nelec=(3, 2), ndet=3, nwalkers=4, seed=7):
    rng = numpy.random.default_rng(seed)
    nocc = sum(nelec)
    dets = random_complex(rng, (ndet, 2 * nbasis, nocc))
    coeffs = random_complex(rng, (ndet,))
    phi = random_complex(rng, (nwalkers, 2 * nbasis, nocc))
    trial = NOCIGHF(dets, coeffs, nelec, nbasis)
    walkers = GHFWalkers(phi[0], nelec[0], nelec[1], nbasis, nwalkers, MPIHandler())
    walkers.phi = phi.copy()
    walkers.build(trial)
    return trial, walkers, phi


def clone_walkers(trial, phi, nelec, nbasis):
    walkers = GHFWalkers(phi[0], nelec[0], nelec[1], nbasis, phi.shape[0], MPIHandler())
    walkers.phi = phi.copy()
    walkers.build(trial)
    return walkers


@pytest.mark.unit
def test_noci_calc_overlap_matches_per_det():
    trial, walkers, _ = build_random_noci_case(seed=11)

    O_full, _ = trial._per_det(walkers)
    O_fast = trial._per_det_overlaps(walkers)
    numpy.testing.assert_allclose(O_fast, O_full, atol=1e-12)

    ovlp = trial.calc_overlap(walkers)
    ref = numpy.sum(trial.coeffs.conj()[:, None] * O_full, axis=0)
    numpy.testing.assert_allclose(ovlp, ref, atol=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize("U", [4.0, -4.0])
def test_noci_kernel_matches_einsum_two_body(U):
    nbasis, nelec = 5, (3, 2)
    trial, w_kernel, phi = build_random_noci_case(nbasis=nbasis, nelec=nelec, seed=13)
    w_einsum = clone_walkers(trial, phi, nelec, nbasis)
    ham = Hubbard(numpy.zeros((2, nbasis, nbasis)), U=U)
    prop = HubbardSingleSite(0.01)
    prop.build(ham, trial, w_kernel, w_kernel.mpi_handler)

    rng = numpy.random.default_rng(99)
    random_fields = rng.random((nbasis, w_kernel.nwalkers))

    prop._noci_ghf_two_body(w_kernel, ham, trial, random_fields)
    prop._noci_ghf_two_body_einsum(w_einsum, ham, trial, random_fields)

    numpy.testing.assert_allclose(w_kernel.phi, w_einsum.phi, atol=1e-10)
    numpy.testing.assert_allclose(w_kernel.weight, w_einsum.weight, atol=1e-10)
    numpy.testing.assert_allclose(w_kernel.ovlp, w_einsum.ovlp, atol=1e-10)
    alive = numpy.abs(w_kernel.weight) > 0.0
    assert numpy.any(alive)
    numpy.testing.assert_allclose(
        w_kernel.ovlp[alive], trial.calc_overlap(w_kernel)[alive], rtol=1e-9
    )


@pytest.mark.unit
def test_noci_two_body_single_det_matches_ghf():
    rng = numpy.random.default_rng(17)
    nbasis, nelec = 4, (2, 2)
    nocc = sum(nelec)
    nwalkers = 3
    psi0 = random_complex(rng, (2 * nbasis, nocc))
    phi = random_complex(rng, (nwalkers, 2 * nbasis, nocc))
    h1e = numpy.zeros((2, nbasis, nbasis))
    ham = Hubbard(h1e, U=3.0)
    random_fields = numpy.full((nbasis, nwalkers), 0.4)

    trial_ghf = SingleDetGHF(psi0.copy(), nelec, nbasis)
    w_ghf = clone_walkers(trial_ghf, phi, nelec, nbasis)
    prop = HubbardSingleSite(0.01)
    prop.build(ham, trial_ghf, w_ghf, w_ghf.mpi_handler)
    prop._ghf_two_body(w_ghf, ham, trial_ghf, random_fields)

    trial_noci = NOCIGHF(psi0[None].copy(), numpy.ones(1, dtype=numpy.complex128), nelec, nbasis)
    w_noci = clone_walkers(trial_noci, phi, nelec, nbasis)
    prop._noci_ghf_two_body(w_noci, ham, trial_noci, random_fields)

    numpy.testing.assert_allclose(w_noci.phi, w_ghf.phi, atol=1e-10)
    numpy.testing.assert_allclose(w_noci.weight, w_ghf.weight, atol=1e-10)
    numpy.testing.assert_allclose(w_noci.ovlp, w_ghf.ovlp, atol=1e-10)


@pytest.mark.unit
def test_noci_full_propagation_smoke():
    numpy.random.seed(23)
    rng = numpy.random.default_rng(23)
    nbasis, nelec = 4, (2, 2)
    trial, walkers, _ = build_random_noci_case(nbasis=nbasis, nelec=nelec, seed=23)
    A = random_complex(rng, (nbasis, nbasis))
    T = A + A.conj().T
    ham = Hubbard(numpy.array([T, T]), U=4.0)
    prop = HubbardSingleSite(0.01)
    prop.build(ham, trial, walkers, walkers.mpi_handler)

    for _ in range(3):
        prop.propagate_walkers(walkers, ham, trial, eshift=0.0)

    assert numpy.all(numpy.isfinite(walkers.weight))
    assert numpy.all(numpy.isfinite(walkers.phi))
    # dead walkers keep their pre-death overlap by construction
    alive = numpy.abs(walkers.weight) > 0.0
    assert numpy.any(alive)
    numpy.testing.assert_allclose(walkers.ovlp[alive], trial.calc_overlap(walkers)[alive], atol=1e-8)
