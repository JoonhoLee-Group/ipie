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

from ipie.trial_wavefunction.single_det_ghf import SingleDetGHF
from ipie.utils.mpi import MPIHandler
from ipie.walkers.ghf_walkers import GHFWalkers


def build_walkers(seed=3, nbasis=7, nelec=(3, 2), nwalkers=4):
    rng = numpy.random.default_rng(seed)
    nocc = sum(nelec)
    psi0 = rng.normal(size=(2 * nbasis, nocc)) + 1j * rng.normal(size=(2 * nbasis, nocc))
    trial = SingleDetGHF(psi0, nelec, nbasis)
    walkers = GHFWalkers(psi0, nelec[0], nelec[1], nbasis, nwalkers, MPIHandler())
    walkers.phi = 2.0 * rng.normal(size=walkers.phi.shape) + 1j * rng.normal(size=walkers.phi.shape)
    walkers.build(trial)
    return trial, walkers


@pytest.mark.unit
def test_ghf_reortho_cpu_matches_qr():
    trial, walkers = build_walkers()
    walkers.detR_shift = numpy.random.default_rng(8).normal(size=walkers.nwalkers)
    phi, ovlp, log_detR = walkers.phi.copy(), walkers.ovlp.copy(), walkers.log_detR.copy()

    detR = walkers.reortho()

    for iw in range(walkers.nwalkers):
        q, r = numpy.linalg.qr(phi[iw])
        phase = numpy.diag(r) / numpy.abs(numpy.diag(r))
        detR_ref = numpy.exp(
            numpy.sum(numpy.log(numpy.abs(numpy.diag(r)))) - walkers.detR_shift[iw]
        )
        numpy.testing.assert_allclose(walkers.phi[iw], q * phase[None, :], rtol=1e-10, atol=1e-12)
        numpy.testing.assert_allclose(detR[iw], detR_ref, rtol=1e-10)
        numpy.testing.assert_allclose(walkers.detR[iw], detR_ref, rtol=1e-10)
        numpy.testing.assert_allclose(walkers.ovlp[iw], ovlp[iw] / detR_ref, rtol=1e-10)
        numpy.testing.assert_allclose(walkers.log_detR[iw], log_detR[iw] + numpy.log(detR_ref))


@pytest.mark.unit
def test_ghf_reortho_cpu_keeps_overlap_consistent():
    trial, walkers = build_walkers(seed=4)
    walkers.reortho()
    numpy.testing.assert_allclose(
        numpy.einsum("wmi,wmj->wij", walkers.phi.conj(), walkers.phi),
        numpy.broadcast_to(
            numpy.eye(walkers.phi.shape[2]), (walkers.nwalkers,) + (walkers.phi.shape[2],) * 2
        ),
        atol=1e-12,
    )
    numpy.testing.assert_allclose(walkers.ovlp, trial.calc_overlap(walkers), rtol=1e-10, atol=1e-12)
