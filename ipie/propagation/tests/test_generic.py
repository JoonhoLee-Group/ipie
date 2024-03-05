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

from ipie.estimators.greens_function import greens_function_single_det_batch
from ipie.propagation.overlap import calc_overlap_single_det_uhf
from ipie.utils.legacy_testing import build_legacy_test_case_handlers
from ipie.utils.misc import dotdict
from ipie.utils.testing import build_test_case_handlers


@pytest.mark.unit
def test_overlap_rhf_batch():
    numpy.random.seed(7)
    nmo = 10
    nelec = (5, 5)
    nwalkers = 2
    nsteps = 25

    qmc = dotdict(
        {
            "dt": 0.005,
            "nstblz": 5,
            "nwalkers": nwalkers,
            "batched": False,
            "hybrid": True,
            "num_steps": nsteps,
            "rhf": True,
        }
    )
    legacy_data = build_legacy_test_case_handlers(
        nelec,
        nmo,
        num_dets=1,
        options=qmc,
        seed=7,
        trial_type="nomsd",
        rhf_trial=True,
    )
    qmc.batched = True
    batched_data = build_test_case_handlers(
        nelec,
        nmo,
        num_dets=1,
        options=qmc,
        seed=7,
        trial_type="single_det",
        rhf_trial=True,
    )
    walkers = legacy_data.walker_handler.walkers
    walker_batch = batched_data.walkers

    ovlp = calc_overlap_single_det_uhf(walker_batch, batched_data.trial)
    ovlp_gf = greens_function_single_det_batch(walker_batch, batched_data.trial)
    ot = [walkers[iw].ot for iw in range(walker_batch.nwalkers)]
    assert numpy.allclose(ovlp, ot)
    assert numpy.allclose(ovlp_gf, ot)

    for iw in range(nwalkers):
        assert numpy.allclose(walker_batch.Ghalfa[iw], walkers[iw].Ghalf[0])


@pytest.mark.unit
def test_overlap_batch():
    numpy.random.seed(7)
    nmo = 10
    nelec = (6, 5)
    nwalkers = 2
    nsteps = 10

    qmc = dotdict(
        {
            "dt": 0.005,
            "nstblz": 5,
            "nwalkers": nwalkers,
            "batched": False,
            "hybrid": True,
            "num_steps": nsteps,
        }
    )
    legacy_data = build_legacy_test_case_handlers(
        nelec,
        nmo,
        num_dets=1,
        options=qmc,
        seed=7,
        trial_type="nomsd",
    )
    qmc.batched = True
    batched_data = build_test_case_handlers(
        nelec,
        nmo,
        num_dets=1,
        options=qmc,
        seed=7,
        trial_type="single_det",
    )
    walkers = legacy_data.walker_handler.walkers
    walker_batch = batched_data.walkers

    ovlp = calc_overlap_single_det_uhf(walker_batch, batched_data.trial)
    ovlp_gf = greens_function_single_det_batch(walker_batch, batched_data.trial)
    ot = [walkers[iw].ot for iw in range(walker_batch.nwalkers)]
    assert numpy.allclose(ovlp, ot)
    assert numpy.allclose(ovlp_gf, ot)

    for iw in range(nwalkers):
        assert numpy.allclose(walker_batch.Ghalfa[iw], walkers[iw].Ghalf[0])
        assert numpy.allclose(walker_batch.Ghalfb[iw], walkers[iw].Ghalf[1])


@pytest.mark.unit
def test_two_body_rhf_batch():
    numpy.random.seed(7)
    nmo = 10
    nelec = (5, 5)
    nwalkers = 8
    nsteps = 25
    qmc = dotdict(
        {
            "dt": 0.005,
            "nstblz": 5,
            "nwalkers": nwalkers,
            "batched": False,
            "hybrid": True,
            "rhf": True,
            "num_steps": nsteps,
        }
    )
    legacy_data = build_legacy_test_case_handlers(
        nelec,
        nmo,
        num_dets=1,
        options=qmc,
        seed=7,
        trial_type="nomsd",
        two_body_only=True,
        rhf_trial=True,
    )
    qmc.batched = True
    batched_data = build_test_case_handlers(
        nelec,
        nmo,
        num_dets=1,
        options=qmc,
        seed=7,
        trial_type="single_det",
        two_body_only=True,
        rhf_trial=True,
    )
    walkers = legacy_data.walker_handler.walkers
    walker_batch = batched_data.walkers

    for iw in range(nwalkers):
        assert numpy.allclose(walker_batch.Ghalfa[iw], walkers[iw].Ghalf[0])


@pytest.mark.unit
def test_two_body_batch():
    numpy.random.seed(7)
    nmo = 10
    nelec = (6, 5)
    nwalkers = 2
    nsteps = 25

    qmc = dotdict(
        {
            "dt": 0.005,
            "nstblz": 5,
            "nwalkers": nwalkers,
            "batched": False,
            "hybrid": True,
            "num_steps": nsteps,
        }
    )
    legacy_data = build_legacy_test_case_handlers(
        nelec,
        nmo,
        num_dets=1,
        options=qmc,
        seed=7,
        trial_type="nomsd",
        two_body_only=True,
    )
    qmc.batched = True
    batched_data = build_test_case_handlers(
        nelec,
        nmo,
        num_dets=1,
        options=qmc,
        seed=7,
        trial_type="single_det",
        two_body_only=True,
    )
    walkers = legacy_data.walker_handler.walkers
    walker_batch = batched_data.walkers

    for iw in range(nwalkers):
        assert numpy.allclose(walker_batch.Ghalfa[iw], walkers[iw].Ghalf[0])
        assert numpy.allclose(walker_batch.Ghalfb[iw], walkers[iw].Ghalf[1])


@pytest.mark.unit
def test_hybrid_rhf_batch():
    numpy.random.seed(7)
    nmo = 10
    nelec = (5, 5)
    nwalkers = 8
    nsteps = 25

    qmc = dotdict(
        {
            "dt": 0.005,
            "nstblz": 5,
            "nwalkers": nwalkers,
            "batched": False,
            "hybrid": True,
            "rhf": True,
            "num_steps": nsteps,
        }
    )
    legacy_data = build_legacy_test_case_handlers(
        nelec,
        nmo,
        num_dets=1,
        options=qmc,
        seed=7,
        trial_type="nomsd",
        rhf_trial=True,
    )
    qmc.batched = True
    batched_data = build_test_case_handlers(
        nelec,
        nmo,
        num_dets=1,
        options=qmc,
        seed=7,
        trial_type="single_det",
        rhf_trial=True,
    )
    walkers = legacy_data.walker_handler.walkers
    walker_batch = batched_data.walkers

    for iw in range(nwalkers):
        assert numpy.allclose(walker_batch.Ghalfa[iw], walkers[iw].Ghalf[0])
        assert numpy.allclose(walker_batch.phia[iw], walkers[iw].phi[:, : nelec[0]])


@pytest.mark.unit
def test_hybrid_batch():
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
            "batched": False,
            "hybrid": True,
            "num_steps": nsteps,
        }
    )
    legacy_data = build_legacy_test_case_handlers(nelec, nmo, num_dets=1, options=qmc, seed=7)
    qmc.batched = True
    batched_data = build_test_case_handlers(nelec, nmo, num_dets=1, options=qmc, seed=7)
    walkers = legacy_data.walker_handler.walkers
    walker_batch = batched_data.walkers
    for iw in range(nwalkers):
        assert numpy.allclose(walker_batch.Ghalfa[iw], walkers[iw].Ghalf[0])
        assert numpy.allclose(walker_batch.Ghalfb[iw], walkers[iw].Ghalf[1])
        assert numpy.allclose(walker_batch.phia[iw], walkers[iw].phi[:, : nelec[0]])
        assert numpy.allclose(walker_batch.phib[iw], walkers[iw].phi[:, nelec[0] :])


@pytest.mark.unit
def test_vhs():
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
            "batched": False,
            "hybrid": True,
            "num_steps": nsteps,
        }
    )
    legacy_data = build_legacy_test_case_handlers(nelec, nmo, num_dets=1, options=qmc, seed=7)
    xshifted = (
        numpy.random.normal(0.0, 1.0, nwalkers * legacy_data.hamiltonian.nfields)
        .reshape(nwalkers, legacy_data.hamiltonian.nfields)
        .astype(numpy.complex128)
    )
    vhs_serial = []
    for iw in range(nwalkers):
        vhs_serial.append(
            legacy_data.propagator.propagator.construct_VHS(legacy_data.hamiltonian, xshifted[iw])
        )

    qmc.batched = True
    batched_data = build_test_case_handlers(nelec, nmo, num_dets=1, options=qmc, seed=7)
    vhs_batch = batched_data.propagator.construct_VHS(batched_data.hamiltonian, xshifted.T.copy())
    for iw in range(nwalkers):
        assert numpy.allclose(vhs_batch[iw], vhs_serial[iw])


if __name__ == "__main__":
    test_overlap_rhf_batch()
    test_overlap_batch()
    test_two_body_batch()
    test_two_body_rhf_batch()
    test_hybrid_rhf_batch()
    test_hybrid_batch()
    test_vhs()
