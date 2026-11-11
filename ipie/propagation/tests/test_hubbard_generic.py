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

from types import SimpleNamespace

import numpy
import pytest

from ipie.hamiltonians.hubbard import Hubbard
from ipie.propagation.hubbard_generic import HubbardSingleSite
from ipie.trial_wavefunction.single_det import SingleDet
from ipie.utils.mpi import MPIHandler
from ipie.walkers.uhf_walkers import UHFWalkers


def _random_orthonormal(nbasis, nocc, rng):
    mat = rng.normal(size=(nbasis, nocc)) + 1.0j * rng.normal(size=(nbasis, nocc))
    q, _ = numpy.linalg.qr(mat)
    return numpy.ascontiguousarray(q[:, :nocc])


def _copy_walkers(walkers):
    return SimpleNamespace(
        nwalkers=walkers.nwalkers,
        nup=walkers.nup,
        ndown=walkers.ndown,
        nbasis=walkers.nbasis,
        rhf=walkers.rhf,
        weight=walkers.weight.copy(),
        ovlp=walkers.ovlp.copy(),
        phia=walkers.phia.copy(),
        phib=walkers.phib.copy(),
        inv_ovlp_a=walkers.inv_ovlp_a.copy(),
        inv_ovlp_b=walkers.inv_ovlp_b.copy(),
    )


def _build_case(U=4.0, spin_decomp=None):
    rng = numpy.random.default_rng(7)
    nbasis = 12
    nup = 5
    ndown = 4
    nwalkers = 3

    h1 = rng.normal(size=(2, nbasis, nbasis))
    h1 = 0.5 * (h1 + h1.transpose(0, 2, 1))
    hamiltonian = Hubbard(h1, U=U)

    psi0a = _random_orthonormal(nbasis, nup, rng)
    psi0b = _random_orthonormal(nbasis, ndown, rng)
    psi = numpy.ascontiguousarray(numpy.hstack([psi0a, psi0b]))

    handler = MPIHandler()
    trial = SingleDet(psi, (nup, ndown), nbasis, handler=handler, verbose=False)
    trial.half_rotate(hamiltonian, handler.scomm)
    walkers = UHFWalkers(psi, nup, ndown, nbasis, nwalkers, handler)
    walkers.build(trial)

    prop = HubbardSingleSite(0.005, spin_decomp=spin_decomp)
    prop.build(hamiltonian, trial, walkers, handler)
    random_fields = rng.random((nbasis, nwalkers))
    return prop, hamiltonian, trial, walkers, random_fields


@pytest.mark.unit
@pytest.mark.parametrize("U, spin_decomp", [(4.0, None), (-4.0, None), (4.0, False)])
def test_hubbard_single_site_cpu_numba_matches_legacy(U, spin_decomp):
    prop, hamiltonian, trial, walkers, random_fields = _build_case(U, spin_decomp)
    ref = _copy_walkers(walkers)
    opt = _copy_walkers(walkers)

    prop._legacy_two_body(ref, hamiltonian, trial, random_fields=random_fields)
    prop._cpu_numba_two_body(opt, hamiltonian, trial, random_fields)

    numpy.testing.assert_allclose(opt.weight, ref.weight, rtol=1.0e-10, atol=1.0e-10)
    numpy.testing.assert_allclose(opt.ovlp, ref.ovlp, rtol=1.0e-10, atol=1.0e-10)
    numpy.testing.assert_allclose(opt.phia, ref.phia, rtol=1.0e-10, atol=1.0e-10)
    numpy.testing.assert_allclose(opt.phib, ref.phib, rtol=1.0e-10, atol=1.0e-10)
    numpy.testing.assert_allclose(opt.inv_ovlp_a, ref.inv_ovlp_a, rtol=1.0e-10, atol=1.0e-10)
    numpy.testing.assert_allclose(opt.inv_ovlp_b, ref.inv_ovlp_b, rtol=1.0e-10, atol=1.0e-10)


@pytest.mark.unit
@pytest.mark.parametrize("path", ["_legacy_two_body", "_cpu_numba_two_body", "_einsum_two_body"])
def test_hubbard_two_body_tracks_overlap_charge_decoupling(path):
    # U < 0 selects the charge decoupling, whose Hirsch constant aux_wfac is field dependent;
    # it must enter the weight, not the tracked overlap <psi_T|phi>.
    prop, hamiltonian, trial, walkers, random_fields = _build_case(U=-4.0)
    assert not prop.spin_decomp
    assert not numpy.allclose(prop.aux_wfac, 1.0)
    getattr(prop, path)(walkers, hamiltonian, trial, random_fields)
    alive = numpy.abs(walkers.weight) > 0.0
    assert numpy.any(alive)
    numpy.testing.assert_allclose(
        walkers.ovlp[alive], trial.calc_overlap(walkers)[alive], rtol=1.0e-9
    )
    ovlp_a = numpy.einsum("mi,wmj->wij", trial.psi0a.conj(), walkers.phia)
    numpy.testing.assert_allclose(
        walkers.inv_ovlp_a[alive], numpy.linalg.inv(ovlp_a)[alive], atol=1.0e-9
    )


@pytest.mark.unit
def test_hirsch_build_rejects_unsupported_setups():
    prop, hamiltonian, trial, walkers, _ = _build_case()
    walkers.rhf = True
    with pytest.raises(ValueError):
        HubbardSingleSite(0.005).build(hamiltonian, trial, walkers, walkers.mpi_handler)

    with pytest.raises(ValueError):
        _build_case(U=-4.0, spin_decomp=True)

    polarized = UHFWalkers(trial.psi0a, trial.nalpha, 0, walkers.nbasis, 2, walkers.mpi_handler)
    with pytest.raises(ValueError):
        HubbardSingleSite(0.005).build(hamiltonian, trial, polarized, walkers.mpi_handler)
