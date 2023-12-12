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

from ipie.hamiltonians.generic import GenericComplexChol
from ipie.utils.misc import dotdict
from ipie.utils.testing import build_test_case_handlers


@pytest.mark.unit
def test_AB_cholesky():
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
        complex_integrals=True,
        complex_trial=True,
        trial_type="single_det",
    )

    ham = test_handler.hamiltonian
    chol = ham.chol  # M^2 x nchol
    nbasis = ham.nbasis
    nchol = ham.nchol

    chol = chol.reshape((nbasis, nbasis, nchol))

    A = 0.5 * (chol + chol.transpose((1, 0, 2)).conj())
    B = 1.0j * 0.5 * (chol - chol.transpose((1, 0, 2)).conj())

    A = A.reshape((nbasis * nbasis, nchol))
    B = B.reshape((nbasis * nbasis, nchol))

    numpy.testing.assert_allclose(A, ham.A, atol=1e-10)
    numpy.testing.assert_allclose(B, ham.B, atol=1e-10)


@pytest.mark.unit
def test_vhs_complex():
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
        complex_integrals=True,
        complex_trial=True,
        trial_type="single_det",
    )

    ham = test_handler.hamiltonian
    xshifted = numpy.random.normal(0.0, 1.0, nwalkers * ham.nfields).reshape(ham.nfields, nwalkers)

    vhs = test_handler.propagator.construct_VHS(ham, xshifted)

    isqrt_dt = 1.0j * numpy.sqrt(qmc.dt)

    nbasis = ham.nbasis
    nchol = ham.nchol

    vhs_ref = isqrt_dt * (ham.A.dot(xshifted[:nchol, :]) + ham.B.dot(xshifted[nchol:, :]))
    vhs_ref = vhs_ref.T.copy()
    vhs_ref = vhs_ref.reshape((nwalkers, nbasis, nbasis))

    numpy.testing.assert_allclose(vhs, vhs_ref, atol=1e-10)


@pytest.mark.unit
def test_vhs_complex_vs_real():
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
    chol = ham.chol

    xshifted = numpy.random.normal(0.0, 1.0, nwalkers * ham.nfields).reshape(ham.nfields, nwalkers)

    vhs = test_handler.propagator.construct_VHS(ham, xshifted)

    cx_chol = numpy.array(chol, dtype=numpy.complex128)
    cx_ham = GenericComplexChol(
        numpy.array(ham.H1, dtype=numpy.complex128), cx_chol, ham.ecore, verbose=False
    )
    nchol = cx_chol.shape[-1]
    cx_xshifted = numpy.zeros((cx_ham.nfields, nwalkers))
    cx_xshifted[:nchol, :] = xshifted.copy()
    cx_xshifted[nchol:, :] = numpy.random.normal(0.0, 1.0, nwalkers * ham.nfields).reshape(
        nchol, nwalkers
    )
    cx_vhs = test_handler.propagator.construct_VHS(cx_ham, cx_xshifted)
    numpy.testing.assert_allclose(vhs, cx_vhs, atol=1e-10)


@pytest.mark.unit
def test_vfb_complex_vs_real():
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

    # using half rotaiton
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
    trial = test_handler.trial
    trial.calc_greens_function(walkers, build_full=True)

    vfb = trial.calc_force_bias(ham, walkers, walkers.mpi_handler)

    nbasis = ham.nbasis
    nchol = ham.nchol

    chol = ham.chol
    G = walkers.Ga + walkers.Gb
    G = G.reshape((nwalkers, nbasis**2))
    vfb_ref = numpy.einsum("wi,ix->wx", G, chol)
    numpy.testing.assert_allclose(vfb, vfb_ref, atol=1e-10)

    cx_chol = numpy.array(chol, dtype=numpy.complex128)
    cx_ham = GenericComplexChol(
        numpy.array(ham.H1, dtype=numpy.complex128), cx_chol, ham.ecore, verbose=False
    )
    trial.half_rotate(cx_ham)
    cx_vfb = trial.calc_force_bias(cx_ham, walkers, walkers.mpi_handler)

    nfields = cx_ham.nfields
    cx_vfb_ref = numpy.zeros((nwalkers, nfields), dtype=numpy.complex128)
    cx_vfb_ref[:, :nchol] = numpy.einsum("wi,ix->wx", G, cx_ham.A)
    cx_vfb_ref[:, nchol:] = numpy.einsum("wi,ix->wx", G, cx_ham.B)

    numpy.testing.assert_allclose(cx_vfb_ref[:, :nchol], vfb_ref, atol=1e-10)
    numpy.testing.assert_allclose(cx_vfb[:, :nchol], vfb_ref, atol=1e-10)
    numpy.testing.assert_allclose(cx_vfb, cx_vfb_ref, atol=1e-10)


@pytest.mark.unit
def test_vfb_complex():
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

    # using half rotaiton
    test_handler = build_test_case_handlers(
        nelec,
        nmo,
        num_dets=1,
        options=qmc,
        seed=7,
        complex_integrals=True,
        complex_trial=True,
        trial_type="single_det",
    )

    ham = test_handler.hamiltonian
    walkers = test_handler.walkers
    trial = test_handler.trial

    trial.calc_greens_function(walkers, build_full=True)

    nbasis = ham.nbasis
    G = walkers.Ga + walkers.Gb
    G = G.reshape((nwalkers, nbasis**2))

    vfb = trial.calc_force_bias(ham, walkers, walkers.mpi_handler)

    nchol = ham.nchol
    nfields = ham.nfields

    vfb_ref = numpy.zeros((nwalkers, nfields), dtype=numpy.complex128)
    vfb_ref[:, :nchol] = numpy.einsum("wi,ix->wx", G, ham.A)
    vfb_ref[:, nchol:] = numpy.einsum("wi,ix->wx", G, ham.B)

    numpy.testing.assert_allclose(vfb, vfb_ref, atol=1e-10)


if __name__ == "__main__":
    test_AB_cholesky()
    test_vhs_complex()
    test_vhs_complex_vs_real()
    test_vfb_complex()
    test_vfb_complex_vs_real()
