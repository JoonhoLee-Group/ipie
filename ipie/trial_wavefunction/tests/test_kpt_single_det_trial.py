import numpy
import pytest

from ipie.config import MPI
from ipie.hamiltonians.kpt_hamiltonian import KptComplexCholSymm
from ipie.hamiltonians.kpt_isdf_hamiltonian import KptISDF
from ipie.trial_wavefunction.single_det_kpt import KptSingleDet
from ipie.utils.testing import (
    _blockdiag_k_orbitals,
    gen_random_test_input_kpt,
    gen_random_test_input_kpt_isdf,
    random_occupations,
    shaped_normal,
)


@pytest.mark.unit
def test_kpt_single_det():
    kmesh = (2, 1, 1)
    nk = kmesh[0] * kmesh[1] * kmesh[2]
    nbasis = 8
    naux = 4 * nbasis
    nalpha, nbeta = (3, 2)

    h1e, chol, wavefunction, kpts = gen_random_test_input_kpt(
        kmesh, nbasis, (nalpha, nbeta), naux, seed=7
    )
    trial = KptSingleDet(wavefunction, nk, (nalpha, nbeta), nbasis)
    hamiltonian = KptComplexCholSymm(
        h1e=numpy.array([h1e, h1e], dtype=numpy.complex128),
        chol=numpy.array(chol, dtype=numpy.complex128),
        kpts=kpts,
    )

    assert trial.num_elec == (nalpha, nbeta)
    assert trial.nbasis == nbasis
    assert trial.nk == nk
    assert trial.num_dets == 1
    assert _blockdiag_k_orbitals(trial.psi0a).shape == (nk * nbasis, nk * nalpha)
    assert _blockdiag_k_orbitals(trial.psi0b).shape == (nk * nbasis, nk * nbeta)

    trial.build()
    trial.half_rotate(hamiltonian, comm=MPI.COMM_WORLD)

    assert trial._rchola.shape == (hamiltonian.unique_nk, nk, nalpha, naux, nbasis)
    assert trial._rcholb.shape == (hamiltonian.unique_nk, nk, nbeta, naux, nbasis)
    assert trial._rcholbara.shape == (hamiltonian.unique_nk, nk, nbasis, naux, nalpha)
    assert trial._rcholbarb.shape == (hamiltonian.unique_nk, nk, nbasis, naux, nbeta)
    assert trial._rH1a.shape == (nk, nalpha, nbasis)
    assert trial._rH1b.shape == (nk, nbeta, nbasis)


@pytest.mark.unit
def test_kpt_single_det_uneven_occ():
    kmesh = (2, 2, 1)
    nk = kmesh[0] * kmesh[1] * kmesh[2]
    nbasis = 8
    naux = 4 * nbasis
    nalpha, nbeta = (3, 2)

    h1e, chol, wavefunction, kpts = gen_random_test_input_kpt(
        kmesh, nbasis, (nalpha, nbeta), naux, seed=11
    )
    noccas = random_occupations(nalpha, nk, alpha=0.3, seed=17)
    noccbs = random_occupations(nbeta, nk, alpha=0.3, seed=23)

    trial = KptSingleDet(
        wavefunction,
        nk,
        (nalpha, nbeta),
        nbasis,
        noccas=noccas,
        noccbs=noccbs,
    )
    hamiltonian = KptComplexCholSymm(
        h1e=numpy.array([h1e, h1e], dtype=numpy.complex128),
        chol=numpy.array(chol, dtype=numpy.complex128),
        kpts=kpts,
    )

    assert trial.noccas is not None
    assert trial.noccbs is not None
    numpy.testing.assert_array_equal(trial.noccas, noccas)
    numpy.testing.assert_array_equal(trial.noccbs, noccbs)
    assert numpy.any(noccas != nalpha)
    assert numpy.any(noccbs != nbeta)

    trial.build()
    trial.half_rotate(hamiltonian, comm=MPI.COMM_WORLD)

    assert trial._rchola.shape == (hamiltonian.unique_nk, nk, nalpha, naux, nbasis)
    assert trial._rcholb.shape == (hamiltonian.unique_nk, nk, nbeta, naux, nbasis)
    assert trial._rcholbara.shape == (hamiltonian.unique_nk, nk, nbasis, naux, nalpha)
    assert trial._rcholbarb.shape == (hamiltonian.unique_nk, nk, nbasis, naux, nbeta)
    assert trial._rH1a.shape == (nk, nalpha, nbasis)
    assert trial._rH1b.shape == (nk, nbeta, nbasis)


@pytest.mark.unit
def test_kpt_single_det_isdf():
    kmesh = (2, 1, 1)
    nk = kmesh[0] * kmesh[1] * kmesh[2]
    nbasis = 8
    naux = 4 * nbasis
    nalpha, nbeta = (3, 2)

    h1e, MPQ, cgto, wavefunction, kpts = gen_random_test_input_kpt_isdf(
        kmesh, nbasis, (nalpha, nbeta), naux, seed=29
    )
    nq = MPQ.shape[0]
    nisdf = cgto.shape[1]
    cholM = shaped_normal((nq, nisdf, naux), cmplx=True, seed=31)

    trial = KptSingleDet(wavefunction, nk, (nalpha, nbeta), nbasis)
    hamiltonian = KptISDF(
        h1e=numpy.array([h1e, h1e], dtype=numpy.complex128),
        MPQ=numpy.array(MPQ, dtype=numpy.complex128),
        cholM=numpy.array(cholM, dtype=numpy.complex128),
        cgto=numpy.array(cgto, dtype=numpy.complex128),
        kpts=kpts,
    )

    assert trial.num_elec == (nalpha, nbeta)
    assert trial.nbasis == nbasis
    assert trial.nk == nk
    assert trial.num_dets == 1

    trial.build()
    trial.half_rotate(hamiltonian, comm=MPI.COMM_WORLD)

    assert trial._rcgtoa.shape == (nk, nisdf, nalpha)
    assert trial._rcgtob.shape == (nk, nisdf, nbeta)
    assert trial._rH1a.shape == (nk, nalpha, nbasis)
    assert trial._rH1b.shape == (nk, nbeta, nbasis)


@pytest.mark.unit
def test_kpt_single_det_isdf_uneven_occ():
    kmesh = (2, 2, 1)
    nk = kmesh[0] * kmesh[1] * kmesh[2]
    nbasis = 8
    naux = 4 * nbasis
    nalpha, nbeta = (3, 2)

    h1e, MPQ, cgto, wavefunction, kpts = gen_random_test_input_kpt_isdf(
        kmesh, nbasis, (nalpha, nbeta), naux, seed=41
    )
    nq = MPQ.shape[0]
    nisdf = cgto.shape[1]
    cholM = shaped_normal((nq, nisdf, naux), cmplx=True, seed=43)
    noccas = random_occupations(nalpha, nk, alpha=0.3, seed=47)
    noccbs = random_occupations(nbeta, nk, alpha=0.3, seed=53)

    trial = KptSingleDet(
        wavefunction,
        nk,
        (nalpha, nbeta),
        nbasis,
        noccas=noccas,
        noccbs=noccbs,
    )
    hamiltonian = KptISDF(
        h1e=numpy.array([h1e, h1e], dtype=numpy.complex128),
        MPQ=numpy.array(MPQ, dtype=numpy.complex128),
        cholM=numpy.array(cholM, dtype=numpy.complex128),
        cgto=numpy.array(cgto, dtype=numpy.complex128),
        kpts=kpts,
    )

    numpy.testing.assert_array_equal(trial.noccas, noccas)
    numpy.testing.assert_array_equal(trial.noccbs, noccbs)
    assert numpy.any(noccas != nalpha)
    assert numpy.any(noccbs != nbeta)

    trial.build()
    trial.half_rotate(hamiltonian, comm=MPI.COMM_WORLD)

    assert trial._rcgtoa.shape == (nk, nisdf, nalpha)
    assert trial._rcgtob.shape == (nk, nisdf, nbeta)
    assert trial._rH1a.shape == (nk, nalpha, nbasis)
    assert trial._rH1b.shape == (nk, nbeta, nbasis)
