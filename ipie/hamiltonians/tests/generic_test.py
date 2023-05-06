import pytest
import numpy
from ipie.utils.linalg import modified_cholesky
from ipie.utils.testing import generate_hamiltonian
from ipie.hamiltonians.generic import GenericRealChol, GenericComplexChol


@pytest.mark.unit
def test_real_modified_cholesky():
    numpy.random.seed(7)
    nmo = 13
    nelec = (4, 3)
    # ERI is already in the right form for cholesky decomposition
    _, _, _, eri = generate_hamiltonian(nmo, nelec, cplx=False, sym=8)

    eri = eri.reshape((nmo**2, nmo**2)).T
    chol = modified_cholesky(eri, tol=1e-8, verbose=False, cmax=30)
    nchol = chol.shape[0]

    chol = chol.reshape((nchol, nmo**2))
    eri_chol = chol.T @ chol.conj()
    numpy.testing.assert_allclose(eri_chol, eri, atol=1e-8)


@pytest.mark.unit
def test_complex_modified_cholesky():
    numpy.random.seed(7)
    nmo = 13
    nelec = (4, 3)
    # ERI is already in the right form for cholesky decomposition
    _, _, _, eri = generate_hamiltonian(nmo, nelec, cplx=True, sym=4)

    eri = eri.reshape((nmo**2, nmo**2)).T
    chol = modified_cholesky(eri, tol=1e-8, verbose=False, cmax=30)
    nchol = chol.shape[0]

    chol = chol.reshape((nchol, nmo**2))
    eri_chol = chol.T @ chol.conj()
    numpy.testing.assert_allclose(eri_chol, eri, atol=1e-8)


@pytest.mark.unit
def test_complex_hamiltonian():
    numpy.random.seed(7)
    nmo = 13
    nelec = (4, 3)
    h1e, chol, nuc, eri = generate_hamiltonian(nmo, nelec, cplx=True, sym=4)
    nchol = chol.shape[0]
    chol = chol.reshape((nchol, nmo**2)).T.copy()
    ham = GenericComplexChol(numpy.array([h1e, h1e], dtype=h1e.dtype), chol, nuc, verbose=True)


@pytest.mark.unit
def test_real_hamiltonian():
    numpy.random.seed(7)
    nmo = 13
    nelec = (4, 3)
    h1e, chol, nuc, eri = generate_hamiltonian(nmo, nelec, cplx=False, sym=8)
    nchol = chol.shape[0]
    chol = chol.reshape((nchol, nmo**2)).T.copy()
    ham = GenericRealChol(numpy.array([h1e, h1e], dtype=h1e.dtype), chol, nuc, verbose=True)


if __name__ == "__main__":
    test_real_modified_cholesky()
    test_complex_modified_cholesky()
    test_complex_hamiltonian()
    test_real_hamiltonian()
