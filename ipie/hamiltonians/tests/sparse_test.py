import pytest
import numpy
from ipie.utils.linalg import modified_cholesky
from ipie.utils.testing import generate_hamiltonian
from ipie.hamiltonians.sparse import SparseComplexChol, SparseRealChol

# TODO: write tests with sparse hamiltonian.
@pytest.mark.unit
def test_sparse_complex_hamiltonian():
    numpy.random.seed(7)
    nmo = 13
    nelec = (4, 3)
    h1e, chol, nuc, eri = generate_hamiltonian(nmo, nelec, cplx=True, sparse=True, sym=4)
    nchol = chol.shape[0]
    chol = chol.reshape((nchol, nmo**2)).T.copy()
    ham = SparseComplexChol(numpy.array([h1e, h1e], dtype=h1e.dtype), chol, nuc, verbose=True)


@pytest.mark.unit
def test_sparse_real_hamiltonian():
    numpy.random.seed(7)
    nmo = 13
    nelec = (4, 3)
    h1e, chol, nuc, eri = generate_hamiltonian(nmo, nelec, cplx=False, sparse=True, sym=8)
    nchol = chol.shape[0]
    chol = chol.reshape((nchol, nmo**2)).T.copy()
    ham = SparseRealChol(numpy.array([h1e, h1e], dtype=h1e.dtype), chol, nuc, verbose=True)


if __name__ == "__main__":
    test_sparse_complex_hamiltonian()
    test_sparse_real_hamiltonian()
