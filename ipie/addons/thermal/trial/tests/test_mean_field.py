import numpy
import pytest

from ipie.systems.generic import Generic
from ipie.utils.testing import generate_hamiltonian
from ipie.hamiltonians.generic import Generic as HamGeneric
from ipie.addons.thermal.trial.mean_field import MeanField


@pytest.mark.unit
def test_mean_field():
    nup = 5
    ndown = 5
    nelec = (nup, ndown)
    nbasis = 10

    mu = -10.
    beta = 0.1
    timestep = 0.01

    complex_integrals = True
    verbose = True

    sym = 8
    if complex_integrals: sym = 4
    
    # Test.
    print('\n----------------------------')
    print('Constructing test objects...')
    print('----------------------------')
    system = Generic(nelec)
    h1e, chol, _, eri = generate_hamiltonian(nbasis, nelec, cplx=complex_integrals, 
                                             sym=sym, tol=1e-10)
    hamiltonian = HamGeneric(h1e=numpy.array([h1e, h1e]),
                             chol=chol.reshape((-1, nbasis**2)).T.copy(),
                             ecore=0)
    trial = MeanField(hamiltonian, nelec, beta, timestep, verbose=verbose)

    assert trial.nelec == nelec
    numpy.testing.assert_almost_equal(trial.nav, numpy.sum(nelec), decimal=6)
    assert trial.rho.shape == (2, nbasis, nbasis)
    assert trial.dmat.shape == (2, nbasis, nbasis)
    assert trial.P.shape == (2, nbasis, nbasis)
    assert trial.G.shape == (2, nbasis, nbasis)


if __name__ == '__main__':
    test_mean_field()
    
   

