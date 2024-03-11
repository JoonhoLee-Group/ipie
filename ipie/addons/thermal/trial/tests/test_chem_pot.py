import numpy
import scipy.linalg
import pytest

from ipie.addons.thermal.trial.chem_pot import find_chemical_potential
from ipie.legacy.trial_density_matrices.chem_pot import find_chemical_potential as legacy_find_chemical_potential


@pytest.mark.unit
def test_find_chemical_potential():
    dt = 0.01
    beta = 1
    nstack = 3
    stack_length = 20
    nav = 7
    nbsf = 14
    alt_convention = False

    dtau = dt * nstack
    h1e = numpy.random.random((nbsf, nbsf))
    rho = numpy.array([scipy.linalg.expm(-dtau * h1e),
                       scipy.linalg.expm(-dtau * h1e)])

    mu = find_chemical_potential(alt_convention, rho, dt, stack_length, nav)
    legacy_mu = legacy_find_chemical_potential(alt_convention, rho, dt, stack_length, nav)

    numpy.testing.assert_allclose(mu, legacy_mu)


if __name__ == '__main__':
    test_find_chemical_potential()
    
   

