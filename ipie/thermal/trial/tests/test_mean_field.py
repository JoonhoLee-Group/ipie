import numpy
import pytest

from pyscf import gto, scf, lo

from ipie.systems.generic import Generic
from ipie.hamiltonians.utils import get_hamiltonian
from ipie.thermal.trial.mean_field import MeanField


def setup_objs():
    nocca = 5
    noccb = 5
    nelec = nocca + noccb
    r0 = 1.75
    mol = gto.M(
            atom=[("H", i * r0, 0, 0) for i in range(nelec)],
            basis='sto-6g',
            unit='Bohr',
            verbose=5)
    
    mu = -10.
    path = "/Users/shufay/Documents/in_prep/ft_moire/ipie/ipie/thermal/tests/"
    options = {
        "hamiltonian": {
            "name": "Generic",
            "integrals": path + "reference_data/generic_integrals.h5",
            "_alt_convention": False,
            "symmetry": False,
            "sparse": False,
            "mu": mu
        },
    }

    # Test.
    print('\n----------------------------')
    print('Constructing test objects...')
    print('----------------------------')
    system = Generic(mol.nelec)
    hamiltonian = get_hamiltonian(system, options["hamiltonian"])
    objs = {'mol': mol, 
            'hamiltonian': hamiltonian}

    return objs


@pytest.mark.unit
def test_mean_field():
    beta = 0.1
    dt = 0.01
    verbose = True
    objs = setup_objs()
    mol = objs['mol']
    hamiltonian = objs['hamiltonian']
    nbasis = hamiltonian.nbasis
    trial = MeanField(hamiltonian, mol.nelec, beta, dt, verbose=verbose)

    assert trial.nelec == mol.nelec
    numpy.testing.assert_almost_equal(trial.nav, numpy.sum(mol.nelec), decimal=6)
    assert trial.rho.shape == (2, nbasis, nbasis)
    assert trial.dmat.shape == (2, nbasis, nbasis)
    assert trial.P.shape == (2, nbasis, nbasis)
    assert trial.G.shape == (2, nbasis, nbasis)


if __name__ == '__main__':
    test_mean_field()
    
   

