import pytest
import numpy

from ipie.systems.ueg import UEG as SysUEG
from ipie.hamiltonians.ueg import UEG

# TODO: write UEG test

@pytest.mark.unit
def test_hamiltonian():
    numpy.random.seed(7)
    nup = 7
    ndown = 5
    ne = nup + ndown
    rs = 1.
    mu = -1.
    ecut = 1.
    sys_opts = {
                "nup": nup,
                "ndown": ndown,
                "rs": rs,
                "mu": mu,
                "ecut": ecut
                }
    ham_opts = {} # Default
    sys = SysUEG(sys_opts)
    ham = UEG(sys, ham_opts, verbose=True)
    assert hasattr(ham, "h1e_mod") == True

@pytest.mark.unit
def test_skip_cholesky():
    numpy.random.seed(7)
    nup = 7
    ndown = 5
    ne = nup + ndown
    rs = 1.
    mu = -1.
    ecut = 1.
    sys_opts = {
                "nup": nup,
                "ndown": ndown,
                "rs": rs,
                "mu": mu,
                "ecut": ecut
                }
    ham_opts = {
                "skip_cholesky": True
                }
    sys = SysUEG(sys_opts)
    ham = UEG(sys, ham_opts, verbose=True)
    assert hasattr(ham, "h1e_mod") == False


if __name__ == "__main__":
    test_hamiltonian()
    test_skip_cholesky()
