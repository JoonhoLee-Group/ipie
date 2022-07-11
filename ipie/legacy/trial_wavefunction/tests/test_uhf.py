import numpy
import pytest

from ipie.legacy.estimators.greens_function import gab
from ipie.legacy.hamiltonians.hubbard import (Hubbard, decode_basis,
                                              encode_basis)
from ipie.legacy.trial_wavefunction.hubbard_uhf import HubbardUHF
from ipie.systems.generic import Generic


@pytest.mark.unit
def test_uhf():
    sys = Generic((8, 8))
    ham = Hubbard({"nup": 8, "ndown": 8, "U": 4.0, "nx": 4, "ny": 4})
    numpy.random.seed(7)
    trial = HubbardUHF(sys, ham)
    assert trial.emin == pytest.approx(-22.638405458100653)


@pytest.mark.unit
def test_uhf_checkerboard():
    sys = Generic((8, 8))
    ham = Hubbard({"nup": 8, "ndown": 8, "U": 4.0, "nx": 4, "ny": 4})
    numpy.random.seed(7)
    trial = HubbardUHF(sys, ham, trial={"ueff": 4.0})
    assert trial.emin == pytest.approx(-12.56655451978628)

    def initialise(self, nbasis, nup, ndown):
        wfn = numpy.zeros((ham.nbasis, sys.ne), dtype=numpy.complex128)
        count = 0
        Ga = numpy.eye(ham.nbasis, dtype=numpy.complex128)
        Gb = numpy.eye(ham.nbasis, dtype=numpy.complex128)
        nalpha = 0
        nbeta = 0
        for i in range(ham.nbasis):
            x, y = decode_basis(4, 4, i)
            if x % 2 == 0 and y % 2 == 0:
                wfn[i, nalpha] = 1.0
                nalpha += 1
            elif x % 2 == 0 and y % 2 == 1:
                wfn[i, sys.nup + nbeta] = -1.0
                nbeta += 1
            elif x % 2 == 1 and y % 2 == 0:
                wfn[i, sys.nup + nbeta] = -1.0
                nbeta += 1
            elif x % 2 == 1 and y % 2 == 1:
                wfn[i, nalpha] = 1.0
                nalpha += 1
        return wfn, 10

    psi, x = initialise(None, 16, 4, 4)
    # trial.initialise = initialise.__get__(trial, UHF)
    G = gab(psi[:, sys.nup :], psi[:, sys.nup :])
    # print(G.diagonal())
    trial.initialise = lambda nbasis, nup, ndown: initialise(trial, nbasis, nup, ndown)
    psi, eigs, emin, error, nav = trial.find_uhf_wfn(
        sys,
        ham,
        trial.ueff,
        trial.ninitial,
        trial.nconv,
        trial.alpha,
        trial.deps,
        False,
    )
    assert trial.emin == pytest.approx(-12.56655451978628)
