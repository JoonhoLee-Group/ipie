import pytest
import numpy
from pauxy.systems.hubbard import Hubbard, decode_basis, encode_basis
from pauxy.trial_wavefunction.uhf import UHF
from pauxy.estimators.greens_function import gab

@pytest.mark.unit
def test_uhf():
    system = Hubbard({'nup': 8, 'ndown': 8, 'U': 4.0, 'nx': 4, 'ny': 4})
    numpy.random.seed(7)
    trial = UHF(system)
    assert trial.emin == pytest.approx(-22.638405458100653)

@pytest.mark.unit
def test_uhf_checkerboard():
    system = Hubbard({'nup': 8, 'ndown': 8, 'U': 4.0, 'nx': 4, 'ny': 4})
    numpy.random.seed(7)
    trial = UHF(system, trial={'ueff': 4.0})
    assert trial.emin == pytest.approx(-12.56655451978628)
    def initialise(self, nbasis, nup, ndown):
        wfn = numpy.zeros((system.nbasis,system.ne), dtype=numpy.complex128)
        count = 0
        Ga = numpy.eye(system.nbasis, dtype=numpy.complex128)
        Gb = numpy.eye(system.nbasis, dtype=numpy.complex128)
        nalpha = 0
        nbeta = 0
        for i in range(system.nbasis):
            x, y = decode_basis(4,4,i)
            if x % 2 == 0 and y % 2 == 0:
                wfn[i,nalpha] = 1.0
                nalpha += 1
            elif x % 2 == 0 and y % 2 == 1:
                wfn[i,system.nup+nbeta] = -1.0
                nbeta += 1
            elif x % 2 == 1 and y % 2 == 0:
                wfn[i,system.nup+nbeta] = -1.0
                nbeta += 1
            elif x % 2 == 1 and y % 2 == 1:
                wfn[i,nalpha] = 1.0
                nalpha += 1
        return wfn, 10
    psi, x = initialise(None, 16, 4, 4)
    # trial.initialise = initialise.__get__(trial, UHF)
    G = gab(psi[:,system.nup:], psi[:,system.nup:])
    # print(G.diagonal())
    trial.initialise = lambda nbasis, nup, ndown: initialise(trial, nbasis, nup, ndown)
    psi, eigs, emin, error, nav = trial.find_uhf_wfn(system, trial.ueff,
                                                     trial.ninitial, trial.nconv,
                                                     trial.alpha, trial.deps, False)
    assert trial.emin == pytest.approx(-12.56655451978628)
