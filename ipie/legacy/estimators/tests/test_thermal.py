import numpy
import pytest

from ipie.legacy.estimators.thermal import entropy
from ipie.legacy.hamiltonians.ueg import UEG as HamUEG
from ipie.legacy.systems.ueg import UEG


def fermi_factor(ek, beta, mu):
    return 1.0 / (numpy.exp(beta * (ek - mu)) + 1.0)


@pytest.mark.unit
def test_entropy():
    system = UEG({"rs": 2.0, "ecut": 2.5, "nup": 7, "ndown": 7})
    ham = HamUEG(system, {"rs": 2.0, "ecut": 2.5, "nup": 7, "ndown": 7})
    mu = -0.9
    beta = 1.0
    S = entropy(1.0, mu, ham.H1)
    eks = ham.H1[0].diagonal()
    N = 2 * sum(fermi_factor(ek, beta, mu) for ek in eks)
    E = 2 * sum(ek * fermi_factor(ek, beta, mu) for ek in eks)
    O = -2 * (sum(numpy.log(1.0 + numpy.exp(-beta * (ek - mu))) for ek in eks))
    assert S - (E - O - mu * N) == pytest.approx(0.0)
