import pytest
import numpy
from pauxy.systems.ueg import UEG
from pauxy.estimators.thermal import entropy


def fermi_factor(ek, beta, mu):
    return 1.0 / (numpy.exp(beta*(ek-mu)) + 1.0)

@pytest.mark.unit
def test_entropy():
    system = UEG({'rs': 2.0, 'ecut': 2.5, 'nup': 7, 'ndown': 7})
    mu = -0.9
    beta = 1.0
    S = entropy(1.0, mu, system.H1)
    eks = system.H1[0].diagonal()
    N = 2*sum(fermi_factor(ek, beta, mu) for ek in eks)
    E = 2*sum(ek*fermi_factor(ek, beta, mu) for ek in eks)
    O = -2*(sum(numpy.log(1.0+numpy.exp(-beta*(ek-mu))) for ek in eks))
    assert S-(E-O-mu*N) == pytest.approx(0.0)
