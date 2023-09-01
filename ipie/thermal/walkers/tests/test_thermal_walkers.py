import numpy
import pytest

from ipie.legacy.hamiltonians.ueg import UEG as HamUEG
from ipie.legacy.systems.ueg import UEG
from ipie.thermal.trial.onebody import OneBody
from ipie.legacy.walkers.thermal import ThermalWalker


@pytest.mark.unit
def test_thermal_walkers():
    options = {"rs": 2, "nup": 7, "ndown": 7, "ecut": 2, "write_integrals": False}
    system = UEG(options=options)
    hamiltonian = HamUEG(system, options=options)

    beta = 1
    trial = OneBody(system, hamiltonian, beta, 0.05)

    nwalkers = 10
    walkers = [ThermalWalker(system, hamiltonian, trial, verbose=i == 0) for i in range(nwalkers)]


if __name__ == "__main__":
    test_thermal_walkers()
