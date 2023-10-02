import numpy
import pytest

from ipie.hamiltonians.ueg import UEG as HamUEG
from ipie.systems.ueg import UEG
from ipie.thermal.trial.one_body import OneBody
from ipie.thermal.walkers.uhf_walkers import UHFThermalWalkers


@pytest.mark.unit
def test_thermal_walkers():
    rs = 2
    ecut = 2
    nup = 7
    ndown = 7
    nelec = (nup, ndown)
    options = {"rs": rs, "nup": nup, "ndown": ndown, "ecut": ecut, "write_integrals": False}

    system = UEG(options=options)
    hamiltonian = HamUEG(system, options=options)

    beta = 1
    dt = 0.05
    verbose = True
    trial = OneBody(hamiltonian, nelec, beta, dt, verbose=verbose)

    nwalkers = 10
    nbasis = trial.dmat.shape[-1]
    walkers = [UHFThermalWalkers(trial, nup, ndown, nbasis, nwalkers, verbose=i == 0) for i in range(nwalkers)]


if __name__ == "__main__":
    test_thermal_walkers()
