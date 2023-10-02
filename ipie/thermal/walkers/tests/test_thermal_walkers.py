import numpy
import pytest

from ipie.hamiltonians.ueg import UEG as HamUEG
from ipie.systems.ueg import UEG
from ipie.thermal.trial.one_body import OneBody
from ipie.thermal.walkers.uhf_walkers import UHFThermalWalkers

from ipie.legacy.hamiltonians.ueg import UEG as LegacyHamUEG
from ipie.legacy.trial_density_matrices.onebody import OneBody as LegacyOneBody
from ipie.legacy.walkers.thermal import ThermalWalker

try:
    from ipie.thermal.estimators.ueg import local_energy_ueg
except ImportError as e:
    print(e)

@pytest.mark.unit
def test_thermal_walkers():
    rs = 2
    ecut = 2
    nup = 7
    ndown = 7
    nelec = (nup, ndown)
    options = {"rs": rs, "nup": nup, "ndown": ndown, "ecut": ecut, "write_integrals": False}

    nwalkers = 5

    system = UEG(options=options)
    hamiltonian = HamUEG(system, options=options)
    
    legacy_hamiltonian = LegacyHamUEG(system, options=options)
    

    beta = 1
    dt = 0.05
    verbose = True
    trial = OneBody(hamiltonian, nelec, beta, dt, verbose=verbose)
    nbasis = trial.dmat.shape[-1]

    legacy_trial = LegacyOneBody(system, hamiltonian, beta, 0.05)
    legacy_walkers = [ThermalWalker(system, legacy_hamiltonian, legacy_trial, verbose=i == 0) for i in range(nwalkers)]

    walkers = UHFThermalWalkers(trial, nbasis, nwalkers, verbose=1)

    for iw in range(nwalkers):
        legacy_eloc = local_energy_ueg(system,hamiltonian, legacy_walkers[iw].G)
        eloc = local_energy_ueg(system,hamiltonian, numpy.array([walkers.Ga[iw],walkers.Gb[iw]]))
        print(eloc)
        numpy.testing.assert_almost_equal(legacy_eloc, eloc, decimal=10)

        numpy.testing.assert_almost_equal(legacy_walkers[iw].G[0], walkers.Ga[iw], decimal=10)
        numpy.testing.assert_almost_equal(legacy_walkers[iw].G[1], walkers.Gb[iw], decimal=10)
        numpy.testing.assert_almost_equal(legacy_walkers[iw].stack.ovlp[0], walkers.stack[iw].ovlp[0], decimal=10)
        numpy.testing.assert_almost_equal(legacy_walkers[iw].stack.ovlp[1], walkers.stack[iw].ovlp[1], decimal=10)

if __name__ == "__main__":
    test_thermal_walkers()
