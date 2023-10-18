import numpy
import pytest

from ipie.systems.ueg import UEG
from ipie.hamiltonians.ueg import UEG as HamUEG
from ipie.thermal.trial.one_body import OneBody
from ipie.thermal.walkers.uhf_walkers import UHFThermalWalkers
from ipie.thermal.estimators.ueg import local_energy_ueg
from ipie.thermal.estimators.thermal import one_rdm_from_G

from ipie.legacy.systems.ueg import UEG as LegacyUEG
from ipie.legacy.hamiltonians.ueg import UEG as LegacyHamUEG
from ipie.legacy.trial_density_matrices.onebody import OneBody as LegacyOneBody
from ipie.legacy.walkers.thermal import ThermalWalker
from ipie.legacy.estimators.ueg import local_energy_ueg as legacy_local_energy_ueg
from ipie.legacy.estimators.thermal import one_rdm_from_G as legacy_one_rdm_from_G

@pytest.mark.unit
def test_thermal_walkers():
    rs = 2
    ecut = 2
    nup = 7
    ndown = 7
    nelec = (nup, ndown)

    nwalkers = 5
    beta = 1
    dt = 0.05
    lowrank = True
    verbose = True
    options = {"rs": rs, "nup": nup, "ndown": ndown, "ecut": ecut, 
               "write_integrals": False, "low_rank" : lowrank}
    
    # Test.
    print('\n----------------------------')
    print('Constructing test objects...')
    print('----------------------------')
    system = UEG(options=options)
    hamiltonian = HamUEG(system, options=options)
    trial = OneBody(hamiltonian, nelec, beta, dt, verbose=verbose)
    nbasis = trial.dmat.shape[-1]
    walkers = UHFThermalWalkers(trial, nbasis, nwalkers, lowrank=lowrank, 
                                verbose=verbose)

    # Legacy.
    print('\n------------------------------')
    print('Constructing legacy objects...')
    print('------------------------------')
    legacy_system = UEG(options=options)
    legacy_hamiltonian = LegacyHamUEG(legacy_system, options=options)
    legacy_trial = LegacyOneBody(legacy_system, legacy_hamiltonian, beta, dt, verbose=verbose)
    legacy_walkers = [
            ThermalWalker(
                legacy_system, legacy_hamiltonian, legacy_trial, 
                walker_opts=options, verbose=i == 0) for i in range(nwalkers)]

    for iw in range(nwalkers):
        P = one_rdm_from_G(numpy.array([walkers.Ga[iw], walkers.Gb[iw]])) 
        eloc = local_energy_ueg(system, hamiltonian, P)

        legacy_P = legacy_one_rdm_from_G(numpy.array(legacy_walkers[iw].G))
        legacy_eloc = legacy_local_energy_ueg(legacy_system, legacy_hamiltonian, legacy_P)

        print(f'\niw = {iw}')
        print(f'eloc = {eloc}')
        print(f'legacy_eloc = {legacy_eloc}')

        numpy.testing.assert_almost_equal(legacy_eloc, eloc, decimal=10)
        numpy.testing.assert_almost_equal(legacy_walkers[iw].G[0], walkers.Ga[iw], decimal=10)
        numpy.testing.assert_almost_equal(legacy_walkers[iw].G[1], walkers.Gb[iw], decimal=10)
        numpy.testing.assert_almost_equal(legacy_walkers[iw].stack.ovlp[0], walkers.stack[iw].ovlp[0], decimal=10)
        numpy.testing.assert_almost_equal(legacy_walkers[iw].stack.ovlp[1], walkers.stack[iw].ovlp[1], decimal=10)

if __name__ == "__main__":
    test_thermal_walkers()
