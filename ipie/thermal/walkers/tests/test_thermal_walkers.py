import numpy
import pytest

from ipie.systems.generic import Generic
from ipie.utils.testing import generate_hamiltonian
from ipie.hamiltonians.generic import Generic as HamGeneric
from ipie.hamiltonians.utils import get_hamiltonian
from ipie.thermal.trial.mean_field import MeanField
from ipie.thermal.trial.one_body import OneBody
from ipie.thermal.walkers.uhf_walkers import UHFThermalWalkers
from ipie.thermal.estimators.generic import local_energy_generic_cholesky
from ipie.thermal.estimators.thermal import one_rdm_from_G

from ipie.legacy.hamiltonians._generic import Generic as LegacyHamGeneric
from ipie.legacy.trial_density_matrices.mean_field import MeanField as LegacyMeanField
from ipie.legacy.trial_density_matrices.onebody import OneBody as LegacyOneBody
from ipie.legacy.walkers.thermal import ThermalWalker
from ipie.legacy.estimators.generic import local_energy_generic_cholesky as legacy_local_energy_generic_cholesky
from ipie.legacy.estimators.thermal import one_rdm_from_G as legacy_one_rdm_from_G


def setup_objs(mf_trial=False, seed=None):
    nocca = 5
    noccb = 5
    nelec = (nocca, noccb)
    nbasis = 10

    mu = -10.
    beta = 0.02
    dt = 0.01
    nwalkers = 2
    numpy.random.seed(seed)

    lowrank = False
    verbose = True
    complex_integrals = False
    sym = 8
    if complex_integrals: sym = 4

    options = {
        "walkers": {
            "low_rank": lowrank
        },

        "hamiltonian": {
            "name": "Generic",
            "_alt_convention": False,
            "sparse": False,
            "mu": mu
        }
    }

    # Test.
    print('\n----------------------------')
    print('Constructing test objects...')
    print('----------------------------')
    h1e, chol, _, _ = generate_hamiltonian(nbasis, nelec, cplx=complex_integrals, 
                                           sym=sym, tol=1e-10)
    hamiltonian = HamGeneric(h1e=numpy.array([h1e, h1e]),
                             chol=chol.reshape((-1, nbasis**2)).T.copy(),
                             ecore=0)
    hamiltonian.name = options["hamiltonian"]["name"]
    hamiltonian._alt_convention = options["hamiltonian"]["_alt_convention"]
    hamiltonian.sparse = options["hamiltonian"]["sparse"]

    trial = OneBody(hamiltonian, nelec, beta, dt, verbose=verbose)

    if mf_trial:
        trial = MeanField(hamiltonian, nelec, beta, dt, verbose=verbose)

    walkers = UHFThermalWalkers(trial, nbasis, nwalkers, lowrank=lowrank, 
                                verbose=verbose)
    
    # Legacy.
    print('\n------------------------------')
    print('Constructing legacy objects...')
    print('------------------------------')
    legacy_system = Generic(nelec, verbose=verbose)
    legacy_system.mu = mu
    legacy_hamiltonian = LegacyHamGeneric(
                            h1e=hamiltonian.H1,
                            chol=hamiltonian.chol,
                            ecore=hamiltonian.ecore)
    legacy_hamiltonian.hs_pot = numpy.copy(hamiltonian.chol)
    legacy_hamiltonian.hs_pot = legacy_hamiltonian.hs_pot.T.reshape(
            (hamiltonian.nchol, hamiltonian.nbasis, hamiltonian.nbasis))
    legacy_hamiltonian._alt_convention = options["hamiltonian"]["_alt_convention"]
    legacy_hamiltonian.mu = options["hamiltonian"]["mu"]
    legacy_trial = LegacyOneBody(legacy_system, legacy_hamiltonian, beta, dt, 
                                 verbose=verbose)
    if mf_trial:
        legacy_trial = LegacyMeanField(legacy_system, legacy_hamiltonian, beta, dt, 
                                       verbose=verbose)
        
    legacy_walkers = [
            ThermalWalker(
                legacy_system, legacy_hamiltonian, legacy_trial, 
                walker_opts=options, verbose=i == 0) for i in range(nwalkers)]

    objs = {'trial': trial,
            'hamiltonian': hamiltonian,
            'walkers': walkers}

    legacy_objs = {'system': legacy_system,
                   'trial': legacy_trial,
                   'hamiltonian': legacy_hamiltonian,
                   'walkers': legacy_walkers}

    return objs, legacy_objs
    

@pytest.mark.unit
def test_thermal_walkers():
    verbose = True
    seed = 7
    objs, legacy_objs = setup_objs(seed=seed)
    trial = objs['trial']
    hamiltonian = objs['hamiltonian']
    walkers = objs['walkers']
    
    legacy_system = legacy_objs['system']
    legacy_trial = legacy_objs['trial']
    legacy_hamiltonian = legacy_objs['hamiltonian']
    legacy_walkers = legacy_objs['walkers']
    
    for iw in range(walkers.nwalkers):
        P = one_rdm_from_G(numpy.array([walkers.Ga[iw], walkers.Gb[iw]])) 
        eloc = local_energy_generic_cholesky(hamiltonian, P)

        legacy_P = legacy_one_rdm_from_G(numpy.array(legacy_walkers[iw].G))
        legacy_eloc = legacy_local_energy_generic_cholesky(
                        legacy_system, legacy_hamiltonian, legacy_P)

        if verbose:
            print(f'\niw = {iw}')
            print(f'eloc = \n{eloc}\n')
            print(f'legacy_eloc = \n{legacy_eloc}\n')
            print(f'walkers.weight = \n{walkers.weight[iw]}\n')
            print(f'legacy_walkers.weight = \n{legacy_walkers[iw].weight}\n')

        numpy.testing.assert_almost_equal(legacy_eloc, eloc, decimal=10)
        numpy.testing.assert_almost_equal(legacy_walkers[iw].G[0], walkers.Ga[iw], decimal=10)
        numpy.testing.assert_almost_equal(legacy_walkers[iw].G[1], walkers.Gb[iw], decimal=10)
        numpy.testing.assert_almost_equal(legacy_walkers[iw].stack.ovlp[0], walkers.stack[iw].ovlp[0], decimal=10)
        numpy.testing.assert_almost_equal(legacy_walkers[iw].stack.ovlp[1], walkers.stack[iw].ovlp[1], decimal=10)


if __name__ == "__main__":
    test_thermal_walkers()
