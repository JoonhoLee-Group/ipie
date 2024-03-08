import numpy
import pytest
from typing import Union

from ipie.config import MPI
from ipie.thermal.estimators.generic import local_energy_generic_cholesky
from ipie.thermal.estimators.thermal import one_rdm_from_G

from ipie.legacy.estimators.generic import local_energy_generic_cholesky as legacy_local_energy_generic_cholesky
from ipie.legacy.estimators.thermal import one_rdm_from_G as legacy_one_rdm_from_G

from ipie.thermal.utils.testing import build_generic_test_case_handlers
from ipie.thermal.utils.legacy_testing import build_legacy_generic_test_case_handlers
from ipie.thermal.utils.legacy_testing import legacy_propagate_walkers

comm = MPI.COMM_WORLD

@pytest.mark.unit
def test_thermal_walkers_fullrank():
    # System params.
    nup = 5
    ndown = 5
    nelec = (nup, ndown)
    nbasis = 10

    # Thermal AFQMC params.
    mu = -10.
    beta = 0.1
    timestep = 0.01
    nwalkers = 10
    # Must be fixed at 1 for Thermal AFQMC--legacy code overides whatever input!
    nsteps_per_block = 1
    nblocks = 12
    stabilize_freq = 10
    pop_control_freq = 1
    pop_control_method = 'pair_branch'
    #pop_control_method = 'comb'
    lowrank = False

    verbose = True
    complex_integrals = False
    debug = True
    mf_trial = True
    propagate = False
    seed = 7
    numpy.random.seed(seed)

    options = {
                'nelec': nelec,
                'nbasis': nbasis,
                'mu': mu,
                'beta': beta,
                'timestep': timestep,
                'nwalkers': nwalkers,
                'seed': seed,
                'nsteps_per_block': nsteps_per_block,
                'nblocks': nblocks,
                'stabilize_freq': stabilize_freq,
                'pop_control_freq': pop_control_freq,
                'pop_control_method': pop_control_method,
                'lowrank': lowrank,
                'complex_integrals': complex_integrals,
                'mf_trial': mf_trial,
                'propagate': propagate,

                "hamiltonian": {
                    "name": "Generic",
                    "_alt_convention": False,
                    "sparse": False,
                    "mu": mu
                },
        
                "propagator": {
                    "optimised": False,
                    "free_projection": False
                },
            }
    
    # Test.
    print('\n----------------------------')
    print('Constructing test objects...')
    print('----------------------------')
    objs =  build_generic_test_case_handlers(options, seed, debug, verbose)
    trial = objs['trial']
    hamiltonian = objs['hamiltonian']
    walkers = objs['walkers']
    
    # Legacy.
    print('\n------------------------------')
    print('Constructing legacy objects...')
    print('------------------------------')
    legacy_objs = build_legacy_generic_test_case_handlers(
            hamiltonian, comm, options, seed=seed, verbose=verbose)
    legacy_system = legacy_objs['system']
    legacy_trial = legacy_objs['trial']
    legacy_hamiltonian = legacy_objs['hamiltonian']
    legacy_walkers = legacy_objs['walkers']
    
    for iw in range(walkers.nwalkers):
        P = one_rdm_from_G(numpy.array([walkers.Ga[iw], walkers.Gb[iw]])) 
        eloc = local_energy_generic_cholesky(hamiltonian, P)

        legacy_P = legacy_one_rdm_from_G(numpy.array(legacy_walkers.walkers[iw].G))
        legacy_eloc = legacy_local_energy_generic_cholesky(
                        legacy_system, legacy_hamiltonian, legacy_P)

        if verbose:
            print(f'\niw = {iw}')
            print(f'eloc = \n{eloc}\n')
            print(f'legacy_eloc = \n{legacy_eloc}\n')
            print(f'walkers.weight = \n{walkers.weight[iw]}\n')
            print(f'legacy_walkers.weight = \n{legacy_walkers.walkers[iw].weight}\n')

        numpy.testing.assert_almost_equal(legacy_eloc, eloc, decimal=10)
        numpy.testing.assert_almost_equal(legacy_walkers.walkers[iw].G[0], walkers.Ga[iw], decimal=10)
        numpy.testing.assert_almost_equal(legacy_walkers.walkers[iw].G[1], walkers.Gb[iw], decimal=10)
        numpy.testing.assert_almost_equal(legacy_walkers.walkers[iw].stack.ovlp[0], walkers.stack[iw].ovlp[0], decimal=10)
        numpy.testing.assert_almost_equal(legacy_walkers.walkers[iw].stack.ovlp[1], walkers.stack[iw].ovlp[1], decimal=10)


@pytest.mark.unit
def test_thermal_walkers_lowrank():
    # System params.
    nup = 5
    ndown = 5
    nelec = (nup, ndown)
    nbasis = 10

    # Thermal AFQMC params.
    mu = -10.
    beta = 0.1
    timestep = 0.01
    nwalkers = 10
    # Must be fixed at 1 for Thermal AFQMC--legacy code overides whatever input!
    nsteps_per_block = 1
    nblocks = 12
    stabilize_freq = 10
    pop_control_freq = 1
    pop_control_method = 'pair_branch'
    #pop_control_method = 'comb'
    lowrank = True

    verbose = True
    complex_integrals = False
    debug = True
    mf_trial = False
    propagate = False
    diagonal = True
    seed = 7
    numpy.random.seed(seed)

    options = {
                'nelec': nelec,
                'nbasis': nbasis,
                'mu': mu,
                'beta': beta,
                'timestep': timestep,
                'nwalkers': nwalkers,
                'seed': seed,
                'nsteps_per_block': nsteps_per_block,
                'nblocks': nblocks,
                'stabilize_freq': stabilize_freq,
                'pop_control_freq': pop_control_freq,
                'pop_control_method': pop_control_method,
                'lowrank': lowrank,
                'complex_integrals': complex_integrals,
                'mf_trial': mf_trial,
                'propagate': propagate,
                'diagonal': diagonal,

                "hamiltonian": {
                    "name": "Generic",
                    "_alt_convention": False,
                    "sparse": False,
                    "mu": mu
                },
        
                "propagator": {
                    "optimised": False,
                    "free_projection": False
                },
            }
    
    # Test.
    print('\n----------------------------')
    print('Constructing test objects...')
    print('----------------------------')
    objs =  build_generic_test_case_handlers(options, seed, debug, verbose)
    trial = objs['trial']
    hamiltonian = objs['hamiltonian']
    walkers = objs['walkers']
    
    # Legacy.
    print('\n------------------------------')
    print('Constructing legacy objects...')
    print('------------------------------')
    legacy_objs = build_legacy_generic_test_case_handlers(
            hamiltonian, comm, options, seed=seed, verbose=verbose)
    legacy_system = legacy_objs['system']
    legacy_trial = legacy_objs['trial']
    legacy_hamiltonian = legacy_objs['hamiltonian']
    legacy_walkers = legacy_objs['walkers']
    
    for iw in range(walkers.nwalkers):
        P = one_rdm_from_G(numpy.array([walkers.Ga[iw], walkers.Gb[iw]])) 
        eloc = local_energy_generic_cholesky(hamiltonian, P)

        legacy_P = legacy_one_rdm_from_G(numpy.array(legacy_walkers.walkers[iw].G))
        legacy_eloc = legacy_local_energy_generic_cholesky(
                        legacy_system, legacy_hamiltonian, legacy_P)

        if verbose:
            print(f'\niw = {iw}')
            print(f'eloc = \n{eloc}\n')
            print(f'legacy_eloc = \n{legacy_eloc}\n')
            print(f'walkers.weight = \n{walkers.weight[iw]}\n')
            print(f'legacy_walkers.weight = \n{legacy_walkers.walkers[iw].weight}\n')

        numpy.testing.assert_almost_equal(legacy_eloc, eloc, decimal=10)
        numpy.testing.assert_almost_equal(legacy_walkers.walkers[iw].G[0], walkers.Ga[iw], decimal=10)
        numpy.testing.assert_almost_equal(legacy_walkers.walkers[iw].G[1], walkers.Gb[iw], decimal=10)
        numpy.testing.assert_almost_equal(legacy_walkers.walkers[iw].stack.ovlp[0], walkers.stack[iw].ovlp[0], decimal=10)
        numpy.testing.assert_almost_equal(legacy_walkers.walkers[iw].stack.ovlp[1], walkers.stack[iw].ovlp[1], decimal=10)

if __name__ == "__main__":
    test_thermal_walkers_fullrank()
    test_thermal_walkers_lowrank()
