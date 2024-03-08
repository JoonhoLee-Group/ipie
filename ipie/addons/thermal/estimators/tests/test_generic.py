import numpy
import pytest
from typing import Tuple, Union

from ipie.config import MPI
from ipie.addons.thermal.estimators.thermal import one_rdm_from_G
from ipie.addons.thermal.estimators.generic import local_energy_generic_cholesky

from ipie.legacy.estimators.thermal import one_rdm_from_G as legacy_one_rdm_from_G
from ipie.legacy.estimators.generic import local_energy_generic_cholesky as legacy_local_energy_generic_cholesky

from ipie.addons.thermal.utils.testing import build_generic_test_case_handlers
from ipie.addons.thermal.utils.legacy_testing import build_legacy_generic_test_case_handlers

comm = MPI.COMM_WORLD

@pytest.mark.unit
def test_local_energy_cholesky(mf_trial=False):
    # System params.
    nup = 5
    ndown = 5
    nelec = (nup, ndown)
    nbasis = 10

    # Thermal AFQMC params.
    mu = -10.
    beta = 0.1
    timestep = 0.01
    nwalkers = 12
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
    P = one_rdm_from_G(trial.G) 
    eloc = local_energy_generic_cholesky(hamiltonian, P)

    # Legacy.
    print('\n------------------------------')
    print('Constructing legacy objects...')
    print('------------------------------')
    legacy_objs = build_legacy_generic_test_case_handlers(
            hamiltonian, comm, options, seed=seed, verbose=verbose)
    legacy_system = legacy_objs['system']
    legacy_trial = legacy_objs['trial']
    legacy_hamiltonian = legacy_objs['hamiltonian']

    legacy_P = legacy_one_rdm_from_G(legacy_trial.G)
    legacy_eloc = legacy_local_energy_generic_cholesky(
                    legacy_system, legacy_hamiltonian, legacy_P)
    
    if verbose:
        print(f'\neloc = \n{eloc}\n')
        print(f'legacy_eloc = \n{legacy_eloc}\n')
            
    numpy.testing.assert_allclose(trial.G, legacy_trial.G, atol=1e-10)
    numpy.testing.assert_allclose(P, legacy_P, atol=1e-10)
    numpy.testing.assert_allclose(eloc, legacy_eloc, atol=1e-10)


if __name__ == '__main__':
    test_local_energy_cholesky(mf_trial=True)
