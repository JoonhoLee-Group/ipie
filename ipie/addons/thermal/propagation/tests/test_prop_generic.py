import numpy
import pytest

try:
    from ipie.addons.thermal.utils.legacy_testing import build_legacy_generic_test_case_handlers
    from ipie.addons.thermal.utils.legacy_testing import legacy_propagate_walkers
    _no_cython = False

except ModuleNotFoundError:
    _no_cython = True

from ipie.config import MPI
from ipie.addons.thermal.estimators.generic import local_energy_generic_cholesky
from ipie.addons.thermal.estimators.thermal import one_rdm_from_G
from ipie.addons.thermal.utils.testing import build_generic_test_case_handlers

from ipie.legacy.estimators.generic import local_energy_generic_cholesky as legacy_local_energy_generic_cholesky
from ipie.legacy.estimators.thermal import one_rdm_from_G as legacy_one_rdm_from_G

comm = MPI.COMM_WORLD

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
nblocks = 12
lowrank = False

mf_trial = True
complex_integrals = False
debug = True
verbose = True
seed = 7
numpy.random.seed(seed)


@pytest.mark.skipif(_no_cython, reason="Need to build cython modules.")
@pytest.mark.unit
def test_mf_shift():
    # Test.
    print('\n----------------------------')
    print('Constructing test objects...')
    print('----------------------------')
    objs =  build_generic_test_case_handlers(
            nelec, nbasis, mu, beta, timestep, nwalkers=nwalkers, lowrank=lowrank, 
            mf_trial=mf_trial, complex_integrals=complex_integrals, debug=debug, 
            seed=seed, verbose=verbose)
    hamiltonian = objs['hamiltonian']
    propagator = objs['propagator']

    # Legacy.
    print('\n------------------------------')
    print('Constructing legacy objects...')
    print('------------------------------')
    legacy_objs = build_legacy_generic_test_case_handlers(
                    hamiltonian, comm, nelec, mu, beta, timestep, 
                    nwalkers=nwalkers, lowrank=lowrank, mf_trial=mf_trial,
                    seed=seed, verbose=verbose)
    legacy_propagator = legacy_objs['propagator']

    if verbose:
        print(f'\nlegacy_mf_shift = \n{legacy_propagator.propagator.mf_shift}\n')
        print(f'mf_shift = \n{propagator.mf_shift}\n')

    numpy.testing.assert_almost_equal(legacy_propagator.propagator.mf_shift, 
                                      propagator.mf_shift, decimal=10)


@pytest.mark.skipif(_no_cython, reason="Need to build cython modules.")
@pytest.mark.unit
def test_BH1():
    # Test.
    print('\n----------------------------')
    print('Constructing test objects...')
    print('----------------------------')
    objs =  build_generic_test_case_handlers(
            nelec, nbasis, mu, beta, timestep, nwalkers=nwalkers, lowrank=lowrank, 
            mf_trial=mf_trial, complex_integrals=complex_integrals, debug=debug, 
            seed=seed, verbose=verbose)
    hamiltonian = objs['hamiltonian']
    propagator = objs['propagator']

    # Legacy.
    print('\n------------------------------')
    print('Constructing legacy objects...')
    print('------------------------------')
    legacy_objs = build_legacy_generic_test_case_handlers(
                    hamiltonian, comm, nelec, mu, beta, timestep, 
                    nwalkers=nwalkers, lowrank=lowrank, mf_trial=mf_trial,
                    seed=seed, verbose=verbose)
    legacy_propagator = legacy_objs['propagator']

    if verbose:
        print(f'\nlegacy_BH1 = \n{legacy_propagator.propagator.BH1}\n')
        print(f'BH1 = \n{propagator.BH1}\n')

    numpy.testing.assert_almost_equal(legacy_propagator.propagator.BH1, 
                                      propagator.BH1, decimal=10)


@pytest.mark.skipif(_no_cython, reason="Need to build cython modules.")
@pytest.mark.unit
def test_construct_two_body_propagator():
    # Test.
    print('\n----------------------------')
    print('Constructing test objects...')
    print('----------------------------')
    objs =  build_generic_test_case_handlers(
            nelec, nbasis, mu, beta, timestep, nwalkers=nwalkers, lowrank=lowrank, 
            mf_trial=mf_trial, complex_integrals=complex_integrals, debug=debug, 
            seed=seed, verbose=verbose)
    trial = objs['trial']
    hamiltonian = objs['hamiltonian']
    walkers = objs['walkers']
    propagator = objs['propagator']

    # Legacy.
    print('\n------------------------------')
    print('Constructing legacy objects...')
    print('------------------------------')
    legacy_objs = build_legacy_generic_test_case_handlers(
                    hamiltonian, comm, nelec, mu, beta, timestep, 
                    nwalkers=nwalkers, lowrank=lowrank, mf_trial=mf_trial,
                    seed=seed, verbose=verbose)
    legacy_trial = legacy_objs['trial']
    legacy_hamiltonian = legacy_objs['hamiltonian']
    legacy_walkers = legacy_objs['walkers']
    legacy_propagator = legacy_objs['propagator']
    
    cmf, cfb, xshifted, VHS = propagator.construct_two_body_propagator(
                                walkers, hamiltonian, trial, debug=True)

    legacy_cmf = []
    legacy_cfb = []
    legacy_xshifted = []
    legacy_VHS = []

    for iw in range(walkers.nwalkers):
        _cmf, _cfb, _xshifted, _VHS = legacy_propagator.two_body_propagator(
                                        legacy_walkers.walkers[iw], legacy_hamiltonian, 
                                        legacy_trial, xi=propagator.xi[iw])
        legacy_cmf.append(_cmf)
        legacy_cfb.append(_cfb)
        legacy_xshifted.append(_xshifted)
        legacy_VHS.append(_VHS)
    
    legacy_xshifted = numpy.array(legacy_xshifted).T

    if verbose:
        print(f'\nlegacy_cmf = {legacy_cmf}')
        print(f'cmf = {cmf}')

        print(f'\nlegacy_cfb = {legacy_cfb}')
        print(f'cfb = {cfb}')

        print(f'\nlegacy_xshifted = \n{legacy_xshifted}\n')
        print(f'xshifted = \n{xshifted}\n')
        
        print(f'legacy_VHS = \n{legacy_VHS}\n')
        print(f'VHS = \n{VHS}\n')

    numpy.testing.assert_almost_equal(legacy_cmf, cmf, decimal=10)
    numpy.testing.assert_almost_equal(legacy_cfb, cfb, decimal=10)
    numpy.testing.assert_almost_equal(legacy_xshifted, xshifted, decimal=10)
    numpy.testing.assert_almost_equal(legacy_VHS, VHS, decimal=10)


@pytest.mark.skipif(_no_cython, reason="Need to build cython modules.")
@pytest.mark.unit
def test_phaseless_generic_propagator():
    # Test.
    print('\n----------------------------')
    print('Constructing test objects...')
    print('----------------------------')
    objs =  build_generic_test_case_handlers(
            nelec, nbasis, mu, beta, timestep, nwalkers=nwalkers, lowrank=lowrank, 
            mf_trial=mf_trial, complex_integrals=complex_integrals, debug=debug, 
            seed=seed, verbose=verbose)
    trial = objs['trial']
    hamiltonian = objs['hamiltonian']
    walkers = objs['walkers']
    propagator = objs['propagator']

    # Legacy.
    print('\n------------------------------')
    print('Constructing legacy objects...')
    print('------------------------------')
    legacy_objs = build_legacy_generic_test_case_handlers(
                    hamiltonian, comm, nelec, mu, beta, timestep, 
                    nwalkers=nwalkers, lowrank=lowrank, mf_trial=mf_trial,
                    seed=seed, verbose=verbose)
    legacy_system = legacy_objs['system']
    legacy_trial = legacy_objs['trial']
    legacy_hamiltonian = legacy_objs['hamiltonian']
    legacy_walkers = legacy_objs['walkers']
    legacy_propagator = legacy_objs['propagator']

    for t in range(walkers.stack[0].nslice):
        for iw in range(walkers.nwalkers):
            P = one_rdm_from_G(numpy.array([walkers.Ga[iw], walkers.Gb[iw]])) 
            eloc = local_energy_generic_cholesky(hamiltonian, P)

            legacy_P = legacy_one_rdm_from_G(numpy.array(legacy_walkers.walkers[iw].G))
            legacy_eloc = legacy_local_energy_generic_cholesky(
                            legacy_system, legacy_hamiltonian, legacy_P)

            if verbose:
                print(f'\nt = {t}')
                print(f'iw = {iw}')
                print(f'eloc = \n{eloc}\n')
                print(f'legacy_eloc = \n{legacy_eloc}\n')
                print(f'walkers.weight = \n{walkers.weight[iw]}\n')
                print(f'legacy_walkers.weight = \n{legacy_walkers.walkers[iw].weight}\n')

            numpy.testing.assert_almost_equal(legacy_eloc, eloc, decimal=10)
            numpy.testing.assert_allclose(legacy_walkers.walkers[iw].G[0], walkers.Ga[iw])
            numpy.testing.assert_almost_equal(legacy_walkers.walkers[iw].G[1], walkers.Gb[iw], decimal=10)
            numpy.testing.assert_almost_equal(legacy_P, P, decimal=10)
            numpy.testing.assert_almost_equal(legacy_walkers.walkers[iw].stack.ovlp[0], walkers.stack[iw].ovlp[0], decimal=10)
            numpy.testing.assert_almost_equal(legacy_walkers.walkers[iw].stack.ovlp[1], walkers.stack[iw].ovlp[1], decimal=10)

        propagator.propagate_walkers(walkers, hamiltonian, trial, debug=True)
        legacy_walkers = legacy_propagate_walkers(
                            legacy_hamiltonian, legacy_trial, legacy_walkers, 
                            legacy_propagator, xi=propagator.xi)


if __name__ == "__main__":
    test_mf_shift()
    test_BH1()
    test_construct_two_body_propagator()
    test_phaseless_generic_propagator()
