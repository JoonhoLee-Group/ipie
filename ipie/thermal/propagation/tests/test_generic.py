import numpy
import pytest

from ipie.qmc.options import QMCOpts

from ipie.systems.generic import Generic
from ipie.utils.testing import generate_hamiltonian
from ipie.hamiltonians.generic import Generic as HamGeneric
from ipie.hamiltonians.utils import get_hamiltonian
from ipie.thermal.trial.mean_field import MeanField
from ipie.thermal.trial.one_body import OneBody
from ipie.thermal.walkers.uhf_walkers import UHFThermalWalkers
from ipie.thermal.propagation.phaseless_generic import PhaselessGeneric
from ipie.thermal.estimators.generic import local_energy_generic_cholesky
from ipie.thermal.estimators.thermal import one_rdm_from_G

from ipie.legacy.hamiltonians._generic import Generic as LegacyHamGeneric
from ipie.legacy.trial_density_matrices.mean_field import MeanField as LegacyMeanField
from ipie.legacy.trial_density_matrices.onebody import OneBody as LegacyOneBody
from ipie.legacy.walkers.thermal import ThermalWalker
from ipie.legacy.thermal_propagation.continuous import Continuous
from ipie.legacy.estimators.generic import local_energy_generic_cholesky as legacy_local_energy_generic_cholesky
from ipie.legacy.estimators.thermal import one_rdm_from_G as legacy_one_rdm_from_G


def legacy_propagate_walkers(legacy_hamiltonian, legacy_trial, legacy_walkers, legacy_propagator, xi=None):
    if xi is None:
        xi = [None] * legacy_walkers.nwalker

    for iw, walker in enumerate(legacy_walkers):
        legacy_propagator.propagate_walker_phaseless(
                legacy_hamiltonian, walker, legacy_trial, xi=xi[iw])

    return legacy_walkers

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
    blocks = 10
    stabilise_freq = 10
    pop_control_freq = 1
    nsteps = 1

    lowrank = False
    verbose = True
    complex_integrals = False
    sym = 8
    if complex_integrals: sym = 4

    options = {
        "qmc": {
            "dt": dt,
            "nwalkers": nwalkers,
            "blocks": blocks,
            "nsteps": nsteps,
            "beta": beta,
            "rng_seed": seed,
            "pop_control_freq": pop_control_freq,
            "stabilise_freq": stabilise_freq,
            "batched": False
        },

        "propagator": {
            "optimised": False
        },

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
    propagator = PhaselessGeneric(dt, mu, lowrank=lowrank, verbose=verbose)
    propagator.build(hamiltonian, trial=trial, walkers=walkers, verbose=verbose)
    
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

    qmc_opts = QMCOpts()
    qmc_opts.nwalkers = nwalkers
    qmc_opts.ntot_walkers = nwalkers
    qmc_opts.beta = beta
    qmc_opts.nsteps = nsteps
    qmc_opts.dt = dt
    qmc_opts.seed = seed

    legacy_propagator = Continuous(
                            options["propagator"], qmc_opts, legacy_system, 
                            legacy_hamiltonian, legacy_trial, verbose=verbose, 
                            lowrank=lowrank)

    objs = {'trial': trial,
            'hamiltonian': hamiltonian,
            'walkers': walkers,
            'propagator': propagator}

    legacy_objs = {'system': legacy_system,
                   'trial': legacy_trial,
                   'hamiltonian': legacy_hamiltonian,
                   'walkers': legacy_walkers,
                   'propagator': legacy_propagator}

    return objs, legacy_objs
    

def test_mf_shift(verbose=False):
    seed = 7
    objs, legacy_objs = setup_objs(seed=seed)
    propagator = objs['propagator']
    legacy_propagator = legacy_objs['propagator']

    if verbose:
        print(f'\nlegacy_mf_shift = \n{legacy_propagator.propagator.mf_shift}\n')
        print(f'mf_shift = \n{propagator.mf_shift}\n')

    numpy.testing.assert_almost_equal(legacy_propagator.propagator.mf_shift, 
                                      propagator.mf_shift, decimal=10)

def test_BH1(verbose=False):
    seed = 7
    objs, legacy_objs = setup_objs(seed=seed)
    propagator = objs['propagator']
    legacy_propagator = legacy_objs['propagator']

    if verbose:
        print(f'\nlegacy_BH1 = \n{legacy_propagator.propagator.BH1}\n')
        print(f'BH1 = \n{propagator.BH1}\n')

    numpy.testing.assert_almost_equal(legacy_propagator.propagator.BH1, 
                                      propagator.BH1, decimal=10)


def test_construct_two_body_propagator(verbose=False):
    seed = 7
    objs, legacy_objs = setup_objs(seed=seed)
    trial = objs['trial']
    hamiltonian = objs['hamiltonian']
    walkers = objs['walkers']
    propagator = objs['propagator']
    
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
                                        legacy_walkers[iw], legacy_hamiltonian, 
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


def test_phaseless_generic_propagator(mf_trial=False, verbose=False):
    seed = 7
    objs, legacy_objs = setup_objs(mf_trial=mf_trial, seed=seed)
    trial = objs['trial']
    hamiltonian = objs['hamiltonian']
    walkers = objs['walkers']
    propagator = objs['propagator']
    
    legacy_system = legacy_objs['system']
    legacy_trial = legacy_objs['trial']
    legacy_hamiltonian = legacy_objs['hamiltonian']
    legacy_walkers = legacy_objs['walkers']
    legacy_propagator = legacy_objs['propagator']

    for t in range(walkers.stack[0].nslice):
        for iw in range(walkers.nwalkers):
            P = one_rdm_from_G(numpy.array([walkers.Ga[iw], walkers.Gb[iw]])) 
            eloc = local_energy_generic_cholesky(hamiltonian, P)

            legacy_P = legacy_one_rdm_from_G(numpy.array(legacy_walkers[iw].G))
            legacy_eloc = legacy_local_energy_generic_cholesky(
                            legacy_system, legacy_hamiltonian, legacy_P)

            if verbose:
                print(f'\nt = {t}')
                print(f'iw = {iw}')
                print(f'eloc = \n{eloc}\n')
                print(f'legacy_eloc = \n{legacy_eloc}\n')
                print(f'walkers.weight = \n{walkers.weight[iw]}\n')
                print(f'legacy_walkers.weight = \n{legacy_walkers[iw].weight}\n')

            numpy.testing.assert_almost_equal(legacy_eloc, eloc, decimal=10)
            numpy.testing.assert_almost_equal(legacy_walkers[iw].G[0], walkers.Ga[iw], decimal=10)
            numpy.testing.assert_almost_equal(legacy_walkers[iw].G[1], walkers.Gb[iw], decimal=10)
            numpy.testing.assert_almost_equal(legacy_P, P, decimal=10)
            numpy.testing.assert_almost_equal(legacy_walkers[iw].stack.ovlp[0], walkers.stack[iw].ovlp[0], decimal=10)
            numpy.testing.assert_almost_equal(legacy_walkers[iw].stack.ovlp[1], walkers.stack[iw].ovlp[1], decimal=10)

        propagator.propagate_walkers(walkers, hamiltonian, trial, debug=True)
        legacy_walkers = legacy_propagate_walkers(
                            legacy_hamiltonian, legacy_trial, legacy_walkers, 
                            legacy_propagator, xi=propagator.xi)


if __name__ == "__main__":
    test_mf_shift(verbose=True)
    test_BH1(verbose=True)
    test_construct_two_body_propagator(verbose=True)
    test_phaseless_generic_propagator(mf_trial=True, verbose=True)
