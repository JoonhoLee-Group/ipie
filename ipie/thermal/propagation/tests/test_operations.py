import numpy
import pytest

from pyscf import gto, scf, lo
from ipie.qmc.options import QMCOpts

from ipie.systems.generic import Generic
from ipie.utils.testing import generate_hamiltonian
from ipie.hamiltonians.generic import Generic as HamGeneric
from ipie.thermal.trial.mean_field import MeanField
from ipie.thermal.trial.one_body import OneBody
from ipie.thermal.walkers.uhf_walkers import UHFThermalWalkers
from ipie.thermal.propagation.operations import apply_exponential
from ipie.thermal.propagation.phaseless_generic import PhaselessGeneric

from ipie.legacy.hamiltonians._generic import Generic as LegacyHamGeneric
from ipie.legacy.trial_density_matrices.mean_field import MeanField as LegacyMeanField
from ipie.legacy.trial_density_matrices.onebody import OneBody as LegacyOneBody
from ipie.legacy.walkers.thermal import ThermalWalker
from ipie.legacy.thermal_propagation.continuous import Continuous

def setup_objs(mf_trial=False, seed=None):
    nocca = 5
    noccb = 5
    nelec = (nocca, noccb)
    nbasis = 10

    mu = -10.
    beta = 0.1
    dt = 0.01
    nwalkers = 5
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
        },
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
                            ecore=hamiltonian.ecore,
                            options=options["hamiltonian"])
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

    #legacy_hamiltonian.chol_vecs = legacy_hamiltonian.chol_vecs.T.reshape(
    #                (hamiltonian.nchol, hamiltonian.nbasis, hamiltonian.nbasis))
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
   

def test_apply_exponential(verbose=False):
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

    _, _, _, VHS = propagator.construct_two_body_propagator(walkers, hamiltonian, trial, debug=True)
    
    exp = []
    legacy_exp = []
    for iw in range(walkers.nwalkers):
        _, _, _, _VHS = legacy_propagator.two_body_propagator(
                                        legacy_walkers[iw], legacy_hamiltonian, 
                                        legacy_trial, xi=propagator.xi[iw])
        _exp = apply_exponential(VHS[iw], propagator.exp_nmax) 
        _legacy_exp = legacy_propagator.exponentiate(_VHS, debug=True)
        exp.append(_exp)
        legacy_exp.append(_legacy_exp)
        numpy.testing.assert_allclose(_VHS, VHS[iw])

    exp = numpy.array(exp)
    legacy_exp = numpy.array(legacy_exp)

    if verbose:
        print(f'\nexp_nmax = {propagator.exp_nmax}')
        print(f'legacy_exp = \n{legacy_exp}\n')
        print(f'exp = \n{exp}\n')

    numpy.testing.assert_almost_equal(legacy_exp, exp, decimal=10)


if __name__ == "__main__":
    test_apply_exponential(verbose=True)
