import numpy
import pytest
from typing import Union

from ipie.config import MPI
from ipie.utils.mpi import MPIHandler
from ipie.utils.testing import generate_hamiltonian
from ipie.walkers.pop_controller import PopController
from ipie.systems.generic import Generic
from ipie.hamiltonians.generic import Generic as HamGeneric
from ipie.qmc.options import QMCOpts

from ipie.thermal.trial.mean_field import MeanField
from ipie.thermal.trial.one_body import OneBody
from ipie.thermal.walkers.uhf_walkers import UHFThermalWalkers
from ipie.thermal.propagation.phaseless_generic import PhaselessGeneric

from ipie.legacy.hamiltonians._generic import Generic as LegacyHamGeneric
from ipie.legacy.trial_density_matrices.mean_field import MeanField as LegacyMeanField
from ipie.legacy.trial_density_matrices.onebody import OneBody as LegacyOneBody
from ipie.legacy.walkers.handler import Walkers
from ipie.legacy.thermal_propagation.continuous import Continuous


def legacy_propagate_walkers(legacy_hamiltonian, 
                             legacy_trial, 
                             legacy_walkers, 
                             legacy_propagator, 
                             xi=None):
    if xi is None:
        xi = [None] * legacy_walkers.nwalkers

    for iw, walker in enumerate(legacy_walkers.walkers):
        legacy_propagator.propagate_walker_phaseless(
                legacy_hamiltonian, walker, legacy_trial, xi=xi[iw])

    return legacy_walkers


def setup_objs(mpi_handler, pop_control_method, seed=None):
    nocca = 5
    noccb = 5
    nelec = (nocca, noccb)
    nbasis = 10

    mu = -10.
    beta = 0.1
    timestep = 0.01
    nwalkers = 10
    nblocks = 3
    stabilize_freq = 10
    pop_control_freq = 1
    nsteps = 1

    lowrank = False
    verbose = True
    complex_integrals = False
    sym = 8
    if complex_integrals: sym = 4
    numpy.random.seed(seed)

    options = {
        "qmc": {
            "dt": timestep,
            "nwalkers": nwalkers,
            "blocks": nblocks,
            "nsteps": nsteps,
            "beta": beta,
            "rng_seed": seed,
            "stabilize_freq": stabilize_freq,
            "batched": False,
        },

        "propagator": {
            "optimised": False
        },

        "walkers": {
            "pop_control_freq": pop_control_freq,
            "pop_control": pop_control_method,
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

    trial = OneBody(hamiltonian, nelec, beta, timestep, verbose=verbose)
    #trial = MeanField(hamiltonian, nelec, beta, timestep, verbose=verbose)

    walkers = UHFThermalWalkers(trial, nbasis, nwalkers, lowrank=lowrank, 
                                verbose=verbose)
    propagator = PhaselessGeneric(timestep, mu, lowrank=lowrank, verbose=verbose)
    propagator.build(hamiltonian, trial=trial, walkers=walkers, verbose=verbose)
    pcontrol = PopController(nwalkers, nsteps, mpi_handler, pop_control_method, verbose=verbose)
       
    
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
    legacy_trial = LegacyOneBody(legacy_system, legacy_hamiltonian, beta, timestep, 
                                 verbose=verbose)
    #legacy_trial = LegacyMeanField(legacy_system, legacy_hamiltonian, beta, timestep, 
    #                               verbose=verbose)
        
    qmc_opts = QMCOpts()
    qmc_opts.nwalkers = nwalkers
    qmc_opts.ntot_walkers = nwalkers
    qmc_opts.beta = beta
    qmc_opts.nsteps = nsteps
    qmc_opts.dt = timestep
    qmc_opts.seed = seed

    legacy_walkers = Walkers(legacy_system, legacy_hamiltonian, legacy_trial, 
                             qmc_opts, walker_opts=options['walkers'],
                             verbose=verbose, comm=mpi_handler.comm)

    legacy_propagator = Continuous(
                            options["propagator"], qmc_opts, legacy_system, 
                            legacy_hamiltonian, legacy_trial, verbose=verbose, 
                            lowrank=lowrank)

    objs = {'trial': trial,
            'hamiltonian': hamiltonian,
            'walkers': walkers,
            'propagator': propagator,
            'pcontrol': pcontrol,
            'nblocks': nblocks}

    legacy_objs = {'system': legacy_system,
                   'trial': legacy_trial,
                   'hamiltonian': legacy_hamiltonian,
                   'walkers': legacy_walkers,
                   'propagator': legacy_propagator}

    return objs, legacy_objs
    

@pytest.mark.unit
def test_pair_branch_batch():
    mpi_handler = MPIHandler()
    pop_control_method = 'pair_branch'
    seed = 7
    objs, legacy_objs = setup_objs(mpi_handler, pop_control_method, seed=seed)
    trial = objs['trial']
    hamiltonian = objs['hamiltonian']
    walkers = objs['walkers']
    propagator = objs['propagator']
    pcontrol = objs['pcontrol']
    nblocks = objs['nblocks']
    
    legacy_system = legacy_objs['system']
    legacy_trial = legacy_objs['trial']
    legacy_hamiltonian = legacy_objs['hamiltonian']
    legacy_walkers = legacy_objs['walkers']
    legacy_propagator = legacy_objs['propagator']

    for block in range(nblocks): 
        for t in range(walkers.stack[0].nslice):
            print('\n------------------------------------------')
            print(f'block = {block}, t = {t}')
            print(f'walkers.weight[0] = {walkers.weight[0]}')
            print(f'legacy_walkers.walkers[0].weight = {legacy_walkers.walkers[0].weight}')
            print(f'diff = {walkers.weight[0] - legacy_walkers.walkers[0].weight}')
            print(f'\nwalkers.unscaled_weight[0] = {walkers.unscaled_weight[0]}')
            print(f'legacy_walkers.walkers[0].unscaled_weight = {legacy_walkers.walkers[0].unscaled_weight}')
            print(f'diff = {walkers.unscaled_weight[0] - legacy_walkers.walkers[0].unscaled_weight}')

            propagator.propagate_walkers(walkers, hamiltonian, trial, debug=True)
            legacy_walkers = legacy_propagate_walkers(
                                legacy_hamiltonian, legacy_trial, legacy_walkers, 
                                legacy_propagator, xi=propagator.xi)
            
            if t > 0:
                pcontrol.pop_control(walkers, mpi_handler.comm)
                legacy_walkers.pop_control(mpi_handler.comm)

        walkers.reset(trial) # Reset stack, weights, phase.
        legacy_walkers.reset(legacy_trial)

    for iw in range(walkers.nwalkers):
        assert numpy.allclose(walkers.Ga[iw], legacy_walkers.walkers[iw].G[0])
        assert numpy.allclose(walkers.Gb[iw], legacy_walkers.walkers[iw].G[1])
        assert numpy.allclose(walkers.weight[iw], legacy_walkers.walkers[iw].weight)
        assert numpy.allclose(walkers.unscaled_weight[iw], legacy_walkers.walkers[iw].unscaled_weight)

#@pytest.mark.unit
def test_comb_batch():
    mpi_handler = MPIHandler()
    pop_control_method = 'comb'
    seed = 7
    objs, legacy_objs = setup_objs(mpi_handler, pop_control_method, seed=seed)
    trial = objs['trial']
    hamiltonian = objs['hamiltonian']
    walkers = objs['walkers']
    propagator = objs['propagator']
    pcontrol = objs['pcontrol']
    nblocks = objs['nblocks']
    
    legacy_system = legacy_objs['system']
    legacy_trial = legacy_objs['trial']
    legacy_hamiltonian = legacy_objs['hamiltonian']
    legacy_walkers = legacy_objs['walkers']
    legacy_propagator = legacy_objs['propagator']

    for block in range(nblocks): 
        for t in range(walkers.stack[0].nslice):
            print('\n------------------------------------------')
            print(f'block = {block}, t = {t}')
            print(f'walkers.weight[0] = {walkers.weight[0]}')
            print(f'legacy_walkers.walkers[0].weight = {legacy_walkers.walkers[0].weight}')
            print(f'diff = {walkers.weight[0] - legacy_walkers.walkers[0].weight}')
            print(f'\nwalkers.unscaled_weight[0] = {walkers.unscaled_weight[0]}')
            print(f'legacy_walkers.walkers[0].unscaled_weight = {legacy_walkers.walkers[0].unscaled_weight}')
            print(f'diff = {walkers.unscaled_weight[0] - legacy_walkers.walkers[0].unscaled_weight}')

            propagator.propagate_walkers(walkers, hamiltonian, trial, debug=True)
            legacy_walkers = legacy_propagate_walkers(
                                legacy_hamiltonian, legacy_trial, legacy_walkers, 
                                legacy_propagator, xi=propagator.xi)
            
            if t > 0:
                pcontrol.pop_control(walkers, mpi_handler.comm)
                legacy_walkers.pop_control(mpi_handler.comm)

        walkers.reset(trial) # Reset stack, weights, phase.
        legacy_walkers.reset(legacy_trial)

    for iw in range(walkers.nwalkers):
        assert numpy.allclose(walkers.Ga[iw], legacy_walkers.walkers[iw].G[0])
        assert numpy.allclose(walkers.Gb[iw], legacy_walkers.walkers[iw].G[1])
        assert numpy.allclose(walkers.weight[iw], legacy_walkers.walkers[iw].weight)
        assert numpy.allclose(walkers.unscaled_weight[iw], legacy_walkers.walkers[iw].unscaled_weight)

#@pytest.mark.unit
def test_stochastic_reconfiguration_batch():
    mpi_handler = MPIHandler()
    pop_control_method = 'stochastic_reconfiguration'
    seed = 7
    objs, legacy_objs = setup_objs(mpi_handler, pop_control_method, seed=seed)
    trial = objs['trial']
    hamiltonian = objs['hamiltonian']
    walkers = objs['walkers']
    propagator = objs['propagator']
    pcontrol = objs['pcontrol']
    nblocks = objs['nblocks']
    
    legacy_system = legacy_objs['system']
    legacy_trial = legacy_objs['trial']
    legacy_hamiltonian = legacy_objs['hamiltonian']
    legacy_walkers = legacy_objs['walkers']
    legacy_propagator = legacy_objs['propagator']

    for block in range(nblocks): 
        for t in range(walkers.stack[0].nslice):
            print('\n------------------------------------------')
            print(f'block = {block}, t = {t}')
            print(f'walkers.weight[0] = {walkers.weight[0]}')
            print(f'legacy_walkers.walkers[0].weight = {legacy_walkers.walkers[0].weight}')
            print(f'diff = {walkers.weight[0] - legacy_walkers.walkers[0].weight}')
            print(f'\nwalkers.unscaled_weight[0] = {walkers.unscaled_weight[0]}')
            print(f'legacy_walkers.walkers[0].unscaled_weight = {legacy_walkers.walkers[0].unscaled_weight}')
            print(f'diff = {walkers.unscaled_weight[0] - legacy_walkers.walkers[0].unscaled_weight}')

            propagator.propagate_walkers(walkers, hamiltonian, trial, debug=True)
            legacy_walkers = legacy_propagate_walkers(
                                legacy_hamiltonian, legacy_trial, legacy_walkers, 
                                legacy_propagator, xi=propagator.xi)
            
            if t > 0:
                pcontrol.pop_control(walkers, mpi_handler.comm)
                legacy_walkers.pop_control(mpi_handler.comm)

        walkers.reset(trial) # Reset stack, weights, phase.
        legacy_walkers.reset(legacy_trial)

    for iw in range(walkers.nwalkers):
        assert numpy.allclose(walkers.Ga[iw], legacy_walkers.walkers[iw].G[0])
        assert numpy.allclose(walkers.Gb[iw], legacy_walkers.walkers[iw].G[1])
        assert numpy.allclose(walkers.weight[iw], legacy_walkers.walkers[iw].weight)
        assert numpy.allclose(walkers.unscaled_weight[iw], legacy_walkers.walkers[iw].unscaled_weight)

if __name__ == "__main__":
    test_pair_branch_batch()
    #test_comb_batch()
    #test_stochastic_reconfiguration_batch()
