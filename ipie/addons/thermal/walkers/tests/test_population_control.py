import numpy
import pytest
from typing import Union

from ipie.config import MPI
from ipie.utils.mpi import MPIHandler
from ipie.walkers.pop_controller import PopController

from ipie.addons.thermal.utils.testing import build_generic_test_case_handlers_mpi
from ipie.addons.thermal.utils.legacy_testing import build_legacy_generic_test_case_handlers_mpi
from ipie.addons.thermal.utils.legacy_testing import legacy_propagate_walkers
    
try:
    from ipie.legacy.estimators.ueg import fock_ueg
    _no_cython = False

except ModuleNotFoundError:
    _no_cython = True

comm = MPI.COMM_WORLD

@pytest.mark.skipif(_no_cython, reason="Need to build cython modules.")
@pytest.mark.unit
def test_pair_branch_batch():
    mpi_handler = MPIHandler()

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
    nblocks = 3
    stabilize_freq = 10
    pop_control_freq = 1
    pop_control_method = 'pair_branch'
    lowrank = False

    verbose = False if (comm.rank != 0) else True
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
            }
    
    # Test.
    if verbose:
        print('\n----------------------------')
        print('Constructing test objects...')
        print('----------------------------')
        
    objs =  build_generic_test_case_handlers_mpi(
                options, mpi_handler, seed, debug, verbose)
    trial = objs['trial']
    hamiltonian = objs['hamiltonian']
    walkers = objs['walkers']
    propagator = objs['propagator']
    pcontrol = PopController(nwalkers, nsteps_per_block, mpi_handler, 
                             pop_control_method, verbose=verbose)
    
    # Legacy.
    if verbose:
        print('\n------------------------------')
        print('Constructing legacy objects...')
        print('------------------------------')

    legacy_objs = build_legacy_generic_test_case_handlers_mpi(
            hamiltonian, mpi_handler, options, seed=seed, verbose=verbose)
    legacy_system = legacy_objs['system']
    legacy_trial = legacy_objs['trial']
    legacy_hamiltonian = legacy_objs['hamiltonian']
    legacy_walkers = legacy_objs['walkers']
    legacy_propagator = legacy_objs['propagator']
    
    for block in range(nblocks): 
        for t in range(walkers.stack[0].nslice):
            if verbose:
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


#@pytest.mark.skipif(_no_cython, reason="Need to build cython modules.")
#@pytest.mark.unit
def test_pair_branch_batch_lowrank():
    mpi_handler = MPIHandler()

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
    nblocks = 3
    stabilize_freq = 10
    pop_control_freq = 1
    pop_control_method = 'pair_branch'
    lowrank = True

    verbose = False if (comm.rank != 0) else True
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
            }
    
    # Test.
    if verbose:
        print('\n----------------------------')
        print('Constructing test objects...')
        print('----------------------------')
        
    objs =  build_generic_test_case_handlers_mpi(options, mpi_handler, seed, debug, verbose)
    trial = objs['trial']
    hamiltonian = objs['hamiltonian']
    walkers = objs['walkers']
    propagator = objs['propagator']
    pcontrol = PopController(nwalkers, nsteps_per_block, mpi_handler,
                             pop_control_method=pop_control_method, verbose=verbose)
    
    # Legacy.
    if verbose:
        print('\n------------------------------')
        print('Constructing legacy objects...')
        print('------------------------------')
        
    legacy_objs = build_legacy_generic_test_case_handlers_mpi(
            hamiltonian, mpi_handler, options, seed=seed, verbose=verbose)
    legacy_system = legacy_objs['system']
    legacy_trial = legacy_objs['trial']
    legacy_hamiltonian = legacy_objs['hamiltonian']
    legacy_walkers = legacy_objs['walkers']
    legacy_propagator = legacy_objs['propagator']
    
    for block in range(nblocks): 
        for t in range(walkers.stack[0].nslice):
            if verbose:
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


@pytest.mark.skipif(_no_cython, reason="Need to build cython modules.")
@pytest.mark.unit
def test_comb_batch():
    mpi_handler = MPIHandler()
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
    nblocks = 3
    stabilize_freq = 10
    pop_control_freq = 1
    pop_control_method = 'comb'
    lowrank = False

    verbose = False if (comm.rank != 0) else True
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
            }
    
    # Test.
    if verbose:
        print('\n----------------------------')
        print('Constructing test objects...')
        print('----------------------------')

    objs =  build_generic_test_case_handlers_mpi(options, mpi_handler, seed, debug, verbose)
    trial = objs['trial']
    hamiltonian = objs['hamiltonian']
    walkers = objs['walkers']
    propagator = objs['propagator']
    pcontrol = PopController(nwalkers, nsteps_per_block, mpi_handler,
                             pop_control_method=pop_control_method, verbose=verbose)
    
    # Legacy.
    if verbose:
        print('\n------------------------------')
        print('Constructing legacy objects...')
        print('------------------------------')
        
    legacy_objs = build_legacy_generic_test_case_handlers_mpi(
            hamiltonian, mpi_handler, options, seed=seed, verbose=verbose)
    legacy_system = legacy_objs['system']
    legacy_trial = legacy_objs['trial']
    legacy_hamiltonian = legacy_objs['hamiltonian']
    legacy_walkers = legacy_objs['walkers']
    legacy_propagator = legacy_objs['propagator']
    
    for block in range(nblocks): 
        for t in range(walkers.stack[0].nslice):
            if verbose:
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


#@pytest.mark.skipif(_no_cython, reason="Need to build cython modules.")
#@pytest.mark.unit
def test_comb_batch_lowrank():
    mpi_handler = MPIHandler()
    
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
    nblocks = 3
    stabilize_freq = 10
    pop_control_freq = 1
    pop_control_method = 'comb'
    lowrank = True

    verbose = False if (comm.rank != 0) else True
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
            }
    
    # Test.
    if verbose:
        print('\n----------------------------')
        print('Constructing test objects...')
        print('----------------------------')

    objs =  build_generic_test_case_handlers_mpi(options, mpi_handler, seed, debug, verbose)
    trial = objs['trial']
    hamiltonian = objs['hamiltonian']
    walkers = objs['walkers']
    propagator = objs['propagator']
    pcontrol = PopController(nwalkers, nsteps_per_block, mpi_handler, 
                             pop_control_method=pop_control_method, verbose=verbose)
    
    # Legacy.
    if verbose:
        print('\n------------------------------')
        print('Constructing legacy objects...')
        print('------------------------------')
        
    legacy_objs = build_legacy_generic_test_case_handlers_mpi(
            hamiltonian, mpi_handler, options, seed=seed, verbose=verbose)
    legacy_system = legacy_objs['system']
    legacy_trial = legacy_objs['trial']
    legacy_hamiltonian = legacy_objs['hamiltonian']
    legacy_walkers = legacy_objs['walkers']
    legacy_propagator = legacy_objs['propagator']
    
    for block in range(nblocks): 
        for t in range(walkers.stack[0].nslice):
            if verbose:
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
    test_comb_batch()
    
    #test_pair_branch_batch_lowrank()
    #test_comb_batch_lowrank()
