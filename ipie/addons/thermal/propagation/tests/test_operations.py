import numpy
import pytest

from ipie.config import MPI
from ipie.addons.thermal.propagation.operations import apply_exponential
from ipie.addons.thermal.utils.testing import build_generic_test_case_handlers
from ipie.addons.thermal.utils.legacy_testing import build_legacy_generic_test_case_handlers
from ipie.addons.thermal.utils.legacy_testing import legacy_propagate_walkers

comm = MPI.COMM_WORLD

@pytest.mark.unit
def test_apply_exponential():
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
            }

    # Test.
    print('\n----------------------------')
    print('Constructing test objects...')
    print('----------------------------')
    objs =  build_generic_test_case_handlers(options, seed, debug, verbose)
    trial = objs['trial']
    hamiltonian = objs['hamiltonian']
    walkers = objs['walkers']
    propagator = objs['propagator']

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
    legacy_propagator = legacy_objs['propagator']

    _, _, _, VHS = propagator.construct_two_body_propagator(walkers, hamiltonian, trial, debug=True)
    
    exp = []
    legacy_exp = []
    for iw in range(walkers.nwalkers):
        _, _, _, _VHS = legacy_propagator.two_body_propagator(
                                        legacy_walkers.walkers[iw], legacy_hamiltonian, 
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
    test_apply_exponential()
