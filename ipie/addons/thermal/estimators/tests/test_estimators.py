import pytest
import tempfile
import numpy
from typing import Tuple, Union

from ipie.config import MPI
from ipie.hamiltonians.generic import Generic as HamGeneric
from ipie.hamiltonians.generic import GenericRealChol, GenericComplexChol

from ipie.addons.thermal.estimators.energy import ThermalEnergyEstimator
from ipie.addons.thermal.estimators.particle_number import ThermalNumberEstimator
from ipie.addons.thermal.estimators.handler import ThermalEstimatorHandler
from ipie.addons.thermal.utils.testing import build_generic_test_case_handlers

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

@pytest.mark.unit
def test_energy_estimator():
    # Test.
    print('\n----------------------------')
    print('Constructing test objects...')
    print('----------------------------')
    objs =  build_generic_test_case_handlers(options, seed, debug, verbose)
    trial = objs['trial']
    hamiltonian = objs['hamiltonian']
    walkers = objs['walkers']

    assert isinstance(hamiltonian, GenericRealChol)
    chol = hamiltonian.chol

    # GenericRealChol.
    re_estim = ThermalEnergyEstimator(hamiltonian=hamiltonian, trial=trial)
    re_estim.compute_estimator(walkers, hamiltonian, trial)
    assert len(re_estim.names) == 5
    tmp = re_estim.data.copy()
    re_estim.post_reduce_hook(tmp)
    assert re_estim.print_to_stdout
    assert re_estim.ascii_filename == None
    assert re_estim.shape == (5,)
    header = re_estim.header_to_text
    data_to_text = re_estim.data_to_text(tmp)
    assert len(data_to_text.split()) == 5
    
    # GenericComplexChol.
    cx_chol = numpy.array(chol, dtype=numpy.complex128)
    cx_hamiltonian = HamGeneric(
        numpy.array(hamiltonian.H1, dtype=numpy.complex128), cx_chol, 
                    hamiltonian.ecore, verbose=False)

    assert isinstance(cx_hamiltonian, GenericComplexChol)

    cx_estim = ThermalEnergyEstimator(hamiltonian=cx_hamiltonian, trial=trial)
    cx_estim.compute_estimator(walkers, cx_hamiltonian, trial)
    assert len(cx_estim.names) == 5
    tmp = cx_estim.data.copy()
    cx_estim.post_reduce_hook(tmp)
    assert cx_estim.print_to_stdout
    assert cx_estim.ascii_filename == None
    assert cx_estim.shape == (5,)
    header = cx_estim.header_to_text
    data_to_text = cx_estim.data_to_text(tmp)
    assert len(data_to_text.split()) == 5
    
    numpy.testing.assert_allclose(re_estim.data, cx_estim.data)


@pytest.mark.unit
def test_number_estimator():
    # Test.
    print('\n----------------------------')
    print('Constructing test objects...')
    print('----------------------------')
    options['complex_integrals'] = True
    objs =  build_generic_test_case_handlers(options, seed, debug, verbose)
    trial = objs['trial']
    hamiltonian = objs['hamiltonian']
    walkers = objs['walkers']

    estim = ThermalNumberEstimator(hamiltonian=hamiltonian, trial=trial)
    estim.compute_estimator(walkers, hamiltonian, trial)
    assert len(estim.names) == 3
    tmp = estim.data.copy()
    estim.post_reduce_hook(tmp)
    assert estim.print_to_stdout
    assert estim.ascii_filename == None
    assert estim.shape == (3,)
    header = estim.header_to_text
    data_to_text = estim.data_to_text(tmp)
    assert len(data_to_text.split()) == 3
    

@pytest.mark.unit
def test_estimator_handler():
    options['complex_integrals'] = True

    with tempfile.NamedTemporaryFile() as tmp1, tempfile.NamedTemporaryFile() as tmp2:
        # Test.
        print('\n----------------------------')
        print('Constructing test objects...')
        print('----------------------------')
        objs =  build_generic_test_case_handlers(options, seed, debug, verbose)
        trial = objs['trial']
        hamiltonian = objs['hamiltonian']
        walkers = objs['walkers']

        estim = ThermalEnergyEstimator(hamiltonian=hamiltonian, trial=trial, 
                                       filename=tmp1.name)
        estim.print_to_stdout = False

        comm = MPI.COMM_WORLD
        handler = ThermalEstimatorHandler(
                    comm,
                    hamiltonian,
                    trial,
                    block_size=10,
                    observables=("energy",),
                    filename=tmp2.name)
        handler["energy1"] = estim
        handler.json_string = ""
        handler.initialize(comm)
        handler.compute_estimators(hamiltonian, trial, walkers)


if __name__ == "__main__":
    test_energy_estimator()
    test_number_estimator()
    test_estimator_handler()



