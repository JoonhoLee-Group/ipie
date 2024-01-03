import pytest
import tempfile
import numpy
from pyscf import gto
from typing import Tuple, Union

from ipie.systems.generic import Generic
from ipie.utils.testing import generate_hamiltonian
from ipie.hamiltonians.generic import Generic as HamGeneric
from ipie.thermal.trial.one_body import OneBody
from ipie.thermal.trial.mean_field import MeanField
from ipie.thermal.walkers.uhf_walkers import UHFThermalWalkers
from ipie.thermal.estimators.energy import ThermalEnergyEstimator
from ipie.thermal.estimators.handler import ThermalEstimatorHandler
from ipie.thermal.propagation.phaseless_generic import PhaselessGeneric

def build_test_case_handlers(nelec: Tuple[int, int],
                             options: Union[dict, None] = None,
                             seed: Union[int, None] = None,
                             choltol: float = 1e-3,
                             complex_integrals: bool = False,
                             verbose: bool = False):
    if seed is not None:
        numpy.random.seed(seed)

    # Unpack options
    mu = options['mu']
    nbasis = options['nbasis']
    beta = options['beta']
    timestep = options['timestep']
    nwalkers = options['nwalkers']
    lowrank = options['lowrank']
    
    sym = 8
    if complex_integrals: sym = 4
    h1e, chol, _, _ = generate_hamiltonian(nbasis, nelec, cplx=complex_integrals, 
                                           sym=sym, tol=choltol)

    hamiltonian = HamGeneric(h1e=numpy.array([h1e, h1e]),
                             chol=chol.reshape((-1, nbasis**2)).T.copy(),
                             ecore=0)
    hamiltonian.name = options['hamiltonian']['name']
    hamiltonian._alt_convention = options['hamiltonian']['_alt_convention']
    hamiltonian.sparse = options['hamiltonian']['sparse']

    trial = MeanField(hamiltonian, nelec, beta, timestep)
    walkers = UHFThermalWalkers(trial, nbasis, nwalkers, lowrank=lowrank)
    propagator = PhaselessGeneric(timestep, mu, lowrank=lowrank, verbose=verbose)
    propagator.build(hamiltonian, trial=trial, walkers=walkers, verbose=verbose)
        
    for t in range(walkers.stack[0].nslice):
        propagator.propagate_walkers(walkers, hamiltonian, trial)

    objs = {'trial': trial,
            'hamiltonian': hamiltonian,
            'walkers': walkers,
            'propagator': propagator}
    return objs


def test_energy_estimator():
    nocca = 5
    noccb = 5
    nelec = (nocca, noccb)
    nbasis = 10

    mu = -10.0
    beta = 0.1
    timestep = 0.01
    nwalkers = 1
    seed = 7
    lowrank = False
    verbose = True
    
    options = {
                'mu': mu,
                'nbasis': nbasis,
                'beta': beta,
                'timestep': timestep,
                'nwalkers': nwalkers,
                'seed': 7,
                'lowrank': lowrank,

                'hamiltonian': {
                    'name': 'Generic',
                    '_alt_convention': False,
                    'sparse': False,
                    'mu': mu
                }
            }
    
    objs = build_test_case_handlers(nelec, options, seed, choltol=1e-10, 
                                    complex_integrals=False, verbose=verbose)
    trial = objs['trial']
    hamiltonian = objs['hamiltonian']
    walkers = objs['walkers']

    estim = ThermalEnergyEstimator(hamiltonian=hamiltonian, trial=trial)
    estim.compute_estimator(walkers, hamiltonian, trial)
    assert len(estim.names) == 5
    #assert estim["ENumer"].real == pytest.approx(-754.0373585215561)
    #assert estim["ETotal"] == pytest.approx(0.0)
    tmp = estim.data.copy()
    estim.post_reduce_hook(tmp)
    #assert tmp[estim.get_index("ETotal")] == pytest.approx(-75.40373585215562)
    assert estim.print_to_stdout
    assert estim.ascii_filename == None
    assert estim.shape == (5,)
    header = estim.header_to_text
    data_to_text = estim.data_to_text(tmp)
    assert len(data_to_text.split()) == 5

def test_estimator_handler():
    nocca = 5
    noccb = 5
    nelec = (nocca, noccb)
    nbasis = 10

    mu = -10.0
    beta = 0.1
    timestep = 0.01
    nwalkers = 1
    seed = 7
    lowrank = False
    verbose = True
    
    options = {
                'mu': mu,
                'nbasis': nbasis,
                'beta': beta,
                'timestep': timestep,
                'nwalkers': nwalkers,
                'seed': 7,
                'lowrank': lowrank,

                'hamiltonian': {
                    'name': 'Generic',
                    '_alt_convention': False,
                    'sparse': False,
                    'mu': mu
                }
            }
    
    with tempfile.NamedTemporaryFile() as tmp1, tempfile.NamedTemporaryFile() as tmp2:
        objs = build_test_case_handlers(nelec, options, seed, choltol=1e-10, 
                                        complex_integrals=False, verbose=verbose)
        trial = objs['trial']
        hamiltonian = objs['hamiltonian']
        walkers = objs['walkers']

        estim = ThermalEnergyEstimator(hamiltonian=hamiltonian, trial=trial, 
                                       filename=tmp1.name)
        estim.print_to_stdout = False
        from ipie.config import MPI

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
    test_estimator_handler()



