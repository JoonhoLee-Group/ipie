import numpy
import pytest
from typing import Union

from ipie.config import MPI
from ipie.utils.mpi import MPIHandler
from ipie.utils.testing import generate_hamiltonian
from ipie.walkers.pop_controller import get_buffer, set_buffer
from ipie.systems.generic import Generic
from ipie.hamiltonians.generic import Generic as HamGeneric
from ipie.qmc.options import QMCOpts

from ipie.thermal.trial.mean_field import MeanField
from ipie.thermal.trial.one_body import OneBody
from ipie.thermal.walkers.uhf_walkers import UHFThermalWalkers

from ipie.legacy.hamiltonians._generic import Generic as LegacyHamGeneric
from ipie.legacy.trial_density_matrices.mean_field import MeanField as LegacyMeanField
from ipie.legacy.trial_density_matrices.onebody import OneBody as LegacyOneBody
from ipie.legacy.walkers.handler import Walkers


def setup_objs(mpi_handler, pop_control_method, seed=None):
    nocca = 5
    noccb = 5
    nelec = (nocca, noccb)
    nbasis = 10

    mu = -10.
    beta = 0.03
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
    walkers.buff_names = ['weight', 'unscaled_weight', 'phase', 'hybrid_energy', 'ovlp']
    
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

    for iw in range(legacy_walkers.nwalkers):
        legacy_walkers.walkers[iw].buff_names = walkers.buff_names
        legacy_walkers.walkers[iw].buff_size = walkers.buff_size

    objs = {'trial': trial,
            'hamiltonian': hamiltonian,
            'walkers': walkers,
            'nblocks': nblocks}

    legacy_objs = {'system': legacy_system,
                   'trial': legacy_trial,
                   'hamiltonian': legacy_hamiltonian,
                   'walkers': legacy_walkers}

    return objs, legacy_objs
    

#@pytest.mark.unit
def test_get_buffer():
    mpi_handler = MPIHandler()
    pop_control_method = 'stochastic_reconfiguration'
    seed = 7
    objs, legacy_objs = setup_objs(mpi_handler, pop_control_method, seed=seed)
    trial = objs['trial']
    hamiltonian = objs['hamiltonian']
    walkers = objs['walkers']
    nblocks = objs['nblocks']

    legacy_system = legacy_objs['system']
    legacy_trial = legacy_objs['trial']
    legacy_hamiltonian = legacy_objs['hamiltonian']
    legacy_walkers = legacy_objs['walkers']

    print(f'\nwalkers.buff_names : \n{walkers.buff_names}\n')
    print(f'legacy_walkers.buff_names : \n{legacy_walkers.walkers[0].buff_names}\n')
    
    # Test equivalence of `stack.__dict__`.
    for iw in range(walkers.nwalkers):
        for key in walkers.stack[iw].__dict__.keys():  
            legacy_key = key
            if key == 'nstack': legacy_key = 'stack_size'
            elif key == 'nslice': legacy_key = 'ntime_slices'
            elif key == 'stack_length': legacy_key = 'nbins'
            val = walkers.stack[iw].__dict__[key]
            legacy_val = legacy_walkers.walkers[iw].stack.__dict__[legacy_key]

            if isinstance(val, (int, float, complex, numpy.float64, numpy.complex128, numpy.ndarray)):
                numpy.testing.assert_allclose(val, legacy_val)

            elif isinstance(val, str):
                assert val == legacy_val

            else: continue

    # Test equivalence of `stack.get_buffer()` buffer size.
    for iw in range(walkers.nwalkers):
        buff = walkers.stack[iw].get_buffer()
        legacy_buff = legacy_walkers.walkers[iw].stack.get_buffer()
        assert buff.size == legacy_buff.size
    
    # Test equivalence of `walker.buff_names`.
    assert walkers.buff_names == legacy_walkers.walkers[0].buff_names

    # Test equivalence of `walkers.get_buffer()` buffer size with the given
    # `walkers.buff_names`.
    for iw in range(walkers.nwalkers):
        for name in walkers.buff_names:
            val = walkers.__dict__[name]
            legacy_val = legacy_walkers.walkers[iw].__dict__[name]

            if isinstance(val, (int, float, complex, numpy.float64, numpy.complex128, numpy.ndarray)):
                numpy.testing.assert_allclose(val, legacy_val)

            elif isinstance(val, str):
                assert val == legacy_val

            else: continue
    
    # 
    for iw in range(walkers.nwalkers):
        buff = get_buffer(walkers, iw)
        legacy_buff = legacy_walkers.walkers[iw].get_buffer()
        assert buff.size == legacy_buff.size


if __name__ == "__main__":
    test_get_buffer()

